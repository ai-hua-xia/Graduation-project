"""
CARLA data collection tuned for action-visual correlation.

Goal: make large action changes correspond to large visual changes.
"""

import argparse
import random
import shutil
import time
from pathlib import Path

import carla
import cv2
import numpy as np
from tqdm import tqdm


CARLA_HOST = "localhost"
CARLA_PORT = 2000
TARGET_MAP = "Town01"

IMAGE_WIDTH = 256
IMAGE_HEIGHT = 256
FPS = 20

DEFAULT_EPISODES = 100
FRAMES_PER_EPISODE = 100
DEFAULT_SAMPLE_INTERVAL = 6  # ticks per saved frame

DEFAULT_DATA_DIR = Path("data/raw_action_corr_v2")

DEFAULT_MIN_CORR = 0.3
DEFAULT_MIN_SPEED = 0.6
DEFAULT_MAX_BAD_FRAMES = 10
DEFAULT_MAX_ATTEMPTS = 0
DEFAULT_MAX_LAG = 3
DEFAULT_MIN_DELTA = 0.02
DEFAULT_LOW_QUANTILE = 0.3
DEFAULT_HIGH_QUANTILE = 0.7

DEFAULT_LOW_THROTTLE = 0.1
DEFAULT_HIGH_THROTTLE = 0.85
DEFAULT_MID_STEER = 0.4
DEFAULT_HARD_STEER = 0.7
DEFAULT_SEGMENT_LEN = 3
DEFAULT_SEGMENT_JITTER = 1
DEFAULT_STRAIGHT_RATIO = 0.3
DEFAULT_MID_RATIO = 0.3
DEFAULT_HARD_RATIO = 0.4
DEFAULT_WARMUP_TICKS = 10
DEFAULT_TURN_THRESHOLD = 0.3
DEFAULT_HARD_THRESHOLD = 0.6
DEFAULT_MIN_TURN_RATIO = 0.6
DEFAULT_MIN_MID_RATIO = 0.2
DEFAULT_MIN_HARD_RATIO = 0.2


def setup_camera(world, vehicle):
    bp_lib = world.get_blueprint_library()
    camera_bp = bp_lib.find("sensor.camera.rgb")
    camera_bp.set_attribute("image_size_x", str(IMAGE_WIDTH))
    camera_bp.set_attribute("image_size_y", str(IMAGE_HEIGHT))
    camera_bp.set_attribute("fov", "90")

    camera_transform = carla.Transform(
        carla.Location(x=1.5, z=1.8),
        carla.Rotation(pitch=-10),
    )

    camera = world.spawn_actor(camera_bp, camera_transform, attach_to=vehicle)
    return camera


def clear_world(world):
    actors = world.get_actors()
    for actor in actors.filter("vehicle.*"):
        actor.destroy()
    for actor in actors.filter("walker.*"):
        actor.destroy()
    for actor in actors.filter("controller.*"):
        actor.destroy()


def check_image_quality(image, min_brightness=20, max_black_ratio=0.7):
    mean_brightness = image.mean()
    black_pixels = np.sum(image < 30) / image.size
    if mean_brightness < min_brightness:
        return False, f"dark({mean_brightness:.1f})"
    if black_pixels > max_black_ratio:
        return False, f"black({black_pixels:.1%})"
    return True, "ok"


def calculate_speed(vehicle):
    vel = vehicle.get_velocity()
    return (vel.x ** 2 + vel.y ** 2 + vel.z ** 2) ** 0.5


def compute_visual_change(img1, img2):
    diff = np.abs(img1.astype(np.float32) - img2.astype(np.float32))
    return float((diff > 30).mean())


def compute_lagged_corr(action_changes, visual_changes, max_lag):
    best_corr = -1.0
    best_lag = 0
    for lag in range(max_lag + 1):
        if lag == 0:
            a = action_changes
            v = visual_changes
        else:
            a = action_changes[:-lag]
            v = visual_changes[lag:]
        if len(a) < 2 or len(v) < 2:
            continue
        if np.std(a) < 1e-6 or np.std(v) < 1e-6:
            continue
        corr = float(np.corrcoef(a, v)[0, 1])
        if corr > best_corr:
            best_corr = corr
            best_lag = lag
    return best_corr, best_lag


def compute_delta(action_changes, visual_changes, low_q, high_q):
    if len(action_changes) == 0:
        return 0.0
    low_thresh = np.quantile(action_changes, low_q)
    high_thresh = np.quantile(action_changes, high_q)
    low_mask = action_changes <= low_thresh
    high_mask = action_changes >= high_thresh
    if not np.any(low_mask) or not np.any(high_mask):
        return 0.0
    return float(np.mean(visual_changes[high_mask]) - np.mean(visual_changes[low_mask]))


def compute_action_signal(actions):
    steering = np.abs(actions[:, 0])
    throttle = actions[:, 1]
    return steering * throttle


def parse_episode_number(path):
    name = path.name
    if not name.startswith("episode_"):
        return None
    suffix = name.split("_", 1)[1]
    if not suffix.isdigit():
        return None
    return int(suffix)


def build_action_sequence(
    num_frames,
    segment_len,
    segment_jitter,
    straight_ratio,
    mid_ratio,
    hard_ratio,
    mid_steer,
    hard_steer,
    low_throttle,
    high_throttle,
):
    actions = []
    mid_throttle = (low_throttle + high_throttle) / 2.0
    modes = [
        ("straight", 0.0, low_throttle),
        ("mid_left", -mid_steer, mid_throttle),
        ("mid_right", mid_steer, mid_throttle),
        ("hard_left", -hard_steer, high_throttle),
        ("hard_right", hard_steer, high_throttle),
    ]
    weights = [
        straight_ratio,
        mid_ratio / 2.0,
        mid_ratio / 2.0,
        hard_ratio / 2.0,
        hard_ratio / 2.0,
    ]

    while len(actions) < num_frames:
        mode, steer_base, throttle_base = random.choices(modes, weights=weights, k=1)[0]
        seg_len = max(1, segment_len + random.randint(-segment_jitter, segment_jitter))
        for _ in range(seg_len):
            if len(actions) >= num_frames:
                break
            if mode == "straight":
                steer = random.uniform(-0.02, 0.02)
                throttle = throttle_base + random.uniform(-0.02, 0.02)
            else:
                steer = steer_base + random.uniform(-0.03, 0.03)
                throttle = throttle_base + random.uniform(-0.02, 0.02)
            actions.append([steer, throttle])

    return np.array(actions, dtype=np.float32)


def collect_episode(world, episode_num, spawn_points, args):
    bp_lib = world.get_blueprint_library()
    vehicle_bp = random.choice(bp_lib.filter("vehicle.*"))

    vehicle = None
    random.shuffle(spawn_points)
    for spawn_point in spawn_points[:30]:
        try:
            vehicle = world.spawn_actor(vehicle_bp, spawn_point)
            break
        except RuntimeError:
            continue

    if vehicle is None:
        print(f"  x spawn failed for episode {episode_num}")
        return False

    camera = setup_camera(world, vehicle)

    latest = {"image": None}

    def camera_callback(image):
        array = np.frombuffer(image.raw_data, dtype=np.uint8)
        array = array.reshape((IMAGE_HEIGHT, IMAGE_WIDTH, 4))[:, :, :3]
        latest["image"] = array.copy()

    camera.listen(camera_callback)

    for _ in range(5):
        world.tick()
        time.sleep(0.01)

    for _ in range(args.warmup_ticks):
        control = carla.VehicleControl(
            throttle=float((args.low_throttle + args.high_throttle) / 2.0),
            steer=0.0,
            brake=0.0,
        )
        vehicle.apply_control(control)
        world.tick()
        time.sleep(0.01)

    actions = build_action_sequence(
        FRAMES_PER_EPISODE,
        args.segment_len,
        args.segment_jitter,
        args.straight_ratio,
        args.mid_ratio,
        args.hard_ratio,
        args.mid_steer,
        args.hard_steer,
        args.low_throttle,
        args.high_throttle,
    )

    sampled_images = []
    sampled_actions = []
    sampled_speeds = []
    bad_frames = 0

    pbar = tqdm(range(FRAMES_PER_EPISODE), desc=f"Episode {episode_num}", leave=False)
    for frame_idx in pbar:
        steer, throttle = actions[frame_idx]
        control = carla.VehicleControl(
            throttle=float(throttle),
            steer=float(steer),
            brake=0.0,
        )

        for _ in range(args.sample_interval):
            vehicle.apply_control(control)
            world.tick()
            time.sleep(0.01)

        img = latest["image"]
        if img is None:
            bad_frames += 1
            if bad_frames > args.max_bad_frames:
                break
            continue

        ok, reason = check_image_quality(img)
        if not ok:
            bad_frames += 1
            pbar.set_postfix({"bad": bad_frames, "reason": reason})
            if bad_frames > args.max_bad_frames:
                break
            continue

        sampled_images.append(img)
        sampled_actions.append([steer, throttle])
        sampled_speeds.append(calculate_speed(vehicle))

    camera.stop()
    camera.destroy()
    vehicle.destroy()

    if len(sampled_images) != FRAMES_PER_EPISODE:
        return False

    sampled_actions = np.array(sampled_actions, dtype=np.float32)
    action_changes = np.abs(np.diff(sampled_actions, axis=0)).sum(axis=1)
    visual_changes = np.array(
        [
            compute_visual_change(sampled_images[i], sampled_images[i + 1])
            for i in range(len(sampled_images) - 1)
        ],
        dtype=np.float32,
    )

    action_signal = compute_action_signal(sampled_actions)[:-1]
    corr, best_lag = compute_lagged_corr(action_signal, visual_changes, args.max_lag)
    if corr < -0.5:
        corr = 0.0
    delta = compute_delta(
        action_signal,
        visual_changes,
        args.low_quantile,
        args.high_quantile,
    )
    corr_change, lag_change = compute_lagged_corr(action_changes, visual_changes, args.max_lag)

    avg_speed = float(np.mean(sampled_speeds)) if sampled_speeds else 0.0
    burst_threshold = (args.low_throttle + args.high_throttle) / 2.0
    speed_mask = sampled_actions[:, 1] >= burst_threshold
    if np.any(speed_mask):
        avg_speed_burst = float(np.mean(np.array(sampled_speeds)[speed_mask]))
    else:
        avg_speed_burst = avg_speed

    abs_steer = np.abs(sampled_actions[:, 0])
    turn_ratio = float(np.mean(abs_steer >= args.turn_threshold))
    mid_ratio = float(np.mean((abs_steer >= args.turn_threshold) & (abs_steer < args.hard_threshold)))
    hard_ratio = float(np.mean(abs_steer >= args.hard_threshold))

    print(
        "Quality: corr=%.3f lag=%d delta=%.3f corr_change=%.3f lag_change=%d "
        "speed=%.2f burst=%.2f turn=%.2f mid=%.2f hard=%.2f "
        "(min_corr=%.2f, min_delta=%.2f, min_speed=%.2f)"
        % (
            corr,
            best_lag,
            delta,
            corr_change,
            lag_change,
            avg_speed,
            avg_speed_burst,
            turn_ratio,
            mid_ratio,
            hard_ratio,
            args.min_corr,
            args.min_delta,
            args.min_speed,
        )
    )

    if (
        corr < args.min_corr
        or delta < args.min_delta
        or avg_speed_burst < args.min_speed
        or turn_ratio < args.min_turn_ratio
        or mid_ratio < args.min_mid_ratio
        or hard_ratio < args.min_hard_ratio
    ):
        return False

    episode_dir = Path(args.data_dir) / f"episode_{episode_num:04d}"
    images_dir = episode_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    for i, img in enumerate(sampled_images):
        cv2.imwrite(str(images_dir / f"{i:04d}.png"), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

    np.save(episode_dir / "actions.npy", sampled_actions)

    metadata = {
        "episode_number": episode_num,
        "sample_interval": args.sample_interval,
        "min_corr": args.min_corr,
        "min_speed": args.min_speed,
        "segment_len": args.segment_len,
        "segment_jitter": args.segment_jitter,
        "straight_ratio": args.straight_ratio,
        "mid_ratio": args.mid_ratio,
        "hard_ratio": args.hard_ratio,
        "mid_steer": args.mid_steer,
        "hard_steer": args.hard_steer,
        "low_throttle": args.low_throttle,
        "high_throttle": args.high_throttle,
        "corr": corr,
        "corr_lag": best_lag,
        "corr_delta": delta,
        "corr_change": corr_change,
        "corr_change_lag": lag_change,
        "avg_speed_mps": avg_speed,
        "avg_speed_burst_mps": avg_speed_burst,
        "turn_ratio": turn_ratio,
        "mid_turn_ratio": mid_ratio,
        "hard_turn_ratio": hard_ratio,
    }
    np.save(episode_dir / "metadata.npy", metadata)

    return True


def parse_args():
    parser = argparse.ArgumentParser(description="Collect CARLA dataset with action-visual correlation")
    parser.add_argument("--host", type=str, default=CARLA_HOST)
    parser.add_argument("--port", type=int, default=CARLA_PORT)
    parser.add_argument("--episodes", type=int, default=DEFAULT_EPISODES)
    parser.add_argument("--episode-start", type=int, default=None)
    parser.add_argument("--episode-end", type=int, default=None)
    parser.add_argument("--sample-interval", type=int, default=DEFAULT_SAMPLE_INTERVAL)
    parser.add_argument("--data-dir", type=str, default=str(DEFAULT_DATA_DIR))
    parser.add_argument("--map", type=str, default=TARGET_MAP)
    parser.add_argument("--min-corr", type=float, default=DEFAULT_MIN_CORR)
    parser.add_argument("--min-delta", type=float, default=DEFAULT_MIN_DELTA)
    parser.add_argument("--min-speed", type=float, default=DEFAULT_MIN_SPEED)
    parser.add_argument("--max-bad-frames", type=int, default=DEFAULT_MAX_BAD_FRAMES)
    parser.add_argument("--max-lag", type=int, default=DEFAULT_MAX_LAG)
    parser.add_argument("--low-quantile", type=float, default=DEFAULT_LOW_QUANTILE)
    parser.add_argument("--high-quantile", type=float, default=DEFAULT_HIGH_QUANTILE)
    parser.add_argument("--low-throttle", type=float, default=DEFAULT_LOW_THROTTLE)
    parser.add_argument("--high-throttle", type=float, default=DEFAULT_HIGH_THROTTLE)
    parser.add_argument("--mid-steer", type=float, default=DEFAULT_MID_STEER)
    parser.add_argument("--hard-steer", type=float, default=DEFAULT_HARD_STEER)
    parser.add_argument("--segment-len", type=int, default=DEFAULT_SEGMENT_LEN)
    parser.add_argument("--segment-jitter", type=int, default=DEFAULT_SEGMENT_JITTER)
    parser.add_argument("--straight-ratio", type=float, default=DEFAULT_STRAIGHT_RATIO)
    parser.add_argument("--mid-ratio", type=float, default=DEFAULT_MID_RATIO)
    parser.add_argument("--hard-ratio", type=float, default=DEFAULT_HARD_RATIO)
    parser.add_argument("--warmup-ticks", type=int, default=DEFAULT_WARMUP_TICKS)
    parser.add_argument("--turn-threshold", type=float, default=DEFAULT_TURN_THRESHOLD)
    parser.add_argument("--hard-threshold", type=float, default=DEFAULT_HARD_THRESHOLD)
    parser.add_argument("--min-turn-ratio", type=float, default=DEFAULT_MIN_TURN_RATIO)
    parser.add_argument("--min-mid-ratio", type=float, default=DEFAULT_MIN_MID_RATIO)
    parser.add_argument("--min-hard-ratio", type=float, default=DEFAULT_MIN_HARD_RATIO)
    parser.add_argument("--max-attempts", type=int, default=DEFAULT_MAX_ATTEMPTS)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    data_dir = Path(args.data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    use_range = args.episode_start is not None or args.episode_end is not None
    if use_range:
        if args.episode_start is None or args.episode_end is None:
            print("Both --episode-start and --episode-end are required.")
            return
        if args.episode_start <= 0 or args.episode_end < args.episode_start:
            print("Invalid episode range. Use positive numbers and end >= start.")
            return

    print("=" * 70)
    print("  CARLA data collection - action correlated")
    print("=" * 70)
    if use_range:
        print(f"Episode range: {args.episode_start}-{args.episode_end}")
    else:
        print(f"Episodes: {args.episodes}")
    print(f"Frames per episode: {FRAMES_PER_EPISODE}")
    print(f"Sample interval: {args.sample_interval} ticks")
    print(f"Host: {args.host}  Port: {args.port}")
    print(f"Output dir: {data_dir.resolve()}")
    print("=" * 70)

    client = carla.Client(args.host, args.port)
    client.set_timeout(30.0)
    world = client.get_world()
    if args.map not in world.get_map().name:
        world = client.load_world(args.map)
        time.sleep(3)

    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = 1.0 / FPS
    world.apply_settings(settings)

    world.set_weather(carla.WeatherParameters.ClearNoon)

    spawn_points = world.get_map().get_spawn_points()
    if not spawn_points:
        print("No spawn points found.")
        return

    successful = 0
    failed = 0
    skipped = 0

    existing_numbers = [
        number
        for number in (parse_episode_number(path) for path in data_dir.glob("episode_*"))
        if number is not None
    ]
    next_episode = (max(existing_numbers) + 1) if existing_numbers else 1
    if not use_range and next_episode > 1:
        print(f"Resuming at episode {next_episode}")

    try:
        if use_range:
            targets = list(range(args.episode_start, args.episode_end + 1))
            for idx, episode_num in enumerate(targets, start=1):
                episode_dir = data_dir / f"episode_{episode_num:04d}"
                if episode_dir.exists():
                    if args.overwrite:
                        shutil.rmtree(episode_dir)
                    else:
                        print(f"\nEpisode {episode_num} ({idx}/{len(targets)}) exists, skipping.")
                        skipped += 1
                        continue

                attempts = 0
                while True:
                    if args.max_attempts > 0 and attempts >= args.max_attempts:
                        print(f"  failed: max attempts reached for episode {episode_num}")
                        break

                    attempts += 1
                    clear_world(world)
                    print(f"\nEpisode {episode_num} ({idx}/{len(targets)}) attempt {attempts}")

                    if collect_episode(world, episode_num, spawn_points, args):
                        successful += 1
                        print(f"  ok: episode {episode_num}")
                        break

                    failed += 1
                    print("  rejected, recollecting...")

                if successful > 0 and successful % 10 == 0:
                    print(f"Progress: {successful} episodes")
        else:
            episode_num = next_episode
            while successful < args.episodes:
                clear_world(world)
                print(f"\nEpisode {successful + 1}/{args.episodes} (episode={episode_num})")

                if collect_episode(world, episode_num, spawn_points, args):
                    successful += 1
                    print(f"  ok: episode {episode_num}")
                else:
                    failed += 1
                    print("  rejected, recollecting...")

                episode_num += 1

                if successful > 0 and successful % 10 == 0:
                    print(f"Progress: {successful}/{args.episodes} episodes")

    except KeyboardInterrupt:
        print("Interrupted by user.")

    finally:
        settings = world.get_settings()
        settings.synchronous_mode = False
        world.apply_settings(settings)

    print("=" * 70)
    print("Done.")
    print(f"Successful episodes: {successful}")
    print(f"Failed attempts: {failed}")
    if skipped:
        print(f"Skipped episodes: {skipped}")
    print("=" * 70)


if __name__ == "__main__":
    main()
