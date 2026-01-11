import argparse
import os
import random

import cv2
import numpy as np
from metadrive.envs.metadrive_env import MetaDriveEnv
from metadrive.component.sensors.rgb_camera import RGBCamera


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect rich-action MetaDrive data")
    parser.add_argument("--dataset", default="dataset_rich_actions")
    parser.add_argument("--episodes", type=int, default=60)
    parser.add_argument("--steps", type=int, default=400)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--start-seed", type=int, default=3000)
    parser.add_argument("--map", type=int, default=7)
    parser.add_argument("--traffic-density", type=float, default=0.15)
    parser.add_argument("--random-traffic", action="store_true", default=True)
    parser.add_argument("--seed", type=int, default=42)

    # Action ranges (align with inference clamp)
    parser.add_argument("--steer-max", type=float, default=0.25)
    parser.add_argument("--throttle-max", type=float, default=0.6)
    parser.add_argument("--throttle-min", type=float, default=-0.6)

    # Segment length range
    parser.add_argument("--segment-min", type=int, default=12)
    parser.add_argument("--segment-max", type=int, default=30)

    # Smooth action changes
    parser.add_argument("--smooth-steer-step", type=float, default=0.06)
    parser.add_argument("--smooth-throttle-step", type=float, default=0.08)

    # Action mode weights
    parser.add_argument("--weight-straight", type=float, default=0.35)
    parser.add_argument("--weight-turn", type=float, default=0.45)
    parser.add_argument("--weight-brake", type=float, default=0.20)
    return parser.parse_args()


def sample_target_action(
    rng: np.random.Generator,
    steer_max: float,
    throttle_max: float,
    throttle_min: float,
    weight_straight: float,
    weight_turn: float,
    weight_brake: float,
):
    weights = np.array([weight_straight, weight_turn, weight_brake], dtype=np.float32)
    weights = weights / max(weights.sum(), 1e-6)
    mode = rng.choice(["straight", "turn", "brake"], p=weights)

    if mode == "straight":
        steer = rng.uniform(-0.06, 0.06)
        throttle = rng.uniform(0.25, throttle_max)
    elif mode == "turn":
        sign = rng.choice([-1.0, 1.0])
        steer = sign * rng.uniform(0.15, steer_max)
        throttle = rng.uniform(0.15, min(0.8 * throttle_max, throttle_max))
    else:
        steer = rng.uniform(-0.12, 0.12)
        throttle = rng.uniform(throttle_min, -0.15)

    return np.array([steer, throttle], dtype=np.float32), mode


def apply_action_smoothing(current: np.ndarray, target: np.ndarray, max_delta: np.ndarray):
    delta = target - current
    delta = np.clip(delta, -max_delta, max_delta)
    return current + delta


def collect_data() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    random.seed(args.seed)

    os.makedirs(os.path.join(args.dataset, "images"), exist_ok=True)

    env_config = {
        "use_render": False,
        "image_observation": True,
        "image_on_cuda": False,
        "window_size": (args.image_size, args.image_size),
        "stack_size": 1,
        "num_scenarios": args.episodes,
        "start_seed": args.start_seed,
        "map": args.map,
        "traffic_density": args.traffic_density,
        "random_traffic": args.random_traffic,
        "random_agent_model": False,
        "vehicle_config": {
            "image_source": "rgb_camera",
            "random_color": True,
        },
        "sensors": {
            "rgb_camera": (RGBCamera, args.image_size, args.image_size),
        },
    }

    env = MetaDriveEnv(env_config)
    print("🚗 环境初始化完成，开始采集富动作数据...")

    all_actions = []
    global_step = 0
    max_delta = np.array([args.smooth_steer_step, args.smooth_throttle_step], dtype=np.float32)
    clamp_min = np.array([-args.steer_max, args.throttle_min], dtype=np.float32)
    clamp_max = np.array([args.steer_max, args.throttle_max], dtype=np.float32)

    for episode in range(args.episodes):
        obs, info = env.reset()
        current_action = np.array([0.0, 0.0], dtype=np.float32)
        target_action, mode = sample_target_action(
            rng,
            args.steer_max,
            args.throttle_max,
            args.throttle_min,
            args.weight_straight,
            args.weight_turn,
            args.weight_brake,
        )
        segment_left = int(rng.integers(args.segment_min, args.segment_max + 1))

        for step in range(args.steps):
            if segment_left <= 0:
                target_action, mode = sample_target_action(
                    rng,
                    args.steer_max,
                    args.throttle_max,
                    args.throttle_min,
                    args.weight_straight,
                    args.weight_turn,
                    args.weight_brake,
                )
                segment_left = int(rng.integers(args.segment_min, args.segment_max + 1))

            current_action = apply_action_smoothing(current_action, target_action, max_delta)
            current_action = np.clip(current_action, clamp_min, clamp_max)

            next_obs, reward, terminated, truncated, info = env.step(current_action.tolist())
            raw_image = next_obs["image"]
            if raw_image.ndim == 4:
                raw_image = raw_image[..., -1]
            if raw_image.dtype != np.uint8:
                image_uint8 = (raw_image * 255).clip(0, 255).astype(np.uint8)
            else:
                image_uint8 = raw_image
            image_bgr = cv2.cvtColor(image_uint8, cv2.COLOR_RGB2BGR)

            img_filename = os.path.join(args.dataset, "images", f"img_{global_step:05d}.png")
            cv2.imwrite(img_filename, image_bgr)
            all_actions.append(current_action.copy())
            global_step += 1
            segment_left -= 1

            if (step + 1) % 50 == 0:
                print(
                    f"Episode {episode + 1}/{args.episodes} | Step {step + 1} | "
                    f"Mode: {mode} | Saved: {img_filename}"
                )

            if terminated or truncated:
                print(f"   ⚠️ 场景 {episode + 1} 结束 (撞车或超时)")
                break

    actions_np = np.array(all_actions, dtype=np.float32)
    np.save(os.path.join(args.dataset, "actions.npy"), actions_np)
    env.close()
    print("✅ 采集完成！")


if __name__ == "__main__":
    collect_data()
