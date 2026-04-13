import argparse
import shutil
import subprocess
from pathlib import Path

import cv2
import numpy as np


def wasd_to_action(key: str) -> np.ndarray:
    steering = 0.0
    throttle = 0.55

    key = key.upper()
    if key == "W":
        throttle = 0.65
    elif key == "S":
        throttle = 0.42
    elif key == "A":
        steering = -0.4
    elif key == "D":
        steering = 0.4
    elif key == "Q":
        steering = -0.4
        throttle = 0.65
    elif key == "E":
        steering = 0.4
        throttle = 0.65
    elif key == "N":
        pass

    return np.array([steering, throttle], dtype=np.float32)


def load_actions_from_txt(txt_path: Path) -> np.ndarray:
    actions = []
    with open(txt_path, "r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            if len(line) == 1 and line.upper() in "WASDQEN":
                actions.append(wasd_to_action(line))
                continue
            parts = line.split()
            if len(parts) != 2:
                continue
            try:
                actions.append([float(parts[0]), float(parts[1])])
            except ValueError:
                continue
    if not actions:
        raise ValueError(f"No valid actions found in {txt_path}")
    return np.asarray(actions, dtype=np.float32)


def _expand_actions(actions: np.ndarray, target_len: int) -> np.ndarray:
    if len(actions) >= target_len:
        return actions[:target_len]
    pad = np.repeat(actions[-1][None], target_len - len(actions), axis=0)
    return np.concatenate([actions, pad], axis=0)


def _watermark_lines(watermark: str) -> list[str]:
    if "\n" in watermark:
        return [line.strip() for line in watermark.splitlines() if line.strip()]
    if " / " in watermark:
        return [line.strip() for line in watermark.split(" / ") if line.strip()]
    return [watermark.strip()]


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


CONTROL_CENTER_RAW = 0.40
CONTROL_DELTA_LIMIT = 0.12
STEER_BOX_ACTIVE_THRESHOLD = 0.08


def _format_control_labels(steer: float, throttle: float) -> tuple[str, str]:
    steer_header = f"steer {float(steer):+.2f}"
    throttle_header = f"throttle {float(throttle):.2f}"
    return steer_header, throttle_header


def _draw_box(frame: np.ndarray, x: int, y: int, label: str, active: bool) -> None:
    color = (40, 180, 40) if active else (120, 120, 120)
    fill = (30, 90, 30) if active else (35, 35, 35)
    cv2.rectangle(frame, (x, y), (x + 36, y + 36), fill, thickness=-1)
    cv2.rectangle(frame, (x, y), (x + 36, y + 36), color, thickness=2)
    cv2.putText(
        frame,
        label,
        (x + 10, y + 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )


def draw_control_overlay(frame: np.ndarray, action: np.ndarray) -> np.ndarray:
    out = frame.copy()
    h, w = out.shape[:2]
    steer, throttle = float(action[0]), float(action[1])
    active_left = steer <= -STEER_BOX_ACTIVE_THRESHOLD
    active_right = steer >= STEER_BOX_ACTIVE_THRESHOLD

    panel_w = max(220, int(w * 0.22))
    panel_h = max(110, int(h * 0.18))
    base_x = 16
    base_y = h - panel_h - 16

    overlay = out.copy()
    cv2.rectangle(overlay, (base_x, base_y), (base_x + panel_w, base_y + panel_h), (18, 18, 18), thickness=-1)
    out = cv2.addWeighted(overlay, 0.72, out, 0.28, 0)

    header_y = base_y + 18
    header_font = 0.38
    steer_header, throttle_header = _format_control_labels(steer, throttle)
    cv2.putText(
        out,
        steer_header,
        (base_x + 10, header_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        header_font,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    steer_text_size, _ = cv2.getTextSize(steer_header, cv2.FONT_HERSHEY_SIMPLEX, header_font, 1)
    control_x = min(base_x + 10 + steer_text_size[0] + 14, base_x + panel_w - 98)
    cv2.putText(
        out,
        throttle_header,
        (control_x, header_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        header_font,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )

    box_y = base_y + 28
    _draw_box(out, base_x + 10, box_y, "A", active_left)
    _draw_box(out, base_x + 58, box_y, "D", active_right)

    steer_bar_x = base_x + 110
    steer_bar_y = box_y + 10
    steer_bar_w = panel_w - 125
    cv2.rectangle(out, (steer_bar_x, steer_bar_y), (steer_bar_x + steer_bar_w, steer_bar_y + 14), (70, 70, 70), -1)
    center_x = steer_bar_x + steer_bar_w // 2
    cv2.line(out, (center_x, steer_bar_y - 4), (center_x, steer_bar_y + 18), (180, 180, 180), 1)
    steer_norm = _clamp01((steer + 1.0) / 2.0)
    steer_pos = steer_bar_x + int(round(steer_norm * steer_bar_w))
    cv2.circle(out, (steer_pos, steer_bar_y + 7), 6, (60, 220, 255), -1)

    throttle_bar_x = base_x + 10
    throttle_bar_y = base_y + panel_h - 30
    throttle_bar_w = panel_w - 20
    throttle_norm = _clamp01(throttle)
    cv2.rectangle(out, (throttle_bar_x, throttle_bar_y), (throttle_bar_x + throttle_bar_w, throttle_bar_y + 14), (70, 70, 70), -1)
    fill_w = int(round(throttle_norm * throttle_bar_w))
    cv2.rectangle(
        out,
        (throttle_bar_x, throttle_bar_y),
        (throttle_bar_x + fill_w, throttle_bar_y + 14),
        (70, 210, 90),
        -1,
    )
    cv2.rectangle(out, (throttle_bar_x, throttle_bar_y), (throttle_bar_x + throttle_bar_w, throttle_bar_y + 14), (180, 180, 180), 1)
    return out


def draw_reference_watermark(frame: np.ndarray, watermark: str) -> np.ndarray:
    out = frame.copy()
    lines = _watermark_lines(watermark)
    line_height = 26
    box_w = max(240, max(len(line) for line in lines) * 11)
    box_h = 18 + line_height * len(lines)

    overlay = out.copy()
    cv2.rectangle(overlay, (8, 8), (8 + box_w, 8 + box_h), (15, 15, 15), thickness=-1)
    out = cv2.addWeighted(overlay, 0.72, out, 0.28, 0)

    for idx, line in enumerate(lines):
        cv2.putText(
            out,
            line,
            (18, 34 + idx * line_height),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
    return out


def apply_progressive_corruption(
    frame: np.ndarray,
    progress: float,
    crash_ramp: float,
    crash_start: float,
    prev_frame: np.ndarray | None = None,
) -> np.ndarray:
    if crash_ramp <= 0:
        return frame

    if progress <= crash_start:
        return frame

    severity = _clamp01(((progress - crash_start) / max(1e-6, 1.0 - crash_start)) * crash_ramp)
    if severity <= 0:
        return frame

    out = frame.astype(np.float32)
    h, w = out.shape[:2]

    # 1) Global low-resolution collapse: downsample then upsample.
    lowres_ratio = max(0.10, 1.0 - 0.78 * severity)
    low_w = max(12, int(round(w * lowres_ratio)))
    low_h = max(12, int(round(h * lowres_ratio)))
    reduced = cv2.resize(out, (low_w, low_h), interpolation=cv2.INTER_LINEAR)
    out = cv2.resize(reduced, (w, h), interpolation=cv2.INTER_LINEAR)

    # 2) Global blur that strengthens over time.
    kernel = int(round(3 + severity * 16))
    if kernel % 2 == 0:
        kernel += 1
    out = cv2.GaussianBlur(out, (kernel, kernel), sigmaX=0.6 + 2.8 * severity)

    # 3) Mild geometric warping to make structures melt and drift.
    yy, xx = np.indices((h, w), dtype=np.float32)
    amp_x = severity * w * 0.018
    amp_y = severity * h * 0.012
    freq_x = 18.0 + 30.0 * (1.0 - severity)
    freq_y = 14.0 + 24.0 * (1.0 - severity)
    map_x = xx + amp_x * np.sin(yy / freq_y + progress * 7.0)
    map_y = yy + amp_y * np.sin(xx / freq_x + progress * 5.0)
    out = cv2.remap(out, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)

    # 4) Temporal sticking: blend with the previous already-degraded frame.
    if prev_frame is not None:
        prev = prev_frame.astype(np.float32)
        prev = cv2.GaussianBlur(prev, (kernel, kernel), sigmaX=0.4 + 2.0 * severity)
        ghost_alpha = 0.18 + 0.52 * severity
        out = cv2.addWeighted(out, 1.0 - ghost_alpha, prev, ghost_alpha, 0)

    # 5) Wash out contrast slightly so details feel merged instead of noisy.
    gray = cv2.cvtColor(np.clip(out, 0, 255).astype(np.uint8), cv2.COLOR_BGR2GRAY).astype(np.float32)
    gray3 = np.repeat(gray[..., None], 3, axis=2)
    desat = 0.10 + 0.28 * severity
    fade = 0.02 + 0.10 * severity
    out = out * (1.0 - desat) + gray3 * desat
    out = out * (1.0 - fade) + 255.0 * fade

    return np.clip(out, 0, 255).astype(np.uint8)


def _scale_frame(frame: np.ndarray, scale: float) -> np.ndarray:
    if scale <= 1.0:
        return frame
    h, w = frame.shape[:2]
    out_w = max(1, int(round(w * scale)))
    out_h = max(1, int(round(h * scale)))
    return cv2.resize(frame, (out_w, out_h), interpolation=cv2.INTER_CUBIC)


def _open_writer(path: Path, fps: float, frame_size: tuple[int, int]) -> cv2.VideoWriter:
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, frame_size)
    if not writer.isOpened():
        raise RuntimeError(f"Could not open video writer for {path}")
    return writer


def _maybe_transcode(temp_path: Path, output_path: Path) -> None:
    if shutil.which("ffmpeg") is None:
        temp_path.replace(output_path)
        return

    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(temp_path),
        "-c:v",
        "libx264",
        "-preset",
        "medium",
        "-crf",
        "23",
        "-pix_fmt",
        "yuv420p",
        "-loglevel",
        "error",
        str(output_path),
    ]
    try:
        subprocess.run(cmd, check=True, capture_output=True)
        temp_path.unlink()
    except (subprocess.CalledProcessError, FileNotFoundError):
        temp_path.replace(output_path)


def render_reference_video(
    input_video: str | Path,
    output_video: str | Path,
    action_txt: str | Path,
    watermark: str = "REFERENCE / NOT MODEL OUTPUT",
    crash_ramp: float = 0.0,
    crash_start: float = 0.55,
    scale: float = 1.0,
    fps: float | None = None,
) -> None:
    input_video = Path(input_video)
    output_video = Path(output_video)
    action_txt = Path(action_txt)

    actions = load_actions_from_txt(action_txt)

    cap = cv2.VideoCapture(str(input_video))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open input video {input_video}")

    src_fps = cap.get(cv2.CAP_PROP_FPS)
    video_fps = fps if fps is not None else (src_fps if src_fps and src_fps > 0 else 10.0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    actions = _expand_actions(actions, max(frame_count, 1))
    scaled_width = max(1, int(round(width * max(1.0, scale))))
    scaled_height = max(1, int(round(height * max(1.0, scale))))

    output_video.parent.mkdir(parents=True, exist_ok=True)
    temp_path = output_video.parent / f"temp_{output_video.name}"
    writer = _open_writer(temp_path, float(video_fps), (scaled_width, scaled_height))

    frame_idx = 0
    prev_corrupted = None
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        progress = 0.0 if frame_count <= 1 else frame_idx / float(frame_count - 1)
        frame = apply_progressive_corruption(
            frame,
            progress,
            crash_ramp,
            crash_start,
            prev_frame=prev_corrupted,
        )
        prev_corrupted = frame.copy()
        frame = _scale_frame(frame, scale)
        frame = draw_reference_watermark(frame, watermark)
        frame = draw_control_overlay(frame, actions[frame_idx])
        writer.write(frame)
        frame_idx += 1

    cap.release()
    writer.release()
    _maybe_transcode(temp_path, output_video)


def main() -> None:
    parser = argparse.ArgumentParser(description="Overlay watermark and control panel on a reference video.")
    parser.add_argument("--input-video", required=True, help="Path to the source mp4 video")
    parser.add_argument("--output-video", required=True, help="Path to the output mp4 video")
    parser.add_argument("--action-txt", required=True, help="Action text file in WASD or numeric format")
    parser.add_argument(
        "--watermark",
        default="REFERENCE / NOT MODEL OUTPUT",
        help="Explicit watermark shown on the video",
    )
    parser.add_argument(
        "--crash-ramp",
        type=float,
        default=0.0,
        help="Progressive crash/corruption ramp factor from 0.0 to 1.0",
    )
    parser.add_argument(
        "--crash-start",
        type=float,
        default=0.55,
        help="Fraction of the video after which corruption starts to appear",
    )
    parser.add_argument("--scale", type=float, default=1.0, help="Scale output video size, e.g. 2.0 for 2x")
    parser.add_argument("--fps", type=float, default=None, help="Override output fps")
    args = parser.parse_args()

    render_reference_video(
        input_video=args.input_video,
        output_video=args.output_video,
        action_txt=args.action_txt,
        watermark=args.watermark,
        crash_ramp=args.crash_ramp,
        crash_start=args.crash_start,
        scale=args.scale,
        fps=args.fps,
    )


if __name__ == "__main__":
    main()
