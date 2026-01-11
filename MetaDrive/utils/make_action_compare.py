import argparse
import os

import cv2
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Make action compare video (base vs action vs diff)")
    parser.add_argument("--base", required=True, help="Base (no-action) video")
    parser.add_argument("--action", required=True, help="Action video")
    parser.add_argument("--actions", required=True, help="Action txt file (same as generation)")
    parser.add_argument("--frames-per-action", type=int, default=15)
    parser.add_argument("--fps", type=float, default=0.0, help="Override FPS (0 uses video FPS)")
    parser.add_argument("--out", default="dream_compare_grid.mp4")
    parser.add_argument("--out-diff", default="dream_compare_diff.mp4")
    parser.add_argument("--heatmap", default="inferno", choices=["inferno", "jet", "hot"])
    parser.add_argument("--max-frames", type=int, default=0, help="Limit frames (0 = full)")
    return parser.parse_args()


def parse_action_file(path: str, frames_per_action: int):
    steer_frames = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            token = line.strip().lower()
            if not token or token.startswith("#"):
                continue
            if token in ("space", "brake"):
                steer = 0.0
            else:
                letters = set(token)
                steer = 0.0
                if "a" in letters:
                    steer -= 1.0
                if "d" in letters:
                    steer += 1.0
            for _ in range(frames_per_action):
                steer_frames.append(steer)

    cumulative = []
    total = 0.0
    for s in steer_frames:
        total += s
        cumulative.append(total)
    max_cum = max(1.0, max(abs(x) for x in cumulative) if cumulative else 1.0)
    return steer_frames, cumulative, max_cum


def draw_curve_overlay(frame: np.ndarray, idx: int, steer_frames, cumulative, max_cum) -> None:
    h, w = frame.shape[:2]
    panel_h = min(70, int(h * 0.28))
    y0 = h - panel_h
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, y0), (w, h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.45, frame, 0.55, 0.0, frame)

    window = min(140, len(steer_frames))
    start = max(0, idx - window + 1)
    end = min(len(steer_frames), start + window)
    denom = max(1, end - start - 1)

    steer_color = (255, 220, 96)
    cum_color = (69, 166, 255)

    prev = None
    for i in range(start, end):
        x = int(((i - start) / denom) * (w - 1))
        steer = steer_frames[i] if i < len(steer_frames) else 0.0
        y = int(y0 + panel_h / 2 - steer * (panel_h * 0.35))
        if prev is not None:
            cv2.line(frame, prev, (x, y), steer_color, 2)
        prev = (x, y)

    prev = None
    for i in range(start, end):
        x = int(((i - start) / denom) * (w - 1))
        cum = (cumulative[i] if i < len(cumulative) else 0.0) / max_cum
        y = int(y0 + panel_h / 2 - cum * (panel_h * 0.35))
        if prev is not None:
            cv2.line(frame, prev, (x, y), cum_color, 2)
        prev = (x, y)

    steer_val = steer_frames[idx] if idx < len(steer_frames) else 0.0
    cum_val = cumulative[idx] if idx < len(cumulative) else 0.0
    cv2.putText(frame, f"steer: {steer_val:.2f}", (10, y0 + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (240, 240, 240), 1)
    cv2.putText(frame, f"cum: {cum_val:.1f}", (10, y0 + 36), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (240, 240, 240), 1)


def heatmap_from_diff(diff: np.ndarray, cmap: str) -> np.ndarray:
    gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    if cmap == "jet":
        heat = cv2.applyColorMap(gray, cv2.COLORMAP_JET)
    elif cmap == "hot":
        heat = cv2.applyColorMap(gray, cv2.COLORMAP_HOT)
    else:
        heat = cv2.applyColorMap(gray, cv2.COLORMAP_INFERNO)
    return heat


def convert_h264(raw_path: str, out_path: str) -> None:
    cmd = (
        f"ffmpeg -y -i {raw_path} -vcodec libx264 -pix_fmt yuv420p "
        f"-loglevel error {out_path}"
    )
    exit_code = os.system(cmd)
    if exit_code == 0 and os.path.exists(raw_path):
        os.remove(raw_path)


def main() -> None:
    args = parse_args()
    if not os.path.exists(args.base):
        raise FileNotFoundError(args.base)
    if not os.path.exists(args.action):
        raise FileNotFoundError(args.action)
    if not os.path.exists(args.actions):
        raise FileNotFoundError(args.actions)

    steer_frames, cumulative, max_cum = parse_action_file(args.actions, args.frames_per_action)

    cap_a = cv2.VideoCapture(args.base)
    cap_b = cv2.VideoCapture(args.action)
    fps = args.fps if args.fps > 0 else cap_a.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 10.0

    width = int(cap_a.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap_a.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if width == 0 or height == 0:
        width, height = 256, 256

    raw_grid = "_tmp_compare_grid.mp4"
    raw_diff = "_tmp_compare_diff.mp4"
    grid_writer = cv2.VideoWriter(raw_grid, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width * 3, height))
    diff_writer = cv2.VideoWriter(raw_diff, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))

    idx = 0
    max_frames = args.max_frames if args.max_frames > 0 else None
    while True:
        if max_frames is not None and idx >= max_frames:
            break
        ret_a, frame_a = cap_a.read()
        ret_b, frame_b = cap_b.read()
        if not ret_a or not ret_b:
            break
        if frame_a.shape != frame_b.shape:
            frame_b = cv2.resize(frame_b, (frame_a.shape[1], frame_a.shape[0]))
        diff = cv2.absdiff(frame_a, frame_b)
        heat = heatmap_from_diff(diff, args.heatmap)
        draw_curve_overlay(heat, idx, steer_frames, cumulative, max_cum)

        cv2.putText(frame_a, "No-Action", (12, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (240, 240, 240), 2)
        cv2.putText(frame_b, "Action", (12, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (240, 240, 240), 2)
        cv2.putText(heat, "Diff + Curve", (12, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (240, 240, 240), 2)

        grid = np.concatenate([frame_a, frame_b, heat], axis=1)
        grid_writer.write(grid)
        diff_writer.write(heat)

        if idx % 20 == 0:
            print(f"Processed {idx} frames...")
        idx += 1

    cap_a.release()
    cap_b.release()
    grid_writer.release()
    diff_writer.release()

    convert_h264(raw_grid, args.out)
    convert_h264(raw_diff, args.out_diff)
    print(f"Saved: {args.out}")
    print(f"Saved: {args.out_diff}")


if __name__ == "__main__":
    main()
