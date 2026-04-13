from pathlib import Path
import sys

import cv2
import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[1]))

from utils.reference_video_overlay import _format_control_labels, apply_progressive_corruption, draw_control_overlay, render_reference_video


def _make_checker_frame(size=96, shift=0):
    yy, xx = np.indices((size, size))
    pattern = (((xx // 4) + (yy // 4) + shift) % 2) * 255
    frame = np.stack(
        [
            pattern,
            np.roll(pattern, 2, axis=1),
            np.roll(pattern, 2, axis=0),
        ],
        axis=-1,
    ).astype(np.uint8)
    return frame


def _write_video(path: Path, frames, fps=8):
    h, w = frames[0].shape[:2]
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    assert writer.isOpened(), f"failed to open writer for {path}"
    for frame in frames:
        writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    writer.release()


def _read_video(path: Path):
    cap = cv2.VideoCapture(str(path))
    assert cap.isOpened(), f"failed to open video {path}"
    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    return frames


def _laplacian_var(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_32F).var())


def _resize_like(frame, target_shape):
    h, w = target_shape[:2]
    return cv2.resize(frame, (w, h), interpolation=cv2.INTER_CUBIC)


def _control_bar_geometry(frame_shape):
    h, w = frame_shape[:2]
    panel_w = max(220, int(w * 0.22))
    panel_h = max(110, int(h * 0.18))
    base_x = 16
    base_y = h - panel_h - 16
    bar_x = base_x + 10
    bar_y = base_y + panel_h - 30
    bar_w = panel_w - 20
    center_x = bar_x + bar_w // 2
    return bar_x, bar_y, bar_w, center_x


def _pixel(frame, x, y):
    return frame[y, x].astype(int)


def _bright_pixel_count(frame, x0, y0, x1, y1, threshold=120):
    crop = frame[y0:y1, x0:x1]
    return int(np.sum(np.max(crop, axis=2) > threshold))


def test_apply_progressive_corruption_becomes_blurry_and_sticky():
    frame = _make_checker_frame(size=128, shift=0)
    prev_frame = _make_checker_frame(size=128, shift=3)

    early = apply_progressive_corruption(
        frame, progress=0.10, crash_ramp=1.0, crash_start=0.35, prev_frame=prev_frame
    )
    late = apply_progressive_corruption(
        frame, progress=0.95, crash_ramp=1.0, crash_start=0.35, prev_frame=prev_frame
    )

    early_delta = np.mean(np.abs(early.astype(np.int16) - frame.astype(np.int16)))
    assert early_delta < 1.0

    frame_sharpness = _laplacian_var(frame)
    late_sharpness = _laplacian_var(late)
    assert late_sharpness < frame_sharpness * 0.4

    prev_distance = np.mean(np.abs(frame.astype(np.int16) - prev_frame.astype(np.int16)))
    late_prev_distance = np.mean(np.abs(late.astype(np.int16) - prev_frame.astype(np.int16)))
    assert late_prev_distance < prev_distance


def test_render_reference_video_adds_left_controls_scale_and_progressive_corruption(tmp_path):
    input_video = tmp_path / "input.mp4"
    output_video = tmp_path / "output.mp4"
    action_txt = tmp_path / "actions.txt"

    frames = [_make_checker_frame(shift=i) for i in range(6)]
    _write_video(input_video, frames)
    action_txt.write_text("0.0 0.65\n-0.4 0.55\n0.4 0.55\n0.0 0.42\n0.0 0.55\n0.0 0.65\n", encoding="utf-8")

    render_reference_video(
        input_video,
        output_video,
        action_txt,
        watermark="REFERENCE / NOT MODEL OUTPUT",
        crash_ramp=1.0,
        crash_start=0.35,
        scale=2.0,
    )

    assert output_video.exists()

    input_frames = _read_video(input_video)
    output_frames = _read_video(output_video)

    assert len(output_frames) == len(input_frames) == len(frames)
    assert output_frames[0].shape[0] == input_frames[0].shape[0] * 2
    assert output_frames[0].shape[1] == input_frames[0].shape[1] * 2

    resized_input0 = _resize_like(input_frames[0], output_frames[0].shape)
    first_diff = np.mean(np.abs(output_frames[0].astype(np.int16) - resized_input0.astype(np.int16)))
    assert first_diff > 2.0

    left_bottom_crop = np.s_[-120:-10, 10:160]
    panel_energy = np.mean(np.abs(output_frames[1][left_bottom_crop].astype(np.int16)))
    assert panel_energy > 10.0

    center_crop = np.s_[40:150, 40:150]
    first_sharpness = _laplacian_var(output_frames[0][center_crop])
    last_sharpness = _laplacian_var(output_frames[-1][center_crop])
    assert last_sharpness < first_sharpness


def test_draw_control_overlay_uses_one_sided_throttle_fill():
    frame = np.zeros((320, 320, 3), dtype=np.uint8)
    bar_x, bar_y, bar_w, center_x = _control_bar_geometry(frame.shape)
    sample_y = bar_y + 7
    sample_x = bar_x + int(round(bar_w * 0.45))

    smaller = draw_control_overlay(frame, np.array([0.0, 0.30], dtype=np.float32))
    larger = draw_control_overlay(frame, np.array([0.0, 0.60], dtype=np.float32))

    smaller_px = _pixel(smaller, sample_x, sample_y)
    larger_px = _pixel(larger, sample_x, sample_y)

    assert smaller_px[1] - smaller_px[0] < 30
    assert larger_px[1] - larger_px[0] > 60


def test_draw_control_overlay_clamps_one_sided_throttle_fill_to_bar_edge():
    frame = np.zeros((320, 320, 3), dtype=np.uint8)
    bar_x, bar_y, bar_w, center_x = _control_bar_geometry(frame.shape)
    sample_y = bar_y + 7
    right_edge_x = bar_x + bar_w - 20

    rendered = draw_control_overlay(frame, np.array([0.0, 1.20], dtype=np.float32))
    sample_px = _pixel(rendered, right_edge_x, sample_y)

    assert sample_px[1] - sample_px[0] > 60


def test_draw_control_overlay_left_side_stays_unfilled_for_small_throttle():
    frame = np.zeros((320, 320, 3), dtype=np.uint8)
    bar_x, bar_y, bar_w, center_x = _control_bar_geometry(frame.shape)
    sample_y = bar_y + 7
    sample_x = bar_x + int(round(bar_w * 0.75))

    rendered = draw_control_overlay(frame, np.array([0.0, 0.20], dtype=np.float32))
    sample_px = _pixel(rendered, sample_x, sample_y)

    assert sample_px[1] - sample_px[0] < 30


def test_draw_control_overlay_uses_compact_steer_control_header():
    frame = np.zeros((320, 320, 3), dtype=np.uint8)
    _, _, _, _ = _control_bar_geometry(frame.shape)
    rendered = draw_control_overlay(frame, np.array([-0.10, 0.40], dtype=np.float32))

    panel_w = max(220, int(frame.shape[1] * 0.22))
    panel_h = max(110, int(frame.shape[0] * 0.18))
    base_x = 16
    base_y = frame.shape[0] - panel_h - 16

    old_title_bright = _bright_pixel_count(rendered, base_x + 8, base_y + 6, base_x + 100, base_y + 24)
    compact_header_bright = _bright_pixel_count(
        rendered,
        base_x + 105,
        base_y + 6,
        base_x + panel_w - 8,
        base_y + 24,
    )

    assert old_title_bright < 220
    assert compact_header_bright > 35


def test_draw_control_overlay_activates_ad_boxes_for_point_one_steer():
    frame = np.zeros((320, 320, 3), dtype=np.uint8)
    panel_w = max(220, int(frame.shape[1] * 0.22))
    panel_h = max(110, int(frame.shape[0] * 0.18))
    base_x = 16
    base_y = frame.shape[0] - panel_h - 16
    box_y = base_y + 28
    a_x = base_x + 10
    d_x = base_x + 58
    sample_y = box_y + 6
    sample_a_x = a_x + 6
    sample_d_x = d_x + 6

    left_turn = draw_control_overlay(frame, np.array([-0.10, 0.45], dtype=np.float32))
    right_turn = draw_control_overlay(frame, np.array([0.10, 0.45], dtype=np.float32))

    left_a_px = _pixel(left_turn, sample_a_x, sample_y)
    left_d_px = _pixel(left_turn, sample_d_x, sample_y)
    right_a_px = _pixel(right_turn, sample_a_x, sample_y)
    right_d_px = _pixel(right_turn, sample_d_x, sample_y)

    assert left_a_px[1] - left_a_px[0] > 30
    assert left_d_px[1] - left_d_px[0] < 10
    assert right_a_px[1] - right_a_px[0] < 10
    assert right_d_px[1] - right_d_px[0] > 30


def test_format_control_labels_uses_raw_throttle_value_without_raw_label():
    steer_header, throttle_header = _format_control_labels(-0.10, 0.47)

    assert steer_header == "steer -0.10"
    assert throttle_header == "throttle 0.47"
