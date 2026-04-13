from pathlib import Path
import subprocess
import sys

import cv2
import numpy as np
import pytest

sys.path.append(str(Path(__file__).resolve().parents[1]))

from utils.switch_to_prediction_video import (
    build_prediction_inputs,
    compose_mixed_frames,
    compute_switch_indices,
    render_mixed_video_from_frames,
)


def _make_flat_frame(size=72, value=0):
    return np.full((size, size, 3), value, dtype=np.uint8)


def _read_video(path: Path):
    cap = cv2.VideoCapture(str(path))
    assert cap.isOpened(), f"failed to open {path}"
    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(frame)
    cap.release()
    return frames


def test_compose_mixed_frames_switches_source_at_requested_frame():
    reference_frames = [_make_flat_frame(value=10 + i) for i in range(5)]
    predicted_frames = [_make_flat_frame(value=200 + i) for i in range(5)]

    mixed_frames = compose_mixed_frames(reference_frames, predicted_frames, split_frame_count=5)

    assert len(mixed_frames) == 10
    assert int(mixed_frames[4][0, 0, 0]) == 14
    assert int(mixed_frames[5][0, 0, 0]) == 200


def test_build_prediction_inputs_aligns_actions_to_switch_context():
    actions = np.arange(40, dtype=np.float32).reshape(20, 2)
    tokens = np.arange(20 * 4, dtype=np.int64).reshape(20, 2, 2)

    context_tokens, action_override = build_prediction_inputs(
        tokens=tokens,
        actions=actions,
        switch_idx=8,
        predict_frame_count=5,
        context_frames=3,
    )

    assert np.array_equal(context_tokens, tokens[5:8])
    assert np.array_equal(action_override, actions[5:13])


def test_build_prediction_inputs_rejects_out_of_bounds_windows():
    actions = np.arange(24, dtype=np.float32).reshape(12, 2)
    tokens = np.arange(12 * 4, dtype=np.int64).reshape(12, 2, 2)

    with pytest.raises(ValueError):
        build_prediction_inputs(
            tokens=tokens,
            actions=actions,
            switch_idx=2,
            predict_frame_count=8,
            context_frames=3,
        )


def test_compute_switch_indices_uses_split_seconds_and_fps():
    split_frame_count, switch_idx = compute_switch_indices(105680, split_seconds=5, fps=10)

    assert split_frame_count == 50
    assert switch_idx == 105730


def test_render_mixed_video_writes_scaled_frames_with_controls(tmp_path):
    output_video = tmp_path / "mixed.mp4"
    reference_frames = [_make_flat_frame(value=30 + i) for i in range(5)]
    predicted_frames = [_make_flat_frame(value=180 + i) for i in range(5)]
    actions = np.array(
        [
            [0.0, 0.55],
            [-0.4, 0.55],
            [0.4, 0.55],
            [0.0, 0.42],
            [0.0, 0.65],
            [0.0, 0.55],
            [-0.3, 0.55],
            [0.3, 0.55],
            [0.0, 0.42],
            [0.0, 0.65],
        ],
        dtype=np.float32,
    )

    render_mixed_video_from_frames(
        reference_frames=reference_frames,
        predicted_frames=predicted_frames,
        actions=actions,
        output_video=output_video,
        split_frame_count=5,
        scale=2.0,
        fps=10,
    )

    assert output_video.exists()

    output_frames = _read_video(output_video)
    assert len(output_frames) == 10
    assert output_frames[0].shape[:2] == (144, 144)

    # Controls should alter the lower-left region.
    control_crop = output_frames[1][-70:-5, 5:100]
    assert float(np.mean(control_crop)) > 5.0

    # No top-left watermark/label should be introduced.
    top_left = output_frames[0][0:32, 0:100]
    assert float(np.mean(top_left)) < 40.0

    # Frame source should switch after the split.
    assert int(output_frames[4][15, 15, 0]) < 80
    assert int(output_frames[5][15, 15, 0]) > 140


def test_switch_to_prediction_cli_help_runs_from_repo_root():
    repo_root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [sys.executable, "utils/switch_to_prediction_video.py", "--help"],
        cwd=repo_root,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    assert "--start-idx" in result.stdout
