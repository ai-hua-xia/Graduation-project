# Switch-To-Prediction Video Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a small utility that renders one video with 5 seconds of reference frames followed by 5 seconds of world-model prediction starting from the 5-second switch point, using post-switch real actions and no source label overlay.

**Architecture:** Keep the feature isolated in a new utility script so the existing `generate_videos.py` flow stays unchanged. Reuse the control-overlay and output-writing helpers from `utils/reference_video_overlay.py`, and add focused tests that verify switch-frame indexing and mixed rendering behavior with synthetic inputs.

**Tech Stack:** Python, NumPy, OpenCV, pytest, existing VQ-VAE/world-model utilities in this repo

---

### Task 1: Add failing tests for switch-indexed mixing

**Files:**
- Create: `tests/test_switch_to_prediction_video.py`
- Modify: none
- Test: `tests/test_switch_to_prediction_video.py`

- [ ] **Step 1: Write the failing test**

```python
def test_mix_frames_switches_source_at_requested_frame():
    ref = [np.full((4, 4, 3), 10, dtype=np.uint8) for _ in range(5)]
    pred = [np.full((4, 4, 3), 200, dtype=np.uint8) for _ in range(5)]
    mixed = compose_mixed_frames(ref, pred, split_frame_count=5)
    assert np.all(mixed[4] == 10)
    assert np.all(mixed[5] == 200)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest -q tests/test_switch_to_prediction_video.py`
Expected: FAIL because the new utility/functions do not exist yet

- [ ] **Step 3: Write minimal implementation**

```python
def compose_mixed_frames(reference_frames, predicted_frames, split_frame_count):
    return list(reference_frames[:split_frame_count]) + list(predicted_frames)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest -q tests/test_switch_to_prediction_video.py`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add tests/test_switch_to_prediction_video.py utils/switch_to_prediction_video.py
git commit -m "test: cover switch-to-prediction video mixing"
```

### Task 2: Add failing tests for post-switch action slicing and bounds validation

**Files:**
- Modify: `tests/test_switch_to_prediction_video.py`
- Create: `utils/switch_to_prediction_video.py`
- Test: `tests/test_switch_to_prediction_video.py`

- [ ] **Step 1: Write the failing test**

```python
def test_build_prediction_inputs_aligns_actions_to_switch_context():
    actions = np.arange(40, dtype=np.float32).reshape(20, 2)
    tokens = np.arange(20 * 4, dtype=np.int64).reshape(20, 2, 2)
    context, action_override = build_prediction_inputs(
        tokens=tokens,
        actions=actions,
        switch_idx=8,
        predict_frame_count=5,
        context_frames=3,
    )
    assert np.array_equal(context, tokens[5:8])
    assert np.array_equal(action_override, actions[5:13])
```

```python
def test_build_prediction_inputs_rejects_out_of_bounds_windows():
    with pytest.raises(ValueError):
        build_prediction_inputs(tokens=tokens, actions=actions, switch_idx=2, predict_frame_count=50, context_frames=3)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest -q tests/test_switch_to_prediction_video.py`
Expected: FAIL because `build_prediction_inputs` is missing

- [ ] **Step 3: Write minimal implementation**

```python
def build_prediction_inputs(tokens, actions, switch_idx, predict_frame_count, context_frames):
    context_tokens = tokens[switch_idx - context_frames:switch_idx].copy()
    action_override = actions[switch_idx - context_frames:switch_idx + predict_frame_count].copy()
    return context_tokens, action_override
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest -q tests/test_switch_to_prediction_video.py`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add tests/test_switch_to_prediction_video.py utils/switch_to_prediction_video.py
git commit -m "feat: slice prediction inputs from switch point"
```

### Task 3: Implement the rendering entry point

**Files:**
- Create: `utils/switch_to_prediction_video.py`
- Modify: `tests/test_switch_to_prediction_video.py`
- Test: `tests/test_switch_to_prediction_video.py`

- [ ] **Step 1: Write the failing integration-style test**

```python
def test_render_mixed_video_writes_scaled_frames_with_controls(tmp_path):
    ...
    render_mixed_video_from_frames(...)
    assert output_video.exists()
    assert len(output_frames) == 10
    assert output_frames[0].shape[:2] == (192, 192)
    assert top_left_crop_matches_source_content
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest -q tests/test_switch_to_prediction_video.py`
Expected: FAIL because the renderer is incomplete

- [ ] **Step 3: Write minimal implementation**

```python
def render_mixed_video_from_frames(...):
    mixed = compose_mixed_frames(...)
    for idx, frame in enumerate(mixed):
        frame = _scale_frame(frame, scale)
        frame = draw_control_overlay(frame, actions[idx])
        writer.write(frame)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest -q tests/test_switch_to_prediction_video.py`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add tests/test_switch_to_prediction_video.py utils/switch_to_prediction_video.py
git commit -m "feat: render mixed reference and prediction video"
```

### Task 4: Add the real model-backed CLI path and generate the demo

**Files:**
- Modify: `utils/switch_to_prediction_video.py`
- Test: `tests/test_switch_to_prediction_video.py`

- [ ] **Step 1: Write a failing test for argument/default computation if needed**

```python
def test_compute_switch_indices_uses_split_seconds_and_fps():
    split_frame_count, switch_idx = compute_switch_indices(105680, split_seconds=5, fps=10)
    assert split_frame_count == 50
    assert switch_idx == 105730
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest -q tests/test_switch_to_prediction_video.py`
Expected: FAIL because helper or CLI glue is missing

- [ ] **Step 3: Write minimal implementation**

```python
def compute_switch_indices(start_idx, split_seconds, fps):
    split_frame_count = int(round(split_seconds * fps))
    return split_frame_count, start_idx + split_frame_count
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest -q tests/test_switch_to_prediction_video.py`
Expected: PASS

- [ ] **Step 5: Generate the real demo and verify outputs**

Run: `pytest -q tests/test_switch_to_prediction_video.py`
Then run the new CLI against:
- `--start-idx 105680`
- `--split-seconds 5`
- `--predict-seconds 5`
- `--output-video outputs/videos/retrieval_demo/retrieval_01_idx105680_switch_predict.mp4`
- `--scale 2.0`
- the existing VQ-VAE checkpoint
- the existing world-model checkpoint
- `data/tokens_action_corr_f8/tokens_actions.npz`

Expected:
- output mp4 exists
- first 5 seconds match reference content
- last 5 seconds come from prediction started at the 5-second switch point
