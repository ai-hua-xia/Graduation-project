# Centered Throttle Overlay Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the current one-sided bottom control bar with a centered bidirectional control bar using raw `0.40` as the zero point, including signed delta/raw text, center marker, `-0.10 / 0 / +0.10` ticks, and distinct positive/negative fill colors, then regenerate the pure-prediction demo video with the updated panel.

**Architecture:** Keep the change localized to the shared `draw_control_overlay(...)` helper in `utils/reference_video_overlay.py` so every existing call site gets the new visualization without branching. Add focused rendering tests that prove signed position ordering, left/right directionality, clamp behavior, tick/marker presence, text formatting, color separation, and absence of any new top-left label.

**Tech Stack:** Python, NumPy, OpenCV, pytest

---

### Task 1: Add failing tests for centered control-bar semantics

**Files:**
- Modify: `tests/test_reference_video_overlay.py`
- Test: `tests/test_reference_video_overlay.py`

- [ ] **Step 1: Write the failing tests**

```python
def test_draw_control_overlay_places_larger_positive_value_farther_right():
    ...

def test_draw_control_overlay_places_below_center_value_on_left():
    ...

def test_draw_control_overlay_clamps_out_of_range_value_to_bar_edge():
    ...

def test_draw_control_overlay_shows_centered_labels_and_ticks():
    ...
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest -q tests/test_reference_video_overlay.py`
Expected: FAIL because the current bar is still one-sided and does not encode signed delta around `0.40`, nor the centered label/tick treatment required by the spec

- [ ] **Step 3: Write minimal implementation**

```python
control_center = 0.40
control_min = -0.12
control_max = 0.12
delta = np.clip(throttle - control_center, control_min, control_max)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest -q tests/test_reference_video_overlay.py`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add tests/test_reference_video_overlay.py utils/reference_video_overlay.py
git commit -m "feat: center throttle overlay around neutral value"
```

### Task 2: Preserve existing mixed/pure-video rendering behavior

**Files:**
- Modify: `tests/test_switch_to_prediction_video.py`
- Test: `tests/test_switch_to_prediction_video.py`

- [ ] **Step 1: Add or update a regression test**

```python
def test_render_mixed_video_keeps_left_bottom_panel_and_no_top_label(tmp_path):
    ...
```

- [ ] **Step 2: Run test to verify it fails if assumptions are broken**

Run: `pytest -q tests/test_switch_to_prediction_video.py`
Expected: PASS if current behavior is already preserved; otherwise FAIL and adjust helper usage

- [ ] **Step 3: Make the minimal integration fix if needed**

```python
# Only if the new overlay breaks an existing assumption
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest -q tests/test_switch_to_prediction_video.py`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add tests/test_switch_to_prediction_video.py utils/switch_to_prediction_video.py
git commit -m "test: preserve overlay behavior in mixed video outputs"
```

### Task 3: Verify and regenerate the pure-prediction demo

**Files:**
- Modify: `utils/reference_video_overlay.py`
- Verify: `outputs/videos/retrieval_demo/predict_only_idx105680_10s_controls.mp4`

- [ ] **Step 1: Run the targeted test suite**

Run: `pytest -q tests/test_reference_video_overlay.py tests/test_switch_to_prediction_video.py`
Expected: PASS

- [ ] **Step 2: Regenerate the pure-prediction control video**

Run the same GPU-backed generation flow used previously, writing:
`outputs/videos/retrieval_demo/predict_only_idx105680_10s_controls.mp4`

Exact command:

```bash
bash -lc 'source ~/miniconda3/etc/profile.d/conda.sh && conda activate voyager && python - <<\"PY\"
from pathlib import Path
import cv2
import numpy as np
from utils.generate_videos import load_models
from utils.switch_to_prediction_video import _predict_frames_from_context
from utils.reference_video_overlay import _maybe_transcode, _open_writer, _scale_frame, draw_control_overlay

token_file = Path(\"data/tokens_action_corr_f8/tokens_actions.npz\")
vqvae_ckpt = Path(\"checkpoints/vqvae/vqvae_action_corr_f8/best.pth\")
wm_ckpt = Path(\"checkpoints/wm/world_model_f8_adaln_aux/best.pth\")
output_video = Path(\"outputs/videos/retrieval_demo/predict_only_idx105680_10s_controls.mp4\")
start_idx = 105680
predict_frame_count = 100
fps = 10
scale = 2.0

data = np.load(token_file)
tokens = data[\"tokens\"]
actions = data[\"actions\"]
num_embeddings = int(tokens.max()) + 1
vqvae, world_model = load_models(str(vqvae_ckpt), str(wm_ckpt), \"cuda\", num_embeddings=num_embeddings)
context_frames = world_model.context_frames
context_tokens = tokens[start_idx:start_idx + context_frames].copy()
action_override = actions[start_idx:start_idx + context_frames + predict_frame_count].copy()
pred_frames = _predict_frames_from_context(vqvae, world_model, context_tokens, action_override, predict_frame_count, \"cuda\", 0.0)
video_actions = actions[start_idx:start_idx + predict_frame_count]
first = _scale_frame(cv2.cvtColor(pred_frames[0], cv2.COLOR_RGB2BGR), scale)
output_video.parent.mkdir(parents=True, exist_ok=True)
temp_path = output_video.parent / f\"temp_{output_video.name}\"
writer = _open_writer(temp_path, float(fps), (first.shape[1], first.shape[0]))
for idx, frame in enumerate(pred_frames):
    bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    bgr = _scale_frame(bgr, scale)
    bgr = draw_control_overlay(bgr, video_actions[idx])
    writer.write(bgr)
writer.release()
_maybe_transcode(temp_path, output_video)
PY'
```

- [ ] **Step 3: Verify output metadata**

Run a metadata check to confirm:
- `100` frames
- `10.0` fps
- `512x512`

- [ ] **Step 4: Do a visual spot-check on one extracted frame**

Run a single-frame extraction and confirm:
- `steer ...`
- `control +/-0.xx`
- `raw 0.xx`
- visible center marker
- visible `-0.10 / 0 / +0.10` ticks
- positive/negative fill colors differ

- [ ] **Step 4: Commit**

```bash
git add utils/reference_video_overlay.py tests/test_reference_video_overlay.py tests/test_switch_to_prediction_video.py
git commit -m "feat: improve control overlay readability"
```
