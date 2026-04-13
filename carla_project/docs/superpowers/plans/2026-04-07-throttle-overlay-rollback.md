# Throttle Overlay Rollback Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Keep the compact no-title header, but revert the bottom throttle bar to a one-sided left-to-right fill and remove the redundant `raw` label.

**Architecture:** Only touch the shared `draw_control_overlay(...)` helper and the tests that describe its layout/semantics. Preserve the current compact `steer / throttle` header while rolling back the bottom bar rendering from centered bidirectional to simple left-to-right fill based on the raw throttle value.

**Tech Stack:** Python, NumPy, OpenCV, pytest

---

### Task 1: Update tests to describe the rollback

**Files:**
- Modify: `tests/test_reference_video_overlay.py`
- Test: `tests/test_reference_video_overlay.py`

- [ ] **Step 1: Write the failing tests**

```python
def test_draw_control_overlay_uses_one_sided_throttle_fill():
    ...

def test_format_control_labels_removes_raw_label_and_uses_raw_throttle_value():
    ...
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest -q tests/test_reference_video_overlay.py`
Expected: FAIL because the current overlay still uses a centered bar and returns a raw label

- [ ] **Step 3: Write minimal implementation**

```python
throttle_norm = clamp(raw_throttle)
throttle_header = f"throttle {raw_throttle:.2f}"
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest -q tests/test_reference_video_overlay.py`
Expected: PASS

### Task 2: Verify and regenerate the video artifact

**Files:**
- Modify: `utils/reference_video_overlay.py`
- Verify: `outputs/videos/retrieval_demo/predict_only_idx105680_10s_controls.mp4`

- [ ] **Step 1: Run the targeted suite**

Run: `pytest -q tests/test_reference_video_overlay.py tests/test_switch_to_prediction_video.py`
Expected: PASS

- [ ] **Step 2: Regenerate the pure-prediction overlay video**

Re-run the existing GPU-backed generation flow for:
`outputs/videos/retrieval_demo/predict_only_idx105680_10s_controls.mp4`

- [ ] **Step 3: Verify output metadata and do a visual spot-check**

Confirm:
- `100` frames
- `10.0` fps
- `512x512`
- no title
- compact `steer / throttle` header
- no `raw` label
- one-sided throttle fill
