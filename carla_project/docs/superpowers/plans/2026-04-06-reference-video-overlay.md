# Reference Video Overlay Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a standalone post-processing utility that converts an existing reference video into a clearly labeled demo with watermark, WASD overlay, and optional progressive blur.

**Architecture:** Keep the feature isolated from the world-model generation path. Add one utility module for frame processing and one focused test that creates a tiny synthetic video, runs the processor, and checks that overlays are applied and blur increases over time.

**Tech Stack:** Python, OpenCV, NumPy, pytest

---

### Task 1: Add the failing test

**Files:**
- Create: `tests/test_reference_video_overlay.py`
- Test: `tests/test_reference_video_overlay.py`

- [ ] **Step 1: Write the failing test**
- [ ] **Step 2: Run test to verify it fails**

### Task 2: Implement the standalone postprocessor

**Files:**
- Create: `utils/reference_video_overlay.py`
- Modify: `utils/__init__.py`
- Test: `tests/test_reference_video_overlay.py`

- [ ] **Step 1: Write minimal implementation**
- [ ] **Step 2: Run test to verify it passes**

### Task 3: Smoke-check CLI usage

**Files:**
- Modify: `utils/reference_video_overlay.py`

- [ ] **Step 1: Run a small smoke check on a real video path**
- [ ] **Step 2: Confirm output file is created**
