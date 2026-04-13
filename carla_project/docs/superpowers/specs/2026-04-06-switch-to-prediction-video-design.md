# Switch-To-Prediction Video Design

## Goal

Generate a single demo video that starts from a real reference clip at `start_idx=105680`, plays the first 5 seconds from the retrieved frames, then switches to pure world-model prediction for the next 5 seconds. The prediction segment must start from the actual 5-second switch point, using the real context frames immediately before the cut and the correctly aligned real action window required by the current world-model inference contract. The output keeps the existing left-bottom control overlay, scales the video up, and does not show any top-left source labels.

## Scope

This design only covers demo-video generation and composition. It does not modify training, model architecture, dataset generation, or evaluation code.

## Existing Context

- `utils/generate_videos.py` already supports pure prediction from a fixed `start_idx`.
- `utils/reference_video_overlay.py` already provides action loading, control rendering, scaling, and output writing helpers.
- The current need is not simple video concatenation. The second half must be predicted from the switch point at 5 seconds, not from the original clip start.

## Proposed Approach

Add a new utility that:

1. Loads tokens/actions and the trained VQ-VAE + world model.
2. Computes:
   - `switch_frame = split_seconds * fps`
   - `switch_idx = start_idx + switch_frame`
3. Builds prediction context from the real token window ending at `switch_idx`.
4. Uses the real action sequence aligned to the switch context to autoregressively predict the next `predict_seconds * fps` frames.
5. Decodes the first `split_seconds * fps` reference frames directly from tokens.
6. Composes a single output video:
   - first segment = decoded reference frames
   - second segment = predicted frames
   - full-length control overlay
   - no source watermark/label
   - optional output scaling

## File Boundaries

- New utility module/script for the switch-to-prediction flow.
- Reuse overlay helpers from `utils/reference_video_overlay.py` rather than duplicating drawing/output code.
- Add focused tests covering the switch-point indexing and source switching behavior.

## Action Alignment

The existing world-model inference code predicts the next frame from:

- `context_tokens[t : t + context_frames]`
- `actions[t : t + context_frames]`

That means the first predicted frame after the cut at `switch_idx` is conditioned on:

- context tokens from `[switch_idx - context_frames, switch_idx)`
- actions from `[switch_idx - context_frames, switch_idx)`

So the action override passed into autoregressive prediction must start at `switch_idx - context_frames`, not at `switch_idx`. The override length remains `predict_frame_count + context_frames`, so later prediction steps continue consuming the post-cut real actions in order.

## Timeline FPS

This utility is defined against the demo-video timeline, not the raw data-collection FPS. For this feature, `fps` means:

- the output video FPS
- the frame-to-seconds conversion used to compute `split_frame_count`

So with the intended demo default of `fps=10`, a 5-second cut means `split_frame_count = 50` and `switch_idx = start_idx + 50`.

## Data Flow

1. Read token/action arrays.
2. Decode reference frames for `[start_idx, start_idx + split_frame_count)`.
3. Extract prediction context from `[switch_idx - context_frames, switch_idx)`.
4. Feed action slices from `[switch_idx - context_frames, switch_idx + predict_frame_count)` into the world model override buffer.
5. Decode predicted token outputs into RGB/BGR frames.
6. Apply scale + control overlay frame-by-frame and write the final mp4.

## Error Handling

- Fail if `start_idx` or `switch_idx` would make the context or prediction window exceed available tokens/actions.
- Fail if required checkpoints/files are missing.
- Fail if the output video writer cannot be opened.

## Testing

- Verify that the mixed-video renderer switches from reference frames to prediction frames at the requested frame index.
- Verify that the prediction action sequence starts at the switch point rather than the original start index.
- Reuse small synthetic arrays / test doubles so tests do not require loading real checkpoints.
