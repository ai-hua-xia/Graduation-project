import argparse
from pathlib import Path
import sys

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from utils.reference_video_overlay import (
    _maybe_transcode,
    _open_writer,
    _scale_frame,
    draw_control_overlay,
)


def compute_switch_indices(start_idx: int, split_seconds: float, fps: float) -> tuple[int, int]:
    split_frame_count = max(1, int(round(split_seconds * fps)))
    return split_frame_count, start_idx + split_frame_count


def compose_mixed_frames(
    reference_frames: list[np.ndarray],
    predicted_frames: list[np.ndarray],
    split_frame_count: int,
) -> list[np.ndarray]:
    if len(reference_frames) < split_frame_count:
        raise ValueError(
            f"Need at least {split_frame_count} reference frames, got {len(reference_frames)}"
        )
    return list(reference_frames[:split_frame_count]) + list(predicted_frames)


def build_prediction_inputs(
    tokens: np.ndarray,
    actions: np.ndarray,
    switch_idx: int,
    predict_frame_count: int,
    context_frames: int,
) -> tuple[np.ndarray, np.ndarray]:
    if switch_idx < context_frames:
        raise ValueError(
            f"switch_idx {switch_idx} must be >= context_frames {context_frames}"
        )
    if switch_idx > len(tokens):
        raise ValueError(f"switch_idx {switch_idx} exceeds token length {len(tokens)}")

    action_start = switch_idx - context_frames
    action_end = switch_idx + predict_frame_count
    if action_end > len(actions):
        raise ValueError(
            f"Need actions up to {action_end}, but only {len(actions)} are available"
        )

    context_tokens = tokens[switch_idx - context_frames : switch_idx].copy()
    action_override = actions[action_start:action_end].copy()
    return context_tokens, action_override


def render_mixed_video_from_frames(
    reference_frames: list[np.ndarray],
    predicted_frames: list[np.ndarray],
    actions: np.ndarray,
    output_video: str | Path,
    split_frame_count: int,
    scale: float = 1.0,
    fps: float = 10.0,
) -> None:
    output_video = Path(output_video)
    mixed_frames = compose_mixed_frames(reference_frames, predicted_frames, split_frame_count)
    if not mixed_frames:
        raise ValueError("No frames to render")

    actions = np.asarray(actions, dtype=np.float32)
    if len(actions) < len(mixed_frames):
        raise ValueError(
            f"Need at least {len(mixed_frames)} actions for overlay, got {len(actions)}"
        )

    first_frame = _scale_frame(cv2.cvtColor(mixed_frames[0], cv2.COLOR_RGB2BGR), scale)
    output_video.parent.mkdir(parents=True, exist_ok=True)
    temp_path = output_video.parent / f"temp_{output_video.name}"
    writer = _open_writer(temp_path, float(fps), (first_frame.shape[1], first_frame.shape[0]))

    for idx, frame in enumerate(mixed_frames):
        bgr_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        bgr_frame = _scale_frame(bgr_frame, scale)
        bgr_frame = draw_control_overlay(bgr_frame, actions[idx])
        writer.write(bgr_frame)

    writer.release()
    _maybe_transcode(temp_path, output_video)


def _decode_reference_frames(vqvae, tokens: np.ndarray, start_idx: int, frame_count: int, device: str):
    from utils.generate_videos import tokens_to_image

    frames = []
    for offset in range(frame_count):
        frame_idx = start_idx + offset
        if frame_idx >= len(tokens):
            break
        frames.append(tokens_to_image(vqvae, tokens[frame_idx], device))
    return frames


def _predict_frames_from_context(
    vqvae,
    world_model,
    context_tokens: np.ndarray,
    action_override: np.ndarray,
    predict_frame_count: int,
    device: str,
    temperature: float,
):
    import torch

    from utils.generate_videos import tokens_to_image

    context_tokens = context_tokens.copy()
    context_frames = world_model.context_frames
    pred_frames = []

    with torch.no_grad():
        memory = None
        use_memory = getattr(world_model, "use_memory", False)
        for t in range(predict_frame_count):
            context_tensor = torch.from_numpy(context_tokens).long().unsqueeze(0).to(device)
            action_seq = action_override[t : t + context_frames]
            action_tensor = torch.from_numpy(action_seq).float().unsqueeze(0).to(device)

            if use_memory:
                logits, memory = world_model(
                    context_tensor, action_tensor, memory=memory, return_memory=True
                )
            else:
                logits = world_model(context_tensor, action_tensor)

            if temperature > 0:
                probs = torch.softmax(logits / temperature, dim=-1)
                pred_tokens = torch.multinomial(
                    probs.view(-1, probs.size(-1)), 1
                ).view(logits.shape[:-1])
            else:
                pred_tokens = torch.argmax(logits, dim=-1)

            pred_tokens = pred_tokens.squeeze(0).cpu().numpy()
            pred_frames.append(tokens_to_image(vqvae, pred_tokens, device))

            h = w = int(np.sqrt(len(pred_tokens)))
            context_tokens = np.roll(context_tokens, -1, axis=0)
            context_tokens[-1] = pred_tokens.reshape(h, w)

    return pred_frames


def render_switch_to_prediction_video(
    vqvae_checkpoint: str | Path,
    world_model_checkpoint: str | Path,
    token_file: str | Path,
    output_video: str | Path,
    start_idx: int,
    split_seconds: float,
    predict_seconds: float,
    fps: float = 10.0,
    scale: float = 1.0,
    device: str = "cuda",
    temperature: float = 0.0,
) -> tuple[int, int]:
    from utils.generate_videos import load_models

    data = np.load(token_file)
    tokens = data["tokens"]
    actions = data["actions"]
    num_embeddings = int(tokens.max()) + 1

    vqvae, world_model = load_models(
        str(vqvae_checkpoint),
        str(world_model_checkpoint),
        device,
        num_embeddings=num_embeddings,
    )

    split_frame_count, switch_idx = compute_switch_indices(start_idx, split_seconds, fps)
    predict_frame_count = max(1, int(round(predict_seconds * fps)))
    total_frame_count = split_frame_count + predict_frame_count

    if start_idx < 0 or start_idx + total_frame_count > len(tokens):
        raise ValueError(
            f"Need token frames in [{start_idx}, {start_idx + total_frame_count}), "
            f"but token length is {len(tokens)}"
        )

    reference_frames = _decode_reference_frames(vqvae, tokens, start_idx, split_frame_count, device)
    context_tokens, action_override = build_prediction_inputs(
        tokens=tokens,
        actions=actions,
        switch_idx=switch_idx,
        predict_frame_count=predict_frame_count,
        context_frames=world_model.context_frames,
    )
    predicted_frames = _predict_frames_from_context(
        vqvae=vqvae,
        world_model=world_model,
        context_tokens=context_tokens,
        action_override=action_override,
        predict_frame_count=predict_frame_count,
        device=device,
        temperature=temperature,
    )
    overlay_actions = actions[start_idx : start_idx + total_frame_count]
    render_mixed_video_from_frames(
        reference_frames=reference_frames,
        predicted_frames=predicted_frames,
        actions=overlay_actions,
        output_video=output_video,
        split_frame_count=split_frame_count,
        scale=scale,
        fps=fps,
    )
    return split_frame_count, switch_idx


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Render a video with a reference segment followed by switch-point prediction."
    )
    parser.add_argument(
        "--vqvae-checkpoint",
        default="checkpoints/vqvae/vqvae_action_corr_f8/best.pth",
        help="Path to the VQ-VAE checkpoint",
    )
    parser.add_argument(
        "--world-model-checkpoint",
        default="checkpoints/wm/world_model_f8_adaln_aux/best.pth",
        help="Path to the world model checkpoint",
    )
    parser.add_argument(
        "--token-file",
        default="data/tokens_action_corr_f8/tokens_actions.npz",
        help="Path to the tokens/actions npz file",
    )
    parser.add_argument("--output-video", required=True, help="Output mp4 path")
    parser.add_argument("--start-idx", type=int, required=True, help="Reference segment start index")
    parser.add_argument("--split-seconds", type=float, default=5.0, help="Reference segment duration")
    parser.add_argument("--predict-seconds", type=float, default=5.0, help="Prediction segment duration")
    parser.add_argument("--fps", type=float, default=10.0, help="Output fps")
    parser.add_argument("--scale", type=float, default=2.0, help="Output scale factor")
    parser.add_argument("--device", default="cuda", help="Inference device")
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature for prediction; 0.0 uses greedy decoding",
    )
    args = parser.parse_args()

    split_frame_count, switch_idx = render_switch_to_prediction_video(
        vqvae_checkpoint=args.vqvae_checkpoint,
        world_model_checkpoint=args.world_model_checkpoint,
        token_file=args.token_file,
        output_video=args.output_video,
        start_idx=args.start_idx,
        split_seconds=args.split_seconds,
        predict_seconds=args.predict_seconds,
        fps=args.fps,
        scale=args.scale,
        device=args.device,
        temperature=args.temperature,
    )
    print(
        f"Saved mixed video to {args.output_video} "
        f"(split_frame_count={split_frame_count}, switch_idx={switch_idx})"
    )


if __name__ == "__main__":
    main()
