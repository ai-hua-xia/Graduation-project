import os
import sys
import time

import cv2
import numpy as np
import torch
import torch.nn.functional as F

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from train.train_vqvae_256 import VQVAE, IMAGE_SIZE
from train.train_world_model import WorldModelGPT, TOKENS_PER_FRAME, BLOCK_SIZE
from train.train_snow_decoder_noise_hr import SnowOverlayNet

# ================= 配置 =================
VQVAE_PATH = "checkpoints_vqvae_256/vqvae_256_ep99.pth"
WORLD_MODEL_PATH = "checkpoints_world_model/world_model_ep99.pth"
SNOW_CKPT_PATH = "snow/decoder_snow_noise_hr_ep19.pth"

DATA_PATH = "dataset_v2_complex/tokens_actions_vqvae_16x16.npz"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
STEPS_TO_DREAM = 50
TEMPERATURE = 1.2
TOP_K = 5

SNOW_STATIC = True
SNOW_SEED = 42
SNOW_OVERLAY_SCALE = None

OUTPUT_VIDEO = "dream_result.mp4"
# =======================================


def generate_snow_mask(height: int, width: int, rng: np.random.Generator) -> np.ndarray:
    mask = np.zeros((height, width), dtype=np.float32)
    num_flakes = int(height * width * rng.uniform(0.0006, 0.0012))
    for _ in range(num_flakes):
        x = int(rng.integers(0, width))
        y = int(rng.integers(0, height))
        r = int(rng.integers(1, 3))
        cv2.circle(mask, (x, y), r, 1.0, -1)
    if rng.random() < 0.6:
        for _ in range(num_flakes // 4):
            x = int(rng.integers(0, width))
            y = int(rng.integers(0, height))
            length = int(rng.integers(5, 15))
            cv2.line(mask, (x, y), (x + length, y + length), 1.0, 1)
    alpha = rng.uniform(0.2, 0.35)
    mask = np.clip(mask, 0, 1) * alpha
    return mask


def load_models():
    print("⏳ Loading VQ-VAE...")
    vqvae = VQVAE().to(DEVICE)
    vqvae.load_state_dict(torch.load(VQVAE_PATH, map_location=DEVICE)["model"])
    vqvae.eval()

    print(f"⏳ Loading World Model from {WORLD_MODEL_PATH}...")
    gpt = WorldModelGPT().to(DEVICE)
    checkpoint = torch.load(WORLD_MODEL_PATH, map_location=DEVICE)
    gpt.load_state_dict(checkpoint["model"])
    gpt.eval()

    overlay_net = None
    overlay_scale = 0.5
    if SNOW_CKPT_PATH:
        print(f"⏳ Loading Snow HR Decoder from {SNOW_CKPT_PATH}...")
        ckpt = torch.load(SNOW_CKPT_PATH, map_location=DEVICE)
        overlay_hidden = ckpt.get("overlay_hidden", 64)
        overlay_net = SnowOverlayNet(hidden=overlay_hidden).to(DEVICE)
        overlay_net.load_state_dict(ckpt["overlay"], strict=True)
        overlay_net.eval()
        overlay_scale = ckpt.get("overlay_scale", overlay_scale)
        if "decoder" in ckpt:
            vqvae.decoder.load_state_dict(ckpt["decoder"], strict=True)
    if SNOW_OVERLAY_SCALE is not None:
        overlay_scale = SNOW_OVERLAY_SCALE
    return vqvae, gpt, overlay_net, overlay_scale


def decode_indices(vqvae, indices, overlay_net=None, snow_mask=None, overlay_scale=0.5):
    with torch.no_grad():
        indices_tensor = torch.LongTensor(indices).unsqueeze(0).to(DEVICE)
        z_q = vqvae.quantizer.embedding(indices_tensor)
        z_q = z_q.permute(0, 3, 1, 2)
        recon = vqvae.decoder(z_q)
        if overlay_net is not None and snow_mask is not None:
            overlay = overlay_net(recon, snow_mask)
            overlay = torch.tanh(overlay) * overlay_scale
            recon = torch.clamp(recon + overlay, 0.0, 1.0)
        img = recon[0].cpu().permute(1, 2, 0).numpy()
        img = np.clip(img, 0, 1) * 255
        return img.astype(np.uint8)


def sample_next_token(logits, temperature=1.0, top_k=None):
    logits = logits[:, -1, :] / temperature
    if top_k is not None:
        v, _ = torch.topk(logits, top_k)
        logits[logits < v[:, [-1]]] = -float("Inf")
    probs = F.softmax(logits, dim=-1)
    idx = torch.multinomial(probs, num_samples=1)
    return idx


def main():
    vqvae, gpt, overlay_net, overlay_scale = load_models()

    print("🌱 Loading Seed Data...")
    data = np.load(DATA_PATH)
    all_tokens = data["tokens"]
    all_actions = data["actions"]

    start_idx = 500
    context_tokens = torch.from_numpy(all_tokens[start_idx].reshape(1, -1)).long().to(DEVICE)
    context_tokens = context_tokens.unsqueeze(0)

    future_actions = torch.from_numpy(all_actions[start_idx:start_idx + STEPS_TO_DREAM]).float().to(DEVICE)
    future_actions = future_actions.unsqueeze(0)

    generated_frames = []
    rng = np.random.default_rng(SNOW_SEED)
    snow_mask_static = None
    if overlay_net is not None and SNOW_STATIC:
        snow_mask = generate_snow_mask(IMAGE_SIZE, IMAGE_SIZE, rng)
        snow_mask_static = torch.from_numpy(snow_mask).unsqueeze(0).unsqueeze(0).to(DEVICE)

    first_frame = decode_indices(
        vqvae,
        all_tokens[start_idx],
        overlay_net=overlay_net,
        snow_mask=snow_mask_static,
        overlay_scale=overlay_scale,
    )
    generated_frames.append(cv2.cvtColor(first_frame, cv2.COLOR_RGB2BGR))

    print(f"🚀 Dreaming start! Context window: {BLOCK_SIZE} tokens")
    with torch.no_grad():
        current_tokens = context_tokens
        current_actions = future_actions[:, 0:1, :]

        for step in range(STEPS_TO_DREAM - 1):
            t0 = time.time()
            this_step_action = future_actions[:, step:step + 1, :]

            MAX_CONTEXT_FRAMES = 3
            if current_tokens.shape[1] > MAX_CONTEXT_FRAMES:
                current_tokens = current_tokens[:, -MAX_CONTEXT_FRAMES:, :]
                current_actions = current_actions[:, -MAX_CONTEXT_FRAMES:, :]

            pred_tokens_so_far = torch.zeros((1, 1, 256), dtype=torch.long).to(DEVICE)
            full_input_tokens = torch.cat([current_tokens, pred_tokens_so_far], dim=1)
            full_input_actions = torch.cat([current_actions, this_step_action], dim=1)

            for i in range(256):
                logits, _ = gpt(full_input_tokens, full_input_actions)
                seq_len = current_tokens.shape[1]
                target_idx = seq_len * 257 + i - 1
                if target_idx >= logits.shape[1]:
                    target_idx = logits.shape[1] - 1
                next_token_logits = logits[:, target_idx, :]
                idx = sample_next_token(next_token_logits.unsqueeze(1), temperature=TEMPERATURE, top_k=TOP_K)
                full_input_tokens[0, -1, i] = idx

            new_frame_tokens = full_input_tokens[:, -1:, :]
            if overlay_net is not None and not SNOW_STATIC:
                snow_mask = generate_snow_mask(IMAGE_SIZE, IMAGE_SIZE, rng)
                snow_mask_t = torch.from_numpy(snow_mask).unsqueeze(0).unsqueeze(0).to(DEVICE)
            else:
                snow_mask_t = snow_mask_static

            img_np = decode_indices(
                vqvae,
                new_frame_tokens.reshape(16, 16).cpu().numpy(),
                overlay_net=overlay_net,
                snow_mask=snow_mask_t,
                overlay_scale=overlay_scale,
            )
            generated_frames.append(cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR))

            current_tokens = torch.cat([current_tokens, new_frame_tokens], dim=1)
            current_actions = torch.cat([current_actions, this_step_action], dim=1)

            print(f"Frame {step + 1}/{STEPS_TO_DREAM} generated. Time: {time.time() - t0:.2f}s")

    print("💾 Saving video...")
    height, width, layers = generated_frames[0].shape
    video = cv2.VideoWriter(
        OUTPUT_VIDEO, cv2.VideoWriter_fourcc(*"mp4v"), 10, (width, height)
    )
    for frame in generated_frames:
        video.write(frame)
    video.release()
    print(f"✅ Dream video saved to {OUTPUT_VIDEO}")


if __name__ == "__main__":
    main()
