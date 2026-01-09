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

from train.train_vqvae_256 import VQVAE, IMAGE_SIZE, EMBED_DIM
from train.train_world_model import WorldModelGPT, TOKENS_PER_FRAME, BLOCK_SIZE
from train.train_snow_decoder_noise import SnowConditioner

# ================= 配置 =================
VQVAE_PATH = "checkpoints_vqvae_256/vqvae_256_ep99.pth"
WORLD_MODEL_PATH = "checkpoints_world_model/world_model_ep99.pth"
SNOW_CKPT_PATH = "checkpoints_adapter/snow/decoder/decoder_snow_noise_ep9.pth"

DATA_PATH = "dataset_v2_complex/tokens_actions_vqvae_16x16.npz"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
STEPS_TO_DREAM = 50
TEMPERATURE = 1.2
TOP_K = 5

SNOW_STATIC = True
SNOW_SEED = 42
SNOW_COND_SCALE = None

OUTPUT_VIDEO = "dream_result.mp4"
# 键盘控制 (可选)
USE_KEYBOARD = False
KEYBOARD_FALLBACK_TO_DATA = True
KEYBOARD_WAIT_FOR_INPUT = False
KEYBOARD_BACKEND = "terminal"  # "terminal" or "pygame"
KEYBOARD_REPEAT_FRAMES = 10
STEER_SCALE = 1.0
THROTTLE_SCALE = 1.0
TARGET_FPS = 0
OVERLAY_WASD = True
# =======================================


class KeyboardActionSource:
    def __init__(self, steer_scale: float, throttle_scale: float):
        import pygame

        self.pygame = pygame
        self.steer_scale = steer_scale
        self.throttle_scale = throttle_scale
        pygame.init()
        self.screen = pygame.display.set_mode((280, 120))
        pygame.display.set_caption("WASD Control (focus this window)")

    def poll(self):
        pygame = self.pygame
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return None, True
        keys = pygame.key.get_pressed()
        steer = float(keys[pygame.K_d]) - float(keys[pygame.K_a])
        throttle = float(keys[pygame.K_w]) - float(keys[pygame.K_s])
        if steer == 0.0 and throttle == 0.0:
            return None, False
        action = np.array(
            [steer * self.steer_scale, throttle * self.throttle_scale],
            dtype=np.float32,
        )
        action = np.clip(action, -1.0, 1.0)
        return action, False

    def wait_for_action(self):
        while True:
            action, quit_signal = self.poll()
            if quit_signal:
                return None, True
            if action is not None:
                return action, False
            time.sleep(0.01)

    def close(self):
        self.pygame.quit()


class TerminalActionSource:
    def __init__(self, steer_scale: float, throttle_scale: float):
        import termios
        import sys

        if not sys.stdin.isatty():
            raise RuntimeError("stdin is not a TTY")
        self.termios = termios
        self.fd = sys.stdin.fileno()
        self.old_settings = termios.tcgetattr(self.fd)
        new_settings = termios.tcgetattr(self.fd)
        new_settings[3] &= ~(termios.ECHO | termios.ICANON)
        termios.tcsetattr(self.fd, termios.TCSADRAIN, new_settings)
        self.steer_scale = steer_scale
        self.throttle_scale = throttle_scale

    def _key_to_action(self, key: str):
        key = key.lower()
        if key == "q":
            return None, True
        if key == " ":
            return np.array([0.0, 0.0], dtype=np.float32), False
        steer = 0.0
        throttle = 0.0
        if key == "a":
            steer = -1.0
        elif key == "d":
            steer = 1.0
        elif key == "w":
            throttle = 1.0
        elif key == "s":
            throttle = -1.0
        else:
            return None, False
        action = np.array(
            [steer * self.steer_scale, throttle * self.throttle_scale],
            dtype=np.float32,
        )
        action = np.clip(action, -1.0, 1.0)
        return action, False

    def poll(self):
        import select
        import sys

        rlist, _, _ = select.select([sys.stdin], [], [], 0)
        if not rlist:
            return None, False
        key = sys.stdin.read(1)
        return self._key_to_action(key)

    def wait_for_action(self):
        import select
        import sys

        while True:
            rlist, _, _ = select.select([sys.stdin], [], [], 0.1)
            if not rlist:
                continue
            key = sys.stdin.read(1)
            action, quit_signal = self._key_to_action(key)
            if quit_signal:
                return None, True
            if action is not None:
                return action, False

    def close(self):
        self.termios.tcsetattr(self.fd, self.termios.TCSADRAIN, self.old_settings)


def draw_wasd_overlay(frame_bgr: np.ndarray, action: np.ndarray | None, active: bool) -> None:
    if frame_bgr is None:
        return
    h, w = frame_bgr.shape[:2]
    size = max(22, int(h * 0.08))
    gap = int(size * 0.25)
    x0 = 12
    y0 = h - (size * 2 + gap) - 12
    if y0 < 8:
        y0 = 8

    def draw_key(label: str, x: int, y: int, is_on: bool) -> None:
        if is_on:
            fill = (0, 220, 255)
            text = (10, 24, 26)
        else:
            fill = (40, 40, 40)
            text = (210, 210, 210)
        border = (200, 200, 200)
        cv2.rectangle(frame_bgr, (x, y), (x + size, y + size), fill, -1)
        cv2.rectangle(frame_bgr, (x, y), (x + size, y + size), border, 1)
        cv2.putText(
            frame_bgr,
            label,
            (x + int(size * 0.32), y + int(size * 0.68)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            text,
            1,
            cv2.LINE_AA,
        )

    steer = 0.0
    throttle = 0.0
    if active and action is not None:
        steer = float(action[0])
        throttle = float(action[1])

    key_w = throttle > 0.1
    key_s = throttle < -0.1
    key_a = steer < -0.1
    key_d = steer > 0.1

    draw_key("W", x0 + size + gap, y0, key_w)
    draw_key("A", x0, y0 + size + gap, key_a)
    draw_key("S", x0 + size + gap, y0 + size + gap, key_s)
    draw_key("D", x0 + (size + gap) * 2, y0 + size + gap, key_d)

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

    conditioner = None
    cond_scale = 1.0
    if SNOW_CKPT_PATH:
        print(f"⏳ Loading Snow Decoder from {SNOW_CKPT_PATH}...")
        ckpt = torch.load(SNOW_CKPT_PATH, map_location=DEVICE)
        if "decoder" in ckpt:
            vqvae.decoder.load_state_dict(ckpt["decoder"], strict=True)
        if "conditioner" in ckpt:
            hidden = ckpt.get("cond_hidden", 64)
            conditioner = SnowConditioner(EMBED_DIM, hidden=hidden).to(DEVICE)
            conditioner.load_state_dict(ckpt["conditioner"], strict=True)
            conditioner.eval()
        cond_scale = ckpt.get("cond_scale", cond_scale)
    if SNOW_COND_SCALE is not None:
        cond_scale = SNOW_COND_SCALE
    return vqvae, gpt, conditioner, cond_scale


def decode_indices(vqvae, indices, conditioner=None, snow_latent=None, cond_scale=1.0):
    with torch.no_grad():
        indices_tensor = torch.LongTensor(indices).unsqueeze(0).to(DEVICE)
        z_q = vqvae.quantizer.embedding(indices_tensor)
        z_q = z_q.permute(0, 3, 1, 2)
        if conditioner is not None and snow_latent is not None:
            z_q = z_q + cond_scale * conditioner(snow_latent)
        decoded_img = vqvae.decoder(z_q)
        img = decoded_img[0].cpu().permute(1, 2, 0).numpy()
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
    vqvae, gpt, conditioner, cond_scale = load_models()

    print("🌱 Loading Seed Data...")
    data = np.load(DATA_PATH)
    all_tokens = data["tokens"]
    all_actions = data["actions"]

    start_idx = 500
    context_tokens = torch.from_numpy(all_tokens[start_idx].reshape(1, -1)).long().to(DEVICE)
    context_tokens = context_tokens.unsqueeze(0)

    keyboard = None
    if USE_KEYBOARD:
        try:
            if KEYBOARD_BACKEND == "pygame":
                keyboard = KeyboardActionSource(STEER_SCALE, THROTTLE_SCALE)
                print("🕹️ Keyboard control enabled (WASD). Focus the small window.")
            else:
                keyboard = TerminalActionSource(STEER_SCALE, THROTTLE_SCALE)
                print("⌨️ Terminal control enabled: WASD, Space=brake, Q=quit")
        except Exception as e:
            print(f"⚠️ Keyboard init failed: {e}. Fallback to dataset actions.")
            keyboard = None

    auto_actions = torch.from_numpy(all_actions[start_idx:start_idx + STEPS_TO_DREAM]).float().to(DEVICE)
    auto_actions = auto_actions.unsqueeze(0)

    generated_frames = []
    rng = np.random.default_rng(SNOW_SEED)
    snow_latent_static = None
    latent_size = int(TOKENS_PER_FRAME ** 0.5)
    if conditioner is not None and SNOW_STATIC:
        snow_mask = generate_snow_mask(IMAGE_SIZE, IMAGE_SIZE, rng)
        snow_t = torch.from_numpy(snow_mask).unsqueeze(0).unsqueeze(0).to(DEVICE)
        snow_latent_static = F.interpolate(snow_t, size=(latent_size, latent_size), mode="area")

    first_frame = decode_indices(
        vqvae,
        all_tokens[start_idx],
        conditioner=conditioner,
        snow_latent=snow_latent_static,
        cond_scale=cond_scale,
    )
    generated_frames.append(cv2.cvtColor(first_frame, cv2.COLOR_RGB2BGR))

    print(f"🚀 Dreaming start! Context window: {BLOCK_SIZE} tokens")
    stop_dream = False
    manual_repeat = 0
    manual_action_np = None
    manual_action_tensor = None
    manual_active = False
    with torch.no_grad():
        current_tokens = context_tokens
        current_actions = auto_actions[:, 0:1, :]

        for step in range(STEPS_TO_DREAM - 1):
            t0 = time.time()
            auto_step_action = auto_actions[:, step:step + 1, :]
            this_step_action = auto_step_action
            manual_active = False
            if keyboard is not None:
                if manual_repeat > 0 and manual_action_tensor is not None:
                    this_step_action = manual_action_tensor
                    manual_active = True
                    manual_repeat -= 1
                else:
                    if KEYBOARD_WAIT_FOR_INPUT:
                        manual_action, quit_signal = keyboard.wait_for_action()
                    else:
                        manual_action, quit_signal = keyboard.poll()
                    if quit_signal:
                        print("⛔ Keyboard input closed, stop dreaming.")
                        stop_dream = True
                        break
                    if manual_action is not None:
                        manual_action_np = manual_action
                        manual_action_tensor = torch.from_numpy(manual_action).view(1, 1, 2).to(DEVICE)
                        this_step_action = manual_action_tensor
                        manual_active = True
                        manual_repeat = max(KEYBOARD_REPEAT_FRAMES - 1, 0)
                    elif not KEYBOARD_FALLBACK_TO_DATA:
                        this_step_action = torch.zeros_like(auto_step_action)

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
                target_idx = seq_len * TOKENS_PER_FRAME + i - 1
                if target_idx >= logits.shape[1]:
                    target_idx = logits.shape[1] - 1
                next_token_logits = logits[:, target_idx, :]
                idx = sample_next_token(next_token_logits.unsqueeze(1), temperature=TEMPERATURE, top_k=TOP_K)
                full_input_tokens[0, -1, i] = idx

            new_frame_tokens = full_input_tokens[:, -1:, :]
            if conditioner is not None and not SNOW_STATIC:
                snow_mask = generate_snow_mask(IMAGE_SIZE, IMAGE_SIZE, rng)
                snow_t = torch.from_numpy(snow_mask).unsqueeze(0).unsqueeze(0).to(DEVICE)
                snow_latent = F.interpolate(snow_t, size=(latent_size, latent_size), mode="area")
            else:
                snow_latent = snow_latent_static

            img_np = decode_indices(
                vqvae,
                new_frame_tokens.reshape(16, 16).cpu().numpy(),
                conditioner=conditioner,
                snow_latent=snow_latent,
                cond_scale=cond_scale,
            )
            frame_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
            if OVERLAY_WASD:
                draw_wasd_overlay(frame_bgr, manual_action_np, manual_active)
            generated_frames.append(frame_bgr)

            current_tokens = torch.cat([current_tokens, new_frame_tokens], dim=1)
            current_actions = torch.cat([current_actions, this_step_action], dim=1)

            frame_time = time.time() - t0
            if TARGET_FPS and TARGET_FPS > 0:
                sleep_time = max(0.0, 1.0 / TARGET_FPS - frame_time)
                if sleep_time > 0:
                    time.sleep(sleep_time)
            print(f"Frame {step + 1}/{STEPS_TO_DREAM} generated. Time: {frame_time:.2f}s")

            if stop_dream:
                break

    if keyboard is not None:
        keyboard.close()

    print("💾 Saving video (step 1: raw export)...")
    height, width, layers = generated_frames[0].shape
    temp_output = "temp_dream_raw.mp4"
    video = cv2.VideoWriter(
        temp_output, cv2.VideoWriter_fourcc(*"mp4v"), 10, (width, height)
    )
    for frame in generated_frames:
        video.write(frame)
    video.release()

    print("⚙️ Auto-converting to H.264 for VS Code compatibility...")
    convert_cmd = (
        f"ffmpeg -y -i {temp_output} -vcodec libx264 -pix_fmt yuv420p "
        f"-loglevel error {OUTPUT_VIDEO}"
    )
    exit_code = os.system(convert_cmd)
    if exit_code == 0:
        if os.path.exists(temp_output):
            os.remove(temp_output)
        print(f"✅ Dream video saved to {OUTPUT_VIDEO} (VS Code 可直接播放)")
    else:
        print(f"⚠️ 转码失败 (可能未安装 ffmpeg)，请下载 {temp_output} 到本地播放。")


if __name__ == "__main__":
    main()
