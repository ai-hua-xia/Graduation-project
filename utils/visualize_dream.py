import os
import sys
import time

import cv2
import numpy as np
import torch
import torch.nn.functional as F

# 允许从子目录直接运行
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# 导入你的模型定义
# 如果报错说找不到模块，请确保文件名一致
from train.train_vqvae_256 import VQVAE, IMAGE_SIZE, EMBED_DIM
from train.train_adapter import LatentAdapter
from train.train_world_model import WorldModelGPT, VOCAB_SIZE, TOKENS_PER_FRAME, BLOCK_SIZE

# ================= 配置 =================
# 1. 模型路径
VQVAE_PATH = "checkpoints_vqvae_256/vqvae_256_ep99.pth"
# 这里选一个你刚刚训练出来的最新权重，比如 ep15, ep20 等
WORLD_MODEL_PATH = "checkpoints_new_world_model/world_model_ep45.pth" # 👈 修改为你现在的最新模型
# 风格解码器权重（优先级高于 Adapter；留空则使用原始 VQ-VAE 解码）
STYLE_DECODER_PATH = ""
# 适配器权重（留空则使用原始 VQ-VAE 解码）
ADAPTER_PATH = ""
ADAPTER_BOTTLENECK = 64

# 2. 数据路径 (用来提取第一帧作为种子)
DATA_PATH = "dataset_v2_complex/tokens_actions_vqvae_16x16.npz"

# 3. 生成参数
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
STEPS_TO_DREAM = 50    # 想要让它想象多少帧 (比如 100 帧)
TEMPERATURE = 0.1       # 0.8: 保守/稳定; 1.0: 正常; 1.2: 更有创造力但也更可能崩坏
TOP_K = 1             # 只从概率最高的 100 个 token 里采样，防止画面出现乱码

OUTPUT_VIDEO = "dream_result.mp4"
# 4. 键盘控制 (可选)
USE_KEYBOARD = True
KEYBOARD_FALLBACK_TO_DATA = False
KEYBOARD_WAIT_FOR_INPUT = True
KEYBOARD_BACKEND = "terminal"  # "terminal" or "pygame"
KEYBOARD_REPEAT_FRAMES = 10
STEER_SCALE = 1.0
THROTTLE_SCALE = 1.0
TARGET_FPS = 0
OVERLAY_WASD = True
# 5. 动作文件 (可选，每行一个动作: w/a/s/d/space)
USE_ACTION_FILE = True
ACTIONS_FILE_PATH = "action.txt"
# =======================================

def key_to_action(key: str, steer_scale: float, throttle_scale: float):
    key = key.strip().lower()
    if not key:
        return None, False
    if key == "q":
        return None, True
    if key in (" ", "space", "brake"):
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
        [steer * steer_scale, throttle * throttle_scale],
        dtype=np.float32,
    )
    action = np.clip(action, -1.0, 1.0)
    return action, False

def load_actions_from_file(path: str, steer_scale: float, throttle_scale: float):
    if not path:
        return []
    if not os.path.isabs(path):
        path = os.path.join(PROJECT_ROOT, path)
    if not os.path.exists(path):
        print(f"⚠️ Action file not found: {path}")
        return []
    actions = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            token = line.strip()
            if not token or token.startswith("#"):
                continue
            action, quit_signal = key_to_action(token, steer_scale, throttle_scale)
            if quit_signal:
                break
            if action is None:
                print(f"⚠️ Unknown action token in file: {token!r}")
                continue
            actions.append(action)
    return actions

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
        return key_to_action(key, self.steer_scale, self.throttle_scale)

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
    adapter = None
    if STYLE_DECODER_PATH:
        print(f"⏳ Loading Style Decoder from {STYLE_DECODER_PATH}...")
        ckpt = torch.load(STYLE_DECODER_PATH, map_location=DEVICE)
        state = ckpt["decoder"] if isinstance(ckpt, dict) and "decoder" in ckpt else ckpt
        vqvae.decoder.load_state_dict(state, strict=True)
    elif ADAPTER_PATH:
        print(f"⏳ Loading Adapter from {ADAPTER_PATH}...")
        adapter = LatentAdapter(EMBED_DIM, bottleneck=ADAPTER_BOTTLENECK).to(DEVICE)
        ckpt = torch.load(ADAPTER_PATH, map_location=DEVICE)
        state = ckpt["adapter"] if isinstance(ckpt, dict) and "adapter" in ckpt else ckpt
        adapter.load_state_dict(state, strict=True)
        adapter.eval()
    return vqvae, gpt, adapter

def decode_indices(vqvae, indices, adapter=None):
    """把 (16, 16) 的 token 矩阵还原成图片"""
    with torch.no_grad():
        # indices shape: (16, 16) -> (1, 16, 16)
        indices_tensor = torch.LongTensor(indices).unsqueeze(0).to(DEVICE)
        # VQVAE 的 decode_indices 需要 indices 已经是 Embedding 后的还是直接 indices?
        # 查看之前的 VQVAE 代码，通常需要通过 quantizer 查表。
        # 为了方便，我们直接用 quantizer 的 embedding 查表功能
        
        # 1. 查表获取 quant vectors
        z_q = vqvae.quantizer.embedding(indices_tensor) # (1, 16, 16, 64)
        z_q = z_q.permute(0, 3, 1, 2) # (1, 64, 16, 16)
        
        # 2. 可选 Adapter + 解码
        if adapter is not None:
            z_q = adapter(z_q)
        decoded_img = vqvae.decoder(z_q)
        
        # 3. 转回 numpy 图片格式
        img = decoded_img[0].cpu().permute(1, 2, 0).numpy()
        img = np.clip(img, 0, 1) * 255
        return img.astype(np.uint8)

def sample_next_token(logits, temperature=1.0, top_k=None):
    """从预测结果中采样"""
    logits = logits[:, -1, :] / temperature # 只取最后一个时间步
    if top_k is not None:
        v, _ = torch.topk(logits, top_k)
        logits[logits < v[:, [-1]]] = -float('Inf')
    
    probs = F.softmax(logits, dim=-1)
    idx = torch.multinomial(probs, num_samples=1)
    return idx

def main():
    vqvae, gpt, adapter = load_models()
    
    # 1. 加载真实数据作为“种子”
    print("🌱 Loading Seed Data...")
    data = np.load(DATA_PATH)
    all_tokens = data['tokens']   # (N, 16, 16)
    all_actions = data['actions'] # (N, 2)
    
    # 我们从第 500 帧开始，作为起始状态
    start_idx = 500
    
    # ================= 🔴 核心修改在这里 =================
    # 在转为 Tensor 后，必须加上 .long()，把 uint16 强转为 int64
    context_tokens = torch.from_numpy(all_tokens[start_idx].reshape(1, -1)).long().to(DEVICE) 
    # ===================================================
    
    context_tokens = context_tokens.unsqueeze(0) # (1, 1, 256) -> batch=1, seq=1, dim=256

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

    # 预加载自动动作（用于回退）
    auto_actions = torch.from_numpy(all_actions[start_idx:start_idx + STEPS_TO_DREAM]).float().to(DEVICE)
    auto_actions = auto_actions.unsqueeze(0) # (1, STEPS, 2)
    file_actions = []
    if USE_ACTION_FILE:
        file_actions = load_actions_from_file(ACTIONS_FILE_PATH, STEER_SCALE, THROTTLE_SCALE)
        if file_actions:
            print(f"📄 Loaded {len(file_actions)} actions from {ACTIONS_FILE_PATH}")
    
    # 用于保存生成的图片
    generated_frames = []
    
    # 先把第一帧解码出来存着
    first_frame = decode_indices(vqvae, all_tokens[start_idx], adapter=adapter)
    generated_frames.append(cv2.cvtColor(first_frame, cv2.COLOR_RGB2BGR))
    
    print(f"🚀 Dreaming start! Context window: {BLOCK_SIZE} tokens")
    
    stop_dream = False
    manual_repeat = 0
    manual_action_np = None
    manual_action_tensor = None
    file_repeat = 0
    file_action_np = None
    file_action_tensor = None
    file_action_idx = 0
    manual_active = False
    with torch.no_grad():
        current_tokens = context_tokens # (1, seq_len, 256)
        current_actions = auto_actions[:, 0:1, :] # 取第一个动作 (1, 1, 2)
        
        for step in range(STEPS_TO_DREAM - 1):
            t0 = time.time()
            
            # 准备当前要生成的帧的容器
            next_frame_tokens = []
            
            # 获取当前上下文的动作
            auto_step_action = auto_actions[:, step:step+1, :] # (1, 1, 2)
            this_step_action = auto_step_action
            manual_active = False
            if file_actions:
                if file_repeat > 0 and file_action_tensor is not None:
                    this_step_action = file_action_tensor
                    manual_action_np = file_action_np
                    manual_active = True
                    file_repeat -= 1
                elif file_action_idx < len(file_actions):
                    file_action_np = file_actions[file_action_idx]
                    file_action_tensor = torch.from_numpy(file_action_np).view(1, 1, 2).to(DEVICE)
                    this_step_action = file_action_tensor
                    manual_action_np = file_action_np
                    manual_active = True
                    file_repeat = max(KEYBOARD_REPEAT_FRAMES - 1, 0)
                    file_action_idx += 1
                elif not KEYBOARD_FALLBACK_TO_DATA:
                    this_step_action = torch.zeros_like(auto_step_action)
            elif keyboard is not None:
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
            
            # 这里的滑动窗口逻辑
            MAX_CONTEXT_FRAMES = 3
            if current_tokens.shape[1] > MAX_CONTEXT_FRAMES:
                current_tokens = current_tokens[:, -MAX_CONTEXT_FRAMES:, :]
                current_actions = current_actions[:, -MAX_CONTEXT_FRAMES:, :]
            
            # 基础输入构造
            pred_tokens_so_far = torch.zeros((1, 1, 256), dtype=torch.long).to(DEVICE)
            
            # 拼接 Img 和 Act
            # 此时 current_tokens 已经是 long 类型，pred_tokens_so_far 也是 long 类型，不会报错了
            full_input_tokens = torch.cat([current_tokens, pred_tokens_so_far], dim=1) 
            full_input_actions = torch.cat([current_actions, this_step_action], dim=1) 
            
            for i in range(256):
                logits, _ = gpt(full_input_tokens, full_input_actions)
                
                seq_len = current_tokens.shape[1] 
                target_idx = seq_len * TOKENS_PER_FRAME + i - 1
                
                # 安全检查
                if target_idx >= logits.shape[1]:
                    target_idx = logits.shape[1] - 1
                    
                next_token_logits = logits[:, target_idx, :]
                
                # 采样
                idx = sample_next_token(next_token_logits.unsqueeze(1), temperature=TEMPERATURE, top_k=TOP_K)
                
                # 填入 tensor
                full_input_tokens[0, -1, i] = idx
            
            # 一帧生成完毕！
            new_frame_tokens = full_input_tokens[:, -1:, :] # (1, 1, 256)
            
            # 解码显示
            # 注意：decode_indices 需要 numpy 格式
            img_np = decode_indices(vqvae, new_frame_tokens.reshape(16, 16).cpu().numpy(), adapter=adapter)
            frame_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
            if OVERLAY_WASD:
                draw_wasd_overlay(frame_bgr, manual_action_np, manual_active)
            generated_frames.append(frame_bgr)
            
            # 更新上下文
            current_tokens = torch.cat([current_tokens, new_frame_tokens], dim=1)
            current_actions = torch.cat([current_actions, this_step_action], dim=1)
            
            frame_time = time.time() - t0
            if TARGET_FPS and TARGET_FPS > 0:
                sleep_time = max(0.0, 1.0 / TARGET_FPS - frame_time)
                if sleep_time > 0:
                    time.sleep(sleep_time)
            print(f"Frame {step+1}/{STEPS_TO_DREAM} generated. Time: {frame_time:.2f}s")

            if stop_dream:
                break

    if keyboard is not None:
        keyboard.close()

    # 保存视频
    print("💾 Saving video (step 1: raw export)...")
    height, width, layers = generated_frames[0].shape
    
    # 1. 先保存为一个临时文件 (使用 mp4v，因为这是 OpenCV 支持最稳的，不容易报错)
    temp_output = "temp_dream_raw.mp4"
    video = cv2.VideoWriter(temp_output, cv2.VideoWriter_fourcc(*'mp4v'), 10, (width, height))
    
    for frame in generated_frames:
        video.write(frame)
    video.release()
    
    # 2. 调用 FFmpeg 自动转码 (转成 VS Code 能播的 H.264 格式)
    # 注意：这需要你的服务器上安装了 ffmpeg (通常做深度学习环境都有)
    print("⚙️ Auto-converting to H.264 for VS Code compatibility...")
    
    # -y: 覆盖同名文件
    # -vcodec libx264: 使用 H.264 编码
    # -pix_fmt yuv420p: 确保浏览器/VSCode 兼容性
    # -loglevel error: 少输出废话
    convert_cmd = f"ffmpeg -y -i {temp_output} -vcodec libx264 -pix_fmt yuv420p -loglevel error {OUTPUT_VIDEO}"
    
    exit_code = os.system(convert_cmd)
    
    if exit_code == 0:
        # 转码成功，删除临时文件
        if os.path.exists(temp_output):
            os.remove(temp_output)
        print(f"✅ Dream video saved to {OUTPUT_VIDEO} (VS Code 可直接播放)")
    else:
        # 转码失败（可能没装 ffmpeg），保留原文件
        print(f"⚠️ 转码失败 (可能未安装 ffmpeg)，请下载 {temp_output} 到本地播放。")

if __name__ == "__main__":
    main()
