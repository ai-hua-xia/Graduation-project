import torch
import torch.nn.functional as F
import numpy as np
import cv2
import os
import time

# 导入你的模型定义
# 如果报错说找不到模块，请确保文件名一致
from train_vqvae_256 import VQVAE, IMAGE_SIZE
from train_world_model import WorldModelGPT, VOCAB_SIZE, TOKENS_PER_FRAME, BLOCK_SIZE

# ================= 配置 =================
# 1. 模型路径
VQVAE_PATH = "checkpoints_vqvae_256/vqvae_256_ep99.pth"
# 这里选一个你刚刚训练出来的最新权重，比如 ep15, ep20 等
WORLD_MODEL_PATH = "checkpoints_world_model/world_model_ep99.pth" # 👈 修改为你现在的最新模型

# 2. 数据路径 (用来提取第一帧作为种子)
DATA_PATH = "dataset_v2_complex/tokens_actions_vqvae_16x16.npz"

# 3. 生成参数
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
STEPS_TO_DREAM = 100    # 想要让它想象多少帧 (比如 100 帧)
TEMPERATURE = 0.3       # 0.8: 保守/稳定; 1.0: 正常; 1.2: 更有创造力但也更可能崩坏
TOP_K = 10             # 只从概率最高的 100 个 token 里采样，防止画面出现乱码

OUTPUT_VIDEO = "dream_result.mp4"
# =======================================

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
    return vqvae, gpt

def decode_indices(vqvae, indices):
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
        
        # 2. 解码
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
    vqvae, gpt = load_models()
    
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
    
    # 提取未来 100 帧的真实动作
    future_actions = torch.from_numpy(all_actions[start_idx:start_idx + STEPS_TO_DREAM]).float().to(DEVICE)
    future_actions = future_actions.unsqueeze(0) # (1, STEPS, 2)
    
    # 用于保存生成的图片
    generated_frames = []
    
    # 先把第一帧解码出来存着
    first_frame = decode_indices(vqvae, all_tokens[start_idx])
    generated_frames.append(cv2.cvtColor(first_frame, cv2.COLOR_RGB2BGR))
    
    print(f"🚀 Dreaming start! Context window: {BLOCK_SIZE} tokens")
    
    with torch.no_grad():
        current_tokens = context_tokens # (1, seq_len, 256)
        current_actions = future_actions[:, 0:1, :] # 取第一个动作 (1, 1, 2)
        
        for step in range(STEPS_TO_DREAM - 1):
            t0 = time.time()
            
            # 准备当前要生成的帧的容器
            next_frame_tokens = []
            
            # 获取当前上下文的动作
            this_step_action = future_actions[:, step:step+1, :] # (1, 1, 2)
            
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
                target_idx = seq_len * 257 + i - 1
                
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
            img_np = decode_indices(vqvae, new_frame_tokens.reshape(16, 16).cpu().numpy())
            generated_frames.append(cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR))
            
            # 更新上下文
            current_tokens = torch.cat([current_tokens, new_frame_tokens], dim=1)
            current_actions = torch.cat([current_actions, this_step_action], dim=1)
            
            print(f"Frame {step+1}/{STEPS_TO_DREAM} generated. Time: {time.time()-t0:.2f}s")

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