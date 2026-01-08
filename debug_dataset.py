import torch
import numpy as np
import cv2
import os
from train_vqvae_256 import VQVAE, DEVICE

# 配置
DATA_PATH = "dataset_v2_complex/tokens_actions_vqvae_16x16.npz"
VQVAE_PATH = "checkpoints_vqvae_256/vqvae_256_ep99.pth" # 你的 VQVAE 权重

def check_dataset():
    print(f"📂 Loading dataset from {DATA_PATH}...")
    data = np.load(DATA_PATH)
    tokens = data['tokens']  # (N, 16, 16)
    
    print(f"📊 Dataset shape: {tokens.shape}")
    print(f"   Sample token values: {tokens[0, 0, :10]}") # 打印几个看看是不是全是0
    
    # 加载 VQ-VAE 解码器
    model = VQVAE().to(DEVICE)
    model.load_state_dict(torch.load(VQVAE_PATH, map_location=DEVICE)["model"])
    model.eval()
    
    # 随机抽查 5 张图
    indices_to_check = np.linspace(0, len(tokens)-1, 5, dtype=int)
    
    reconstructed_images = []
    
    print("🔄 Decoding tokens back to images...")
    with torch.no_grad():
        for idx in indices_to_check:
            # 取出 Token (16, 16)
            token_idx = tokens[idx]
            
            # 变成 Tensor (1, 16, 16)
            indices_tensor = torch.LongTensor(token_idx).unsqueeze(0).to(DEVICE)
            
            # 解码
            z_q = model.quantizer.embedding(indices_tensor).permute(0, 3, 1, 2)
            decoded_img = model.decoder(z_q)
            
            # 后处理
            img = decoded_img[0].cpu().permute(1, 2, 0).numpy()
            img = np.clip(img, 0, 1) * 255.0
            img = img.astype(np.uint8)
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            
            # 在图上写上序号
            cv2.putText(img, f"Idx: {idx}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            reconstructed_images.append(img)
    
    # 拼图保存
    result = np.hstack(reconstructed_images)
    cv2.imwrite("debug_dataset_check.jpg", result)
    print("✅ Check complete! Please look at 'debug_dataset_check.jpg'.")
    print("👉 如果这张图是乱码/花屏，说明 .npz 数据制作错了！GPT 是无辜的。")
    print("👉 如果这张图清晰，说明数据没问题，问题在 GPT 模型本身。")

if __name__ == "__main__":
    check_dataset()