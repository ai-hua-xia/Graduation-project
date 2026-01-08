import torch
import cv2
import numpy as np
from train_vqvae_256 import VQVAE, DEVICE
from train_world_model import WorldModelGPT, BLOCK_SIZE

# 配置
VQVAE_PATH = "checkpoints_vqvae_256/vqvae_256_ep99.pth"
WORLD_MODEL_PATH = "checkpoints_world_model/world_model_ep10.pth" # 用那个 Loss=2.0 左右的
DATA_PATH = "dataset_v2_complex/tokens_actions_vqvae_16x16.npz"

def test_teacher_forcing():
    # 1. 加载所有东西
    vqvae = VQVAE().to(DEVICE)
    vqvae.load_state_dict(torch.load(VQVAE_PATH, map_location=DEVICE)["model"])
    
    gpt = WorldModelGPT().to(DEVICE)
    gpt.load_state_dict(torch.load(WORLD_MODEL_PATH, map_location=DEVICE)["model"])
    gpt.eval()
    
    data = np.load(DATA_PATH)
    tokens = torch.from_numpy(data['tokens']).long().to(DEVICE) # (N, 16, 16)
    actions = torch.from_numpy(data['actions']).float().to(DEVICE) # (N, 2)
    
    # 2. 选取一段真实的连续数据 (比如第 500-504 帧)
    start_idx = 500
    seq_len = 3
    
    # 历史: 0, 1, 2, 3
    input_tokens = tokens[start_idx : start_idx+seq_len].reshape(1, seq_len, 256) 
    input_actions = actions[start_idx : start_idx+seq_len].unsqueeze(0)
    
    # 真实目标: 第 4 帧 (即 start_idx + 4)
    target_tokens = tokens[start_idx + seq_len] 
    
    print("🤖 GPT is predicting the NEXT frame based on REAL history...")
    
    # 3. 让 GPT 预测下一帧 (Teacher Forcing)
    with torch.no_grad():
        # 构造输入 (B, seq, 256)
        # 还需要加上这一步的动作，用来预测这一步的图
        # 我们这里简化，直接看它能不能根据历史预测未来
        # 注意：训练时输入是 (img, act)，预测下一个 img
        # 我们手动构造一个 dummy 输入来触发预测
        
        # 为了预测第 5 帧，我们需要输入前 4 帧 + 第 5 帧的动作
        next_action = actions[start_idx + seq_len].view(1, 1, 2)
        
        # 此时还没有第 5 帧的图像，我们用全0占位，让 GPT 填空
        dummy_next_token = torch.zeros((1, 1, 256), dtype=torch.long).to(DEVICE)
        
        full_input_tokens = torch.cat([input_tokens, dummy_next_token], dim=1) # seq=5
        full_input_actions = torch.cat([input_actions, next_action], dim=1)    # seq=5
        
        # 逐像素生成第 5 帧
        generated_tokens = []
        for i in range(256):
            logits, _ = gpt(full_input_tokens, full_input_actions)
            
            # 取出对应位置的 logit
            # 历史长度 seq_len=4. 对应 flattened index 是 4 * 257 + i - 1
            idx_in_flat = seq_len * 257 + i - 1
            if idx_in_flat >= logits.shape[1]: idx_in_flat = logits.shape[1]-1

            next_logit = logits[:, idx_in_flat, :]
            
            # 贪婪采样 (取概率最大的，不随机) 看它到底学到了什么
            token_id = torch.argmax(next_logit, dim=-1)
            
            full_input_tokens[0, -1, i] = token_id # 填回去
            generated_tokens.append(token_id.item())
            
    # 4. 解码对比
    # 真实图
    z_q_true = vqvae.quantizer.embedding(target_tokens.unsqueeze(0)).permute(0, 3, 1, 2)
    img_true = vqvae.decoder(z_q_true)[0].cpu().permute(1, 2, 0).detach().numpy()
    
    # 预测图
    gen_tensor = torch.tensor(generated_tokens).reshape(1, 16, 16).to(DEVICE)
    z_q_pred = vqvae.quantizer.embedding(gen_tensor).permute(0, 3, 1, 2)
    img_pred = vqvae.decoder(z_q_pred)[0].cpu().permute(1, 2, 0).detach().numpy()
    
    # 拼图
    img_true = np.clip(img_true, 0, 1) * 255
    img_pred = np.clip(img_pred, 0, 1) * 255
    res = np.hstack([img_true, img_pred])
    cv2.imwrite("debug_single_step.jpg", res)
    print("✅ Prediction done. Check 'debug_single_step.jpg'.")
    print("⬅️ 左边: 真实未来 | ➡️ 右边: GPT预测未来")

if __name__ == "__main__":
    test_teacher_forcing()