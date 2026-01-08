import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import math
import time

# ================= 配置区域 =================
DATA_PATH = "dataset_v2_complex/tokens_actions_vqvae_16x16.npz"
OUT_DIR = "checkpoints_world_model"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# 模型参数
VOCAB_SIZE = 1024       # VQ-VAE 的词表大小
ACTION_DIM = 2          # 动作维度 (转向, 油门)
N_EMBD = 512            # 嵌入维度 (GPT的隐藏层大小)
N_HEAD = 8              # 注意力头数
N_LAYER = 8             # Transformer 层数
DROPOUT = 0.1

# 序列参数
TOKENS_PER_FRAME = 256  # 16x16
BLOCK_SIZE = 257 * 4    # 上下文长度：看过去 4 帧 (256图 + 1动作) * 4
BATCH_SIZE = 16         # 显存不够就改小，比如 8 或 4
LEARNING_RATE = 3e-4
MAX_EPOCHS = 100
SAVE_EVERY = 5          # 每多少轮保存一次

os.makedirs(OUT_DIR, exist_ok=True)

# ================= 1. 数据集定义 =================
class WorldModelDataset(Dataset):
    def __init__(self, data_path, seq_len=4):
        print(f"Loading data from {data_path}...")
        data = np.load(data_path)
        self.tokens = data['tokens']   # (N, 16, 16)
        self.actions = data['actions'] # (N, 2)
        self.indices = data['indices'] # (N,) 用于判断是否连续
        
        # 展平 Token: (N, 16, 16) -> (N, 256)
        self.n_samples = len(self.tokens)
        self.tokens_flat = self.tokens.reshape(self.n_samples, -1).astype(np.int64)
        
        self.seq_len = seq_len # 一次拿几帧训练
        self.frame_struct_len = TOKENS_PER_FRAME + 1 # 一帧的总长度 (256图 + 1动作)

        # 预计算所有有效的起始索引（防止跨视频采样）
        self.valid_starts = []
        for i in range(self.n_samples - self.seq_len):
            # 检查这几帧在原始视频里是否是连续的 (index 必须连号)
            # 例如: indices[i+seq_len] - indices[i] 应该等于 seq_len
            if self.indices[i + self.seq_len] - self.indices[i] == self.seq_len:
                self.valid_starts.append(i)
        
        print(f"Data loaded. Total frames: {self.n_samples}. Valid sequences: {len(self.valid_starts)}")

    def __len__(self):
        return len(self.valid_starts)

    def __getitem__(self, idx):
        # 获取这一段的起始帧索引
        start_idx = self.valid_starts[idx]
        end_idx = start_idx + self.seq_len
        
        # 提取数据段
        batch_tokens = self.tokens_flat[start_idx:end_idx] # (seq_len, 256)
        batch_actions = self.actions[start_idx:end_idx]    # (seq_len, 2)
        
        # 构造输入序列： [Img0, Act0, Img1, Act1, ...]
        # 我们需要把 Image Token 和 Action 拼起来。
        # 为了方便处理，我们只返回原始数据，在 collate_fn 或 forward 里再拼接 embedding
        
        return {
            "tokens": torch.from_numpy(batch_tokens),
            "actions": torch.from_numpy(batch_actions).float()
        }

# ================= 2. GPT 模型定义 =================
class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert N_EMBD % N_HEAD == 0
        self.c_attn = nn.Linear(N_EMBD, 3 * N_EMBD)
        self.c_proj = nn.Linear(N_EMBD, N_EMBD)
        self.attn_dropout = nn.Dropout(DROPOUT)
        self.resid_dropout = nn.Dropout(DROPOUT)
        self.n_head = N_HEAD
        self.n_embd = N_EMBD
        # 因果遮罩 (Mask)
        self.register_buffer("bias", torch.tril(torch.ones(BLOCK_SIZE, BLOCK_SIZE))
                                     .view(1, 1, BLOCK_SIZE, BLOCK_SIZE))

    def forward(self, x):
        B, T, C = x.size()
        q, k, v  = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_dropout(self.c_proj(y))

class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln1 = nn.LayerNorm(N_EMBD)
        self.attn = CausalSelfAttention(config)
        self.ln2 = nn.LayerNorm(N_EMBD)
        self.mlp = nn.Sequential(
            nn.Linear(N_EMBD, 4 * N_EMBD),
            nn.GELU(),
            nn.Linear(4 * N_EMBD, N_EMBD),
            nn.Dropout(DROPOUT),
        )

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x

class WorldModelGPT(nn.Module):
    def __init__(self):
        super().__init__()
        # 1. 嵌入层
        self.token_embedding = nn.Embedding(VOCAB_SIZE, N_EMBD)
        self.action_embedding = nn.Linear(ACTION_DIM, N_EMBD) # 连续动作映射到 embedding 空间
        self.position_embedding = nn.Embedding(BLOCK_SIZE, N_EMBD)
        
        # 2. Transformer Blocks
        self.blocks = nn.Sequential(*[Block(None) for _ in range(N_LAYER)])
        self.ln_f = nn.LayerNorm(N_EMBD)
        
        # 3. 输出头 (预测下一个 Token)
        self.head = nn.Linear(N_EMBD, VOCAB_SIZE, bias=False)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)

    def forward(self, token_seq, action_seq, targets=None):
        """
        token_seq: (B, seq_len, 256)
        action_seq: (B, seq_len, 2)
        """
        B, seq_len, _ = token_seq.shape
        
        # --- 核心逻辑：构造输入序列 ---
        # 每一帧变成了 257 个 token: [256个图token, 1个动作embedding]
        # 总长度 T = seq_len * 257
        
        # 1. 把 Image Tokens 变成 Embedding
        img_embs = self.token_embedding(token_seq) # (B, seq_len, 256, N_EMBD)
        
        # 2. 把 Action 变成 Embedding
        act_embs = self.action_embedding(action_seq) # (B, seq_len, N_EMBD)
        act_embs = act_embs.unsqueeze(2) # (B, seq_len, 1, N_EMBD)
        
        # 3. 拼接: 在每帧的 256 个图 token 后面拼 1 个动作 token
        # 形状变: (B, seq_len, 257, N_EMBD)
        x = torch.cat([img_embs, act_embs], dim=2) 
        
        # 4. 展平为时间序列
        # 形状变: (B, seq_len * 257, N_EMBD) -> (B, T, N_EMBD)
        x = x.view(B, -1, N_EMBD)
        
        # 5. 加上位置编码
        T = x.size(1)
        if T > BLOCK_SIZE:
             # 如果序列太长（比如第一次运行），截断（虽然理论上不会发生）
             x = x[:, :BLOCK_SIZE, :]
             T = BLOCK_SIZE
             
        pos_idxs = torch.arange(T, device=x.device)
        pos_emb = self.position_embedding(pos_idxs)
        x = x + pos_emb
        
        # 6. Transformer Forward
        x = self.blocks(x)
        x = self.ln_f(x)
        
        # 7. 计算 Loss
        logits = self.head(x) # (B, T, VOCAB_SIZE)
        
        loss = None
        if targets is not None:
            # targets 的构造需要稍微费点劲
            # 我们的 x 是: [I0...I0, A0, I1...I1, A1, ...]
            # 我们希望预测: [I0...I0, I1...I1, A1...] 的下一个
            # 其实最简单的自回归目标是：输入 idx 的预测目标是 idx+1
            
            # 构造完整的 target 序列索引
            # Image tokens: 0~1023
            # Action 位置我们不想计算 Loss (因为动作是连续值，且是给定的条件，不是预测目标)
            # 所以我们在 target 里把 Action 的位置设为 -1 (ignore_index)
            
            # 准备 Target Tensor
            # 原始 targets 是输入的 token_seq，但是我们需要把它们按顺序排好
            # (B, seq_len, 256) -> (B, seq_len * 257) ? 不对，这里只有256个
            
            flat_tokens = token_seq.view(B, -1) # (B, seq_len * 256)
            # 我们需要构造一个和 x 一样长的 (B, seq_len * 257) 的 target 矩阵
            # 其中 Image 位置填 Image Token，Action 位置填 -1
            
            target_seq = torch.full((B, seq_len, 257), -1, dtype=torch.long, device=DEVICE)
            target_seq[:, :, :256] = token_seq
            target_seq = target_seq.view(B, -1) # (B, T)

            # Shift predict:
            # logits预测的是下一个词。所以 logits[:, :-1] 应该预测 target_seq[:, 1:]
            
            logits = logits[:, :-1, :]
            target_seq = target_seq[:, 1:]
            
            # Flatten for loss
            loss = F.cross_entropy(logits.reshape(-1, VOCAB_SIZE), target_seq.reshape(-1), ignore_index=-1)

        return logits, loss

# ================= 3. 训练主循环 =================
def main():
    # 1. 准备数据
    dataset = WorldModelDataset(DATA_PATH, seq_len=int(BLOCK_SIZE/257))
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    
    # 2. 初始化模型
    model = WorldModelGPT().to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    
    # 3. 尝试加载断点
    start_epoch = 0
    checkpoints = sorted([f for f in os.listdir(OUT_DIR) if f.endswith(".pth")])
    if checkpoints:
        latest = os.path.join(OUT_DIR, checkpoints[-1])
        print(f"🔄 Resuming from {latest}")
        ckpt = torch.load(latest, map_location=DEVICE)
        model.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optimizer'])
        start_epoch = ckpt['epoch'] + 1

    # 4. 开始训练
    print(f"🚀 Start Training World Model on {DEVICE}...")
    model.train()
    
    for epoch in range(start_epoch, MAX_EPOCHS):
        total_loss = 0
        start_time = time.time()
        
        for i, batch in enumerate(dataloader):
            tokens = batch['tokens'].to(DEVICE)   # (B, seq, 256)
            actions = batch['actions'].to(DEVICE) # (B, seq, 2)
            
            optimizer.zero_grad()
            
            # Forward (传入 tokens 作为 target)
            _, loss = model(tokens, actions, targets=tokens)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0) # 梯度裁剪防止爆炸
            optimizer.step()
            
            total_loss += loss.item()
            
            if i % 10 == 0:
                print(f"Epoch {epoch} | Step {i}/{len(dataloader)} | Loss: {loss.item():.4f}")
        
        avg_loss = total_loss / len(dataloader)
        print(f"✅ Epoch {epoch} Done. Avg Loss: {avg_loss:.4f}. Time: {time.time()-start_time:.1f}s")
        
        # 保存模型
        if epoch % SAVE_EVERY == 0 or epoch == MAX_EPOCHS - 1:
            save_path = os.path.join(OUT_DIR, f"world_model_ep{epoch}.pth")
            torch.save({
                'epoch': epoch,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'loss': avg_loss
            }, save_path)
            print(f"💾 Saved checkpoint: {save_path}")

if __name__ == "__main__":
    main()