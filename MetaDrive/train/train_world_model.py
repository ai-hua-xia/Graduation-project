import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import math
import time

# ================= 配置区域 =================
DATA_PATH = "dataset_rich_actions/tokens_actions_vqvae_16x16.npz"
OUT_DIR = "checkpoints_new_rich_world_model"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
RESUME_PATH = "checkpoints_new_rich_world_model/world_model_ep129.pth"  # 可选：指定继续训练的checkpoint路径，留空则自动找OUT_DIR里最新的

# 模型参数
VOCAB_SIZE = 1024       # VQ-VAE 的词表大小
ACTION_DIM = 2          # 动作维度 (转向, 油门)
N_EMBD = 512            # 嵌入维度 (GPT的隐藏层大小)
N_HEAD = 8              # 注意力头数
N_LAYER = 8             # Transformer 层数
DROPOUT = 0.1
USE_ACTION_FILM = True   # 动作 FiLM 调制 (更强的条件注入)

# 序列参数
TOKENS_PER_FRAME = 256  # 16x16
CONTEXT_FRAMES = 4      # 上下文帧数
BLOCK_SIZE = TOKENS_PER_FRAME * CONTEXT_FRAMES
BATCH_SIZE = 64         # 显存不够就改小，比如 8 或 4
LEARNING_RATE = 3e-4
MAX_EPOCHS = 180
SAVE_EVERY = 5          # 每多少轮保存一次
TEMPORAL_SMOOTH_WEIGHT = 0.08  # 时间一致性正则权重 (0 关闭)
TEMPORAL_SMOOTH_USE_ACTION = True
TEMPORAL_SMOOTH_ACTION_BETA = 2.0  # 动作越大，平滑约束越弱

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

class CrossAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert N_EMBD % N_HEAD == 0
        self.q_proj = nn.Linear(N_EMBD, N_EMBD)
        self.kv_proj = nn.Linear(N_EMBD, 2 * N_EMBD)
        self.c_proj = nn.Linear(N_EMBD, N_EMBD)
        self.attn_dropout = nn.Dropout(DROPOUT)
        self.resid_dropout = nn.Dropout(DROPOUT)
        self.n_head = N_HEAD
        self.n_embd = N_EMBD

    def forward(self, x, context):
        B, T, C = x.size()
        Bc, S, Cc = context.size()
        if Bc != B or Cc != C:
            raise ValueError("Action context shape mismatch.")

        q = self.q_proj(x)
        k, v = self.kv_proj(context).split(self.n_embd, dim=2)

        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        k = k.view(B, S, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, S, self.n_head, C // self.n_head).transpose(1, 2)

        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
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
        self.action_film = nn.Linear(N_EMBD, 2 * N_EMBD) if USE_ACTION_FILM else None
        self.ln_cross = nn.LayerNorm(N_EMBD)
        self.cross_attn = CrossAttention(config)
        self.ln2 = nn.LayerNorm(N_EMBD)
        self.mlp = nn.Sequential(
            nn.Linear(N_EMBD, 4 * N_EMBD),
            nn.GELU(),
            nn.Linear(4 * N_EMBD, N_EMBD),
            nn.Dropout(DROPOUT),
        )

    def forward(self, x, action_context, action_token_context=None):
        x_norm = self.ln1(x)
        if self.action_film is not None and action_token_context is not None:
            gamma, beta = self.action_film(action_token_context).chunk(2, dim=-1)
            x_norm = x_norm * (1 + gamma) + beta
        x = x + self.attn(x_norm)
        if action_context is not None:
            x = x + self.cross_attn(self.ln_cross(x), action_context)
        x = x + self.mlp(self.ln2(x))
        return x

class WorldModelGPT(nn.Module):
    def __init__(self):
        super().__init__()
        # 1. 嵌入层
        self.token_embedding = nn.Embedding(VOCAB_SIZE, N_EMBD)
        self.action_embedding = nn.Linear(ACTION_DIM, N_EMBD) # 连续动作映射到 embedding 空间
        self.action_pos_embedding = nn.Embedding(CONTEXT_FRAMES, N_EMBD)
        self.position_embedding = nn.Embedding(BLOCK_SIZE, N_EMBD)
        
        # 2. Transformer Blocks
        self.blocks = nn.ModuleList([Block(None) for _ in range(N_LAYER)])
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

        # 1. Image tokens -> embeddings
        flat_tokens = token_seq.view(B, -1)  # (B, seq_len * 256)
        tok_embs = self.token_embedding(flat_tokens)  # (B, T, N_EMBD)

        # 2. Action embeddings + positional encoding (per frame)
        if seq_len > self.action_pos_embedding.num_embeddings:
            raise ValueError("seq_len exceeds action position embedding size.")
        frame_ids = torch.arange(seq_len, device=tok_embs.device)
        act_embs = self.action_embedding(action_seq) + self.action_pos_embedding(frame_ids)[None, :, :]

        # 3. Broadcast action embeddings to tokens
        T = tok_embs.size(1)
        token_frame_ids = torch.arange(T, device=tok_embs.device) // TOKENS_PER_FRAME
        token_frame_ids = torch.clamp(token_frame_ids, max=seq_len - 1)
        act_tok_embs = act_embs[:, token_frame_ids, :]  # (B, T, N_EMBD)

        # 4. Position embedding
        if T > BLOCK_SIZE:
            tok_embs = tok_embs[:, :BLOCK_SIZE, :]
            act_tok_embs = act_tok_embs[:, :BLOCK_SIZE, :]
            T = BLOCK_SIZE
        pos_idxs = torch.arange(T, device=tok_embs.device)
        pos_emb = self.position_embedding(pos_idxs)

        x = tok_embs + act_tok_embs + pos_emb

        # 5. Transformer Forward (self-attn + cross-attn)
        for block in self.blocks:
            x = block(x, act_embs, act_tok_embs)
        x = self.ln_f(x)
        logits = self.head(x)

        loss = None
        if targets is not None:
            target_seq = flat_tokens[:, :T]
            logits_for_loss = logits[:, :-1, :]
            target_seq = target_seq[:, 1:]
            loss = F.cross_entropy(logits_for_loss.reshape(-1, VOCAB_SIZE), target_seq.reshape(-1))

        return logits, loss

# ================= 3. 训练主循环 =================
def main():
    # 1. 准备数据
    dataset = WorldModelDataset(DATA_PATH, seq_len=CONTEXT_FRAMES)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    
    # 2. 初始化模型
    model = WorldModelGPT().to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    
    # 3. 尝试加载断点
    start_epoch = 0
    resume_path = RESUME_PATH
    if resume_path:
        if not os.path.exists(resume_path):
            raise FileNotFoundError(f"Resume checkpoint not found: {resume_path}")
    else:
        checkpoints = sorted([f for f in os.listdir(OUT_DIR) if f.endswith(".pth")])
        if checkpoints:
            resume_path = os.path.join(OUT_DIR, checkpoints[-1])
    if resume_path:
        print(f"🔄 Resuming from {resume_path}")
        ckpt = torch.load(resume_path, map_location=DEVICE)
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
            logits, loss = model(tokens, actions, targets=tokens)
            smooth_loss = None
            if TEMPORAL_SMOOTH_WEIGHT > 0 and logits is not None:
                B, seq_len, _ = tokens.shape
                if seq_len > 1:
                    flat_tokens = tokens.view(B, -1)
                    logits_for_target = logits[:, :-1, :]
                    start = TOKENS_PER_FRAME - 1
                    if logits_for_target.size(1) > start:
                        smooth_logits = logits_for_target[:, start:, :]
                        smooth_logits = smooth_logits.view(B, seq_len - 1, TOKENS_PER_FRAME, VOCAB_SIZE)
                        prev_tokens = flat_tokens[:, :-TOKENS_PER_FRAME].view(B, seq_len - 1, TOKENS_PER_FRAME)
                        smooth_loss = F.cross_entropy(
                            smooth_logits.reshape(-1, VOCAB_SIZE),
                            prev_tokens.reshape(-1),
                            reduction="none",
                        )
                        smooth_loss = smooth_loss.view(B, seq_len - 1, TOKENS_PER_FRAME)
                        if TEMPORAL_SMOOTH_USE_ACTION:
                            action_mag = torch.norm(actions[:, 1:, :], dim=-1)
                            weight = torch.exp(-TEMPORAL_SMOOTH_ACTION_BETA * action_mag).unsqueeze(-1)
                            smooth_loss = (smooth_loss * weight).mean()
                        else:
                            smooth_loss = smooth_loss.mean()
                        loss = loss + TEMPORAL_SMOOTH_WEIGHT * smooth_loss
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0) # 梯度裁剪防止爆炸
            optimizer.step()
            
            total_loss += loss.item()
            
            if i % 10 == 0:
                if smooth_loss is None:
                    print(f"Epoch {epoch} | Step {i}/{len(dataloader)} | Loss: {loss.item():.4f}")
                else:
                    print(f"Epoch {epoch} | Step {i}/{len(dataloader)} | Loss: {loss.item():.4f} | Smooth: {smooth_loss.item():.4f}")
        
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
