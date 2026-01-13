# CARLA World Model - Scripts

便捷脚本工具集

## 脚本说明

### 1. `run_pipeline.sh` - 完整训练流程
运行从数据采集到评估的完整pipeline。

```bash
bash carla_project/script/run_pipeline.sh
```

**包含步骤**：
1. 数据采集（如果需要）
2. 训练 VQ-VAE v2
3. 导出 tokens
4. 训练 World Model (Teacher Forcing)
5. 训练 World Model (Scheduled Sampling)
6. 生成对比视频和dream视频
7. 评估并可视化结果

### 2. `start_training.sh` - 在tmux中启动训练
在tmux会话中启动训练任务，支持后台运行。

```bash
# 训练 VQ-VAE v2
bash carla_project/script/start_training.sh train_vqvae vqvae 0

# 训练 World Model (Teacher Forcing)
bash carla_project/script/start_training.sh train_wm world_model 1

# 训练 World Model (Scheduled Sampling)
bash carla_project/script/start_training.sh train_ss ss 1
```

**参数**：
- `task_name`: tmux会话名称
- `training_type`: 训练类型 (vqvae/world_model/ss)
- `gpu_id`: GPU编号（可选，默认0）

### 3. `quick_eval.sh` - 快速评估模型
评估指定的World Model checkpoint。

```bash
# 评估 Teacher Forcing 模型
bash carla_project/script/quick_eval.sh \
    carla_project/checkpoints/world_model_v2/best.pth tf

# 评估 Scheduled Sampling 模型
bash carla_project/script/quick_eval.sh \
    carla_project/checkpoints/world_model_ss/best.pth ss
```

**生成内容**：
- 对比视频 (comparison)
- Dream视频
- 评估指标 (JSON)
- 可视化图表

### 4. `start_carla_server.sh` - 启动CARLA服务器
启动CARLA Docker容器。

```bash
bash carla_project/script/start_carla_server.sh
```

### 5. `test_wasd.sh` - 测试WASD动作生成
测试WASD动作映射和视频生成功能。

```bash
bash carla_project/script/test_wasd.sh
```

### 6. `quick_start.sh` - WASD快速开始指南
显示WASD动作生成的快速使用指南。

```bash
bash carla_project/script/quick_start.sh
```

## 使用建议

### 首次运行
```bash
# 1. 启动CARLA服务器
bash carla_project/script/start_carla_server.sh

# 2. 运行完整pipeline
bash carla_project/script/run_pipeline.sh
```

### 训练单个模型
```bash
# 在tmux中启动训练
bash carla_project/script/start_training.sh my_training world_model 1

# 查看训练进度
tmux attach -t my_training
# 或
tail -f carla_project/logs/my_training.log
```

### 快速评估
```bash
# 评估某个checkpoint
bash carla_project/script/quick_eval.sh \
    carla_project/checkpoints/world_model_v2/world_model_epoch_099.pth \
    epoch99
```

## 注意事项

1. **权限**: 首次使用需要添加执行权限
   ```bash
   chmod +x carla_project/script/*.sh
   ```

2. **路径**: 所有脚本都假设从项目根目录运行

3. **GPU**: 确保指定的GPU可用且有足够显存

4. **tmux**: 使用tmux可以在关闭终端后继续训练
