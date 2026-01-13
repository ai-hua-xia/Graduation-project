# WASD键盘控制

使用WASD按键控制World Model生成视频。

## 快速开始

### 1. 创建动作文件
```bash
cat > my_actions.txt << 'EOF'
W
W
A
D
N
EOF
```

### 2. 生成视频
```bash
python carla_project/visualize/dream.py \
    --vqvae-checkpoint carla_project/checkpoints/vqvae_v2/best.pth \
    --world-model-checkpoint carla_project/checkpoints/world_model_v2/best.pth \
    --token-file carla_project/data/tokens_v2/tokens_actions.npz \
    --action-txt my_actions.txt \
    --output output.mp4
```

## 按键说明

| 按键 | 动作 | Steering | Throttle |
|------|------|----------|----------|
| W | 加速 | 0.0 | 0.65 |
| S | 减速 | 0.0 | 0.42 |
| A | 左转 | -0.4 | 0.55 |
| D | 右转 | 0.4 | 0.55 |
| Q | 左转+加速 | -0.4 | 0.65 |
| E | 右转+加速 | 0.4 | 0.65 |
| N | 直行 | 0.0 | 0.55 |

## 文件格式

### WASD格式
每行一个字母：
```
W
W
A
D
N
```

### 数值格式
每行两个数字（steering throttle）：
```
0.0 0.65
-0.4 0.55
0.4 0.55
0.0 0.55
```

### 注释
使用 `#` 开头：
```
# 加速段
W
W
# 左转
A
A
```

## 重要提示

### 动作范围
模型训练数据范围：
- **Steering**: [-0.6, 0.6]
- **Throttle**: [0.4, 0.7]

**超出范围会影响生成质量！**

### 关于刹车
训练数据中**没有刹车**（throttle全部>0），因此：
- S键是"减速"而非"刹车"
- 不要使用负的throttle值

## 示例

### 示例1：简单测试
```
W
W
W
A
A
D
D
N
```

### 示例2：复杂驾驶
```
# 起步加速
W
W
W
# 左转弯
Q
Q
Q
# 直行
N
N
# 右转弯
E
E
E
# 减速
S
S
```

## 测试

```bash
# 测试WASD功能
bash carla_project/script/test_wasd.sh

# 使用示例文件
python carla_project/visualize/dream.py \
    --vqvae-checkpoint carla_project/checkpoints/vqvae_v2/best.pth \
    --world-model-checkpoint carla_project/checkpoints/world_model_v2/best.pth \
    --token-file carla_project/data/tokens_v2/tokens_actions.npz \
    --action-txt carla_project/visualize/example_actions.txt \
    --output example.mp4
```

## 常见问题

### Q: 为什么S不是刹车？
A: 训练数据中没有刹车动作，模型从未见过负的throttle值。

### Q: 可以使用更大的转向角度吗？
A: 可以，但建议在[-0.6, 0.6]范围内，超出可能影响质量。

### Q: 如何提高生成质量？
A:
1. 使用训练范围内的动作值
2. 避免频繁切换动作
3. 使用Scheduled Sampling模型

## 下一步

- 查看 [QUICKSTART.md](QUICKSTART.md) 了解完整流程
- 查看 [README.md](README.md) 了解项目概况
