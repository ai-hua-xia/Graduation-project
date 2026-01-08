import metadrive
from metadrive.envs.metadrive_env import MetaDriveEnv
from metadrive.policy.idm_policy import IDMPolicy
from metadrive.component.sensors.rgb_camera import RGBCamera
import cv2
import numpy as np
import os

# ================= 配置升级 =================
DATASET_NAME = "dataset_v2_complex"
TOTAL_EPISODES = 50                  # 采集 50 个不同的场景
MAX_STEPS_PER_EPISODE = 400          # 每个场景跑 400 步
IMAGE_WIDTH = 256                    # ⚡️ 强烈建议改用 256，性价比最高
IMAGE_HEIGHT = 256                   # 512 对于 VAE 真的太难练了，容易崩
# ===========================================

def collect_data():
    os.makedirs(os.path.join(DATASET_NAME, "images"), exist_ok=True)

    env_config = {
        "use_render": False,
        "image_observation": True,
        "image_on_cuda": False,
        "window_size": (IMAGE_WIDTH, IMAGE_HEIGHT),
        "stack_size": 1,
        
        # ✅ [关键修改 1] 开启无限随机地图
        # MetaDrive 会根据 seed 自动生成完全不同的道路网络（十字路口、弯道、直道）
        "num_scenarios": TOTAL_EPISODES,  # 有多少圈，就有多少张不同的地图
        "start_seed": 1000,               # 随机种子
        "map": 7,                         # 这里的数字代表地图生成的“积木”数量，7块积木拼出的路够复杂了
        
        # ✅ [关键修改 2] 增加交通复杂度
        "traffic_density": 0.3,           # 提高密度 (0.3 以上可能会经常堵车，0.2 比较顺畅且有车)
        "random_traffic": True,           # 这里的车会随机生成
        
        # ✅ [关键修改 3] 视觉多样性
        "random_agent_model": False,       # 关闭主车随机模型，防止随机出大卡车挡住视野
        "vehicle_config": {
            "image_source": "rgb_camera",
            "random_color": True,         # 你的车颜色随机
        },
        
        "sensors": {
            "rgb_camera": (RGBCamera, IMAGE_WIDTH, IMAGE_HEIGHT),
        },
        "agent_policy": IDMPolicy,
    }
    
    try:
        env = MetaDriveEnv(env_config)
    except Exception as e:
        print(f"❌ 环境初始化失败: {e}")
        return

    print("🚗 环境初始化完成，开始自动采集数据...")
    
    all_actions = []
    global_step = 0
    
    for episode in range(TOTAL_EPISODES):
        obs, info = env.reset()
        episode_actions = []
        
        for step in range(MAX_STEPS_PER_EPISODE):
            next_obs, reward, terminated, truncated, info = env.step([0, 0])
            
            raw_image = next_obs["image"]
            
            # [调试] 再次打印形状
            if global_step == 0:
                print(f"🔍 [Debug] 修正后图像形状: {raw_image.shape}")

            # ✅ [关键修改 2] 更稳健的形状处理
            # 即使设置了 stack_size=1，有时候它还是会返回 (64, 64, 3, 1)
            # 或者万一它是 (64, 64, 3, 3)，我们也只取最后一帧 (当前帧)
            if raw_image.ndim == 4:
                # 取最后一帧: [..., -1]
                # 这样无论是 1 还是 3，都只会拿最新的一张图
                raw_image = raw_image[..., -1]
            
            # 此时 raw_image 应该是 (64, 64, 3)
            
            # 数值处理：如果是 0-1 的浮点数，转为 0-255
            if raw_image.dtype != np.uint8:
                 image_uint8 = (raw_image * 255).clip(0, 255).astype(np.uint8)
            else:
                 image_uint8 = raw_image

            # 颜色空间转换 RGB -> BGR
            image_bgr = cv2.cvtColor(image_uint8, cv2.COLOR_RGB2BGR)

            # 获取动作
            current_action = np.array([env.vehicle.steering, env.vehicle.throttle_brake])
            episode_actions.append(current_action)

            # 保存
            img_filename = os.path.join(DATASET_NAME, "images", f"img_{global_step:05d}.png")
            cv2.imwrite(img_filename, image_bgr)

            global_step += 1
            
            if (step + 1) % 50 == 0:
                print(f"Episode {episode+1}/{TOTAL_EPISODES} | Step {step+1} | Saved: {img_filename}")

            # 如果撞车了或者跑完了，就提前结束，换下一个地图
            if terminated or truncated:
                print(f"   ⚠️ 场景 {episode+1} 结束 (撞车或超时)")
                break
        
        all_actions.extend(episode_actions)

    actions_np = np.array(all_actions)
    np.save(os.path.join(DATASET_NAME, "actions.npy"), actions_np)
    env.close()
    print("✅ 采集完成！")

if __name__ == "__main__":
    collect_data()
