"""
CARLA大规模数据采集 - Town03城市场景
目标: 100 episodes × 100 frames = 10,000 帧
"""

import carla
import numpy as np
import cv2
import time
from pathlib import Path
from tqdm import tqdm
import random

# 配置
CARLA_HOST = 'localhost'
CARLA_PORT = 2000
TARGET_MAP = 'Town03'

# 大规模采集配置
NUM_EPISODES = 100        # 100个episodes
FRAMES_PER_EPISODE = 100  # 每个episode 100帧
# 总计: 10,000 帧

IMAGE_WIDTH = 256
IMAGE_HEIGHT = 256
FPS = 20

# 数据保存路径
DATA_DIR = Path('../data/raw')


def setup_camera(world, vehicle):
    """设置相机"""
    bp_lib = world.get_blueprint_library()
    camera_bp = bp_lib.find('sensor.camera.rgb')
    camera_bp.set_attribute('image_size_x', str(IMAGE_WIDTH))
    camera_bp.set_attribute('image_size_y', str(IMAGE_HEIGHT))
    camera_bp.set_attribute('fov', '90')

    # 相机位置（车顶前方）
    camera_transform = carla.Transform(
        carla.Location(x=1.5, z=1.8),
        carla.Rotation(pitch=-10)
    )

    camera = world.spawn_actor(camera_bp, camera_transform, attach_to=vehicle)
    return camera


def random_action():
    """生成随机动作（偏向转向）"""
    # 70%概率大转向
    if random.random() < 0.7:
        steering = random.uniform(-0.6, 0.6)
    else:
        steering = random.uniform(-0.2, 0.2)

    throttle = random.uniform(0.4, 0.7)
    return steering, throttle


def collect_episode(world, episode_idx, spawn_points):
    """采集单个episode"""
    bp_lib = world.get_blueprint_library()

    # 随机选择车辆
    vehicle_bp = bp_lib.filter('vehicle.*')[0]

    # 尝试多个出生点
    vehicle = None
    random.shuffle(spawn_points)
    for spawn_point in spawn_points[:30]:
        try:
            vehicle = world.spawn_actor(vehicle_bp, spawn_point)
            break
        except RuntimeError:
            continue

    if vehicle is None:
        print(f"  ✗ 无法生成车辆，跳过episode {episode_idx}")
        return False

    # 设置相机
    camera = setup_camera(world, vehicle)

    # 图像存储
    images = []

    def camera_callback(image):
        array = np.frombuffer(image.raw_data, dtype=np.uint8)
        array = array.reshape((IMAGE_HEIGHT, IMAGE_WIDTH, 4))[:, :, :3]
        images.append(array.copy())

    camera.listen(camera_callback)

    # 等待相机初始化
    for _ in range(5):
        world.tick()
        time.sleep(0.01)

    images.clear()

    # 采集数据
    actions = []
    pbar = tqdm(range(FRAMES_PER_EPISODE), desc=f"Episode {episode_idx}", leave=False)

    for frame_idx in pbar:
        # 生成随机动作
        steering, throttle = random_action()
        actions.append([steering, throttle])

        # 应用动作
        control = carla.VehicleControl(
            throttle=throttle,
            steer=steering,
            brake=0.0
        )
        vehicle.apply_control(control)

        # 前进一帧
        world.tick()
        time.sleep(0.01)

    # 停止相机
    camera.stop()

    # 保存数据
    episode_dir = DATA_DIR / f'episode_{episode_idx:04d}'
    episode_dir.mkdir(parents=True, exist_ok=True)

    # 保存图像
    images_dir = episode_dir / 'images'
    images_dir.mkdir(exist_ok=True)

    # 取前FRAMES_PER_EPISODE帧
    for i, img in enumerate(images[:FRAMES_PER_EPISODE]):
        cv2.imwrite(str(images_dir / f'{i:04d}.png'), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

    # 保存动作
    np.save(episode_dir / 'actions.npy', np.array(actions))

    # 清理
    camera.destroy()
    vehicle.destroy()

    return True


def main():
    print("=" * 60)
    print("  CARLA大规模数据采集 - Town03城市场景")
    print(f"  目标: {NUM_EPISODES} episodes × {FRAMES_PER_EPISODE} frames = {NUM_EPISODES * FRAMES_PER_EPISODE:,} 帧")
    print("=" * 60)

    # 连接CARLA
    print(f"\n连接到CARLA服务器 {CARLA_HOST}:{CARLA_PORT}...")
    client = carla.Client(CARLA_HOST, CARLA_PORT)
    client.set_timeout(30.0)
    world = client.get_world()
    print(f"✓ 已连接！当前地图: {world.get_map().name}")

    # 检查是否需要切换地图
    current_map = world.get_map().name
    if TARGET_MAP not in current_map:
        print(f"\n切换地图到 {TARGET_MAP}（城市场景，建筑密集）...")
        world = client.load_world(TARGET_MAP)
        time.sleep(3)
        print(f"✓ 地图已切换到 {TARGET_MAP}")

    # 启用同步模式
    print("\n启用同步模式...")
    settings = world.get_settings()
    original_settings = settings
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = 1.0 / FPS
    world.apply_settings(settings)
    print("✓ 同步模式已启用")

    # 获取出生点
    spawn_points = world.get_map().get_spawn_points()
    print(f"\n可用出生点: {len(spawn_points)}")

    # 检查已有数据
    existing_episodes = list(DATA_DIR.glob('episode_*'))
    start_idx = len(existing_episodes)
    if start_idx > 0:
        print(f"\n发现已有 {start_idx} 个episodes，从 episode_{start_idx:04d} 继续采集")

    # 采集数据
    successful = 0
    failed = 0

    try:
        for i in range(start_idx, NUM_EPISODES):
            print(f"\n{'='*60}")
            print(f"  Episode {i+1}/{NUM_EPISODES}")
            print(f"{'='*60}")

            if collect_episode(world, i, spawn_points):
                successful += 1
                print(f"✓ Episode {i} 完成！保存了 {FRAMES_PER_EPISODE} 帧")
            else:
                failed += 1

            # 每10个episode显示进度
            if (i + 1) % 10 == 0:
                total_frames = (successful) * FRAMES_PER_EPISODE
                print(f"\n📊 进度: {i+1}/{NUM_EPISODES} episodes, {total_frames:,} 帧已采集")

    except KeyboardInterrupt:
        print("\n\n⚠ 用户中断采集")

    finally:
        # 恢复异步模式
        print("\n恢复异步模式...")
        settings = world.get_settings()
        settings.synchronous_mode = False
        world.apply_settings(settings)

    # 统计
    total_frames = successful * FRAMES_PER_EPISODE
    print("\n" + "=" * 60)
    print("  采集完成！")
    print("=" * 60)
    print(f"成功: {successful} episodes")
    print(f"失败: {failed} episodes")
    print(f"总帧数: {total_frames:,}")
    print(f"\n数据保存在: {DATA_DIR.absolute()}")

    print("\n下一步：")
    print("  python verify_data_v2.py  # 验证数据质量")
    print("  cd ../train && python train_vqvae.py  # 训练VQ-VAE")


if __name__ == '__main__':
    main()
