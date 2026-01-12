"""
CARLA数据采集 - Town03城市场景版本
专门用于城市场景采集，建筑物密集，转向效果明显
"""

import carla
import numpy as np
import cv2
from pathlib import Path
from tqdm import tqdm
import time
import json

# 配置
HOST = 'localhost'
PORT = 2000
OUTPUT_DIR = '../data/raw'
NUM_EPISODES = 10  # 增加到10个episodes
FRAMES_PER_EPISODE = 100  # 增加到100帧
IMAGE_WIDTH = 256
IMAGE_HEIGHT = 256
TARGET_MAP = 'Town03'  # ⭐ 城市地图


def save_image(image, filepath):
    """保存图像"""
    array = np.frombuffer(image.raw_data, dtype=np.uint8)
    array = array.reshape((image.height, image.width, 4))
    array = array[:, :, :3]
    array = array[:, :, ::-1]
    cv2.imwrite(str(filepath), array)


def main():
    print("=" * 60)
    print("  CARLA数据采集 - Town03城市场景")
    print("=" * 60)
    print()

    # 连接CARLA
    print(f"连接到CARLA服务器 {HOST}:{PORT}...")
    client = carla.Client(HOST, PORT)
    client.set_timeout(10.0)

    world = client.get_world()
    current_map = world.get_map().name
    print(f"✓ 已连接！当前地图: {current_map}")

    # ⭐ 切换到城市地图
    if TARGET_MAP not in current_map:
        print(f"\n切换地图到 {TARGET_MAP}（城市场景，建筑密集）...")
        world = client.load_world(TARGET_MAP)
        time.sleep(2)  # 等待地图加载
        print(f"✓ 地图已切换到 {TARGET_MAP}")

    # 启用同步模式
    print("\n启用同步模式...")
    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = 0.05  # 20 FPS
    world.apply_settings(settings)
    print("✓ 同步模式已启用")
    print()

    # 创建输出目录
    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 获取blueprint
    blueprint_library = world.get_blueprint_library()
    vehicle_bp = blueprint_library.find('vehicle.tesla.model3')
    camera_bp = blueprint_library.find('sensor.camera.rgb')
    camera_bp.set_attribute('image_size_x', str(IMAGE_WIDTH))
    camera_bp.set_attribute('image_size_y', str(IMAGE_HEIGHT))
    camera_bp.set_attribute('fov', '90')

    spawn_points = world.get_map().get_spawn_points()
    print(f"可用出生点: {len(spawn_points)}")

    # 开始采集
    for episode_id in range(NUM_EPISODES):
        print(f"\n{'='*60}")
        print(f"  Episode {episode_id + 1}/{NUM_EPISODES}")
        print(f"{'='*60}")

        # 创建episode目录
        episode_dir = output_dir / f"episode_{episode_id:04d}"
        episode_dir.mkdir(exist_ok=True)
        (episode_dir / "images").mkdir(exist_ok=True)

        # 随机选择出生点
        spawn_point = np.random.choice(spawn_points)

        vehicle = None
        camera = None

        try:
            # 生成车辆
            vehicle = world.spawn_actor(vehicle_bp, spawn_point)
            print(f"✓ 车辆已生成")

            # 等待车辆稳定
            for _ in range(5):
                world.tick()

            # 设置相机
            camera_transform = carla.Transform(
                carla.Location(x=1.5, y=0.0, z=1.4),
                carla.Rotation(pitch=-5.0)
            )

            camera = world.spawn_actor(camera_bp, camera_transform, attach_to=vehicle)

            # 图像队列
            image_queue = []

            def camera_callback(image):
                image_queue.append(image)

            camera.listen(camera_callback)

            # 等待相机就绪
            for _ in range(5):
                world.tick()

            # 采集数据
            print(f"采集 {FRAMES_PER_EPISODE} 帧...")
            actions = []
            frame_count = 0

            pbar = tqdm(total=FRAMES_PER_EPISODE)

            while frame_count < FRAMES_PER_EPISODE:
                # 生成随机动作（转向优先）
                if np.random.random() < 0.7:
                    steering = np.random.uniform(-0.6, 0.6)
                else:
                    steering = np.random.uniform(-0.1, 0.1)

                throttle = np.random.uniform(0.4, 0.7)

                # 应用控制
                control = carla.VehicleControl(
                    steer=float(steering),
                    throttle=float(throttle),
                    brake=0.0
                )
                vehicle.apply_control(control)

                # 推进世界
                world.tick()

                # 保存图像
                if len(image_queue) > 0:
                    image = image_queue.pop(0)
                    image_path = episode_dir / "images" / f"frame_{frame_count:06d}.png"
                    save_image(image, str(image_path))

                    # 保存动作
                    actions.append([steering, throttle])

                    frame_count += 1
                    pbar.update(1)

            pbar.close()

            # 保存动作序列
            np.save(episode_dir / "actions.npy", np.array(actions))

            # 保存元数据
            metadata = {
                'episode_id': episode_id,
                'num_frames': frame_count,
                'map': world.get_map().name,
            }

            with open(episode_dir / "metadata.json", 'w') as f:
                json.dump(metadata, f, indent=2)

            print(f"✓ Episode {episode_id} 完成！保存了 {frame_count} 帧")

        except Exception as e:
            print(f"✗ 错误: {e}")
            import traceback
            traceback.print_exc()

        finally:
            # 清理
            if camera is not None:
                camera.stop()
                camera.destroy()
            if vehicle is not None:
                vehicle.destroy()

            for _ in range(5):
                world.tick()

    # 恢复异步模式
    print("\n恢复异步模式...")
    settings.synchronous_mode = False
    world.apply_settings(settings)

    print()
    print("=" * 60)
    print("  采集完成！")
    print("=" * 60)
    print(f"数据保存在: {output_dir}")
    print()
    print("下一步：")
    print("  python verify_data.py  # 验证数据质量")


if __name__ == '__main__':
    main()
