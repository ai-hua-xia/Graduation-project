"""
CARLA数据采集主程序

使用方法：
    python carla_collector.py --host localhost --port 2000 --episodes 100
"""

import argparse
import time
import numpy as np
from pathlib import Path
from tqdm import tqdm
import sys

try:
    import carla
except ImportError:
    print("Error: CARLA Python API not found!")
    print("Please install: pip install carla==0.9.15")
    sys.exit(1)

import config
import utils


class CARLACollector:
    def __init__(self, host='localhost', port=2000, output_dir='../data/raw'):
        """初始化CARLA采集器"""
        self.host = host
        self.port = port
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # 连接到CARLA服务器
        print(f"Connecting to CARLA server at {host}:{port}...")
        self.client = carla.Client(host, port)
        self.client.set_timeout(10.0)

        # 获取版本信息
        version = self.client.get_server_version()
        print(f"Connected to CARLA {version}")

        self.world = None
        self.vehicle = None
        self.camera = None
        self.traffic_actors = []
        self.collision_sensor = None
        self.collision_count = 0

    def setup_world(self, town_name, weather_name):
        """设置世界"""
        print(f"Loading world: {town_name}, Weather: {weather_name}")

        # 加载地图
        self.world = self.client.load_world(town_name)

        # 设置同步模式
        if config.SYNC_MODE:
            settings = self.world.get_settings()
            settings.synchronous_mode = True
            settings.fixed_delta_seconds = config.FIXED_DELTA_SECONDS
            self.world.apply_settings(settings)

        # 设置天气
        utils.set_weather(self.world, weather_name)

        # 生成交通
        self.traffic_actors = utils.spawn_traffic(
            self.world,
            num_vehicles=config.NUM_VEHICLES,
            num_pedestrians=config.NUM_PEDESTRIANS
        )

    def spawn_vehicle(self):
        """生成主车辆"""
        blueprint = self.world.get_blueprint_library().find(config.VEHICLE_FILTER)
        spawn_point = utils.get_random_spawn_point(self.world)

        self.vehicle = self.world.spawn_actor(blueprint, spawn_point)
        print(f"Spawned vehicle at {spawn_point.location}")

        # 等待车辆稳定
        if config.SYNC_MODE:
            self.world.tick()
        else:
            time.sleep(0.5)

    def setup_sensors(self):
        """设置传感器"""
        # RGB相机
        self.camera = utils.setup_camera(
            self.world,
            self.vehicle,
            config.CAMERA_CONFIG,
            config.CAMERA_TRANSFORM
        )

        # 碰撞传感器
        collision_bp = self.world.get_blueprint_library().find('sensor.other.collision')
        self.collision_sensor = self.world.spawn_actor(
            collision_bp,
            carla.Transform(),
            attach_to=self.vehicle
        )
        self.collision_sensor.listen(lambda event: self.on_collision(event))

        self.collision_count = 0

    def on_collision(self, event):
        """碰撞回调"""
        self.collision_count += 1

    def collect_episode(self, episode_id, driving_mode='normal'):
        """采集一个episode"""
        # 创建episode目录
        episode_dir = utils.create_episode_dir(self.output_dir, episode_id)

        # 元数据
        metadata = {
            'episode_id': episode_id,
            'town': self.world.get_map().name,
            'weather': str(self.world.get_weather()),
            'driving_mode': driving_mode,
            'num_frames': 0,
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        }

        # 数据缓存
        images = []
        actions = []
        frame_count = 0
        prev_action = None

        # 图像队列（用于异步保存）
        image_queue = []

        def camera_callback(image):
            """相机回调"""
            image_queue.append(image)

        self.camera.listen(camera_callback)

        # 获取驾驶模式配置
        mode_config = config.DRIVING_MODES.get(
            driving_mode,
            config.DRIVING_MODES['normal']
        )

        print(f"\nCollecting episode {episode_id}...")
        pbar = tqdm(total=config.FRAMES_PER_EPISODE, desc=f"Episode {episode_id}")

        try:
            while frame_count < config.FRAMES_PER_EPISODE:
                # 检查碰撞
                if self.collision_count > config.MAX_COLLISION_COUNT:
                    print(f"\nToo many collisions ({self.collision_count}), terminating episode")
                    break

                # 生成动作
                action = utils.get_driving_action(mode_config)

                # 平滑动作
                action = utils.smooth_action(action, prev_action, alpha=0.7)
                prev_action = action

                # 应用动作
                control = carla.VehicleControl(
                    steer=float(action[0]),
                    throttle=max(0, float(action[1])),
                    brake=max(0, -float(action[1])),
                )
                self.vehicle.apply_control(control)

                # Tick世界
                if config.SYNC_MODE:
                    self.world.tick()
                else:
                    time.sleep(config.FIXED_DELTA_SECONDS)

                # 检查速度
                speed = utils.calculate_speed(self.vehicle)
                if speed < config.MIN_SPEED:
                    continue  # 跳过静止/慢速帧

                # 保存图像
                if len(image_queue) > 0:
                    image = image_queue.pop(0)
                    image_path = episode_dir / "images" / f"frame_{frame_count:06d}.png"
                    utils.save_image(image, str(image_path))

                    # 保存动作
                    actions.append(action)

                    frame_count += 1
                    pbar.update(1)

        except KeyboardInterrupt:
            print("\nInterrupted by user")

        finally:
            # 停止相机监听
            self.camera.stop()
            pbar.close()

        # 保存动作序列
        if len(actions) > 0:
            utils.save_actions(episode_dir, actions)

        # 更新元数据
        metadata['num_frames'] = frame_count
        metadata['collision_count'] = self.collision_count
        utils.save_metadata(episode_dir, metadata)

        print(f"Saved {frame_count} frames to {episode_dir}")

        return frame_count

    def cleanup(self):
        """清理资源"""
        print("\nCleaning up...")

        if self.camera is not None:
            self.camera.destroy()

        if self.collision_sensor is not None:
            self.collision_sensor.destroy()

        if self.vehicle is not None:
            self.vehicle.destroy()

        utils.cleanup_actors(self.traffic_actors)
        self.traffic_actors = []

        # 恢复异步模式
        if config.SYNC_MODE and self.world is not None:
            settings = self.world.get_settings()
            settings.synchronous_mode = False
            self.world.apply_settings(settings)

    def run(self, num_episodes):
        """运行数据采集"""
        total_frames = 0

        try:
            for episode_id in range(num_episodes):
                # 随机选择场景和天气
                town = np.random.choice(config.TOWNS)
                weather = np.random.choice(config.WEATHERS)
                driving_mode = config.DEFAULT_DRIVING_MODE

                # 设置世界
                self.setup_world(town, weather)

                # 生成车辆和传感器
                self.spawn_vehicle()
                self.setup_sensors()

                # 采集数据
                frames = self.collect_episode(episode_id, driving_mode)
                total_frames += frames

                # 清理当前episode
                self.cleanup()

                print(f"Total frames collected: {total_frames}")

        except KeyboardInterrupt:
            print("\n\nCollection interrupted by user")

        finally:
            self.cleanup()
            print(f"\nCollection complete! Total frames: {total_frames}")


def main():
    parser = argparse.ArgumentParser(description='CARLA Data Collector')
    parser.add_argument('--host', type=str, default='localhost',
                        help='CARLA server host')
    parser.add_argument('--port', type=int, default=2000,
                        help='CARLA server port')
    parser.add_argument('--episodes', type=int, default=100,
                        help='Number of episodes to collect')
    parser.add_argument('--output', type=str, default='../data/raw',
                        help='Output directory')

    args = parser.parse_args()

    # 创建采集器
    collector = CARLACollector(
        host=args.host,
        port=args.port,
        output_dir=args.output
    )

    # 运行采集
    collector.run(num_episodes=args.episodes)


if __name__ == '__main__':
    main()
