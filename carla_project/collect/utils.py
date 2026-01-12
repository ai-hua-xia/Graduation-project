"""
CARLA数据采集工具函数
"""

import numpy as np
import cv2
import json
import os
from pathlib import Path


def save_image(image, filepath):
    """保存图像"""
    array = np.frombuffer(image.raw_data, dtype=np.uint8)
    array = array.reshape((image.height, image.width, 4))  # BGRA
    array = array[:, :, :3]  # 去掉alpha通道
    array = array[:, :, ::-1]  # BGR -> RGB
    cv2.imwrite(filepath, array)
    return array


def process_action(control):
    """
    从CARLA control提取动作向量

    Args:
        control: carla.VehicleControl对象

    Returns:
        action: np.array([steering, throttle])
    """
    steering = control.steer  # [-1, 1]
    throttle = control.throttle  # [0, 1]

    # 如果有刹车，可以用负油门表示
    if control.brake > 0:
        throttle = -control.brake

    return np.array([steering, throttle], dtype=np.float32)


def create_episode_dir(base_dir, episode_id):
    """创建episode目录"""
    episode_dir = Path(base_dir) / f"episode_{episode_id:04d}"
    episode_dir.mkdir(parents=True, exist_ok=True)

    # 创建子目录
    (episode_dir / "images").mkdir(exist_ok=True)

    return episode_dir


def save_metadata(episode_dir, metadata):
    """保存元数据"""
    filepath = episode_dir / "metadata.json"
    with open(filepath, 'w') as f:
        json.dump(metadata, f, indent=2)


def load_metadata(episode_dir):
    """加载元数据"""
    filepath = episode_dir / "metadata.json"
    with open(filepath, 'r') as f:
        return json.load(f)


def save_actions(episode_dir, actions):
    """保存动作序列"""
    filepath = episode_dir / "actions.npy"
    np.save(filepath, np.array(actions))


def load_actions(episode_dir):
    """加载动作序列"""
    filepath = episode_dir / "actions.npy"
    return np.load(filepath)


def calculate_speed(vehicle):
    """计算车辆速度（m/s）"""
    vel = vehicle.get_velocity()
    return np.sqrt(vel.x**2 + vel.y**2 + vel.z**2)


def get_random_spawn_point(world):
    """获取随机出生点"""
    spawn_points = world.get_map().get_spawn_points()
    return np.random.choice(spawn_points)


def set_weather(world, weather_name):
    """设置天气"""
    weather_presets = {
        'ClearNoon': carla.WeatherParameters.ClearNoon,
        'CloudyNoon': carla.WeatherParameters.CloudyNoon,
        'WetNoon': carla.WeatherParameters.WetNoon,
        'ClearSunset': carla.WeatherParameters.ClearSunset,
        'WetSunset': carla.WeatherParameters.WetSunset,
        'SoftRainNoon': carla.WeatherParameters.SoftRainNoon,
    }

    if weather_name in weather_presets:
        world.set_weather(weather_presets[weather_name])
    else:
        print(f"Warning: Unknown weather '{weather_name}', using ClearNoon")
        world.set_weather(carla.WeatherParameters.ClearNoon)


def setup_camera(world, vehicle, camera_config, camera_transform):
    """设置相机"""
    import carla

    # 创建相机蓝图
    camera_bp = world.get_blueprint_library().find('sensor.camera.rgb')
    camera_bp.set_attribute('image_size_x', str(camera_config['image_size_x']))
    camera_bp.set_attribute('image_size_y', str(camera_config['image_size_y']))
    camera_bp.set_attribute('fov', str(camera_config['fov']))

    # 设置相机位置
    transform = carla.Transform(
        carla.Location(
            x=camera_transform['x'],
            y=camera_transform['y'],
            z=camera_transform['z']
        ),
        carla.Rotation(
            pitch=camera_transform['pitch'],
            yaw=camera_transform['yaw'],
            roll=camera_transform['roll']
        )
    )

    # 生成相机
    camera = world.spawn_actor(camera_bp, transform, attach_to=vehicle)

    return camera


def spawn_traffic(world, num_vehicles=30, num_pedestrians=20):
    """生成交通参与者"""
    import carla
    import random

    actors = []

    # 生成车辆
    vehicle_bps = world.get_blueprint_library().filter('vehicle.*')
    spawn_points = world.get_map().get_spawn_points()
    random.shuffle(spawn_points)

    for i in range(min(num_vehicles, len(spawn_points))):
        bp = random.choice(vehicle_bps)
        try:
            vehicle = world.spawn_actor(bp, spawn_points[i])
            vehicle.set_autopilot(True)
            actors.append(vehicle)
        except:
            pass

    # 生成行人
    pedestrian_bps = world.get_blueprint_library().filter('walker.pedestrian.*')
    spawn_points = []

    for i in range(num_pedestrians):
        spawn_point = carla.Transform()
        loc = world.get_random_location_from_navigation()
        if loc is not None:
            spawn_point.location = loc
            try:
                bp = random.choice(pedestrian_bps)
                pedestrian = world.spawn_actor(bp, spawn_point)
                actors.append(pedestrian)
            except:
                pass

    print(f"Spawned {len(actors)} traffic actors")
    return actors


def cleanup_actors(actors):
    """清理生成的actor"""
    for actor in actors:
        if actor is not None:
            actor.destroy()


def get_driving_action(mode_config, waypoint=None):
    """
    根据驾驶模式生成动作

    Args:
        mode_config: 驾驶模式配置
        waypoint: 当前路点（可选，用于更智能的驾驶）

    Returns:
        action: [steering, throttle]
    """
    # 随机决定是否转向
    should_turn = np.random.random() < mode_config['turn_probability']

    if should_turn:
        # 转向
        steering = np.random.uniform(*mode_config['steering_range'])
    else:
        # 直行
        steering = np.random.uniform(-0.1, 0.1)

    # 油门
    throttle = np.random.uniform(*mode_config['throttle_range'])

    return np.array([steering, throttle], dtype=np.float32)


def smooth_action(current_action, prev_action, alpha=0.7):
    """
    平滑动作序列

    Args:
        current_action: 当前动作
        prev_action: 上一个动作
        alpha: 平滑系数（越大越平滑）

    Returns:
        smoothed_action
    """
    if prev_action is None:
        return current_action

    return alpha * prev_action + (1 - alpha) * current_action


# 在文件开头导入carla（如果未导入）
try:
    import carla
except ImportError:
    print("Warning: CARLA module not found. Some functions may not work.")
    carla = None
