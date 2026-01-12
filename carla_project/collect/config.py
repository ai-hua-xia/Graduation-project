"""
CARLA数据采集配置文件
"""

# ============= 图像配置 =============
IMAGE_WIDTH = 256
IMAGE_HEIGHT = 256
FOV = 90  # Field of View，增大以看到更多场景

# ============= 采集配置 =============
EPISODES = 100  # 采集的episode数量
FRAMES_PER_EPISODE = 200  # 每个episode的帧数
FPS = 20  # 采集帧率
SKIP_FRAMES = 2  # 每隔多少帧采集一次（降低数据冗余）

# ============= CARLA场景配置 =============
TOWNS = [
    'Town01',  # 小城市，T型路口多
    'Town02',  # 小城市，曲线路多
    'Town03',  # 大城市，复杂路网
    'Town04',  # 高速公路+城市
    'Town05',  # 大城市，开放路口
]

WEATHERS = [
    'ClearNoon',      # 晴天中午
    'CloudyNoon',     # 多云中午
    'WetNoon',        # 雨天中午
    'ClearSunset',    # 晴天黄昏
]

# ============= 车辆配置 =============
VEHICLE_FILTER = 'vehicle.tesla.model3'  # 使用特斯拉Model 3

# ============= 驾驶策略配置 =============
DRIVING_MODES = {
    'normal': {  # 正常驾驶
        'steering_range': (-0.5, 0.5),
        'throttle_range': (0.3, 0.7),
        'turn_probability': 0.3,  # 30%概率转向
    },
    'turning_focus': {  # 专注转向采集
        'steering_range': (-0.8, 0.8),
        'throttle_range': (0.4, 0.6),
        'turn_probability': 0.7,  # 70%概率转向
    },
    'mixed': {  # 混合模式
        'steering_range': (-0.7, 0.7),
        'throttle_range': (0.2, 0.8),
        'turn_probability': 0.5,
    }
}

DEFAULT_DRIVING_MODE = 'turning_focus'  # 默认使用转向优先模式

# ============= 交通配置 =============
NUM_VEHICLES = 30  # 场景中其他车辆数量
NUM_PEDESTRIANS = 20  # 行人数量

# ============= 输出配置 =============
SAVE_RGB = True  # 保存RGB图像
SAVE_DEPTH = False  # 是否保存深度图（可选）
SAVE_SEMANTIC = False  # 是否保存语义分割（可选）

# ============= 相机配置 =============
CAMERA_CONFIG = {
    'image_size_x': IMAGE_WIDTH,
    'image_size_y': IMAGE_HEIGHT,
    'fov': FOV,
}

# 相机位置（相对于车辆）
CAMERA_TRANSFORM = {
    'x': 1.5,  # 前方1.5米
    'y': 0.0,  # 居中
    'z': 1.4,  # 高度1.4米（驾驶员视角）
    'pitch': -5.0,  # 稍微向下
    'yaw': 0.0,
    'roll': 0.0,
}

# ============= 同步模式配置 =============
SYNC_MODE = True  # 使用同步模式（确保数据时间对齐）
FIXED_DELTA_SECONDS = 1.0 / FPS  # 固定时间步长

# ============= 过滤配置 =============
MIN_SPEED = 2.0  # 最小速度（m/s），低于此速度的帧不保存（避免静止画面）
MAX_COLLISION_COUNT = 3  # 最大碰撞次数，超过则终止episode

# ============= 动作配置 =============
ACTION_DIM = 2  # 动作维度：[转向, 油门]
# 注意：刹车可以通过负油门表示，或者扩展到3维

# ============= 路径配置 =============
DEFAULT_OUTPUT_DIR = '../data/raw'
