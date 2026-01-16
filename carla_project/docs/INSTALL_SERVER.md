# CARLA 服务器安装（0.9.16）

本文档用于安装 CARLA 0.9.16 服务器，并与本项目脚本保持一致。

## 1. 下载与解压

```bash
cd ~
wget https://carla-releases.s3.us-east-005.backblazeb2.com/Linux/CARLA_0.9.16.tar.gz
mkdir -p ~/CARLA_0.9.16
tar -xzf CARLA_0.9.16.tar.gz -C ~/CARLA_0.9.16
```

> `bin/start_carla_server.sh` 默认使用 `~/CARLA_0.9.16` 作为服务器目录。
> 如果你安装在其他路径，请修改脚本中的 `CARLA_DIR`。

## 2. 启动服务器

```bash
cd ~/CARLA_0.9.16
./CarlaUE4.sh -RenderOffScreen
```

也可以直接使用脚本：
```bash
./bin/start_carla_server.sh
```

## 3. Python API 版本

Python API 版本需要与服务器版本一致。当前仓库的 `requirements_carla.txt` 仍固定为 `carla==0.9.15`，如使用 0.9.16 服务器，请同步更新为：

```bash
pip install carla==0.9.16
```

## 4. 连接验证

```bash
python - << 'PY'
import carla
client = carla.Client('localhost', 2000)
client.set_timeout(10.0)
world = client.get_world()
print('Connected to:', world.get_map().name)
PY
```

如果连接失败，请确认：
- 服务器已启动（端口 2000）
- 本机防火墙未阻止连接
- Python API 版本与服务器一致
