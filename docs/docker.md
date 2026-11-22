# Docker 使用说明

本文补充 [项目 README](../README.md) 中的环境配置步骤。使用 Docker 时，Python、PyTorch 和系统依赖由 `Dockerfile` 安装。论文权重、数据集和评测命令沿用 README。

## 1. 准备目录

按 README 克隆官方 RIFE 仓库，将它与本项目并排放置：

    Project/
    ├── CS_543_Vedio_Interpolation/
    └── ECCV2022-RIFE/

`compose.yaml` 会将本项目挂载到容器的 `/workspace`，将官方 RIFE 仓库挂载到 `/rife`。

## 2. 构建镜像

本配置需要容器能够访问 NVIDIA GPU。在 Windows 上，启动 Docker Desktop 并启用 WSL2 后端。

在本项目根目录运行以下命令，构建 Docker 镜像：

    docker build -t cs543-vfi:cu128 .

## 3. 进入开发容器

**使用 VS Code**：在 VS Code 中安装 Dev Containers 扩展并打开本项目。执行 **Dev Containers: Reopen in Container**，然后打开终端。扩展会通过项目配置连接 Compose 的 `dev` 服务。

## 4. 准备数据并运行评测

在容器终端进入官方 RIFE 仓库：

    cd /rife

从 README 第 3 节的 `gdown` 命令开始准备论文权重，再按第 4 节准备 Middlebury 和 UCF101 数据。通过浏览器下载的 UCF101 压缩包可以放到宿主机上的 `ECCV2022-RIFE/`，容器会在 `/rife` 下看到它。

权重和数据备齐后，在 `/rife` 运行：

    python benchmark/MiddleBury_Other.py
    python benchmark/UCF101.py

评测参考值见 [项目 README](../README.md)。

如需保存预测帧、对比图和逐样本分数，在容器终端运行以下命令。结果保存在本项目的 `outputs/` 目录。

    python /workspace/tools/save_predictions.py --rife /rife --out /workspace/outputs
