# Go1 三课快速开始

本仓库基于 MotrixLab `v0.4.0b1`，包含 Go1 平地行走、坡道平台停稳、平地路点导航三个任务。环境、配置和资产引用已并入仓库，无需手动复制课程文件。

上游源码基线：[Motphys/MotrixLab，c9f9c7c42f65a573aaf6e3a53cede2aac28bf3c3](https://github.com/Motphys/MotrixLab/commit/c9f9c7c42f65a573aaf6e3a53cede2aac28bf3c3)。本仓库保留上游许可和第三方资产声明。

## 1. 克隆与资产

准备 Git、Git LFS 和 uv。Windows 建议使用英文目录，例如 `F:\Robotics`，避免 Python 3.10 的可编辑安装在中文路径下出现编码问题。

```bash
git clone https://github.com/Oder2333/MotrixLab-Go1-Course.git
cd MotrixLab-Go1-Course
git lfs install --local
git lfs pull --include="motrix_robots/src/motrix_robots/assets/go1/**,motrix_envs/src/motrix_envs/common/motphys-ground.png" --exclude=""
```

Go1 XML 引用的 STL 网格和地面纹理由 Git LFS 管理。第二课的高度场由 Python 在创建环境时生成；第三课的路点标记 XML 已随源码提交。

未修改的上游 LFS 资产通过仓库内的 `.lfsconfig` 从 Motphys/MotrixLab 的公开资产端点下载。请保持该配置，确保 Git LFS 能获取网格与纹理。

## 2. 安装课程依赖

在仓库根目录执行。三课均使用 RSL-RL，因此需要显式选择 `--extra rslrl`。

```bash
uv sync --frozen --inexact --no-default-groups --extra cuda --extra rslrl --extra tbb
```

这会使用锁文件中的 Python 3.10 环境和 CUDA 版 PyTorch。没有可用 CUDA 显卡时，训练器会自动使用 CPU；CPU 适合先验证流程，正式训练耗时取决于机器性能。渲染另外需要可用的图形驱动。

Linux 激活环境：

```bash
source .venv/bin/activate
```

Windows PowerShell 激活环境：

```powershell
.\.venv\Scripts\Activate.ps1
```

下文命令都在已激活的虚拟环境中、仓库根目录运行。后续同步依赖时保持上述 extras，避免普通 `uv sync` 改变课程依赖组合。运行脚本直接使用 `python`，或使用 `uv run --no-sync python`。

## 3. 先检查配置和小规模训练

以第二课为例，先确认 Hydra 能组合任务与算法配置：

```bash
python -X utf8 scripts/train.py task=go1-slope-platform/rslrl.ppo --cfg job
```

再运行 16 个环境、2 轮训练：

```bash
python -X utf8 scripts/train.py task=go1-slope-platform/rslrl.ppo num_envs=16 seed=42 algo.max_iterations=2 render=false play=false
```

出现 `Training completed` 表示环境、算法和模型保存流程已跑通。两轮训练不足以学会任务。CPU 机器先观察每轮耗时，再决定正式训练规模；减少环境数会减少每轮采样量，通常需要重新评估迭代次数与学习效果。

## 4. 三课正式训练

下列为课程云端实验使用的规模，使用同一组奖励、观测和任务设置。训练结果受硬件、运行环境及随机过程影响，应结合日志和回放验收。

第一课：跟踪速度指令，学习平地行走。

```bash
python -X utf8 scripts/train.py task=go1-walk-flat/rslrl.ppo num_envs=1024 seed=42 algo.max_iterations=500 render=false play=false
```

第二课：从坡前平地出发，爬坡到平台目标附近，并连续停稳 3 秒。

```bash
python -X utf8 scripts/train.py task=go1-slope-platform/rslrl.ppo num_envs=512 seed=42 algo.max_iterations=3000 render=false play=false
```

第三课：在平地依次经过 A、B、C 路点，并在最终目标附近连续停稳 3 秒。

```bash
python -X utf8 scripts/train.py task=go1-waypoint-flat/rslrl.ppo num_envs=512 seed=42 algo.max_iterations=3000 render=false play=false
```

任务规则根据目标位置生成速度指令，策略学习机器人关节控制。第三课提供固定路点路线；修改路线时，配置与可视化标记应保持一致。

## 5. 预览、回放与日志

预览环境和地形，不加载策略：

```bash
python -X utf8 scripts/view.py env=go1-slope-platform num_envs=1
python -X utf8 scripts/view.py env=go1-waypoint-flat num_envs=1
```

训练结果保存在 `runs/<环境名>/rslrl/torch/ppo/<运行时间>/`。将下面路径中的 `<运行时间>` 替换成所需训练目录，再回放对应模型：

```bash
python -X utf8 scripts/play.py env=go1-walk-flat policy="runs/go1-walk-flat/rslrl/torch/ppo/<运行时间>/checkpoints/latest.pt" num_envs=1 seed=42
python -X utf8 scripts/play.py env=go1-slope-platform policy="runs/go1-slope-platform/rslrl/torch/ppo/<运行时间>/checkpoints/latest.pt" num_envs=1 seed=42
python -X utf8 scripts/play.py env=go1-waypoint-flat policy="runs/go1-waypoint-flat/rslrl/torch/ppo/<运行时间>/checkpoints/latest.pt" num_envs=1 seed=42
```

需要录制视频时，在对应回放命令末尾添加：

```bash
record_video=true record_seconds=30 record_width=1280 record_height=720
```

查看全部课程日志：

```bash
tensorboard --logdir runs
```

第二、三课的 `success/Episode` 表示已结束回合中的成功比例；`task/success` 是逐步事件统计，两者口径不同。验收时同时检查路线完成、最终停稳和失败情况。

## 6. 源码位置

| 内容 | 文件位置 |
| --- | --- |
| 通用训练、预览、回放入口 | `scripts/train.py`、`scripts/view.py`、`scripts/play.py` |
| 三课训练配置 | `configs/task/go1-walk-flat/`、`configs/task/go1-slope-platform/`、`configs/task/go1-waypoint-flat/` |
| 机器人参数与模型 | `motrix_robots/src/motrix_robots/unitree.py`、`motrix_robots/src/motrix_robots/assets/go1/` |
| 平地任务与共用行走逻辑 | `motrix_envs/src/motrix_envs/locomotion/quadruped/go1.py`、同目录 `cfg.py`、`walk_np.py`、`velocity_command.py` |
| 第二课配置、任务、判断与地形 | `motrix_envs/src/motrix_envs/locomotion/quadruped/go1_platform/` |
| 第三课配置、任务、判断与标记 | `motrix_envs/src/motrix_envs/locomotion/quadruped/go1_waypoint/` |
| 新任务导入注册 | `motrix_envs/src/motrix_envs/locomotion/quadruped/__init__.py` |
| RSL-RL 回合统计转发 | `motrix_rl_rslrl/src/motrix_rl_rslrl/torch/wrap_np.py` |

本仓库提供任务源码与训练入口，策略权重由使用者训练生成。上游项目和第三方资产的许可分别见 `LICENSE`、`THIRD_PARTY_NOTICES.md` 及资产目录中的许可证。
