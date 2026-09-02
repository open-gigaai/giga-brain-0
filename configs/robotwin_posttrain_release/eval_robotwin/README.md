# GigaBrain 0.7 RoboTwin 2.0 Evaluation

本目录提供 GigaBrain 0.7 在 RoboTwin 2.0 上 50 个双臂任务的训练和评测脚本。评测协议固定为：
- 14D observation state
- 14D model action
- 一个服务端端口，对应 N 个并行仿真客户端（请求在服务端排队串行推理）
```text
state  = left_joints(6) + left_gripper(1)
       + right_joints(6) + right_gripper(1)             = 14D

action = left_joints(6) + left_gripper(1)
       + right_joints(6) + right_gripper(1)             = 14D
```

每个任务在 `demo_clean` 和 `demo_randomized` 两种场景配置下各跑 100 集，共 50 × 2 = 100 项、10000 集。

> **训练只用 clean 数据，评测覆盖 randomized。**

## 仿真环境

评测依赖 RoboTwin 2.0 仿真环境，请按官方文档自行搭建：
- 官方仓库：https://github.com/RoboTwin-Platform/RoboTwin
- 安装文档：https://robotwin-platform.github.io/doc/usage/robotwin-install.html

本目录只提供策略侧脚本，不包含 RoboTwin 本体。环境装好后，需要把客户端脚本拷进 RoboTwin 仓库运行（客户端要 `import` RoboTwin 的 `envs` 模块）：

```bash
cd /path/to/RoboTwin
cp /path/to/eval_robotwin/scripts/robotwin_eval_client.py communication/
```

`communication/` 是本目录约定的存放位置，RoboTwin 仓库里没有这个目录时先 `mkdir communication`。

## 训练

训练数据只取 50 个任务的 `demo_clean` 采集轨迹，合并成一个 LeRobot 数据集根目录。
**不要把 `demo_randomized` 数据混进训练集**，否则 randomized 评测就不再是跨域泛化。

```bash
## 1. 把 RoboTwin demo_clean 数据转成 LeRobot v2.1格式

## 2. 计算 norm stats

## 3. 启动后训练（8 卡 FSDP2，80k steps）
export GIGABRAIN_ROOT=/path/to/giga-brain-0
bash configs/robotwin_posttrain_release/scripts/train_robotwin.sh
```

训练前先改配置里的用户配置区：`DATA_ROOT` / `NORM_STATS_PATH` / `PRETRAINED_CKPT` / `PROJECT_DIR`。
checkpoint 落在 `<PROJECT_DIR>/models/checkpoint_*/`，评测用其中的 `model_ema/`。

数据集的 `meta/info.json` 必须包含 `"robot_type": "agilex_cobot_magic"`。发布配置通过
`robot_type_embodiment_id_overrides` 将它显式映射到 embodiment ID 0，以对齐参考训练和发布
checkpoint；`run_giga_server.sh` 因此也默认使用 ID 0。若自行改动训练映射，评测时必须同步设置
`EMBODIMENT_ID`，否则会调用错误的 embodiment-specific action projection。

### 参考训练日志

`logs/train_robotwin_reference.log` 是我们实际跑这个任务的日志，可以用来对齐 loss 曲线和各阶段耗时。
日志中的 `GigaBrain0Trainer` / `GigaBrain0Transform` 是发布前的旧类名；当前发布入口分别为
`GigaBrain07Trainer` / `GigaBrain07Transform`，模型协议不变。

注意日志的并行规模和本目录配置不同：日志是多机 32 进程跑的（`batch_size_per_gpu=32`，全局 batch 1024），
所以只跑到 25000 步；本目录的 config 是 8 卡（全局 batch 256），要 100000 步才等价。
最终评测用的是 32进程 **20000 步** 的权重，等价于8卡 80000步权重。
config 里 `max_steps=80000`、`checkpoint_keeps` 按 1 万步取点，就是照这个来的。

## 评测

```bash
## 终端 1：启动 GigaBrain Policy Server
cd /path/to/eval_robotwin
conda activate gigabrain
export GIGABRAIN_ROOT=/path/to/giga-brain-0                 # 项目目录
export MODEL_PATH=/path/to/checkpoint/model_ema             # 模型目录
export NORM_STATS_PATH=/path/to/norm_stats.json             # norm_stats.json
export PRETRAINED_PATH=/path/to/models--google--paligemma2-3b-pt-224
export FAST_TOKENIZER_PATH=/path/to/models--physical-intelligence--fast
export PORT=8081
CUDA_VISIBLE_DEVICES=0 bash scripts/run_giga_server.sh
# 同一个 server 同时服务 demo_clean 和 demo_randomized，不需要换权重

## 终端 2：启动 RoboTwin 并行评测客户端
cd /path/to/RoboTwin
conda activate RoboTwin
export SERVER_HOST=127.0.0.1
export SERVER_PORT=8081
export CLIENT_GPUS=0,1,2,3,4,5,6,7
export CLIENT_SCRIPT=$(pwd)/communication/robotwin_eval_client.py
# 参数：NUM_CLIENTS TEST_NUM SAVE_VIDEO POS_LOOKAHEAD_STEP
bash /path/to/eval_robotwin/scripts/run_robotwin_parallel.sh 24 100 0 30

## 查看结果
python /path/to/eval_robotwin/scripts/parse_eval_results.py \
  /path/to/RoboTwin/eval_result_gigabrain07_parallel/<timestamp> --accepted-totals 100
```

单任务调试：

```bash
cd /path/to/RoboTwin
python communication/robotwin_eval_client.py \
  --config task_config/demo_clean.yml \
  --task_name place_shoe \
  --ckpt_setting demo_clean \
  --test_num 5 \
  --host 127.0.0.1 --port 8081 \
  --image_mode float_native \
  --pos_lookahead_step 30
```

`image_mode` 保持 `float_native`，resize 交给服务端，避免两侧各 resize 一次。

### 预测 50 步、只执行 30 步

模型一次推理输出 `chunk_size=50` 步动作，但我们只执行前 30 步（`pos_lookahead_step 30`），
剩下 20 步丢掉、重新取观测再推理。

## 脚本说明

| 脚本 | 运行位置 | 说明 |
| --- | --- | --- |
| `run_giga_server.sh` | 服务端机器 | 调用仓库 `scripts/inference/inference_agilex_server_unified.py` |
| `robotwin_eval_client.py` | RoboTwin 仓库 `communication/` | 单任务仿真客户端，需拷进 RoboTwin 后运行 |
| `run_robotwin_parallel.sh` | RoboTwin 仓库根目录 | 拉起 N 个客户端并行跑 100 项 |
| `parse_eval_results.py` | 任意 | 汇总 `_result.txt` 输出成功率 |

`robotwin_eval_client.py` 由仓库里的 `scripts/inference/inference_agilex_client_unified_smooth.py`
改造而来：原脚本面向真机 AgileX，通过 ROS 收发观测和动作，仿真里跑不了。改造后观测走
`env.get_obs()`、动作走 `env.take_action()`，并补上回合管理、成功率统计和断线重连。
ZMQ 协议、`observation.*` 键名、CHW 布局、`float_native` 归一化、14 维拼接顺序保持不变。
