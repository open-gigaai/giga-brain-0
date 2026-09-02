# GigaBrain 0.7 EBench Evaluation

本目录提供 GigaBrain 0.7 的 EBench 本地和在线评测脚本。评测协议固定为：
- 14D observation state
- 17D model action
- 一个服务端端口，对应一个客户端，一个 GenManip worker
```text
state  = left_joints(6) + left_gripper(1)
       + right_joints(6) + right_gripper(1)             = 14D

action = left_joints(6) + left_gripper(1)
       + right_joints(6) + right_gripper(1)
       + base_step_delta(dx, dy, dtheta)                = 17D
```

## 在线评测

```bash
## 1. 启动 GigaBrain Policy Server
cd /path/to/eval_ebench
conda activate gigabrain
export GIGABRAIN_ROOT=/path/giga-brain-0            # 项目目录
export MODEL_PATH=/path/to/checkpoint/model_ema     # 模型目录
export NORM_STATS_PATH=/path/to/norm_stats.json     # norm_stats.json
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export BASE_PORT=8000
export NUM_PORTS=8
bash scripts/run_giga_policy_mul.sh

## 2. 提交评测任务，启动云端 GenManip Eval Server
# 获取个人 token：https://internrobotics.shlab.org.cn/eval/api-keys
export EBENCH_SUBMIT_TOKEN="<your-token>"
gmp online submit \
  --base_url https://internrobotics.shlab.org.cn/eval \
  --token "${EBENCH_SUBMIT_TOKEN}" \
  --benchmark_set ebench_generalist \
  --model_name "GigaBrain-0.7" \
  --model_type VLA \
  --submitter_name "<your-name>" \
  --submitter_homepage "<your-homepage>" \
  --is_public 0

## 3. 记录返回的 `RUN_ID` 和 `EVAL_ENDPOINT`
export RUN_ID="<returned-run-id>"
export EVAL_ENDPOINT="<returned-eval-endpoint>"

## 4. 启动本地客户端
cd /path/to/eval_ebench
conda activate genmanip
export MODEL_HOST=127.0.0.1
export BASE_PORT=8000
export NUM_PORTS=8
export HORIZON=20
bash scripts/run_ebench_mul.sh

## 5. 查看在线进度：
# 查看进度 https://internrobotics.shlab.org.cn/eval/online-evaluation
gmp status \
  --url "${EVAL_ENDPOINT}" \
  --token "${EBENCH_SUBMIT_TOKEN}" \
  --run_id "${RUN_ID}"

```

## 本地评测

```bash
## 终端 1：启动 GigaBrain Policy Server
cd /path/to/eval_ebench
conda activate gigabrain
export GIGABRAIN_ROOT=/path/giga-brain-0            # 项目目录
export MODEL_PATH=/path/to/checkpoint/model_ema     # 模型目录
export NORM_STATS_PATH=/path/to/norm_stats.json     # norm_stats.json
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export BASE_PORT=8000
export NUM_PORTS=8
bash scripts/run_giga_policy_mul.sh

### 终端 2：启动 GenManip Eval Server
cd /path/to/GenManip
conda activate genmanip 
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
python ray_eval_server.py --host 0.0.0.0 --port 8087

### 终端 3：提交任务并启动客户端
cd /path/to/eval_ebench
conda activate genmanip 
export RUN_ID=gb07_ebench_eval
export TASK="ebench/generalist/test_mini"
gmp submit "${TASK}" --run_id "${RUN_ID}"

export MODEL_HOST=127.0.0.1
export BASE_PORT=8000
export NUM_PORTS=8
export HORIZON=20
bash scripts/run_ebench_mul.sh

### 查看进度
watch -n 60 gmp status --url "http://0.0.0.0:8087" --run_id "$RUN_ID"

### 查看结果
python scripts/get_ebench_val.py \
  --dir /path/to/GenManip/saved/eval_results/ebench/<run-id>

```

