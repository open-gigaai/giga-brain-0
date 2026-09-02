#!/usr/bin/env python3
"""GigaBrain 0.7 EBench evaluation client with native step-delta base actions."""

from __future__ import annotations

import argparse
from io import BytesIO
import os
import sys
import time
from typing import Any

import numpy as np


STATE_DIM = 14  # EBench 双臂状态维度
ACTION_DIM = 17  # EBench 双臂动作和 base step delta 维度
WARMUP_STEPS = 3  # 正式评测前的固定预热次数
MODEL_TIMEOUT_MS = 600_000  # 单次模型推理超时时间
PING_TIMEOUT_MS = 15_000  # 模型服务检查超时时间
STEP_TIMEOUT = 600.0  # GenManip step 请求超时时间
RESET_TIMEOUT = 3_000.0  # GenManip reset 请求超时时间
MAX_RECONNECTS = 3  # GenManip 请求失败后的最大重连次数
RECONNECT_SLEEP = 2.0  # GenManip 重连等待时间


class GigaBrainPolicyClient:
    """连接 GigaBrain ZMQ 服务并执行固定的 EBench step-delta 协议。"""

    def __init__(self, host: str, port: int, horizon: int):
        import zmq

        self.zmq = zmq
        self.host = host
        self.port = port
        self.horizon = horizon
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REQ)
        self.socket.setsockopt(zmq.LINGER, 0)
        self.socket.setsockopt(zmq.IMMEDIATE, 1)
        self.socket.setsockopt(zmq.RCVTIMEO, MODEL_TIMEOUT_MS)
        self.socket.setsockopt(zmq.SNDTIMEO, MODEL_TIMEOUT_MS)
        self.socket.connect(f"tcp://{host}:{port}")
        self.reset_model_memory = True

    @staticmethod
    def _serialize(data: Any) -> bytes:
        import torch

        buffer = BytesIO()
        torch.save(data, buffer)
        return buffer.getvalue()

    @staticmethod
    def _deserialize(data: bytes) -> Any:
        import torch

        return torch.load(BytesIO(data), map_location="cpu", weights_only=False)

    def _call(self, endpoint: str, data: dict[str, Any] | None = None) -> Any:
        request: dict[str, Any] = {"endpoint": endpoint}
        if data is not None:
            request["data"] = data
        try:
            self.socket.send(self._serialize(request))
            message = self.socket.recv()
        except self.zmq.Again as exc:
            raise TimeoutError(
                f"GigaBrain request {endpoint!r} timed out at "
                f"tcp://{self.host}:{self.port}"
            ) from exc
        if message == b"ERROR":
            raise RuntimeError(f"GigaBrain server returned ERROR for endpoint {endpoint!r}")
        return self._deserialize(message)

    def check_server(self) -> dict[str, Any]:
        old_recv_timeout = self.socket.getsockopt(self.zmq.RCVTIMEO)
        old_send_timeout = self.socket.getsockopt(self.zmq.SNDTIMEO)
        self.socket.setsockopt(self.zmq.RCVTIMEO, PING_TIMEOUT_MS)
        self.socket.setsockopt(self.zmq.SNDTIMEO, PING_TIMEOUT_MS)
        try:
            ping_result = self._call("ping")
            server_info = self._call("server_info")
        finally:
            self.socket.setsockopt(self.zmq.RCVTIMEO, old_recv_timeout)
            self.socket.setsockopt(self.zmq.SNDTIMEO, old_send_timeout)

        if not isinstance(server_info, dict):
            raise TypeError(f"server_info must be a dict, got {type(server_info).__name__}")
        expected = {
            "state_input_dim": STATE_DIM,
            "action_output_dim": ACTION_DIM,
            "is_robot_moving": True,
        }
        mismatches = {
            key: (server_info.get(key), value)
            for key, value in expected.items()
            if server_info.get(key) != value
        }
        if mismatches:
            details = ", ".join(
                f"{key}={actual!r} (expected {wanted!r})"
                for key, (actual, wanted) in mismatches.items()
            )
            raise ValueError(f"GigaBrain server is not an EBench 14D/17D deployment: {details}")

        print(f"[INFO] GigaBrain ping: {ping_result}", flush=True)
        print(
            "[INFO] GigaBrain server_info: "
            f"state_dim={server_info['state_input_dim']}, "
            f"action_dim={server_info['action_output_dim']}, "
            "base_mode=step_delta",
            flush=True,
        )
        return server_info

    @staticmethod
    def _image_tensor(image: Any) -> Any:
        import torch

        array = np.asarray(image)
        if array.ndim != 3:
            raise ValueError(f"Expected image shape (H,W,C) or (C,H,W), got {array.shape}")
        if array.shape[0] in (1, 3, 4) and array.shape[-1] not in (1, 3, 4):
            chw = array
        else:
            chw = np.transpose(array, (2, 0, 1))
        tensor = torch.as_tensor(chw, dtype=torch.float32)
        if tensor.numel() and tensor.max().item() > 1.5:
            tensor = tensor / 255.0
        return tensor.contiguous()

    @staticmethod
    def _field(worker_obs: dict[str, Any], key: str, worker_id: str) -> Any:
        if key not in worker_obs:
            available = ", ".join(sorted(map(str, worker_obs)))
            raise KeyError(
                f"worker {worker_id!r} observation missing {key!r}; "
                f"available keys: {available}"
            )
        return worker_obs[key]

    def _prepare_observation(self, obs: dict[str, Any], worker_id: str) -> dict[str, Any]:
        import torch

        worker_data = obs.get(worker_id)
        if not isinstance(worker_data, dict) or not isinstance(worker_data.get("obs"), dict):
            raise RuntimeError(f"worker {worker_id!r} returned an invalid observation")
        worker_obs = worker_data["obs"]

        joints = np.asarray(
            self._field(worker_obs, "state.joints", worker_id), dtype=np.float32
        ).reshape(-1)
        gripper = np.asarray(
            self._field(worker_obs, "state.gripper", worker_id), dtype=np.float32
        ).reshape(-1)
        if joints.shape[0] != 12 or gripper.shape[0] != 4:
            raise ValueError(
                f"worker {worker_id!r} expected joints/gripper dims 12/4, "
                f"got {joints.shape[0]}/{gripper.shape[0]}"
            )
        state = np.concatenate(
            [joints[:6], gripper[0:1], joints[6:12], gripper[2:3]]
        ).astype(np.float32)
        if state.shape != (STATE_DIM,):
            raise ValueError(f"Expected observation.state shape ({STATE_DIM},), got {state.shape}")

        request = {
            "video.overlook_camera_view": self._image_tensor(
                self._field(worker_obs, "video.overlook_camera_view", worker_id)
            ),
            "video.left_camera_view": self._image_tensor(
                self._field(worker_obs, "video.left_camera_view", worker_id)
            ),
            "video.right_camera_view": self._image_tensor(
                self._field(worker_obs, "video.right_camera_view", worker_id)
            ),
            "observation.state": torch.as_tensor(state, dtype=torch.float32),
            "task": self._field(worker_obs, "instruction", worker_id),
        }
        if self.reset_model_memory:
            request["reset_observation_memory"] = True
        return request

    @staticmethod
    def _action_array(result: Any) -> np.ndarray:
        if isinstance(result, dict):
            if "actions" in result:
                result = result["actions"]
            elif "action" in result:
                result = result["action"]
            else:
                raise KeyError(f"GigaBrain result has no action key: {list(result)}")
        if hasattr(result, "detach") and hasattr(result, "cpu"):
            result = result.detach().cpu().numpy()
        actions = np.asarray(result)
        if actions.ndim != 2 or actions.shape[1] != ACTION_DIM:
            raise ValueError(
                f"Expected model action shape (T, {ACTION_DIM}), got {actions.shape}"
            )
        if not np.isfinite(actions).all():
            raise ValueError("Model action contains NaN or Inf")
        return actions

    def convert_action_chunk(self, result: Any) -> list[dict[str, Any]]:
        actions = self._action_array(result)
        usable_horizon = min(self.horizon, actions.shape[0])
        if usable_horizon <= 0:
            raise ValueError(f"Model returned an empty action chunk: {actions.shape}")

        converted = []
        for action in actions[:usable_horizon]:
            joint_action = np.concatenate(
                [
                    action[:6],
                    np.repeat(action[6], 2),
                    action[7:13],
                    np.repeat(action[13], 2),
                ]
            ).astype(np.float32)
            converted.append(
                {
                    "action": joint_action,
                    "base_motion": np.asarray(action[14:17], dtype=np.float32),
                    "control_type": "joint_position",
                    "is_rel": False,
                    "base_is_rel": True,
                }
            )
        return converted

    def infer(self, obs: dict[str, Any], worker_id: str) -> list[dict[str, Any]]:
        request = self._prepare_observation(obs, worker_id)
        start = time.time()
        result = self._call("inference", request)
        self.reset_model_memory = False
        print(
            f"[MODEL] worker={worker_id} inference returned in {time.time() - start:.3f}s",
            flush=True,
        )
        return self.convert_action_chunk(result)

    def warmup(self) -> None:
        import torch

        image = torch.full((3, 224, 224), 0.5, dtype=torch.float32)
        request = {
            "video.overlook_camera_view": image,
            "video.left_camera_view": image,
            "video.right_camera_view": image,
            "observation.state": torch.zeros(STATE_DIM, dtype=torch.float32),
            "task": "do nothing",
            "reset_observation_memory": True,
        }
        for index in range(WARMUP_STEPS):
            start = time.time()
            self._call("inference", request)
            print(
                f"[WARMUP] inference {index + 1}/{WARMUP_STEPS} "
                f"finished in {time.time() - start:.3f}s",
                flush=True,
            )
        self.reset_episode()

    def reset_episode(self) -> None:
        self.reset_model_memory = True

    def close(self) -> None:
        self.socket.close(linger=0)
        self.context.term()


class EBenchEvalRunner:
    """管理单个 GenManip worker 的 reset、step、重连和结束状态。"""

    def __init__(
        self,
        policy: GigaBrainPolicyClient,
        eval_endpoint: str,
        run_id: str,
        worker_id: str,
        token: str,
    ):
        self.policy = policy
        self.eval_endpoint = eval_endpoint.rstrip("/")
        self.run_id = run_id
        self.worker_id = worker_id
        self.token = token
        self.eval_client: Any | None = None

    def _new_eval_client(self) -> Any:
        from genmanip_client import EvalClient

        return EvalClient(
            base_url=self.eval_endpoint,
            worker_ids=[self.worker_id],
            run_id=self.run_id,
            token=self.token or None,
            save_result=False,
            save_process=False,
            step_timeout=STEP_TIMEOUT,
            reset_timeout=RESET_TIMEOUT,
        )

    def _reset(self) -> dict[str, Any] | None:
        try:
            obs = self.eval_client.reset()
        except RuntimeError as exc:
            if "InsufficientResourcesError" in str(exc):
                print(
                    f"[INFO] worker {self.worker_id} 已经没有可分配的任务或执行 slot，客户端正常结束。",
                    flush=True,
                )
                return None
            raise
        self.policy.reset_episode()
        worker_data = obs.get(self.worker_id)
        if isinstance(worker_data, dict) and worker_data.get("obs") is None:
            print(
                f"[INFO] worker {self.worker_id} 已经没有可分配的任务，客户端正常结束。",
                flush=True,
            )
            return None
        print(f"[RESET] received obs for worker {self.worker_id}", flush=True)
        return obs

    def _started_new_episode(self, obs: dict[str, Any]) -> bool:
        worker_data = obs.get(self.worker_id)
        if not isinstance(worker_data, dict):
            return False
        if worker_data.get("episode_result") is not None:
            return True
        worker_obs = worker_data.get("obs")
        return isinstance(worker_obs, dict) and bool(worker_obs.get("reset"))

    def run(self) -> int:
        self.eval_client = self._new_eval_client()
        obs = self._reset()
        if obs is None:
            return 0

        reconnect_count = 0
        total_steps = 0
        finished = False
        while not finished:
            if self.worker_id not in obs:
                raise RuntimeError(
                    f"worker {self.worker_id!r} not found in observation: {list(obs)}"
                )
            actions = self.policy.infer(obs, self.worker_id)
            action_chunk = [{self.worker_id: action} for action in actions]
            try:
                start = time.time()
                obs, finished = self.eval_client.step(action_chunk)
                reconnect_count = 0
                total_steps += len(action_chunk)
                if not finished and self._started_new_episode(obs):
                    self.policy.reset_episode()
                    print("[RESET] cleared GigaBrain observation memory", flush=True)
                print(
                    f"[STEP] sent_chunk={len(action_chunk)} total_steps={total_steps} "
                    f"elapsed={time.time() - start:.3f}s done={finished}",
                    flush=True,
                )
            except Exception as exc:
                reconnect_count += 1
                print(
                    f"[WARN] eval step failed ({reconnect_count}/{MAX_RECONNECTS}): {exc}",
                    file=sys.stderr,
                    flush=True,
                )
                if reconnect_count > MAX_RECONNECTS:
                    raise
                try:
                    self.eval_client.close()
                except Exception as close_exc:
                    print(f"[WARN] eval client close failed: {close_exc}", file=sys.stderr)
                time.sleep(RECONNECT_SLEEP)
                self.eval_client = self._new_eval_client()
                obs = self._reset()
                if obs is None:
                    return 0

        print("[INFO] Evaluation finished.", flush=True)
        return 0

    def close(self) -> None:
        if self.eval_client is not None:
            self.eval_client.close()


# 解析评测客户端命令行参数
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="GigaBrain 0.7 EBench client using native 17D step-delta actions."
    )
    parser.add_argument("--worker-id", default=os.environ.get("WORKER_ID", "0"))
    parser.add_argument(
        "--eval-endpoint",
        default=os.environ.get("EVAL_ENDPOINT", "http://127.0.0.1:8087"),
    )
    parser.add_argument("--run-id", default=os.environ.get("RUN_ID", ""))
    parser.add_argument(
        "--eval-token",
        default=os.environ.get("EBENCH_SUBMIT_TOKEN", ""),
    )
    parser.add_argument("--model-host", default=os.environ.get("MODEL_HOST", "127.0.0.1"))
    parser.add_argument(
        "--model-port",
        type=int,
        default=int(os.environ.get("MODEL_PORT", "8000")),
    )
    parser.add_argument(
        "--horizon",
        type=int,
        default=int(os.environ.get("HORIZON", "20")),
    )
    return parser.parse_args()


# 启动单 worker 的 EBench 评测流程
def main() -> int:
    args = parse_args()
    args.model_host = args.model_host.strip()
    args.eval_endpoint = args.eval_endpoint.strip().rstrip("/")
    if not args.run_id:
        raise ValueError("RUN_ID is required. Set RUN_ID or pass --run-id.")
    if not args.model_host:
        raise ValueError("MODEL_HOST must not be empty.")
    if args.horizon <= 0:
        raise ValueError("HORIZON must be a positive integer.")

    print("=" * 72)
    print("GigaBrain 0.7 EBench evaluation client")
    print(f"Eval endpoint : {args.eval_endpoint}")
    print(f"Run ID        : {args.run_id}")
    print(f"Worker ID     : {args.worker_id}")
    print(f"Model server  : tcp://{args.model_host}:{args.model_port}")
    print(f"Horizon       : {args.horizon}")
    print("Base mode     : step_delta")
    print(f"Warmup steps  : {WARMUP_STEPS}")
    print("=" * 72, flush=True)

    policy: GigaBrainPolicyClient | None = None
    runner: EBenchEvalRunner | None = None
    try:
        policy = GigaBrainPolicyClient(args.model_host, args.model_port, args.horizon)
        policy.check_server()
        policy.warmup()
        runner = EBenchEvalRunner(
            policy=policy,
            eval_endpoint=args.eval_endpoint,
            run_id=args.run_id,
            worker_id=str(args.worker_id),
            token=args.eval_token,
        )
        return runner.run()
    except KeyboardInterrupt:
        print("\n[INFO] Interrupted by user.", flush=True)
        return 130
    finally:
        if runner is not None:
            runner.close()
        if policy is not None:
            policy.close()


if __name__ == "__main__":
    raise SystemExit(main())
