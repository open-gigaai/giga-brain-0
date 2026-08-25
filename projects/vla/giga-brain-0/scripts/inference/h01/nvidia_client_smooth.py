"""H01 远程推理 client —— smooth(平滑滑动窗口)版本。

对标 ubuntu 上的 inference_agilex_client_unified_smooth.py，把它的平滑机制搬到 H01
(nvidia Jetson) 客户端上，但复用本目录 nvidia_client.py 的 H01 专有部分
(RobotController: 鱼眼相机 / 关节订阅 / 臂+爪+头+腰 publisher / get_frame / ready pose)
和图像预处理。默认使用 ZMQ 连接 inference_agilex_server_unified.py，旧 H01
websocket server 可通过 --transport websocket 继续使用。

平滑机制(与参考版一致，索引按 H01 16/22 维动作布局适配):
  1. 异步滑动窗口推理: 后台线程推理，主循环按 publish_rate 匀速发布，缓冲区剩余
     <= inference_trigger_remaining 时提前发起下一次推理，避免"推完再动"的同步卡顿。
  2. 自适应 horizon: calculate_adaptive_horizon 在预测轨迹发散快时(看双爪 7/15 的 L2)
     缩短执行 horizon，越早重推理。
  3. 拼接处平滑: apply_sliding_window_result 把最近已执行的一段 + 新预测拼起来做低通，
     用 align_search_window 按真实位姿就近对齐拼接起点，再用余弦 smoothstep 连续性偏置
     (continuity_blend_steps)把拼接处的位置阶跃抹平，消除切换卡顿。

H01 动作前 16 维布局: [左臂0-6, 左爪7, 右臂8-14, 右爪15]；22 维模式再包含
[头16-17, 腰18-21]。臂关节做低通，夹爪和头/腰保留模型原值。
只支持 qpose(关节控制，pick_fork/H01 用的就是 joint 控制)；endpose 需要 IK，本 smooth 版不支持。

安全: 默认 --execute_action 关闭(只推理+平滑，不下发)，与同步版一致，避免机械臂乱动。
"""

import argparse
import pickle
from pathlib import Path
import threading
import time

import cv2
import numpy as np
import rclpy

from openpi_client import image_tools

from h01_client_contract import build_unified_inference_request

# 复用同步版 (deployed 名为 nvidia_client.py) 的 H01 专有实现与工具函数
from nvidia_client import (
    RobotController,
    build_policy_state,
    build_server_state,
    validate_server_state,
    slice_actions_to_robot_dim,
    publish_action_to_robot,
    butterworth_lowpass_filter,
    READY_POSE,
    set_seed,
    SERVER_STATE_DIM,
    ROBOT_ACTION_DIM,
    FULL_BODY_ACTION_DIM,
    SUPPORTED_ACTION_DIMS,
)


TRAINING_FIRST_FRAME_PATH = (
    Path(__file__).resolve().parent
    / "gb1_pg2_push_buttons_h01_260815_first_frame.npy"
)


def load_ready_pose_npy(state_path):
    """Load a 16D/20D ready pose or convert a 22D H01 state to 20D."""
    state = np.load(state_path, allow_pickle=False).astype(np.float64).reshape(-1)
    if not np.isfinite(state).all():
        raise ValueError(f"ready-pose state contains non-finite values: {state_path}")
    if state.size == SERVER_STATE_DIM:
        # move_to_ready_pose accepts arms/grippers 16D followed by waist 4D;
        # head state (indices 16:18) is observed but is not part of ready pose.
        state = np.concatenate([state[:16], state[18:22]])
    elif state.size not in (ROBOT_ACTION_DIM, 20):
        raise ValueError(
            "ready-pose state must be 16D, 20D, or 22D, "
            f"got {state.size}D from {state_path}"
        )
    return state.tolist()


def load_training_first_frame_ready_pose(state_path=TRAINING_FIRST_FRAME_PATH):
    """Load the saved training state as a robot-ready pose."""
    return load_ready_pose_npy(state_path)


class UnifiedZmqClient:
    """Client adapter for ``inference_agilex_server_unified.py``."""

    _PICKLE_MAGIC = b"GIGA_NUMPY_PICKLE_V1\x00"

    def __init__(self, host: str, port: int, timeout_ms: int = 30000):
        import zmq

        self._context = zmq.Context()
        self._socket = self._context.socket(zmq.REQ)
        self._socket.setsockopt(zmq.LINGER, 0)
        self._socket.setsockopt(zmq.RCVTIMEO, timeout_ms)
        self._socket.setsockopt(zmq.SNDTIMEO, timeout_ms)
        self._socket.connect(f"tcp://{host}:{port}")

    def _serialize(self, value) -> bytes:
        return self._PICKLE_MAGIC + pickle.dumps(value, protocol=5)

    def _deserialize(self, value: bytes):
        if not value.startswith(self._PICKLE_MAGIC):
            raise ValueError(
                "unified server response does not support GIGA_NUMPY_PICKLE_V1"
            )
        return pickle.loads(value[len(self._PICKLE_MAGIC):])

    def _call_endpoint(self, endpoint: str, data=None, requires_input: bool = True):
        request = {"endpoint": endpoint}
        if requires_input:
            request["data"] = data
        self._socket.send(self._serialize(request))
        message = self._socket.recv()
        if message == b"ERROR":
            raise RuntimeError(
                f"Unified inference server failed at endpoint {endpoint!r}"
            )
        return self._deserialize(message)

    def get_server_info(self):
        return self._call_endpoint("server_info", requires_input=False)

    def infer(self, obs):
        obs = {**obs, "state": validate_server_state(obs["state"])}
        request = build_unified_inference_request(obs)
        actions = self._call_endpoint("inference", request)
        if hasattr(actions, "detach"):
            actions = actions.detach().cpu().numpy()
        return {"actions": actions}

    def close(self):
        self._socket.close(linger=0)
        self._context.term()


def validate_server_io(server_info, *, action_dim: int, chunk_size: int) -> int:
    """Validate fixed 22D state input and action-prefix availability."""
    server_state_dim = server_info.get("state_input_dim")
    if server_state_dim != SERVER_STATE_DIM:
        raise ValueError(
            "client/server state schema mismatch: "
            f"server={server_state_dim!r}, client={SERVER_STATE_DIM}"
        )

    server_action_dim = server_info.get(
        "action_output_dim", server_info.get("original_action_dim")
    )
    if server_action_dim is None:
        raise ValueError("server_info is missing action_output_dim")
    server_action_dim = int(server_action_dim)
    if server_action_dim < action_dim:
        raise ValueError(
            "server action width is smaller than the configured ROS publish width: "
            f"server={server_action_dim}, action_dim={action_dim}"
        )

    server_chunk_size = server_info.get("n_action_steps")
    if server_chunk_size != chunk_size:
        raise ValueError(
            "client/server action chunk mismatch: "
            f"server={server_chunk_size!r}, client={chunk_size}"
        )

    expected_image_keys = {
        "observation.images.cam_high",
        "observation.images.cam_left_wrist",
        "observation.images.cam_right_wrist",
    }
    actual_image_keys = set(server_info.get("present_img_keys") or [])
    if actual_image_keys != expected_image_keys:
        raise ValueError(
            "client/server image schema mismatch: "
            f"server={sorted(actual_image_keys)!r}, "
            f"expected={sorted(expected_image_keys)!r}"
        )

    return server_action_dim

# ==================== H01 16/22 维动作布局 ====================
LEFT_ARM_COLS = list(range(0, 7))      # 0..6  左臂 7 关节
LEFT_GRIP_COL = 7                      # 左夹爪
RIGHT_ARM_COLS = list(range(8, 15))    # 8..14 右臂 7 关节
RIGHT_GRIP_COL = 15                    # 右夹爪
GRIPPER_COLS = (LEFT_GRIP_COL, RIGHT_GRIP_COL)
ARM_COLS = LEFT_ARM_COLS + RIGHT_ARM_COLS  # 就近对齐只比臂关节，排除夹爪

# --dump_obs 已保存帧数(限量)
_dump_obs_saved = 0
_dump_act_saved = 0   # [debug] 已保存的动作块数量


# ==================== 平滑核心(H01 索引适配) ====================
def _maybe_apply_gripper_rescale(action, apply_rescale: bool):
    """Apply the AgileX gripper rescale to H01 gripper columns 7 and 15."""
    if not apply_rescale:
        return
    for gripper_col in GRIPPER_COLS:
        gripper = action[gripper_col]
        action[gripper_col] = -0.03 if gripper < 0.04 else gripper * 1.5


def smooth_action_sequence(actions: np.ndarray) -> np.ndarray:
    """对 (N,16/22) 双臂动作做低通，夹爪和可选头/腰维度保留。

    序列过短(filtfilt 需要一定长度)或滤波异常时原样返回。
    """
    if actions is None or len(actions) <= 6:
        return actions
    try:
        out = np.asarray(actions).copy()
        out[:, 0:7] = butterworth_lowpass_filter(np.asarray(actions[:, 0:7]).copy())
        out[:, 8:15] = butterworth_lowpass_filter(np.asarray(actions[:, 8:15]).copy())
        return out
    except Exception as exc:  # filtfilt 对过短序列会报错，退回不平滑
        print(f'[SMOOTH] filter skipped: {exc}')
        return actions


def calculate_adaptive_horizon(actions: np.ndarray, max_horizon: int,
                               inference_trigger_remaining: int, distance_thresh: float) -> int:
    """轨迹发散快时缩短 horizon: 双爪列(7/15)放大 10 倍后看若干检查点与首帧的 L2。"""
    if len(actions) < max_horizon:
        return len(actions)
    scaled = np.asarray(actions).copy()
    scaled[:, LEFT_GRIP_COL] *= 10.0
    scaled[:, RIGHT_GRIP_COL] *= 10.0
    scaled = scaled[:, [LEFT_GRIP_COL, RIGHT_GRIP_COL]]
    base = scaled[0]
    for check_point in (15, 20, 25, 30, 40):
        if check_point >= max_horizon:
            break
        l2 = np.linalg.norm(scaled[check_point] - base)
        if l2 > distance_thresh:
            adaptive_horizon = check_point + inference_trigger_remaining
            print(f'[ADAPTIVE_HORIZON] L2({l2:.4f}) > thresh({distance_thresh}) at cp={check_point} '
                  f'-> horizon={adaptive_horizon}')
            return adaptive_horizon
    return max_horizon


def apply_sliding_window_result(current_buffer, current_index, inference_result, num1, adaptive_num2,
                                adaptive_horizon, continuity_blend_steps=0, align_search_window=8):
    """拼接滑动窗口结果并在拼接处平滑(H01 索引)。返回新动作段，或 None 表示不更新。

    取最后(最多) num1 个已执行动作作低通上下文，与新预测(从就近匹配到的起点到
    adaptive_horizon)拼接、低通，丢弃上下文后返回新段；continuity_blend_steps>0 时叠加
    余弦 smoothstep 连续性偏置，强制新段首帧对齐到最近已执行动作、在该步数内衰减到 0。
    """
    try:
        old_count = max(0, min(num1, current_index))
        if old_count > 0:
            remaining_old = current_buffer[current_index - old_count: current_index]
        else:
            remaining_old = np.empty((0, current_buffer.shape[1]), dtype=current_buffer.dtype)

        # 就近匹配拼接起点: 以 adaptive_num2 为中心，在窗口内找与最近已执行动作臂关节最接近的预测帧
        new_start = adaptive_num2
        if current_index > 0 and align_search_window > 0 and len(inference_result) > 0:
            last_action = np.asarray(current_buffer[current_index - 1])
            arm_cols = [c for c in ARM_COLS if c < inference_result.shape[1]]
            lo = max(0, adaptive_num2 - align_search_window)
            hi = min(len(inference_result), adaptive_num2 + align_search_window + 1)
            if hi > lo and arm_cols:
                seg = inference_result[lo:hi][:, arm_cols]
                dist = np.linalg.norm(seg - last_action[arm_cols], axis=1)
                new_start = lo + int(np.argmin(dist))

        if new_start < adaptive_horizon:
            new_inf = inference_result[new_start:adaptive_horizon]
        elif new_start < len(inference_result):
            # 推理比裁剪后的 horizon 还慢: 至少执行最新一帧，避免拼接为空导致缓冲不更新卡死
            new_inf = inference_result[new_start:new_start + 1]
        else:
            new_inf = np.empty((0, current_buffer.shape[1]), dtype=current_buffer.dtype)

        if len(remaining_old) > 0 and len(new_inf) > 0:
            concatenated = np.concatenate([remaining_old, new_inf], axis=0)
        elif len(new_inf) > 0:
            concatenated = new_inf
        else:
            return None

        smoothed = smooth_action_sequence(concatenated)
        new_segment = smoothed[len(remaining_old):]
        if len(new_segment) == 0:
            return None

        # 连续性偏置: 低通只能把拼接处阶跃抹成半阶跃，这里强制新段首帧对齐到最近已执行动作，
        # 偏置在 continuity_blend_steps 步内余弦衰减到 0 -> 拼接处零跳变。夹爪(7/15)与 >=16 不偏置。
        if continuity_blend_steps > 0 and current_index > 0:
            new_segment = new_segment.copy()
            last_action = np.asarray(current_buffer[current_index - 1], dtype=new_segment.dtype)
            delta = (last_action - new_segment[0]).astype(new_segment.dtype)
            for g in GRIPPER_COLS:
                if g < delta.shape[0]:
                    delta[g] = 0.0
            if delta.shape[0] > 16:
                delta[16:] = 0.0
            k = min(continuity_blend_steps, len(new_segment))
            ramp = np.zeros(len(new_segment), dtype=new_segment.dtype)
            ramp[:k] = 0.5 * (1.0 + np.cos(np.pi * np.arange(k, dtype=new_segment.dtype) / k))
            new_segment = new_segment + delta[None, :] * ramp[:, None]
            # 首帧 == 上一帧已执行动作，上一拍刚发过，丢掉首帧避免保持一拍不动
            new_segment = new_segment[1:]
            if len(new_segment) == 0:
                return None
        return new_segment
    except Exception as exc:
        print(f'[SLIDING_APPLY] concat error: {exc}')
        return None


# ==================== smooth 控制器 ====================
class SmoothInferenceController:
    """异步滑动窗口推理 + 拼接平滑主控制器(H01 / openpi websocket)。"""

    def __init__(self, args, controller: RobotController, client):
        self.args = args
        self.controller = controller
        self.client = client

        self.publish_rate = args.publish_rate
        self.chunk_size = args.chunk_size
        self.inference_trigger_remaining = args.inference_trigger_remaining
        self.max_action_execute_horizon = args.max_action_execute_horizon
        self.distance_thresh = args.distance_thresh
        self.continuity_blend_steps = args.continuity_blend_steps
        self.align_search_window = args.align_search_window
        self.max_publish_step = args.max_publish_step
        self.apply_gripper_rescale = args.apply_gripper_rescale
        self.gripper_offset = args.gripper_offset
        self.action_dim = args.action_dim

        self.action_buffer = None
        self.action_index = 0
        self.action_buffer_lock = threading.Lock()

        self.pending_inference_result = None
        self.pending_inference_lock = threading.Lock()
        # 同一时刻只允许一个推理在飞：client 是单条 websocket，两个线程同时 infer 会报
        # "cannot call recv while another thread is already running recv"，并读到错位的旧响应。
        self.infer_inflight_lock = threading.Lock()
        self.infer_in_flight = False

        self.inference_thread = None
        self.inference_trigger_step = None

    # --------------------------------------------------------------------- #
    def _build_obs(self, trigger_step):
        """从最新一帧构建 openpi obs(与同步版 async_inference_thread 完全一致)。"""
        global _dump_obs_saved
        result = self.controller.get_frame()
        if not result:
            return None

        img_msgs = result[:-2]
        imgs = [
            cv2.cvtColor(
                image_tools.convert_to_uint8(image_tools.resize_with_pad(img, 224, 224)),
                cv2.COLOR_BGR2RGB,
            )
            for img in img_msgs
        ]
        qpos = np.array(result[-2] + result[-1], dtype=np.float32)
        policy_state = build_policy_state(qpos, self.args.policy_action_space)
        policy_state = build_server_state(policy_state, self.controller)

        images = {
            "cam_high": imgs[0].transpose(2, 0, 1),
            "cam_left_wrist_up": imgs[1].transpose(2, 0, 1),
            "cam_right_wrist_up": imgs[2].transpose(2, 0, 1),
        }
        if self.args.fisheye_mode == '5fisheye':
            images["cam_left_wrist_down"] = imgs[3].transpose(2, 0, 1)
            images["cam_right_wrist_down"] = imgs[4].transpose(2, 0, 1)

        obs = {
            "state": policy_state,
            "images": images,
            "prompt": self.args.prompt,
            "base_vel": [0.0, 0.0],
        }

        # 可选 dump(前 N 帧)：CHW RGB -> HWC BGR 存 PNG，state/prompt 存 txt
        if self.args.dump_obs and _dump_obs_saved < self.args.dump_obs_count:
            from pathlib import Path
            dump_dir = Path(self.args.dump_obs_dir)
            dump_dir.mkdir(parents=True, exist_ok=True)
            for key, chw in images.items():
                hwc_bgr = cv2.cvtColor(np.asarray(chw).transpose(1, 2, 0), cv2.COLOR_RGB2BGR)
                cv2.imwrite(str(dump_dir / f'step{trigger_step}_{key}.png'), hwc_bgr)
            with open(dump_dir / f'step{trigger_step}_obs.txt', 'w') as f:
                f.write(f'prompt: {self.args.prompt}\n')
                f.write(f'state(len={len(policy_state)}): '
                        f'{np.round(np.asarray(policy_state, dtype=np.float32), 5).tolist()}\n')
                f.write(f'image_keys: {list(images.keys())}\n')
            _dump_obs_saved += 1
            print(f'[DUMP] step {trigger_step} obs 已存到 {dump_dir} ({_dump_obs_saved}/{self.args.dump_obs_count})')

        return obs

    def _normalize_actions_shape(self, all_actions):
        all_actions = np.asarray(all_actions, dtype=np.float32)
        if all_actions.ndim == 3 and all_actions.shape[0] == 1:
            all_actions = all_actions[0]
        elif all_actions.ndim == 1:
            all_actions = all_actions[None, :]
        if all_actions.ndim != 2 or all_actions.shape[1] != self.action_dim:
            raise ValueError(
                f"expected action chunk (N,{self.action_dim}), got {all_actions.shape}"
            )
        return all_actions

    def _inference_async(self, trigger_step):
        """后台推理: 取帧 -> openpi 推理 -> 截到配置的 16/22 维 -> 写 pending。"""
        inference_start = time.time()
        try:
            obs = self._build_obs(trigger_step)
            if obs is None:
                print('[SMOOTH_INFER] frame sync failed')
                with self.pending_inference_lock:
                    self.pending_inference_result = None
                    self.inference_trigger_step = None
                return

            raw_actions = self.client.infer(obs)['actions']
            all_actions = slice_actions_to_robot_dim(raw_actions, self.action_dim)
            all_actions = self._normalize_actions_shape(all_actions)

            # [debug] dump 原始动作块，核对模型输出是否正常(仅 --dump_obs 时)
            global _dump_act_saved
            if self.args.dump_obs and _dump_act_saved < self.args.dump_obs_count:
                from pathlib import Path
                _dd = Path(self.args.dump_obs_dir); _dd.mkdir(parents=True, exist_ok=True)
                _ra = np.asarray(raw_actions); _aa = np.asarray(all_actions, dtype=np.float32)
                if _aa.ndim == 1:
                    _aa = _aa[None, :]
                with open(_dd / f'step{trigger_step}_actions.txt', 'w') as _f:
                    _f.write(f'raw_actions shape: {_ra.shape}\n')
                    _f.write(f'published_actions({self.action_dim}) shape: {_aa.shape}\n')
                    _f.write(f'first 3 rows ({self.action_dim}d):\n')
                    for _r in _aa[:3]:
                        _f.write('  ' + str(np.round(_r, 4).tolist()) + '\n')
                    _f.write(f'chunk per-dim min: {np.round(_aa.min(0), 4).tolist()}\n')
                    _f.write(f'chunk per-dim max: {np.round(_aa.max(0), 4).tolist()}\n')
                    _f.write(f'chunk row0->rowN delta: {np.round(_aa[-1] - _aa[0], 4).tolist()}\n')
                _dump_act_saved += 1
                print(f'[DUMP-ACT] step {trigger_step} actions 已存 ({_dump_act_saved}/{self.args.dump_obs_count})')

            duration = time.time() - inference_start
            adaptive_num2 = int(duration * self.publish_rate) + 1
            adaptive_horizon = calculate_adaptive_horizon(
                all_actions, self.max_action_execute_horizon,
                self.inference_trigger_remaining, self.distance_thresh,
            )
            if adaptive_num2 >= self.inference_trigger_remaining:
                print(f'[SMOOTH_INFER] WARN infer {duration:.3f}s -> adaptive_num2={adaptive_num2} '
                      f'>= num1={self.inference_trigger_remaining}; 增大 --inference_trigger_remaining 以避免卡顿')
            else:
                print(f'[SMOOTH_INFER] infer {duration:.3f}s -> adaptive_num2={adaptive_num2}, '
                      f'horizon={adaptive_horizon}')

            with self.pending_inference_lock:
                self.pending_inference_result = {
                    'actions': np.asarray(all_actions).copy(),
                    'trigger_step': trigger_step,
                    'adaptive_num2': adaptive_num2,
                    'adaptive_horizon': adaptive_horizon,
                    'inference_duration': duration,
                }
        except Exception as exc:
            print(f'[SMOOTH_INFER] inference error: {exc}')
            with self.pending_inference_lock:
                self.pending_inference_result = None
                self.inference_trigger_step = None
        finally:
            with self.infer_inflight_lock:
                self.infer_in_flight = False

    def _start_inference(self, trigger_step):
        """起一次后台推理；已有推理在飞时直接跳过（返回 False）。

        必须在这里挡住：client 是单条 websocket，并发 infer 会导致 recv 冲突且响应错位。
        用显式标志而非 thread.is_alive()，避免"检查后动作"的竞态。
        """
        with self.infer_inflight_lock:
            if self.infer_in_flight:
                return False
            self.infer_in_flight = True
        self.inference_thread = threading.Thread(target=self._inference_async, args=(trigger_step,))
        self.inference_thread.daemon = True
        self.inference_thread.start()
        self.inference_trigger_step = trigger_step
        return True

    def _publish_action(self, action):
        """把单个 16/22 维动作下发到 H01(仅 --execute_action 时真正发)。"""
        if not self.args.execute_action:
            return
        a = np.asarray(action, dtype=np.float64).copy()
        _maybe_apply_gripper_rescale(a, self.apply_gripper_rescale)
        a[LEFT_GRIP_COL] += self.gripper_offset
        a[RIGHT_GRIP_COL] += self.gripper_offset
        publish_action_to_robot(self.controller, a, self.action_dim)

    # --------------------------------------------------------------------- #
    def run(self):
        args = self.args
        if args.policy_action_space != 'qpose':
            raise ValueError("smooth 版仅支持 --policy_action_space qpose（H01 为关节控制；endpose 需 IK，暂不支持）")
        if self.max_action_execute_horizon > self.chunk_size:
            raise ValueError(f'max_action_execute_horizon({self.max_action_execute_horizon}) > chunk_size({self.chunk_size})')
        if self.inference_trigger_remaining >= self.max_action_execute_horizon:
            raise ValueError(f'inference_trigger_remaining({self.inference_trigger_remaining}) >= '
                             f'max_action_execute_horizon({self.max_action_execute_horizon})')
        if self.inference_trigger_remaining >= self.chunk_size:
            raise ValueError(f'inference_trigger_remaining({self.inference_trigger_remaining}) >= chunk_size({self.chunk_size})')

        # 按启动参数决定是否移动到 ready pose；部署脚本默认关闭以保证 dry-run 安全。
        if args.init_pose:
            if args.ready_pose_npy:
                ready_pose = load_ready_pose_npy(args.ready_pose_npy)
                src = f'[task npy] {args.ready_pose_npy}'
            else:
                try:
                    ready_pose = load_training_first_frame_ready_pose()
                    src = f'[训练数据第一帧] {TRAINING_FIRST_FRAME_PATH}'
                except Exception as exc:
                    ready_pose = READY_POSE
                    src = f'[默认 READY_POSE；首帧 npy 读取失败: {exc}]'
            print(f"[init_pose] 移动双臂到 ready pose（会移动机械臂，2s 平滑插值）{src}...")
            self.controller.move_to_ready_pose(ready_pose)

        input("Press Enter to start inference...")

        print(f'[SMOOTH_MAIN] control loop start, publish_rate={self.publish_rate}Hz, '
              f'execute={args.execute_action}, action_dim={self.action_dim}, chunk={self.chunk_size}, '
              f'num1={self.inference_trigger_remaining}, horizon={self.max_action_execute_horizon}, '
              f'blend={self.continuity_blend_steps}, align={self.align_search_window}, '
              f'gripper_rescale={self.apply_gripper_rescale}')

        t = 0
        need_initial_inference = True
        target_interval = 1.0 / self.publish_rate
        with self.action_buffer_lock:
            self.action_buffer = None
            self.action_index = 0
        with self.pending_inference_lock:
            self.pending_inference_result = None
            self.inference_trigger_step = None

        while t < self.max_publish_step and rclpy.ok():
            step_start = time.time()

            # 1. 初始推理(拿到第一段动作前反复重试，避免首帧未到就卡死)
            if need_initial_inference:
                if self.inference_thread is None or not self.inference_thread.is_alive():
                    with self.pending_inference_lock:
                        has_pending = self.pending_inference_result is not None
                    if not has_pending:
                        self._start_inference(t)
                        need_initial_inference = False

            # 2. 应用待处理推理结果(拼接 + 平滑)
            with self.pending_inference_lock:
                if self.pending_inference_result is not None:
                    with self.action_buffer_lock:
                        remaining = (len(self.action_buffer) - self.action_index
                                     if self.action_buffer is not None else 0)
                        adaptive_num2 = self.pending_inference_result.get('adaptive_num2', 0)
                        # clamp>=0: 推理耗时 >= 触发提前量时 expected 会变负导致永不应用->卡死
                        expected_remaining = max(self.inference_trigger_remaining - adaptive_num2, 0)

                        if self.action_buffer is None or remaining <= expected_remaining:
                            if self.action_buffer is not None:
                                new_buffer = apply_sliding_window_result(
                                    self.action_buffer, self.action_index,
                                    self.pending_inference_result['actions'],
                                    self.inference_trigger_remaining, adaptive_num2,
                                    self.pending_inference_result.get('adaptive_horizon',
                                                                      self.max_action_execute_horizon),
                                    self.continuity_blend_steps, self.align_search_window,
                                )
                                if new_buffer is not None:
                                    self.action_buffer = new_buffer
                                    self.action_index = 0
                            else:
                                buf = self._normalize_actions_shape(self.pending_inference_result['actions'].copy())
                                if buf.ndim == 2:
                                    self.action_buffer = smooth_action_sequence(buf)
                                    self.action_index = 0
                                else:
                                    print(f'[SMOOTH_MAIN] unexpected action shape: {buf.shape}')
                                    self.action_buffer = None
                            self.pending_inference_result = None

            # 3. 触发新推理(缓冲剩余 <= 阈值、无在跑推理、无待应用结果)
            with self.pending_inference_lock:
                has_pending = self.pending_inference_result is not None
            with self.action_buffer_lock:
                if self.action_buffer is not None:
                    remaining = len(self.action_buffer) - self.action_index
                    thread_idle = self.inference_thread is None or not self.inference_thread.is_alive()
                    if remaining <= self.inference_trigger_remaining and thread_idle and not has_pending:
                        self._start_inference(t)
                    elif (self.inference_thread is not None and self.inference_thread.is_alive()
                          and self.inference_trigger_step is not None
                          and (t - self.inference_trigger_step) > 100):
                        # 看门狗：推理超时。注意 _start_inference 现在有 in-flight 保护，
                        # 旧推理还在飞时这里是安全的 no-op，不能再起第二个线程抢同一条连接。
                        self._start_inference(t)

            # 4. 发布一帧动作
            with self.action_buffer_lock:
                if self.action_buffer is not None and self.action_index < len(self.action_buffer):
                    self._publish_action(self.action_buffer[self.action_index])
                    self.action_index += 1

            t += 1
            elapsed = time.time() - step_start
            if elapsed < target_interval:
                time.sleep(target_interval - elapsed)


def build_arg_parser():
    p = argparse.ArgumentParser(description='H01 远程推理 client (smooth 滑动窗口版)')
    p.add_argument('--server_host', default='172.16.100.33')
    p.add_argument('--server_port', type=int, default=8011)
    p.add_argument('--prompt', type=str, default='Grab the fork and put it in the basket.')
    p.add_argument('--robot_type', default='h01', choices=['h01', 't170d'])
    p.add_argument('--fisheye_mode', default='3fisheye', choices=['3fisheye', '5fisheye'])
    p.add_argument('--policy_action_space', default='qpose', choices=['qpose'],
                   help='smooth 版仅支持 qpose(关节控制)')
    p.add_argument('--server_state_dim', type=int, choices=(SERVER_STATE_DIM,),
                   default=SERVER_STATE_DIM,
                   help='保留该参数以兼容启动脚本，但只接受固定值 22')
    p.add_argument('--action_dim', type=int, choices=SUPPORTED_ACTION_DIMS,
                   default=ROBOT_ACTION_DIM,
                   help='从模型 action 截取并发布到 ROS 的固定维度：16=双臂+夹爪，22=再加头2+腰4')
    p.add_argument('--transport', choices=('zmq', 'websocket'), default='zmq',
                   help='unified server 使用 zmq；旧 inference_h01_server 使用 websocket')
    p.add_argument('--server_timeout_ms', type=int, default=30000,
                   help='ZMQ 发送/接收超时，毫秒')
    # 平滑调度参数(默认对齐参考 smooth 版)
    p.add_argument('--publish_rate', type=int, default=30, help='匀速发布频率 Hz')
    p.add_argument('--chunk_size', type=int, default=50, help='服务端单次返回的动作数')
    p.add_argument('--inference_trigger_remaining', type=int, default=10,
                   help='num1: 缓冲剩余这么多时提前发起下一次异步推理')
    p.add_argument('--max_action_execute_horizon', type=int, default=35,
                   help='单次推理最多执行多少个动作 (<= chunk_size)')
    p.add_argument('--distance_thresh', type=float, default=0.5,
                   help='自适应 horizon 的 L2 距离阈值(双爪发散快时缩短 horizon)')
    p.add_argument('--continuity_blend_steps', type=int, default=10,
                   help='拼接处连续性偏置衰减步数(0=关闭)，消除拼接卡顿')
    p.add_argument('--align_search_window', type=int, default=8,
                   help='拼接起点就近匹配的搜索半窗(0=纯时间估计)')
    p.add_argument('--apply_gripper_rescale', action='store_true', default=False,
                   help='发布前缩放双爪列(7/15)：值小于 0.04 置为 -0.03，否则乘 1.5（默认关闭）')
    p.add_argument('--gripper_offset', type=float, default=0.0, help='发布前加到双爪列(7/15)的偏置')
    p.add_argument('--max_publish_step', type=int, default=10000)
    # 安全 / 初始化(与同步版一致)
    p.add_argument('--execute_action', action='store_true', default=False,
                   help='真正下发动作到机器人(默认关闭，仅推理+平滑)')
    p.add_argument('--init_pose', dest='init_pose', action='store_true', default=False,
                   help='推理前移动双臂到 ready pose(默认关闭)')
    p.add_argument('--no_init_pose', dest='init_pose', action='store_false',
                   help='跳过 ready pose(dry-run 不动臂)')
    p.add_argument('--ready_pose_npy', type=str, default='',
                   help='按任务的 ready pose npy(16维臂爪、20维臂爪+腰，或22维完整state);'
                        '空=训练首帧 npy，读取失败时回退内置 READY_POSE')
    # dump
    p.add_argument('--dump_obs', action='store_true', default=False)
    p.add_argument('--dump_obs_dir', type=str, default='debug/obs_dump')
    p.add_argument('--dump_obs_count', type=int, default=3)
    p.add_argument('--seed', type=int, default=0)
    return p


def main():
    parser = build_arg_parser()
    args = parser.parse_args()
    if args.action_dim == FULL_BODY_ACTION_DIM and args.robot_type != 'h01':
        parser.error('--action_dim 22 只支持 robot_type=h01（需要头部和腰部 ROS topic）')
    set_seed(args.seed)

    rclpy.init()
    controller = RobotController(
        args.robot_type,
        fisheye_mode=args.fisheye_mode,
    )
    def spin_controller():
        try:
            rclpy.spin(controller)
        except Exception:
            # SIGINT can shut down the context while spin_once is active.
            if rclpy.ok():
                raise

    threading.Thread(target=spin_controller, daemon=True).start()

    time.sleep(2.0)
    print(f"连接推理服务器 {args.server_host}:{args.server_port} ({args.transport})...")
    if args.transport == 'zmq':
        client = UnifiedZmqClient(
            host=args.server_host,
            port=args.server_port,
            timeout_ms=args.server_timeout_ms,
        )
        server_info = client.get_server_info()
        server_action_dim = validate_server_io(
            server_info,
            action_dim=args.action_dim,
            chunk_size=args.chunk_size,
        )
        print(
            "[SERVER] I/O schema verified: "
            f"state={server_info['state_input_dim']}, "
            f"server_action={server_action_dim}, "
            f"published_action={args.action_dim}, "
            f"steps={server_info['n_action_steps']}"
        )
    else:
        from openpi_client import websocket_client_policy

        client = websocket_client_policy.WebsocketClientPolicy(
            host=args.server_host, port=args.server_port
        )

    smooth = SmoothInferenceController(args, controller, client)
    try:
        smooth.run()
    except KeyboardInterrupt:
        print("\n中断")
    finally:
        if hasattr(client, 'close'):
            client.close()
        try:
            controller.close_debug_writers()
        except Exception:
            pass
        controller.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
