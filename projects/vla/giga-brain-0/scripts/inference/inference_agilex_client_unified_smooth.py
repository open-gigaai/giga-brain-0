"""Unified ROS client for VLA flow-matching inference — smooth (平滑) 版本.

在 ``inference_agilex_client_unified.py`` 的基础上加入两个来自
``client_gigabrain0.1.1_smooth_dev.py`` 的平滑机制:

1. 异步滑动窗口推理 (sliding-window async inference)
   - 后台线程在缓冲区剩余 ``inference_trigger_remaining`` (num1) 个动作时就提前
     发起下一次推理, 新 chunk 在旧 chunk 执行完之前就绪, 消除"推理时停顿"。
   - ``adaptive_num2`` 根据推理耗时估计推理期间已消耗的动作数 (duration*publish_rate),
     用于对齐新旧序列的拼接点。
   - ``calculate_adaptive_horizon`` 在预测轨迹快速发散时 (joint 6/13 的 L2 距离)
     缩短执行 horizon。

2. 拼接处低通平滑 (cross-boundary low-pass smoothing)
   - ``apply_sliding_window_result`` 把最后 num1 个已执行动作与新预测拼接, 做
     Butterworth 低通滤波 (夹爪列不滤波), 再丢掉前 num1 个 (仅作滤波上下文),
     使新 chunk 平滑接入旧轨迹, 消除拼接跳变。

ZMQ 协议 / image-mode / gripper-rescale / tyro CLI 等均沿用 unified 客户端, 因此
和现有 unified server 完全兼容。

参数说明 (与旧 clinet.sh 调用 client_gigabrain0.1.1_smooth_dev.py 时的一组参数保持一致,
所以同一套 flag 可以直接沿用; tyro 下划线/连字符两种写法都支持):
    --publish_rate              发布频率 Hz
    --task_name                 语言指令
    --inference_trigger_remaining   num1, 缓冲区剩余这么多动作时提前发起下一次异步推理
    --max_action_execute_horizon    单次推理最多执行多少个动作 (<= chunk_size)
    --distance_thresh           自适应 horizon 的 L2 距离阈值, 轨迹发散快时缩短执行 horizon
    --continuity_blend_steps    拼接处连续性偏置的衰减步数 (默认 10, 0=关闭), 消除拼接卡顿
    --align_search_window       拼接起点就近匹配的搜索半窗 (默认 8); 0=关闭, 退回旧脚本的
                                纯时间估计拼接 (new_actions_start = adaptive_num2)
    --gripper_offset            发布前加到两个夹爪列 (6/13) 的偏置 (默认 0.0);
                                旧 smooth_dev 脚本发布前固定做 gripper -= 0.015, 对照时传 -0.015
    --gripper_zero_threshold    将最终夹爪命令中小于该阈值的值置 0; 默认 None=关闭
    --init_pose_mode            unified (默认, 回全零位) / smooth_dev (旧脚本硬编码的初始位姿,
                                夹爪最终到 -0.339)
    --pos_lookahead_step        遗留参数, 滑动窗口调度不使用, 仅为兼容旧脚本不报错而暴露
    --observation_memory_offsets    遗留参数; 本 unified 协议客户端只发送最新单帧, 仅 (0,) 有
                                意义, 传入多帧历史会被忽略并打印 warning
    --chunk_size                服务端单次推理返回的动作数 (默认 50, 需与真实 chunk 对齐)
    --host / --port             unified server 地址 (默认 0.0.0.0:8080)
    --image-mode / --force-rgb / --apply-gripper-rescale   见 unified 客户端

用法示例:
    # Raw uint8 images with gripper rescaling.
    python inference_agilex_client_unified_smooth.py

    # PaliGemma2 server.
    python inference_agilex_client_unified_smooth.py --image-mode float_native --no-apply-gripper-rescale

    # 调平滑参数 (等价于旧 clinet.sh 的那组 flag)
    python inference_agilex_client_unified_smooth.py \
        --pos_lookahead_step 50 --publish_rate 30 \
        --max_action_execute_horizon 50 --inference_trigger_remaining 10 \
        --distance_thresh 0.3 --observation_memory_offsets 0 \
        --task_name "put apple into woven basket"

    # 严格对齐旧 client_gigabrain0.1.1_smooth_dev.py 行为 (PG2):
    # 关闭本脚本新增的就近匹配/连续性偏置, 补上旧脚本的夹爪偏置和初始位姿
    python inference_agilex_client_unified_smooth.py \
        --image-mode float_native --no-apply-gripper-rescale \
        --publish_rate 30 --max_action_execute_horizon 50 \
        --inference_trigger_remaining 10 --distance_thresh 0.3 \
        --continuity_blend_steps 0 --align_search_window 0 \
        --gripper_offset -0.015 --init_pose_mode smooth_dev \
        --task_name "flatten and fold the shirt"
"""

import threading
import time
from collections import deque
from io import BytesIO
from typing import Any, Dict, Literal, Optional, Tuple

import numpy as np
import rospy
import torch
import tyro
import zmq
from cv_bridge import CvBridge
from einops import rearrange
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from PIL import Image as PILImage
from scipy.signal import butter, filtfilt
from sensor_msgs.msg import Image, JointState
from std_msgs.msg import Header


ImageMode = Literal['uint8', 'float_native', 'float_resize224']


# --------------------------------------------------------------------------- #
# 平滑 (low-pass filter) 工具                                                  #
# --------------------------------------------------------------------------- #
def butterworth_lowpass_filter(data: np.ndarray, cutoff_freq: float = 1.0, sampling_freq: float = 15.0, order: int = 2) -> np.ndarray:
    """对输入序列沿 axis=0 应用低通 Butterworth 滤波。cutoff 越小越平滑。"""
    nyquist = 0.5 * sampling_freq
    normal_cutoff = cutoff_freq / nyquist
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return filtfilt(b, a, data, axis=0)


def smoothen_actions_numpy(actions: np.ndarray) -> np.ndarray:
    """平滑一个 7 维 (单臂) 动作块, 但夹爪列 (最后一列) 不滤波。"""
    actions_np = actions
    filtered_actions_np = butterworth_lowpass_filter(actions_np.copy())
    # 夹爪关节不做平滑, 保留原始信号
    filtered_actions_np[:, -1] = actions_np[:, -1]
    return filtered_actions_np.copy()


def smooth_action_sequence(actions: np.ndarray) -> np.ndarray:
    """对 (N, >=14) 的双臂动作序列做平滑: 左臂 [:7] / 右臂 [7:14] 分别滤波,
    夹爪列保留, 14 之后的列 (如底盘速度) 原样保留。滤波失败时返回原序列。"""
    if len(actions) <= 1:
        return actions
    try:
        left_smooth = smoothen_actions_numpy(actions[:, :7])
        right_smooth = smoothen_actions_numpy(actions[:, 7:14])
        smoothed = np.concatenate([left_smooth, right_smooth], axis=1)
        if actions.shape[1] > 14:
            smoothed = np.concatenate([smoothed, actions[:, 14:]], axis=1)
        return smoothed
    except Exception as exc:  # filtfilt 对过短序列会报错, 此时退回不平滑
        print(f'[SMOOTH] filter skipped: {exc}')
        return actions


def calculate_adaptive_horizon(actions: np.ndarray, max_horizon: int, inference_trigger_remaining: int, distance_thresh: float) -> int:
    """根据预测轨迹的 L2 距离自适应裁剪 action_execute_horizon。

    对第 7 个关节 (索引 6/13, 夹爪) 放大 10 倍后, 在若干检查点上计算与首帧的 L2
    距离; 一旦超过阈值就把 horizon 截到 ``check_point + inference_trigger_remaining``,
    否则用 ``max_horizon``。轨迹发散越快, 执行得越短, 越早重推理。
    """
    if len(actions) < max_horizon:
        return len(actions)

    actions_scaled = actions.copy()
    actions_scaled[:, 6] *= 10   # 左臂夹爪
    actions_scaled[:, 13] *= 10  # 右臂夹爪
    actions_scaled = actions_scaled[:, [6, 13]]

    base_action = actions_scaled[0]
    check_points = [15, 20, 25, 30, 40]
    for check_point in check_points:
        if check_point >= max_horizon:
            break
        l2_distance = np.linalg.norm(actions_scaled[check_point] - base_action)
        if l2_distance > distance_thresh:
            adaptive_horizon = check_point + inference_trigger_remaining
            print(f'[ADAPTIVE_HORIZON] L2({l2_distance:.4f}) > thresh({distance_thresh}) at cp={check_point} -> horizon={adaptive_horizon}')
            return adaptive_horizon

    return max_horizon


def apply_sliding_window_result(current_buffer, current_index, inference_result, num1, adaptive_num2,
                                adaptive_horizon, continuity_blend_steps=0, align_search_window=8):
    """拼接滑动窗口推理结果并在拼接处做平滑 (使用自适应 horizon)。

    取最后 (最多) num1 个已执行动作作为低通滤波上下文, 与新预测 (从就近匹配到的起点到
    adaptive_horizon) 拼接、平滑, 再丢弃这段上下文, 返回平滑后的新动作序列。

    new_actions_start 不再死用 adaptive_num2 (纯时间估计), 而是以它为中心、在
    align_search_window 窗口内找与"最近已执行动作"关节最接近的预测帧 -> 按机器人实际位姿对齐,
    避免时间估计偏差导致重放已走过轨迹 (起点偏小) 或跳步 (起点偏大)。

    continuity_blend_steps > 0 时再叠加"连续性偏置": 强制新段首帧对齐到最近已执行动作,
    偏置在该步数内衰减到 0, 消除拼接处残留的位置跳变 (低通去不掉直流偏移)。
    """
    try:
        # 实际可用的旧上下文 = min(num1, current_index)。current_index 可能 < num1
        # (当 num1 调大或 horizon 被裁短时), 必须按实际数量裁剪; 否则
        # current_buffer[current_index-num1:current_index] 会取到负索引 -> 空上下文,
        # 再配合下面固定丢弃 num1 个, 会让返回序列塌缩成极少动作 -> 抖动甚至卡死。
        old_count = max(0, min(num1, current_index))
        if old_count > 0:
            remaining_old_actions = current_buffer[current_index - old_count: current_index]
        else:
            remaining_old_actions = np.array([]).reshape(0, current_buffer.shape[1])

        # 就近匹配选拼接起点: 以 adaptive_num2 为中心, 在窗口内找与最近已执行动作最接近的预测帧
        # (仅比手臂关节, 排除夹爪 6/13 与底盘)。时间估计受线程调度/观测延迟影响会偏小 -> 接到
        # 已走过的索引 -> 机器人往回重放; 就近匹配按真实位姿对齐, 双向纠正。
        new_actions_start = adaptive_num2
        if current_index > 0 and align_search_window > 0 and len(inference_result) > 0:
            last_action = np.asarray(current_buffer[current_index - 1])
            n_cols = min(14, inference_result.shape[1])
            arm_cols = [c for c in range(n_cols) if c not in (6, 13)]
            lo = max(0, adaptive_num2 - align_search_window)
            hi = min(len(inference_result), adaptive_num2 + align_search_window + 1)
            if hi > lo and len(arm_cols) > 0:
                seg = inference_result[lo:hi][:, arm_cols]
                dist = np.linalg.norm(seg - last_action[arm_cols], axis=1)
                new_actions_start = lo + int(np.argmin(dist))

        if new_actions_start < adaptive_horizon:
            new_inference_actions = inference_result[new_actions_start:adaptive_horizon]
        elif new_actions_start < len(inference_result):
            # 推理比裁剪后的 horizon 还慢 (adaptive_num2 >= adaptive_horizon): 至少执行最新的
            # 一个动作, 避免拼接结果为空导致缓冲区不更新、机器人卡死。
            new_inference_actions = inference_result[new_actions_start:new_actions_start + 1]
        else:
            new_inference_actions = np.array([]).reshape(0, current_buffer.shape[1])

        if len(remaining_old_actions) > 0 and len(new_inference_actions) > 0:
            concatenated_actions = np.concatenate([remaining_old_actions, new_inference_actions], axis=0)
        elif len(new_inference_actions) > 0:
            concatenated_actions = new_inference_actions
        else:
            return None

        smoothed_actions = smooth_action_sequence(concatenated_actions)
        # 丢弃实际预置的上下文数量 (而非固定 num1), 保证返回完整的新动作段
        new_segment = smoothed_actions[len(remaining_old_actions):]
        if len(new_segment) == 0:
            return None

        # 连续性偏置: 低通只能把拼接处的位置阶跃抹成"半阶跃"(残留约 delta/2 的跳变),
        # 这里强制新段首帧对齐到最近已执行动作, 并把偏置在 continuity_blend_steps 步内线性
        # 衰减到 0 -> 拼接处零跳变且平滑收敛到模型新轨迹; 同时吸收 adaptive_num2 估计误差。
        # 夹爪(6/13) 与底盘(>=14) 列不做偏置。
        if continuity_blend_steps > 0 and current_index > 0:
            new_segment = new_segment.copy()
            last_action = np.asarray(current_buffer[current_index - 1], dtype=new_segment.dtype)
            delta = (last_action - new_segment[0]).astype(new_segment.dtype)
            for gripper_col in (6, 13):
                if gripper_col < delta.shape[0]:
                    delta[gripper_col] = 0.0
            if delta.shape[0] > 14:
                delta[14:] = 0.0
            k = min(continuity_blend_steps, len(new_segment))
            # 余弦缓入 (smoothstep): 两端斜率为 0, 偏置不再引入速度突变; 线性 ramp 会在拼接处
            # 和衰减结束处各留一个速度拐点 -> 轻微抖动。
            ramp = np.zeros(len(new_segment), dtype=new_segment.dtype)
            ramp[:k] = 0.5 * (1.0 + np.cos(np.pi * np.arange(k, dtype=new_segment.dtype) / k))
            new_segment = new_segment + delta[None, :] * ramp[:, None]
            # 此时 new_segment[0] == 上一帧已执行动作; 上一拍刚发过它, 再发一次会让机械臂保持
            # 一拍 (1/publish_rate) 不动 -> 周期性微停顿。丢掉首帧, 从下一帧发布 (恰好与
            # "apply 这一拍本身也会发布一次动作" 对齐, 不丢实际进度)。
            new_segment = new_segment[1:]
            if len(new_segment) == 0:
                return None
        return new_segment
    except Exception as exc:
        print(f'[SLIDING_APPLY] concat error: {exc}')
        return None


# --------------------------------------------------------------------------- #
# ZMQ inference client (与 unified 客户端一致)                                  #
# --------------------------------------------------------------------------- #
class TorchSerializer:
    @staticmethod
    def to_bytes(data):
        buffer = BytesIO()
        torch.save(data, buffer)
        return buffer.getvalue()

    @staticmethod
    def from_bytes(data):
        buffer = BytesIO(data)
        obj = torch.load(buffer, map_location='cpu')
        return obj


class BaseInferenceClient:
    def __init__(self, host: str = 'localhost', port: int = 5555, timeout_ms: int = 15000):
        self.context = zmq.Context()
        self.host = host
        self.port = port
        self.timeout_ms = timeout_ms
        self._init_socket()

    def _init_socket(self):
        self.socket = self.context.socket(zmq.REQ)
        self.socket.connect(f'tcp://{self.host}:{self.port}')

    def ping(self) -> bool:
        try:
            self.call_endpoint('ping', requires_input=False)
            return True
        except zmq.error.ZMQError:
            self._init_socket()
            return False

    def kill_server(self):
        self.call_endpoint('kill', requires_input=False)

    def call_endpoint(self, endpoint: str, data: dict = None, requires_input: bool = True) -> dict:
        request: dict = {'endpoint': endpoint}
        if requires_input:
            request['data'] = data

        self.socket.send(TorchSerializer.to_bytes(request))
        message = self.socket.recv()
        if message == b'ERROR':
            raise RuntimeError('Server error')
        return TorchSerializer.from_bytes(message)

    def __del__(self):
        self.socket.close()
        self.context.term()


class RobotInferenceClient(BaseInferenceClient):
    """Client for communicating with the RobotInferenceServer."""

    def inference(self, observations: Dict[str, Any]):
        return self.call_endpoint('inference', observations)


def validate_server_dimensions(
    client,
    *,
    state_dim: int,
    action_dim: Optional[int] = None,
) -> Tuple[Dict[str, Any], int]:
    """Resolve the action schema without assuming it matches the state width."""
    info = client.call_endpoint('server_info', requires_input=False)
    server_state_dim = info.get('state_input_dim', info.get('expected_state_dim'))
    if server_state_dim is not None and int(server_state_dim) != int(state_dim):
        raise ValueError(
            'Client/server state schema mismatch: '
            f'server state_input_dim={server_state_dim}, client state_dim={state_dim}'
        )

    server_action_dim = info.get(
        'action_output_dim', info.get('original_action_dim')
    )
    if (
        server_action_dim is not None
        and info.get('action_output_dim_state_dependent', False)
    ):
        server_action_dim = min(int(server_action_dim), int(state_dim))
    if action_dim is None:
        resolved_action_dim = (
            int(server_action_dim)
            if server_action_dim is not None
            else int(state_dim)
        )
    else:
        resolved_action_dim = int(action_dim)
        if (
            server_action_dim is not None
            and int(server_action_dim) != resolved_action_dim
        ):
            raise ValueError(
                'Client/server action schema mismatch: '
                f'server action_output_dim={server_action_dim}, '
                f'client action_dim={resolved_action_dim}'
            )
    if resolved_action_dim <= 0:
        raise ValueError(f'action_dim must be positive, got {resolved_action_dim}')

    print(
        'Connected to inference server: '
        f'state_input_dim={server_state_dim}, '
        f'action_output_dim={resolved_action_dim}, '
        f'training_action_dim={info.get("training_action_dim", "<unknown>")}, '
        f'mask_unsupervised_action_dims_for_noise='
        f'{info.get("mask_unsupervised_action_dims_for_noise", "<unknown>")}'
    )
    return info, resolved_action_dim


# --------------------------------------------------------------------------- #
# 图像预处理 (与 unified 客户端一致)                                            #
# --------------------------------------------------------------------------- #
def resize_with_pad(images: np.ndarray, height: int, width: int, method=PILImage.BILINEAR) -> np.ndarray:
    """Replicates tf.image.resize_with_pad for multiple images using PIL."""

    def _resize_with_pad_pil(image: PILImage.Image, height: int, width: int, method: int) -> PILImage.Image:
        cur_width, cur_height = image.size
        if cur_width == width and cur_height == height:
            return image

        ratio = max(cur_width / width, cur_height / height)
        resized_height = int(cur_height / ratio)
        resized_width = int(cur_width / ratio)
        resized_image = image.resize((resized_width, resized_height), resample=method)

        zero_image = PILImage.new(resized_image.mode, (width, height), 0)
        pad_height = max(0, int((height - resized_height) / 2))
        pad_width = max(0, int((width - resized_width) / 2))
        zero_image.paste(resized_image, (pad_width, pad_height))
        assert zero_image.size == (width, height)
        return zero_image

    if images.shape[-3:-1] == (height, width):
        return images

    original_shape = images.shape
    images = images.reshape(-1, *original_shape[-3:])
    resized = np.stack([_resize_with_pad_pil(PILImage.fromarray(im), height, width, method=method) for im in images])
    return resized.reshape(*original_shape[:-3], *resized.shape[-3:])


def _preprocess_rgb(img: np.ndarray, image_mode: ImageMode) -> np.ndarray:
    """Apply the selected client-side image preprocessing to a single RGB frame."""
    if image_mode == 'uint8':
        return img.astype(np.uint8)
    if image_mode == 'float_native':
        return img.astype(np.float32) / 255.0
    if image_mode == 'float_resize224':
        return resize_with_pad(img, 224, 224).astype(np.float32) / 255.0
    raise ValueError(f'Unknown image_mode: {image_mode!r}')


def _preprocess_depth_to_uint8(depth: np.ndarray, pad_to_480x640: bool = False) -> np.ndarray:
    """Common agilex depth preprocessing → uint8 RGB-stacked frame."""
    depth = depth.copy()[..., None].repeat(3, axis=-1)
    if pad_to_480x640:
        padded = np.zeros([480, 640, 3])
        padded[:400, :, :] = depth[:, :, :]
        depth = padded
    return np.clip(depth / 10.0, 0, 255).astype(np.uint8)


def _preprocess_depth(depth: np.ndarray, image_mode: ImageMode, pad_to_480x640: bool = False) -> np.ndarray:
    depth_u8 = _preprocess_depth_to_uint8(depth, pad_to_480x640=pad_to_480x640)
    if image_mode == 'uint8':
        return depth_u8
    if image_mode == 'float_native':
        return depth_u8.astype(np.float32) / 255.0
    if image_mode == 'float_resize224':
        return resize_with_pad(depth_u8, 224, 224).astype(np.float32) / 255.0
    raise ValueError(f'Unknown image_mode: {image_mode!r}')


def make_infer_data(camera_high, camera_left, camera_right, camera_high_depth, camera_left_depth, camera_right_depth, task_name, qpos):
    assert qpos.shape == (14,)

    camera_high_chw = rearrange(camera_high, 'h w c -> c h w')
    camera_left_chw = rearrange(camera_left, 'h w c -> c h w')
    camera_right_chw = rearrange(camera_right, 'h w c -> c h w')

    observation = {
        'observation.state': torch.from_numpy(qpos).to(torch.float32),
        'observation.images.cam_high': torch.from_numpy(camera_high_chw),
        'observation.images.cam_left_wrist': torch.from_numpy(camera_left_chw),
        'observation.images.cam_right_wrist': torch.from_numpy(camera_right_chw),
        'task': task_name,
    }

    if camera_high_depth is not None:
        camera_high_depth_chw = rearrange(camera_high_depth, 'h w c -> c h w')
        observation['observation.depth_images.cam_high'] = torch.from_numpy(camera_high_depth_chw)
    if camera_left_depth is not None:
        camera_left_depth_chw = rearrange(camera_left_depth, 'h w c -> c h w')
        observation['observation.depth_images.cam_left_wrist'] = torch.from_numpy(camera_left_depth_chw)
    if camera_right_depth is not None:
        camera_right_depth_chw = rearrange(camera_right_depth, 'h w c -> c h w')
        observation['observation.depth_images.cam_right_wrist'] = torch.from_numpy(camera_right_depth_chw)

    return observation


def _maybe_apply_gripper_rescale(action, apply_rescale: bool):
    """Rescale the left and right gripper actions."""
    if not apply_rescale:
        return
    left_gripper = action[6]
    right_gripper = action[13]
    #action[6] = -0.03 if left_gripper < 0.04 else left_gripper * 1.5
    #action[13] = -0.03 if right_gripper < 0.04 else right_gripper * 1.5
    action[6] -= 0.003
    action[13] -= 0.003


# --------------------------------------------------------------------------- #
# 平滑滑动窗口控制器                                                            #
# --------------------------------------------------------------------------- #
class SmoothInferenceController:
    """封装异步滑动窗口推理 + 拼接平滑的主控制器。"""

    MIN_QPOS = [-2.618, 0.0, -2.967, -1.745, -1.22, -2.0944]
    MAX_QPOS = [2.618, 3.14, 0.0, 1.745, 1.22, 2.0944]

    def __init__(
        self,
        client,
        ros_operator,
        task_name: str,
        publish_rate: int,
        chunk_size: int,
        inference_trigger_remaining: int,
        max_action_execute_horizon: int,
        distance_thresh: float,
        max_publish_step: int,
        state_dim: int,
        action_dim: int,
        use_robot_base: bool,
        image_mode: ImageMode,
        apply_gripper_rescale: bool,
        continuity_blend_steps: int = 10,
        align_search_window: int = 8,
        gripper_offset: float = 0.0,
        gripper_zero_threshold: Optional[float] = None,
        init_pose_mode: Literal['unified', 'smooth_dev'] = 'unified',
    ):
        self.client = client
        self.ros_operator = ros_operator
        self.task_name = task_name
        self.publish_rate = publish_rate
        self.chunk_size = chunk_size
        self.inference_trigger_remaining = inference_trigger_remaining
        self.max_action_execute_horizon = max_action_execute_horizon
        self.distance_thresh = distance_thresh
        self.max_publish_step = max_publish_step
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.use_robot_base = use_robot_base
        self.image_mode = image_mode
        self.apply_gripper_rescale = apply_gripper_rescale
        self.continuity_blend_steps = continuity_blend_steps
        self.align_search_window = align_search_window
        self.gripper_offset = gripper_offset
        self.gripper_zero_threshold = gripper_zero_threshold
        self.init_pose_mode = init_pose_mode

        # 动作缓冲区 + 推理状态 (跨线程共享, 用锁保护)
        self.action_buffer = None
        self.action_index = 0
        self.action_buffer_lock = threading.Lock()

        self.pending_inference_result = None
        self.pending_inference_lock = threading.Lock()

        self.inference_thread = None
        self.inference_trigger_step = None
        self.current_episode_id = 0

    # --------------------------------------------------------------------- #
    def _build_obs(self):
        """从最新一帧 ROS 观测构建推理输入 dict (与 unified 客户端格式一致)。"""
        result = self.ros_operator.get_frame()
        if not result:
            return None

        (img_front, img_left, img_right, img_front_depth, img_left_depth, img_right_depth,
         puppet_arm_left, puppet_arm_right, robot_base) = result

        img_front = _preprocess_rgb(img_front, self.image_mode)
        img_left = _preprocess_rgb(img_left, self.image_mode)
        img_right = _preprocess_rgb(img_right, self.image_mode)
        if img_front_depth is not None:
            img_front_depth = _preprocess_depth(img_front_depth, self.image_mode, pad_to_480x640=True)
        if img_left_depth is not None:
            img_left_depth = _preprocess_depth(img_left_depth, self.image_mode, pad_to_480x640=False)
        if img_right_depth is not None:
            img_right_depth = _preprocess_depth(img_right_depth, self.image_mode, pad_to_480x640=False)

        qpos = np.concatenate((np.array(puppet_arm_left.position), np.array(puppet_arm_right.position)), axis=0)
        obs = make_infer_data(
            img_front, img_left, img_right,
            img_front_depth, img_left_depth, img_right_depth,
            self.task_name, qpos,
        )
        obs['qpos'] = qpos
        obs['qvel'] = np.concatenate((np.array(puppet_arm_left.velocity), np.array(puppet_arm_right.velocity)), axis=0)
        obs['effort'] = np.concatenate((np.array(puppet_arm_left.effort), np.array(puppet_arm_right.effort)), axis=0)
        if self.use_robot_base:
            obs['base_vel'] = [robot_base.twist.twist.linear.x, robot_base.twist.twist.angular.z]
            obs['qpos'] = np.concatenate((obs['qpos'], obs['base_vel']), axis=0)
        else:
            obs['base_vel'] = [0.0, 0.0]
        return obs

    def _normalize_actions_shape(self, all_actions):
        """把推理输出整理成 (N, action_dim) 的二维数组。"""
        all_actions = np.asarray(all_actions)
        if all_actions.ndim == 3 and all_actions.shape[0] == 1:
            all_actions = all_actions[0]
        elif all_actions.ndim == 1:
            if all_actions.shape[0] % self.action_dim != 0:
                raise ValueError(
                    f'Flat action output with {all_actions.shape[0]} values cannot be '
                    f'reshaped to action_dim={self.action_dim}'
                )
            all_actions = all_actions.reshape(-1, self.action_dim)
        if all_actions.ndim != 2 or int(all_actions.shape[-1]) != self.action_dim:
            raise ValueError(
                f'Expected action output [T, {self.action_dim}], got '
                f'{tuple(all_actions.shape)}'
            )
        return all_actions

    def _inference_async(self, trigger_step, episode_id):
        """后台推理线程: 取帧 -> 推理 -> 计算自适应参数 -> 写入 pending 结果。"""
        if rospy.is_shutdown():
            return

        inference_start_time = time.time()
        try:
            if rospy.is_shutdown():
                return

            obs = self._build_obs()
            if obs is None:
                print('[SLIDING_INFERENCE] frame sync failed')
                with self.pending_inference_lock:
                    self.pending_inference_result = None
                    self.inference_trigger_step = None
                return

            if rospy.is_shutdown():
                return

            actions = self.client.inference(obs)
            all_actions = actions.float().cpu().numpy()

            if rospy.is_shutdown():
                return

            all_actions = self._normalize_actions_shape(all_actions)

            inference_end_time = time.time()
            total_inference_duration = inference_end_time - inference_start_time
            # 推理期间大约会消耗多少个动作 (+1 安全裕量)
            adaptive_num2 = int(total_inference_duration * self.publish_rate) + 1
            adaptive_horizon = calculate_adaptive_horizon(
                all_actions, self.max_action_execute_horizon,
                self.inference_trigger_remaining, self.distance_thresh,
            )
            # 调参用: 若 adaptive_num2 >= inference_trigger_remaining(num1), 说明推理比触发
            # 提前量还慢, 动作会断流/卡顿, 应调大 --inference_trigger_remaining。
            if adaptive_num2 >= self.inference_trigger_remaining:
                print(f'[SLIDING_INFERENCE] WARN infer {total_inference_duration:.3f}s -> '
                      f'adaptive_num2={adaptive_num2} >= num1={self.inference_trigger_remaining}; '
                      f'增大 --inference_trigger_remaining 以避免卡顿')
            else:
                print(f'[SLIDING_INFERENCE] infer {total_inference_duration:.3f}s -> '
                      f'adaptive_num2={adaptive_num2}, horizon={adaptive_horizon}')

            if rospy.is_shutdown():
                return

            with self.pending_inference_lock:
                if episode_id is not None and episode_id != self.current_episode_id:
                    return  # 跨 episode 的旧结果, 丢弃
                self.pending_inference_result = {
                    'actions': all_actions.copy(),
                    'trigger_step': trigger_step,
                    'episode_id': episode_id,
                    'completion_time': inference_end_time,
                    'inference_duration': total_inference_duration,
                    'adaptive_num2': adaptive_num2,
                    'adaptive_horizon': adaptive_horizon,
                }
        except Exception as exc:
            if not rospy.is_shutdown():
                print(f'[SLIDING_INFERENCE] inference error: {exc}')

    def _start_inference(self, trigger_step, episode_id):
        self.inference_thread = threading.Thread(target=self._inference_async, args=(trigger_step, episode_id))
        self.inference_thread.daemon = True
        self.inference_thread.start()
        self.inference_trigger_step = trigger_step

    def _publish_action(self, action):
        """对单个动作做夹爪后处理和关节限位, 然后发布。"""
        action = np.array(action, dtype=np.float64)
        _maybe_apply_gripper_rescale(action, self.apply_gripper_rescale)
        # 旧 smooth_dev 脚本发布前固定 gripper -= 0.015 (闭合余量); 对照时传 --gripper_offset -0.015
        action[6] += self.gripper_offset
        action[13] += self.gripper_offset
        if self.gripper_zero_threshold is not None:
            for gripper_col in (6, 13):
                if action[gripper_col] < self.gripper_zero_threshold:
                    action[gripper_col] = 0.0
        left_action = list(action[:7])
        right_action = list(action[7:14])
        left_action[0:6] = np.clip(left_action[0:6], self.MIN_QPOS, self.MAX_QPOS)
        right_action[0:6] = np.clip(right_action[0:6], self.MIN_QPOS, self.MAX_QPOS)
        self.ros_operator.puppet_arm_publish(left_action, right_action)
        if self.use_robot_base and len(action) >= 16:
            self.ros_operator.robot_base_publish(action[14:16])

    # --------------------------------------------------------------------- #
    def run(self):
        # 参数验证
        if self.max_action_execute_horizon > self.chunk_size:
            raise ValueError(f'max_action_execute_horizon({self.max_action_execute_horizon}) > chunk_size({self.chunk_size})')
        if self.inference_trigger_remaining >= self.max_action_execute_horizon:
            raise ValueError(f'inference_trigger_remaining({self.inference_trigger_remaining}) >= max_action_execute_horizon({self.max_action_execute_horizon})')
        if self.inference_trigger_remaining >= self.chunk_size:
            raise ValueError(f'inference_trigger_remaining({self.inference_trigger_remaining}) >= chunk_size({self.chunk_size})')

        # 初始姿态: unified=回全零位 (与 unified 客户端一致); smooth_dev=旧
        # client_gigabrain0.1.1_smooth_dev.py 硬编码的初始位姿 (采数据的起始位姿)
        if self.init_pose_mode == 'smooth_dev':
            left0 = [-0.00133514404296875, 0.00209808349609375, 0.01583099365234375, -0.032616615295410156,
                     -0.00286102294921875, 0.00095367431640625, 0.06]
            right0 = [-0.00133514404296875, 0.00438690185546875, 0.034523963928222656, -0.053597450256347656,
                      -0.00476837158203125, -0.00209808349609375, 0.06]
            left1 = [-0.00133514404296875, 0.00209808349609375, 0.01583099365234375, -0.032616615295410156,
                     -0.00286102294921875, 0.00095367431640625, -0.3393220901489258]
            right1 = [-0.00133514404296875, 0.00547955322265625, 0.01583099365234375, -0.032616615295410156,
                      -0.00286102294921875, 0.00095367431640625, -0.3397035598754883]
        else:
            left0 = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.3]
            right0 = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.3]
            left1 = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
            right1 = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        self.ros_operator.puppet_arm_publish_continuous(left0, right0)
        input('Enter any key to continue :')
        self.ros_operator.puppet_arm_publish_continuous(left1, right1)

        while not rospy.is_shutdown():
            t = 0
            need_initial_inference = True
            self.current_episode_id += 1
            episode_id = self.current_episode_id
            with self.action_buffer_lock:
                self.action_buffer = None
                self.action_index = 0
            with self.pending_inference_lock:
                self.pending_inference_result = None
                self.inference_trigger_step = None
            rate = rospy.Rate(self.publish_rate)
            print(f'[SLIDING_MAIN] control loop start, publish_rate={self.publish_rate}Hz, episode={episode_id}')

            while t < self.max_publish_step and not rospy.is_shutdown():
                # 1. 启动初始推理
                if need_initial_inference and not rospy.is_shutdown():
                    if self.inference_thread is None or not self.inference_thread.is_alive():
                        self._start_inference(t, episode_id)
                        need_initial_inference = False

                # 2. 应用待处理的推理结果 (拼接 + 平滑)
                if not rospy.is_shutdown():
                    with self.pending_inference_lock:
                        if self.pending_inference_result is not None:
                            with self.action_buffer_lock:
                                remaining_actions = len(self.action_buffer) - self.action_index if self.action_buffer is not None else 0
                                adaptive_num2 = self.pending_inference_result.get('adaptive_num2', 0)
                                # clamp 到 >=0: 当推理耗时 >= 触发提前量 (adaptive_num2 >= num1) 时,
                                # 不 clamp 会让 expected_remaining 变负, 应用条件 remaining<=expected
                                # 永远不成立 -> 缓冲区耗尽后再也不更新, 机器人卡死并不停重推理。
                                # clamp 后缓冲耗尽 (remaining==0) 时一定会应用新结果。
                                # 若长期触发此 clamp, 说明推理比 num1 个动作还慢, 应调大
                                # --inference_trigger_remaining 以恢复平滑重叠。
                                expected_remaining = max(self.inference_trigger_remaining - adaptive_num2, 0)

                                if remaining_actions <= expected_remaining or self.action_buffer is None:
                                    if self.action_buffer is not None:
                                        new_buffer = apply_sliding_window_result(
                                            self.action_buffer, self.action_index,
                                            self.pending_inference_result['actions'],
                                            self.inference_trigger_remaining, adaptive_num2,
                                            self.pending_inference_result.get('adaptive_horizon', self.max_action_execute_horizon),
                                            self.continuity_blend_steps,
                                            self.align_search_window,
                                        )
                                        if new_buffer is not None:
                                            self.action_buffer = new_buffer
                                            self.action_index = 0
                                    else:
                                        # 初始推理结果: 直接整形 + 平滑后使用
                                        buf = self._normalize_actions_shape(self.pending_inference_result['actions'].copy())
                                        if buf.ndim == 2:
                                            self.action_buffer = smooth_action_sequence(buf)
                                            self.action_index = 0
                                        else:
                                            print(f'[SLIDING_MAIN] unexpected action shape: {buf.shape}')
                                            self.action_buffer = None

                                    self.pending_inference_result = None

                # 3. 触发新推理
                if not rospy.is_shutdown():
                    # 先按 pending_lock -> action_buffer_lock 的顺序读取 (与 step 2 一致, 避免死锁)
                    with self.pending_inference_lock:
                        has_pending = self.pending_inference_result is not None
                    with self.action_buffer_lock:
                        if self.action_buffer is not None:
                            remaining_actions = len(self.action_buffer) - self.action_index
                            thread_idle = self.inference_thread is None or not self.inference_thread.is_alive()
                            need_new_inference = False

                            # 正常/恢复触发: 剩余 <= 阈值, 无在跑推理, 且没有未应用的结果就发起。
                            # 用 <= 而非 ==: 当 buffer 被 horizon 裁短到一开始就 < num1, 或上一次推理
                            # 抓帧失败/超时未产出时, 也能立刻补发, 不必等缓冲耗尽到 0 才恢复 (那段就是
                            # "等推理"卡顿)。has_pending 守卫避免推理提前完成时重复发起、覆盖待应用结果。
                            if remaining_actions <= self.inference_trigger_remaining and thread_idle and not has_pending:
                                need_new_inference = True
                            # 推理线程长时间无结果, 强制重推理
                            elif (self.inference_thread is not None and self.inference_thread.is_alive() and
                                  self.inference_trigger_step is not None and (t - self.inference_trigger_step) > 100):
                                need_new_inference = True

                            if need_new_inference:
                                self._start_inference(t, episode_id)

                # 4. 发布动作
                if not rospy.is_shutdown():
                    with self.action_buffer_lock:
                        if self.action_buffer is not None and self.action_index < len(self.action_buffer):
                            self._publish_action(self.action_buffer[self.action_index])
                            self.action_index += 1

                t += 1
                if not rospy.is_shutdown():
                    rate.sleep()

            if rospy.is_shutdown():
                break
            rate.sleep()


class RosOperator:
    def __init__(
        self,
        publish_rate: int = 30,
        arm_steps_length=None,
        use_depth_image: bool = False,
        use_robot_base: bool = False,
        force_rgb_encoding: bool = False,
        img_left_topic: str = '/camera_l/color/image_raw',
        img_right_topic: str = '/camera_r/color/image_raw',
        img_front_topic: str = '/camera_f/color/image_raw',
        img_left_depth_topic: str = '/camera_l/aligned_depth_to_color/image_raw',
        img_right_depth_topic: str = '/camera_r/aligned_depth_to_color/image_raw',
        img_front_depth_topic: str = '/camera_f/depth/image_raw',
        puppet_arm_left_topic: str = '/puppet/joint_left',
        puppet_arm_right_topic: str = '/puppet/joint_right',
        robot_base_topic: str = '/odom_raw',
        puppet_arm_left_cmd_topic: str = '/master/joint_left',
        puppet_arm_right_cmd_topic: str = '/master/joint_right',
        robot_base_cmd_topic: str = '/cmd_vel',
    ):
        # If True: cv_bridge transcodes to rgb8 regardless of source encoding.
        # If False: 'passthrough' — whatever the camera publishes (usually rgb8 on
        # RealSense default launch, but may be bgr8 if the driver was reconfigured).
        self.force_rgb_encoding = force_rgb_encoding
        self.robot_base_deque = None
        self.puppet_arm_right_deque = None
        self.puppet_arm_left_deque = None
        self.img_front_deque = None
        self.img_right_deque = None
        self.img_left_deque = None
        self.img_front_depth_deque = None
        self.img_right_depth_deque = None
        self.img_left_depth_deque = None
        self.bridge = None
        self.puppet_arm_left_publisher = None
        self.puppet_arm_right_publisher = None
        self.robot_base_publisher = None
        self.puppet_arm_publish_thread = None
        self.puppet_arm_publish_lock = None

        self.publish_rate = publish_rate
        self.arm_steps_length = arm_steps_length if arm_steps_length is not None else [0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.2]
        self.use_depth_image = use_depth_image
        self.use_robot_base = use_robot_base
        self.img_left_topic = img_left_topic
        self.img_right_topic = img_right_topic
        self.img_front_topic = img_front_topic
        self.img_left_depth_topic = img_left_depth_topic
        self.img_right_depth_topic = img_right_depth_topic
        self.img_front_depth_topic = img_front_depth_topic
        self.puppet_arm_left_topic = puppet_arm_left_topic
        self.puppet_arm_right_topic = puppet_arm_right_topic
        self.robot_base_topic = robot_base_topic
        self.puppet_arm_left_cmd_topic = puppet_arm_left_cmd_topic
        self.puppet_arm_right_cmd_topic = puppet_arm_right_cmd_topic
        self.robot_base_cmd_topic = robot_base_cmd_topic

        self.ctrl_state = False
        self.ctrl_state_lock = threading.Lock()
        self.init()
        self.init_ros()

    def init(self):
        self.bridge = CvBridge()
        self.img_left_deque = deque()
        self.img_right_deque = deque()
        self.img_front_deque = deque()
        self.img_left_depth_deque = deque()
        self.img_right_depth_deque = deque()
        self.img_front_depth_deque = deque()
        self.puppet_arm_left_deque = deque()
        self.puppet_arm_right_deque = deque()
        self.robot_base_deque = deque()
        self.puppet_arm_publish_lock = threading.Lock()
        self.puppet_arm_publish_lock.acquire()

    def puppet_arm_publish(self, left, right):
        joint_state_msg = JointState()
        joint_state_msg.header = Header()
        joint_state_msg.header.stamp = rospy.Time.now()
        joint_state_msg.name = ['joint0', 'joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'joint6']
        joint_state_msg.position = left
        self.puppet_arm_left_publisher.publish(joint_state_msg)
        joint_state_msg.position = right
        self.puppet_arm_right_publisher.publish(joint_state_msg)

    def robot_base_publish(self, vel):
        vel_msg = Twist()
        vel_msg.linear.x = vel[0]
        vel_msg.linear.y = 0
        vel_msg.linear.z = 0
        vel_msg.angular.x = 0
        vel_msg.angular.y = 0
        vel_msg.angular.z = vel[1]
        self.robot_base_publisher.publish(vel_msg)

    def puppet_arm_publish_continuous(self, left, right):
        rate = rospy.Rate(self.publish_rate)
        left_arm = None
        right_arm = None
        while True and not rospy.is_shutdown():
            if len(self.puppet_arm_left_deque) != 0:
                left_arm = list(self.puppet_arm_left_deque[-1].position)
            if len(self.puppet_arm_right_deque) != 0:
                right_arm = list(self.puppet_arm_right_deque[-1].position)
            if left_arm is None or right_arm is None:
                rate.sleep()
                continue
            else:
                break
        left_symbol = [1 if left[i] - left_arm[i] > 0 else -1 for i in range(len(left))]
        right_symbol = [1 if right[i] - right_arm[i] > 0 else -1 for i in range(len(right))]
        flag = True
        step = 0
        while flag and not rospy.is_shutdown():
            if self.puppet_arm_publish_lock.acquire(False):
                return
            left_diff = [abs(left[i] - left_arm[i]) for i in range(len(left))]
            right_diff = [abs(right[i] - right_arm[i]) for i in range(len(right))]
            flag = False
            for i in range(len(left)):
                if left_diff[i] < self.arm_steps_length[i]:
                    left_arm[i] = left[i]
                else:
                    left_arm[i] += left_symbol[i] * self.arm_steps_length[i]
                    flag = True
            for i in range(len(right)):
                if right_diff[i] < self.arm_steps_length[i]:
                    right_arm[i] = right[i]
                else:
                    right_arm[i] += right_symbol[i] * self.arm_steps_length[i]
                    flag = True
            joint_state_msg = JointState()
            joint_state_msg.header = Header()
            joint_state_msg.header.stamp = rospy.Time.now()
            joint_state_msg.name = ['joint0', 'joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'joint6']
            joint_state_msg.position = left_arm
            self.puppet_arm_left_publisher.publish(joint_state_msg)
            joint_state_msg.position = right_arm
            self.puppet_arm_right_publisher.publish(joint_state_msg)
            step += 1
            print('puppet_arm_publish_continuous:', step)
            rate.sleep()

    def puppet_arm_publish_linear(self, left, right):
        num_step = 100
        rate = rospy.Rate(200)

        left_arm = None
        right_arm = None

        while True and not rospy.is_shutdown():
            if len(self.puppet_arm_left_deque) != 0:
                left_arm = list(self.puppet_arm_left_deque[-1].position)
            if len(self.puppet_arm_right_deque) != 0:
                right_arm = list(self.puppet_arm_right_deque[-1].position)
            if left_arm is None or right_arm is None:
                rate.sleep()
                continue
            else:
                break

        traj_left_list = np.linspace(left_arm, left, num_step)
        traj_right_list = np.linspace(right_arm, right, num_step)

        for i in range(len(traj_left_list)):
            traj_left = traj_left_list[i]
            traj_right = traj_right_list[i]
            traj_left[-1] = left[-1]
            traj_right[-1] = right[-1]
            joint_state_msg = JointState()
            joint_state_msg.header = Header()
            joint_state_msg.header.stamp = rospy.Time.now()
            joint_state_msg.name = ['joint0', 'joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'joint6']
            joint_state_msg.position = traj_left
            self.puppet_arm_left_publisher.publish(joint_state_msg)
            joint_state_msg.position = traj_right
            self.puppet_arm_right_publisher.publish(joint_state_msg)
            rate.sleep()

    def puppet_arm_publish_continuous_thread(self, left, right):
        if self.puppet_arm_publish_thread is not None:
            self.puppet_arm_publish_lock.release()
            self.puppet_arm_publish_thread.join()
            self.puppet_arm_publish_lock.acquire(False)
            self.puppet_arm_publish_thread = None
        self.puppet_arm_publish_thread = threading.Thread(target=self.puppet_arm_publish_continuous, args=(left, right))
        self.puppet_arm_publish_thread.start()

    def get_frame(self):
        if (
            len(self.img_left_deque) == 0
            or len(self.img_right_deque) == 0
            or len(self.img_front_deque) == 0
            or (
                self.use_depth_image
                and (len(self.img_left_depth_deque) == 0 or len(self.img_right_depth_deque) == 0 or len(self.img_front_depth_deque) == 0)
            )
        ):
            return False
        if self.use_depth_image:
            frame_time = min(
                [
                    self.img_left_deque[-1].header.stamp.to_sec(),
                    self.img_right_deque[-1].header.stamp.to_sec(),
                    self.img_front_deque[-1].header.stamp.to_sec(),
                    self.img_left_depth_deque[-1].header.stamp.to_sec(),
                    self.img_right_depth_deque[-1].header.stamp.to_sec(),
                    self.img_front_depth_deque[-1].header.stamp.to_sec(),
                ]
            )
        else:
            frame_time = min(
                [
                    self.img_left_deque[-1].header.stamp.to_sec(),
                    self.img_right_deque[-1].header.stamp.to_sec(),
                    self.img_front_deque[-1].header.stamp.to_sec(),
                ]
            )

        if len(self.img_left_deque) == 0 or self.img_left_deque[-1].header.stamp.to_sec() < frame_time:
            return False
        if len(self.img_right_deque) == 0 or self.img_right_deque[-1].header.stamp.to_sec() < frame_time:
            return False
        if len(self.img_front_deque) == 0 or self.img_front_deque[-1].header.stamp.to_sec() < frame_time:
            return False
        if len(self.puppet_arm_left_deque) == 0 or self.puppet_arm_left_deque[-1].header.stamp.to_sec() < frame_time:
            return False
        if len(self.puppet_arm_right_deque) == 0 or self.puppet_arm_right_deque[-1].header.stamp.to_sec() < frame_time:
            return False
        if self.use_depth_image and (len(self.img_left_depth_deque) == 0 or self.img_left_depth_deque[-1].header.stamp.to_sec() < frame_time):
            return False
        if self.use_depth_image and (len(self.img_right_depth_deque) == 0 or self.img_right_depth_deque[-1].header.stamp.to_sec() < frame_time):
            return False
        if self.use_depth_image and (len(self.img_front_depth_deque) == 0 or self.img_front_depth_deque[-1].header.stamp.to_sec() < frame_time):
            return False
        if self.use_robot_base and (len(self.robot_base_deque) == 0 or self.robot_base_deque[-1].header.stamp.to_sec() < frame_time):
            return False

        color_encoding = 'rgb8' if self.force_rgb_encoding else 'passthrough'
        while self.img_left_deque[0].header.stamp.to_sec() < frame_time:
            self.img_left_deque.popleft()
        img_left = self.bridge.imgmsg_to_cv2(self.img_left_deque.popleft(), color_encoding)

        while self.img_right_deque[0].header.stamp.to_sec() < frame_time:
            self.img_right_deque.popleft()
        img_right = self.bridge.imgmsg_to_cv2(self.img_right_deque.popleft(), color_encoding)

        while self.img_front_deque[0].header.stamp.to_sec() < frame_time:
            self.img_front_deque.popleft()
        img_front = self.bridge.imgmsg_to_cv2(self.img_front_deque.popleft(), color_encoding)

        while self.puppet_arm_left_deque[0].header.stamp.to_sec() < frame_time:
            self.puppet_arm_left_deque.popleft()
        puppet_arm_left = self.puppet_arm_left_deque.popleft()

        while self.puppet_arm_right_deque[0].header.stamp.to_sec() < frame_time:
            self.puppet_arm_right_deque.popleft()
        puppet_arm_right = self.puppet_arm_right_deque.popleft()

        img_left_depth = None
        if self.use_depth_image:
            while self.img_left_depth_deque[0].header.stamp.to_sec() < frame_time:
                self.img_left_depth_deque.popleft()
            img_left_depth = self.bridge.imgmsg_to_cv2(self.img_left_depth_deque.popleft(), 'passthrough')

        img_right_depth = None
        if self.use_depth_image:
            while self.img_right_depth_deque[0].header.stamp.to_sec() < frame_time:
                self.img_right_depth_deque.popleft()
            img_right_depth = self.bridge.imgmsg_to_cv2(self.img_right_depth_deque.popleft(), 'passthrough')

        img_front_depth = None
        if self.use_depth_image:
            while self.img_front_depth_deque[0].header.stamp.to_sec() < frame_time:
                self.img_front_depth_deque.popleft()
            img_front_depth = self.bridge.imgmsg_to_cv2(self.img_front_depth_deque.popleft(), 'passthrough')

        robot_base = None
        if self.use_robot_base:
            while self.robot_base_deque[0].header.stamp.to_sec() < frame_time:
                self.robot_base_deque.popleft()
            robot_base = self.robot_base_deque.popleft()

        return (img_front, img_left, img_right, img_front_depth, img_left_depth, img_right_depth, puppet_arm_left, puppet_arm_right, robot_base)

    def img_left_callback(self, msg):
        if len(self.img_left_deque) >= 2000:
            self.img_left_deque.popleft()
        self.img_left_deque.append(msg)

    def img_right_callback(self, msg):
        if len(self.img_right_deque) >= 2000:
            self.img_right_deque.popleft()
        self.img_right_deque.append(msg)

    def img_front_callback(self, msg):
        if len(self.img_front_deque) >= 2000:
            self.img_front_deque.popleft()
        self.img_front_deque.append(msg)

    def img_left_depth_callback(self, msg):
        if len(self.img_left_depth_deque) >= 2000:
            self.img_left_depth_deque.popleft()
        self.img_left_depth_deque.append(msg)

    def img_right_depth_callback(self, msg):
        if len(self.img_right_depth_deque) >= 2000:
            self.img_right_depth_deque.popleft()
        self.img_right_depth_deque.append(msg)

    def img_front_depth_callback(self, msg):
        if len(self.img_front_depth_deque) >= 2000:
            self.img_front_depth_deque.popleft()
        self.img_front_depth_deque.append(msg)

    def puppet_arm_left_callback(self, msg):
        if len(self.puppet_arm_left_deque) >= 2000:
            self.puppet_arm_left_deque.popleft()
        self.puppet_arm_left_deque.append(msg)

    def puppet_arm_right_callback(self, msg):
        if len(self.puppet_arm_right_deque) >= 2000:
            self.puppet_arm_right_deque.popleft()
        self.puppet_arm_right_deque.append(msg)

    def robot_base_callback(self, msg):
        if len(self.robot_base_deque) >= 2000:
            self.robot_base_deque.popleft()
        self.robot_base_deque.append(msg)

    def ctrl_callback(self, msg):
        self.ctrl_state_lock.acquire()
        self.ctrl_state = msg.data
        self.ctrl_state_lock.release()

    def get_ctrl_state(self):
        self.ctrl_state_lock.acquire()
        state = self.ctrl_state
        self.ctrl_state_lock.release()
        return state

    def init_ros(self):
        rospy.init_node('joint_state_publisher', anonymous=True)
        rospy.Subscriber(self.img_left_topic, Image, self.img_left_callback, queue_size=1000, tcp_nodelay=True)
        rospy.Subscriber(self.img_right_topic, Image, self.img_right_callback, queue_size=1000, tcp_nodelay=True)
        rospy.Subscriber(self.img_front_topic, Image, self.img_front_callback, queue_size=1000, tcp_nodelay=True)
        if self.use_depth_image:
            rospy.Subscriber(self.img_left_depth_topic, Image, self.img_left_depth_callback, queue_size=1000, tcp_nodelay=True)
            rospy.Subscriber(self.img_right_depth_topic, Image, self.img_right_depth_callback, queue_size=1000, tcp_nodelay=True)
            rospy.Subscriber(self.img_front_depth_topic, Image, self.img_front_depth_callback, queue_size=1000, tcp_nodelay=True)
        rospy.Subscriber(self.puppet_arm_left_topic, JointState, self.puppet_arm_left_callback, queue_size=1000, tcp_nodelay=True)
        rospy.Subscriber(self.puppet_arm_right_topic, JointState, self.puppet_arm_right_callback, queue_size=1000, tcp_nodelay=True)
        rospy.Subscriber(self.robot_base_topic, Odometry, self.robot_base_callback, queue_size=1000, tcp_nodelay=True)
        self.puppet_arm_left_publisher = rospy.Publisher(self.puppet_arm_left_cmd_topic, JointState, queue_size=10)
        self.puppet_arm_right_publisher = rospy.Publisher(self.puppet_arm_right_cmd_topic, JointState, queue_size=10)
        self.robot_base_publisher = rospy.Publisher(self.robot_base_cmd_topic, Twist, queue_size=10)


def main(
    image_mode: ImageMode = 'uint8',
    force_rgb: bool = False,
    apply_gripper_rescale: bool = True,
    host: str = '0.0.0.0',
    port: int = 8080,
    task_name: str = 'Clean desk',
    publish_rate: int = 30,
    max_publish_step: int = 10000,
    chunk_size: int = 50,
    state_dim: int = 14,
    action_dim: Optional[int] = None,
    use_robot_base: bool = False,
    use_depth_image: bool = False,
    inference_trigger_remaining: int = 10,
    max_action_execute_horizon: int = 35,
    distance_thresh: float = 0.5,
    continuity_blend_steps: int = 10,
    align_search_window: int = 8,
    gripper_offset: float = 0.0,
    gripper_zero_threshold: Optional[float] = None,
    init_pose_mode: Literal['unified', 'smooth_dev'] = 'unified',
    pos_lookahead_step: int = 50,
    observation_memory_offsets: Tuple[int, ...] = (0,),
):
    """Run the unified ROS inference client — smooth (sliding-window) variant.

    Args:
        image_mode: Client-side image preprocessing.
            'uint8'           - send raw uint8; the server resizes (default).
            'float_native'    - convert to float32 [0,1] at native resolution; server resizes in float
                                space (matches PG2-aligned client).
            'float_resize224' - client-side resize_with_pad to 224 + float32 (matches legacy PG2 baseline client).
        force_rgb: cv_bridge transcode to 'rgb8' (True) vs 'passthrough' (False).
        apply_gripper_rescale: Rescale gripper actions before publishing.
            (left/right gripper < 0.04 → -0.03, else *= 1.5). Default True.
        host/port: server address (defaults 0.0.0.0:8080; use 127.0.0.1 for the legacy PG2 baseline server).
        task_name: language instruction sent with each observation.
        state_dim: Width of the observation.state sent to the server.
        action_dim: Expected server action width. None reads action_output_dim from
            server_info, so mobile 14-D state / 16-D action deployments work by default.
        chunk_size: 服务端单次推理返回的动作数。
        inference_trigger_remaining: num1, 缓冲区剩余这么多动作时提前发起下一次异步推理。
        max_action_execute_horizon: 单次推理最多执行多少个动作 (<= chunk_size)。
        distance_thresh: 自适应 horizon 的 L2 距离阈值, 轨迹发散快时缩短执行 horizon。
        continuity_blend_steps: 拼接处连续性偏置的衰减步数, 强制新段首帧对齐到最近已执行动作并
            在该步数内衰减到 0, 消除拼接卡顿。0=关闭; 越大越平滑但对新轨迹收敛越慢 (默认 10)。
        align_search_window: 拼接起点就近匹配的搜索半窗 (以 adaptive_num2 为中心 ±该值)。
            0=关闭, 严格退回旧 smooth_dev 的纯时间估计拼接 (new_actions_start=adaptive_num2)。
        gripper_offset: 发布前加到两个夹爪列 (6/13) 的偏置。旧 smooth_dev 发布前固定
            gripper -= 0.015, 严格对照时传 -0.015 (默认 0.0 不加)。
        gripper_zero_threshold: rescale 和 offset 后，夹爪值小于该阈值时置为 0.0。
            None 表示关闭（默认）。
        init_pose_mode: episode 起始位姿。'unified'=回全零位 (默认, 与 unified 客户端一致);
            'smooth_dev'=旧 client_gigabrain0.1.1_smooth_dev.py 硬编码的初始位姿。
        pos_lookahead_step: 兼容旧 clinet.sh 的遗留参数, 滑动窗口调度不使用, 仅为不报错而暴露。
        observation_memory_offsets: 兼容旧 clinet.sh; 本 unified 协议客户端只发送最新单帧,
            因此仅 (0,) 有意义, 传入多帧历史会被忽略并打印 warning。
    """
    if set(observation_memory_offsets) != {0}:
        print(
            f'[WARN] observation_memory_offsets={list(observation_memory_offsets)} ignored: '
            'this unified-protocol client sends only the single newest frame '
            '(multi-frame memory needs server-side support not in make_infer_data).'
        )

    ros_operator = RosOperator(
        publish_rate=publish_rate,
        use_robot_base=use_robot_base,
        use_depth_image=use_depth_image,
        force_rgb_encoding=force_rgb,
    )
    client = RobotInferenceClient(host=host, port=port)
    _, resolved_action_dim = validate_server_dimensions(
        client,
        state_dim=state_dim,
        action_dim=action_dim,
    )
    controller = SmoothInferenceController(
        client=client,
        ros_operator=ros_operator,
        task_name=task_name,
        publish_rate=publish_rate,
        chunk_size=chunk_size,
        inference_trigger_remaining=inference_trigger_remaining,
        max_action_execute_horizon=max_action_execute_horizon,
        distance_thresh=distance_thresh,
        max_publish_step=max_publish_step,
        state_dim=state_dim,
        action_dim=resolved_action_dim,
        use_robot_base=use_robot_base,
        image_mode=image_mode,
        apply_gripper_rescale=apply_gripper_rescale,
        continuity_blend_steps=continuity_blend_steps,
        align_search_window=align_search_window,
        gripper_offset=gripper_offset,
        gripper_zero_threshold=gripper_zero_threshold,
        init_pose_mode=init_pose_mode,
    )
    controller.run()


if __name__ == '__main__':
    tyro.cli(main)
