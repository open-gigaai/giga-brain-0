#!/usr/bin/env python3
"""
inference_h01_smooth.py - h01/t170d 机器人异步推理与动作平滑

基于 inference.py 架构，引入：
- 异步推理：推理与动作执行并行
- Butterworth 低通滤波平滑（仅关节，夹爪不平滑）
- 滑动窗口动作拼接
- cam_high 使用 /camera_fisheye_front/image 的头部鱼眼图像
"""

import sys
import numpy as np
from cv_bridge import CvBridge
import argparse
import cv2
import random
import rclpy
from rclpy.node import Node
from std_srvs.srv import Trigger
from sensor_msgs.msg import JointState, Image
from rclpy.qos import qos_profile_sensor_data
import threading
import time
import datetime
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict
from scipy.signal import butter, filtfilt

from openpi_client import image_tools

try:
    from fixed_image_msgs.msg import FixedYUYVImage
except Exception:  # pragma: no cover - ROS env only
    FixedYUYVImage = None
# ==================== 全局变量 ====================
# 异步推理相关
inference_thread = None
inference_lock = threading.Lock()

# 动作缓冲区
action_buffer = None
action_buffer_lock = threading.Lock()
action_index = 0

# 待应用的推理结果
pending_inference_result = None
pending_inference_lock = threading.Lock()
inference_trigger_step = None

# --dump_obs 已保存的帧数（限量，避免刷屏/占盘）
_dump_obs_saved = 0

# 延迟加载的 H01 FK/IK 转换函数（仅 endpose 模式使用）
_h01_fk_ik_funcs = None

class JointNames:
    LEFT_ARM = [
        "L_SHOULDER_P",
        "L_SHOULDER_R",
        "L_SHOULDER_Y",
        "L_ELBOW_Y",
        "L_WRIST_P",
        "L_WRIST_Y",
        "L_WRIST_R",
    ]

    RIGHT_ARM = [
        "R_SHOULDER_P",
        "R_SHOULDER_R",
        "R_SHOULDER_Y",
        "R_ELBOW_Y",
        "R_WRIST_P",
        "R_WRIST_Y",
        "R_WRIST_R",
    ]

    LEFT_GRIPPER = ["left_gripper"]
    RIGHT_GRIPPER = ["right_gripper"]
    # 腰 4 关节，真机 /waist/joint_state 的 name（大写、1-indexed），对应数据集 state dim 18-21
    # (waist_joint_0..3) 的顺序：WAIST_1↔waist_0, WAIST_2↔waist_1, WAIST_3↔waist_2, WAIST_4↔waist_3。
    WAIST = ["WAIST_1", "WAIST_2", "WAIST_3", "WAIST_4"]
    # 头 2 关节，真机 /head/joint_state 的 name，对应数据集 state dim 16-17。
    HEAD = ["HEAD_YAW", "HEAD_PITCH"]

class JointLimits:
    GRIPPER = (0.0, 0.06)

class ServiceNames:
    LEFT_ARM_RESET = "/left_arm/joint_reset"
    RIGHT_ARM_RESET = "/right_arm/joint_reset"

    LEFT_GRIPPER_CALIBRATE = "/left_gripper/calibrate_gripper_zero"
    LEFT_GRIPPER_ENABLE = "/left_gripper/enable_gripper"
    LEFT_GRIPPER_DISABLE = "/left_gripper/disable_gripper"
    RIGHT_GRIPPER_CALIBRATE = "/right_gripper/calibrate_gripper_zero"
    RIGHT_GRIPPER_ENABLE = "/right_gripper/enable_gripper"
    RIGHT_GRIPPER_DISABLE = "/right_gripper/disable_gripper"

class TopicNames:
    LEFT_ARM_STATE = "/left_arm/joint_state"
    LEFT_ARM_COMMAND = "/left_arm/joint_command"
    LEFT_GRIPPER_STATE = "/left_gripper/joint_state"
    LEFT_GRIPPER_COMMAND = "/left_gripper/joint_command"

    RIGHT_ARM_STATE = "/right_arm/joint_state"
    RIGHT_ARM_COMMAND = "/right_arm/joint_command"
    RIGHT_GRIPPER_STATE = "/right_gripper/joint_state"
    RIGHT_GRIPPER_COMMAND = "/right_gripper/joint_command"

    WAIST_STATE = "/waist/joint_state"
    WAIST_COMMAND = "/waist/joint_command"

    HEAD_STATE = "/head/joint_state"
    HEAD_COMMAND = "/head/joint_command"

@dataclass
class JointData:
    position: float = 0.0
    velocity: float = 0.0
    effort: float = 0.0
    timestamp: float = 0.0

@dataclass
class JointGroupState:
    joints: Dict[str, JointData] = field(default_factory=dict)

    def update_from_msg(self, msg: JointState):
        for index, name in enumerate(msg.name):
            if name not in self.joints:
                continue
            joint = self.joints[name]
            joint.position = msg.position[index] if index < len(msg.position) else 0.0
            joint.velocity = msg.velocity[index] if index < len(msg.velocity) else 0.0
            joint.effort = msg.effort[index] if index < len(msg.effort) else 0.0
            joint.timestamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9

    def positions(self, joint_names):
        return [self.joints[name].position for name in joint_names]

    def is_ready(self, joint_names):
        return all(self.joints[name].timestamp > 0.0 for name in joint_names)

def create_joint_group(joint_names):
    state = JointGroupState()
    for name in joint_names:
        state.joints[name] = JointData()
    return state

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)

# ==================== 平滑函数 ====================
def butterworth_lowpass_filter(data: np.ndarray, cutoff_freq: float = 1.0,
                                sampling_freq: float = 15.0, order: int = 2) -> np.ndarray:
    """
    Butterworth 低通滤波器

    Args:
        data: 输入数据 (N, D)
        cutoff_freq: 截止频率 (Hz)，越低越平滑
        sampling_freq: 采样频率 (Hz)
        order: 滤波器阶数

    Returns:
        滤波后的数据
    """
    nyquist = 0.5 * sampling_freq
    normal_cutoff = cutoff_freq / nyquist
    b, a = butter(order, normal_cutoff, btype="low", analog=False)
    # filtfilt needs enough samples for padding; fallback to no-op if too short
    padlen = 3 * (max(len(a), len(b)) - 1)
    if data.shape[0] <= padlen:
        return data
    filtered_data = filtfilt(b, a, data, axis=0)
    return filtered_data

def smoothen_actions(actions: np.ndarray) -> np.ndarray:
    """
    平滑动作序列，夹爪保持原始值

    Args:
        actions: 动作序列 (N, D)，D 为 16 或 22；前 16 维格式为
            [左臂7关节, 左夹爪, 右臂7关节, 右夹爪]
            注意：endpose 动作需要先通过 IK 转成 qpose 后再调用本函数。

    Returns:
        平滑后的动作序列
    """
    if len(actions) <= 1:
        return actions

    actions_smoothed = actions.copy()

    # 左臂关节平滑 (索引 0-6)
    actions_smoothed[:, :7] = butterworth_lowpass_filter(actions[:, :7].copy())
    # 右臂关节平滑 (索引 8-14)
    actions_smoothed[:, 8:15] = butterworth_lowpass_filter(actions[:, 8:15].copy())
    # 夹爪保持原始值 (索引 7 和 15)

    return actions_smoothed

def smooth_action_transition(actions: np.ndarray, boundary_index: int, blend_radius: int = 3) -> np.ndarray:
    """
    对拼接边界做局部过渡平滑，避免新旧动作块在接缝处发生突变。

    Args:
        actions: 拼接后的动作序列 (N, D)，D 为 16 或 22
        boundary_index: 新动作块开始的位置
        blend_radius: 边界两侧参与过渡的帧数

    Returns:
        局部平滑后的动作序列
    """
    if len(actions) <= 1 or boundary_index <= 0 or boundary_index >= len(actions):
        return actions

    start = max(0, boundary_index - blend_radius)
    end = min(len(actions) - 1, boundary_index + blend_radius - 1)
    if end <= start:
        return actions

    blended = actions.copy()
    left_anchor = actions[start].copy()
    right_anchor = actions[end].copy()
    total = end - start

    # 仅平滑双臂关节，不修改夹爪开合值。
    joint_slices = [slice(0, 7), slice(8, 15)]
    for idx in range(start, end + 1):
        alpha = (idx - start) / total if total > 0 else 1.0
        for joint_slice in joint_slices:
            blended[idx, joint_slice] = (
                (1.0 - alpha) * left_anchor[joint_slice] +
                alpha * right_anchor[joint_slice]
            )

    return blended

def find_nearest_action(actions: np.ndarray, target_action: np.ndarray) -> int:
    """
    在动作序列中找到与目标动作 L2 距离最近的位置

    Args:
        actions: 动作序列 (N, D)，D 为 16 或 22
        target_action: 目标动作 (D,)

    Returns:
        最近动作的索引
    """
    action_copy = actions.copy()
    target_copy = target_action.copy()

    # 放大夹爪权重（索引 7 和 15），因为夹爪范围是 0-0.06，数值太小
    gripper_weight = 30.0
    target_copy[7] *= gripper_weight
    target_copy[15] *= gripper_weight
    action_copy[:, 7] *= gripper_weight
    action_copy[:, 15] *= gripper_weight

    distances = np.linalg.norm(action_copy - target_copy, axis=1)
    return int(np.argmin(distances))

def _get_h01_fk_ik_funcs():
    """Lazy import H01 FK/IK helpers to avoid impacting qpose mode startup."""
    global _h01_fk_ik_funcs
    if _h01_fk_ik_funcs is not None:
        return _h01_fk_ik_funcs

    try:
        from h01_fk_ik import transform_state_to_end_pose, transform_epose_to_qpose
    except (ImportError, ModuleNotFoundError) as first_exc:
        try:
            # Fallback for module-style execution from repo root.
            from h01.h01_fk_ik import transform_state_to_end_pose, transform_epose_to_qpose
        except (ImportError, ModuleNotFoundError) as second_exc:
            for exc in (first_exc, second_exc):
                if isinstance(exc, ModuleNotFoundError) and exc.name == "placo":
                    raise ImportError(
                        "H01 endpose IK requires the 'placo' package. "
                        "Install it in this environment (for example: `pip install placo`) "
                        "or run with --policy_action_space qpose."
                    ) from exc
            raise

    _h01_fk_ik_funcs = (transform_state_to_end_pose, transform_epose_to_qpose)
    return _h01_fk_ik_funcs

def build_policy_state(qpos: np.ndarray, action_space: str) -> np.ndarray:
    """
    Build policy input state according to action space.

    - qpose mode: returns 16D qpose
    - endpose mode: converts qpose -> 14D epose via FK
    """
    qpos = np.asarray(qpos, dtype=np.float32)
    if qpos.ndim != 1:
        raise ValueError(f"qpos must be 1D, got shape {qpos.shape}")

    if action_space == "qpose":
        if qpos.shape[0] != 16:
            raise ValueError(f"qpose state must be 16D, got {qpos.shape}")
        return qpos

    if action_space == "endpose":
        if qpos.shape[0] != 16:
            raise ValueError(f"endpose FK input qpos must be 16D, got {qpos.shape}")
        transform_state_to_end_pose, _ = _get_h01_fk_ik_funcs()
        epose = transform_state_to_end_pose(qpos[None, :])[0]
        epose = np.asarray(epose, dtype=np.float32)
        if epose.shape[0] != 14:
            raise ValueError(f"epose state must be 14D, got {epose.shape}")
        return epose

    raise ValueError(f"Unknown policy action space: {action_space}")

def normalize_action_chunk(raw_actions) -> np.ndarray:
    """Normalize server action chunk to shape (N, D)."""
    actions = np.asarray(raw_actions, dtype=np.float32)
    if actions.ndim == 3 and actions.shape[0] == 1:
        actions = actions[0]
    elif actions.ndim == 1:
        actions = actions[None, :]
    if actions.ndim != 2:
        raise ValueError(
            f"actions must be (D,), (N,D), or (1,N,D), got shape {actions.shape}"
        )
    return actions

def convert_policy_actions_to_qpose(actions: np.ndarray, current_qpos: np.ndarray, args):
    """
    Convert policy output actions to a qpose chunk.

    qpose input keeps the configured 16/22D layout; endpose input is converted
    by IK to the 16D arm/gripper layout.

    Returns:
        qpose_actions, ik_duration_sec
    """
    if args.policy_action_space == "qpose":
        if actions.shape[1] != args.action_dim:
            raise ValueError(
                f"qpose mode expects actions shape (N,{args.action_dim}), "
                f"got {actions.shape}"
            )
        return actions, 0.0

    if args.policy_action_space == "endpose":
        if actions.shape[1] != 14:
            raise ValueError(
                f"endpose mode expects actions shape (N,14), got {actions.shape}"
            )
        _, transform_epose_to_qpose = _get_h01_fk_ik_funcs()
        ik_start = time.time()
        qpose_actions = transform_epose_to_qpose(
            actions,
            init_qpose=np.asarray(current_qpos, dtype=np.float32),
            urdf_path=Path(args.h01_urdf_path),
            use_prev_as_ref=True,
            use_joint_limits=True,
        )
        ik_duration = time.time() - ik_start
        qpose_actions = np.asarray(qpose_actions, dtype=np.float32)
        if qpose_actions.ndim != 2 or qpose_actions.shape[1] != 16:
            raise ValueError(
                f"IK converted qpose actions must be (N,16), got {qpose_actions.shape}"
            )
        return qpose_actions, ik_duration

    raise ValueError(f"Unknown policy action space: {args.policy_action_space}")

# ==================== State / Action 维度对齐 ====================
# 发给模型的 state 固定为 22 维：双臂+双夹爪 16 维，头 2 维，腰 4 维。
SERVER_STATE_DIM = 22
ROBOT_ACTION_DIM = 16
FULL_BODY_ACTION_DIM = 22
SUPPORTED_ACTION_DIMS = (ROBOT_ACTION_DIM, FULL_BODY_ACTION_DIM)

# ==================== Ready pose ====================
# 每个任务一个 ready pose = 该任务训练集 episode 起始帧 observation.state 的中位数（分布内起始位姿）。
# ready pose 是一个列表：16 维 = [左臂×7, 左夹爪, 右臂×7, 右夹爪]（不控腰）；
# 20 维 = 前 16 维 + [腰×4]（末 4 维 = WAIST_1..4，见 JointNames.WAIST）。
# 臂关节顺序对应 JointNames.LEFT_ARM/RIGHT_ARM = [SHOULDER_P, SHOULDER_R, SHOULDER_Y, ELBOW_Y,
# WRIST_P, WRIST_Y, WRIST_R]。
# 单一 ready pose(20 维 = 左臂7,左爪,右臂7,右爪,腰4[WAIST_1..4])。取自 push_button 起始帧中位数
# (pick_fork/push_button 臂几乎相同,腰前探够按钮),button 家族(含 beside_plate)通用。
READY_POSE = [
    -0.799971, -0.799971, 0.499982, 1.200052, 0.299793, -0.000192, -0.000192, 0.056217,
    0.800067, 0.799971, -0.499982, -1.200148, -0.300176, -0.000192, -0.000192, 0.058804,
    0.013998, -0.768381, 1.487650, 0.979063,
]


def split_ready_pose(pose):
    """把 ready pose 列表拆成 (左臂7, 左夹爪, 右臂7, 右夹爪, 腰4或None)。

    支持 16 维（臂+爪，无腰）或 20 维（臂+爪+腰4）。16 维时腰返回 None（不控腰）。
    """
    n = len(pose)
    if n not in (16, 20):
        raise ValueError(f"ready pose 必须是 16 维[臂+爪] 或 20 维[臂+爪+腰4]，收到 {n} 维")
    left_arm = list(pose[0:7])
    left_grip = float(pose[7])
    right_arm = list(pose[8:15])
    right_grip = float(pose[15])
    waist = list(pose[16:20]) if n == 20 else None
    return left_arm, left_grip, right_arm, right_grip, waist

def pad_state_to_server_dim(state: np.ndarray, target_dim: int = SERVER_STATE_DIM) -> np.ndarray:
    """将 state 末尾补零或截断，对齐到服务器期望的维度（默认固定为 22 维）。

    Args:
        state: 原始 state（1D），例如 qpose 的 16 维。
        target_dim: 目标维度，模型请求链路固定传入 22。

    Returns:
        长度为 target_dim 的 1D float32 数组；若原始维度已 >= 目标则截断到目标维度。
    """
    state = np.asarray(state, dtype=np.float32).reshape(-1)
    if state.shape[0] == target_dim:
        return state
    if state.shape[0] > target_dim:
        return state[:target_dim]
    padded = np.zeros(target_dim, dtype=np.float32)
    padded[:state.shape[0]] = state
    return padded

# state dim16-21 = [HEAD_YAW, HEAD_PITCH, WAIST_1..4]，训练数据里这 6 维是真实关节值。
# 旧版本在这里补零，归一化后严重越界（实测 pound_clay 的 WAIST_3 补零后 ≈ -12，正常应在 ±1），
# 模型本体感知错乱 → 动作退化成几乎不动。
_head_waist_warned = False


def fill_head_waist_into_state(state: np.ndarray, controller) -> np.ndarray:
    """就地把真实的头(dim16-17)/腰(dim18-21)反馈写入 22 维 state。

    读不到反馈时保留原值(0)并告警一次——最差退回旧行为，不会更糟。
    """
    global _head_waist_warned
    if state.shape[0] < 22:
        return state
    missing = []
    with controller.state_lock:
        if controller.head_state.is_ready(JointNames.HEAD):
            state[16:18] = controller.head_state.positions(JointNames.HEAD)
        else:
            missing.append("head")
        if controller.waist_state.is_ready(JointNames.WAIST):
            state[18:22] = controller.waist_state.positions(JointNames.WAIST)
        else:
            missing.append("waist")
    if missing and not _head_waist_warned:
        _head_waist_warned = True
        print("[STATE] 警告: 收不到 %s 反馈，state dim16-21 保持为 0（会偏离训练分布）"
              % "/".join(missing))
    return state


def validate_server_state(state: np.ndarray) -> np.ndarray:
    """校验并返回模型协议要求的 22 维 float32 state。"""
    state = np.asarray(state, dtype=np.float32)
    if state.shape != (SERVER_STATE_DIM,):
        raise ValueError(
            f"model state must have shape ({SERVER_STATE_DIM},), got {state.shape}"
        )
    return state


def build_server_state(state: np.ndarray, controller) -> np.ndarray:
    """构造固定 22 维的模型输入 state，并填入真实头/腰反馈。"""
    state = pad_state_to_server_dim(state, SERVER_STATE_DIM)
    state = fill_head_waist_into_state(state, controller)
    return validate_server_state(state)


def slice_actions_to_robot_dim(raw_actions, dim: int = ROBOT_ACTION_DIM) -> np.ndarray:
    """规范化动作块并截取 client 配置的 16/22 维发布前缀。

    返回宽度可以大于配置宽度；短于配置宽度时拒绝该动作块，避免用补零的
    绝对关节命令驱动缺失维度。
    """
    actions = normalize_action_chunk(raw_actions)
    if actions.shape[1] < dim:
        raise ValueError(
            f"server action width {actions.shape[1]} is smaller than "
            f"configured action_dim={dim}"
        )
    return actions[:, :dim]


def publish_action_to_robot(controller, action, action_dim: int) -> None:
    """按 H01 16/22 维布局把一帧动作发布到各 ROS joint_command topic。"""
    if action_dim not in SUPPORTED_ACTION_DIMS:
        raise ValueError(
            f"action_dim must be one of {SUPPORTED_ACTION_DIMS}, got {action_dim}"
        )
    action = np.asarray(action, dtype=np.float64)
    if action.shape != (action_dim,):
        raise ValueError(
            f"action must have shape ({action_dim},), got {action.shape}"
        )

    controller.send_left_arm_command(JointNames.LEFT_ARM, action[0:7].tolist())
    controller.send_left_gripper_command(action[7])
    controller.send_right_arm_command(JointNames.RIGHT_ARM, action[8:15].tolist())
    controller.send_right_gripper_command(action[15])
    if action_dim == FULL_BODY_ACTION_DIM:
        controller.send_head_command(action[16:18].tolist())
        controller.send_waist_command(action[18:22].tolist())

# ==================== RobotController ====================
class RobotController(Node):
    def __init__(self, robot_type, dump_fisheye_videos=False,
                 dump_fisheye_dir='debug/fisheye_videos', dump_fisheye_fps=30.0,
                 fisheye_mode='5fisheye'):
        super().__init__('robot_controller')
        self.robot_type = robot_type
        self.fisheye_mode = fisheye_mode
        self.bridge = CvBridge()
        self.state_lock = threading.Lock()
        self.dump_fisheye_videos = dump_fisheye_videos and robot_type == 'h01'
        self.dump_fisheye_dir = Path(dump_fisheye_dir)
        self.dump_fisheye_fps = float(dump_fisheye_fps)
        self.fisheye_video_writers = {}
        self.fisheye_video_paths = {}

        # 机器人状态
        self.left_arm_state = create_joint_group(JointNames.LEFT_ARM)
        self.right_arm_state = create_joint_group(JointNames.RIGHT_ARM)
        self.left_gripper_state = create_joint_group(JointNames.LEFT_GRIPPER)
        self.right_gripper_state = create_joint_group(JointNames.RIGHT_GRIPPER)
        self.waist_state = create_joint_group(JointNames.WAIST)  # 仅 h01 有腰反馈
        self.head_state = create_joint_group(JointNames.HEAD)    # 仅 h01 有头反馈

        # 图像状态
        self.cam_high_image = None
        self.cam_fisheye_front_image = None
        self.cam_left_wrist_up_image = None
        self.cam_left_wrist_down_image = None
        self.cam_right_wrist_up_image = None
        self.cam_right_wrist_down_image = None

        # 话题配置
        topics = {
            't170d': [
                '/camera_head/color/image',
                '/camera_head/color/image',
                '/camera_left_hand/color/image',
                '/camera_left_hand/color/image',
                '/camera_right_hand/color/image',
                '/camera_right_hand/color/image'
            ],
            'h01': [
                '/camera_fisheye_front/image',
                '/camera_fisheye_front/image',
                '/camera_fisheye_lefthand_up/image',
                '/camera_fisheye_lefthand_down/image',
                '/camera_fisheye_righthand_up/image',
                '/camera_fisheye_righthand_down/image'
            ]
        }
        camera_topics = topics[robot_type]

        if robot_type == 'h01':
            left_arm_state_topic = TopicNames.LEFT_ARM_STATE
            right_arm_state_topic = TopicNames.RIGHT_ARM_STATE
            left_gripper_state_topic = TopicNames.LEFT_GRIPPER_STATE
            right_gripper_state_topic = TopicNames.RIGHT_GRIPPER_STATE
            left_arm_command_topic = TopicNames.LEFT_ARM_COMMAND
            right_arm_command_topic = TopicNames.RIGHT_ARM_COMMAND
            left_gripper_command_topic = TopicNames.LEFT_GRIPPER_COMMAND
            right_gripper_command_topic = TopicNames.RIGHT_GRIPPER_COMMAND
        else:
            left_arm_state_topic = '/left_arm/joint_states'
            right_arm_state_topic = '/right_arm/joint_states'
            left_gripper_state_topic = '/left_gripper/gripper_state'
            right_gripper_state_topic = '/right_gripper/gripper_state'
            left_arm_command_topic = '/left_arm/joint_command'
            right_arm_command_topic = '/right_arm/joint_command'
            left_gripper_command_topic = '/left_gripper/gripper_command'
            right_gripper_command_topic = '/right_gripper/gripper_command'

        self.sensor_topics = {
            'cam_high_image': camera_topics[0],
            'cam_fisheye_front_image': camera_topics[1],
            'cam_left_wrist_up_image': camera_topics[2],
            'cam_left_wrist_down_image': camera_topics[3],
            'cam_right_wrist_up_image': camera_topics[4],
            'cam_right_wrist_down_image': camera_topics[5],
            'left_arm_state': left_arm_state_topic,
            'right_arm_state': right_arm_state_topic,
            'left_gripper_state': left_gripper_state_topic,
            'right_gripper_state': right_gripper_state_topic,
        }

        # 状态订阅器
        self.create_subscription(JointState, left_arm_state_topic, self.left_arm_callback, 10)
        self.create_subscription(JointState, right_arm_state_topic, self.right_arm_callback, 10)
        self.create_subscription(JointState, left_gripper_state_topic, self.left_gripper_callback, 10)
        self.create_subscription(JointState, right_gripper_state_topic, self.right_gripper_callback, 10)

        # 图像订阅器
        if robot_type == 'h01':
            if FixedYUYVImage is None:
                raise ImportError("fixed_image_msgs is required for h01 fisheye camera topics")
            # head fisheye: /camera_fisheye_front/image (FixedYUYVImage)
            self.create_subscription(
                FixedYUYVImage, camera_topics[1], self.cam_fisheye_front_callback, qos_profile_sensor_data
            )
            # wrist fisheye yuyv
            self.create_subscription(FixedYUYVImage, camera_topics[2], self.cam_left_wrist_up_callback, qos_profile_sensor_data)
            if self.fisheye_mode == '5fisheye':
                self.create_subscription(FixedYUYVImage, camera_topics[3], self.cam_left_wrist_down_callback, qos_profile_sensor_data)
            self.create_subscription(FixedYUYVImage, camera_topics[4], self.cam_right_wrist_up_callback, qos_profile_sensor_data)
            if self.fisheye_mode == '5fisheye':
                self.create_subscription(FixedYUYVImage, camera_topics[5], self.cam_right_wrist_down_callback, qos_profile_sensor_data)
        else:
            self.create_subscription(Image, camera_topics[0], self.cam_high_callback, qos_profile_sensor_data)
            self.create_subscription(Image, camera_topics[1], self.cam_fisheye_front_callback, qos_profile_sensor_data)
            self.create_subscription(Image, camera_topics[2], self.cam_left_wrist_up_callback, qos_profile_sensor_data)
            if self.fisheye_mode == '5fisheye':
                self.create_subscription(Image, camera_topics[3], self.cam_left_wrist_down_callback, qos_profile_sensor_data)
            self.create_subscription(Image, camera_topics[4], self.cam_right_wrist_up_callback, qos_profile_sensor_data)
            if self.fisheye_mode == '5fisheye':
                self.create_subscription(Image, camera_topics[5], self.cam_right_wrist_down_callback, qos_profile_sensor_data)

        # 控制发布器
        self.left_arm_pub = self.create_publisher(JointState, left_arm_command_topic, 10)
        self.right_arm_pub = self.create_publisher(JointState, right_arm_command_topic, 10)
        self.left_gripper_pub = self.create_publisher(JointState, left_gripper_command_topic, 10)
        self.right_gripper_pub = self.create_publisher(JointState, right_gripper_command_topic, 10)

        # 腰（仅 h01）：/waist/joint_command 下发 + /waist/joint_state 反馈（JointState, name=WAIST_1..4）
        self.waist_pub = None
        self.head_pub = None
        if robot_type == 'h01':
            self.create_subscription(JointState, TopicNames.WAIST_STATE, self.waist_callback, 10)
            self.waist_pub = self.create_publisher(JointState, TopicNames.WAIST_COMMAND, 10)
            # 头反馈：state dim16-17 要填真实 HEAD_YAW/HEAD_PITCH（训练数据非零）
            self.create_subscription(JointState, TopicNames.HEAD_STATE, self.head_callback, 10)
            self.head_pub = self.create_publisher(JointState, TopicNames.HEAD_COMMAND, 10)

        # 服务客户端（对齐 h01_ros2_interface_examples.py）
        self.left_arm_reset_client = self.create_client(Trigger, ServiceNames.LEFT_ARM_RESET)
        self.right_arm_reset_client = self.create_client(Trigger, ServiceNames.RIGHT_ARM_RESET)
        self.left_gripper_calibrate_client = self.create_client(Trigger, ServiceNames.LEFT_GRIPPER_CALIBRATE)
        self.left_gripper_enable_client = self.create_client(Trigger, ServiceNames.LEFT_GRIPPER_ENABLE)
        self.left_gripper_disable_client = self.create_client(Trigger, ServiceNames.LEFT_GRIPPER_DISABLE)
        self.right_gripper_calibrate_client = self.create_client(Trigger, ServiceNames.RIGHT_GRIPPER_CALIBRATE)
        self.right_gripper_enable_client = self.create_client(Trigger, ServiceNames.RIGHT_GRIPPER_ENABLE)
        self.right_gripper_disable_client = self.create_client(Trigger, ServiceNames.RIGHT_GRIPPER_DISABLE)

        if self.dump_fisheye_videos:
            self.dump_fisheye_dir.mkdir(parents=True, exist_ok=True)
            self.get_logger().info(
                f'fisheye video dump enabled, dir={self.dump_fisheye_dir}, fps={self.dump_fisheye_fps}'
            )

        self.get_logger().info(
            f'RobotController ({robot_type}) initialized with async smooth mode (head fisheye + fisheye wrist)'
        )

    # 回调函数
    def left_arm_callback(self, msg):
        with self.state_lock:
            self.left_arm_state.update_from_msg(msg)

    def right_arm_callback(self, msg):
        with self.state_lock:
            self.right_arm_state.update_from_msg(msg)

    def left_gripper_callback(self, msg):
        with self.state_lock:
            self.left_gripper_state.update_from_msg(msg)

    def right_gripper_callback(self, msg):
        with self.state_lock:
            self.right_gripper_state.update_from_msg(msg)

    def waist_callback(self, msg):
        with self.state_lock:
            self.waist_state.update_from_msg(msg)

    def head_callback(self, msg):
        with self.state_lock:
            self.head_state.update_from_msg(msg)

    def cam_high_callback(self, msg):
        with self.state_lock: self.cam_high_image = msg
    def cam_fisheye_front_callback(self, msg):
        with self.state_lock: self.cam_fisheye_front_image = msg
    def cam_left_wrist_up_callback(self, msg):
        with self.state_lock: self.cam_left_wrist_up_image = msg
    def cam_left_wrist_down_callback(self, msg):
        with self.state_lock: self.cam_left_wrist_down_image = msg
    def cam_right_wrist_up_callback(self, msg):
        with self.state_lock: self.cam_right_wrist_up_image = msg
    def cam_right_wrist_down_callback(self, msg):
        with self.state_lock: self.cam_right_wrist_down_image = msg

    # 控制函数（对齐 h01_ros2_interface_examples.py）
    def send_left_arm_command(self, joint_names, positions):
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = list(joint_names)
        msg.position = [float(position) for position in positions]
        self.left_arm_pub.publish(msg)

    def send_right_arm_command(self, joint_names, positions):
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = list(joint_names)
        msg.position = [float(position) for position in positions]
        self.right_arm_pub.publish(msg)

    def send_left_gripper_command(self, position):
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = JointNames.LEFT_GRIPPER
        clipped_position = max(JointLimits.GRIPPER[0], min(JointLimits.GRIPPER[1], float(position)))
        msg.position = [float(clipped_position)]
        self.left_gripper_pub.publish(msg)

    def send_right_gripper_command(self, position):
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = JointNames.RIGHT_GRIPPER
        clipped_position = max(JointLimits.GRIPPER[0], min(JointLimits.GRIPPER[1], float(position)))
        msg.position = [float(clipped_position)]
        self.right_gripper_pub.publish(msg)

    def send_waist_command(self, positions):
        if self.waist_pub is None:
            self.get_logger().warn("[waist] 无 /waist/joint_command publisher（robot_type 非 h01），跳过腰下发")
            return
        positions = list(positions)
        if len(positions) != len(JointNames.WAIST):
            raise ValueError(
                f"waist command must contain {len(JointNames.WAIST)} positions, "
                f"got {len(positions)}"
            )
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = list(JointNames.WAIST)
        msg.position = [float(p) for p in positions]
        self.waist_pub.publish(msg)

    def send_head_command(self, positions):
        if self.head_pub is None:
            self.get_logger().warn("[head] 无 /head/joint_command publisher（robot_type 非 h01），跳过头部下发")
            return
        positions = list(positions)
        if len(positions) != len(JointNames.HEAD):
            raise ValueError(
                f"head command must contain {len(JointNames.HEAD)} positions, "
                f"got {len(positions)}"
            )
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = list(JointNames.HEAD)
        msg.position = [float(p) for p in positions]
        self.head_pub.publish(msg)

    def control_left_arm(self, pos):
        self.send_left_arm_command(JointNames.LEFT_ARM, pos)

    def control_right_arm(self, pos):
        self.send_right_arm_command(JointNames.RIGHT_ARM, pos)

    def control_left_gripper(self, pos):
        self.send_left_gripper_command(pos)

    def control_right_gripper(self, pos):
        self.send_right_gripper_command(pos)

    def _call_trigger_service(self, client, service_name: str, timeout_sec: float = 10.0) -> bool:
        if not client.wait_for_service(timeout_sec=timeout_sec):
            self.get_logger().error(f"{service_name} service unavailable")
            return False

        future = client.call_async(Trigger.Request())
        deadline = time.time() + timeout_sec

        while not future.done() and time.time() < deadline:
            time.sleep(0.05)

        if not future.done():
            self.get_logger().error(f"{service_name} service call timed out")
            return False

        try:
            result = future.result()
        except Exception as exc:
            self.get_logger().error(f"{service_name} service call failed: {exc}")
            return False

        if result.success:
            self.get_logger().info(f"{service_name} succeeded: {result.message}")
        else:
            self.get_logger().error(f"{service_name} failed: {result.message}")
        return result.success

    def _service_available(self, client, timeout_sec: float = 0.2) -> bool:
        return client.wait_for_service(timeout_sec=timeout_sec)

    def reset_left_arm(self, timeout_sec: float = 10.0) -> bool:
        return self._call_trigger_service(self.left_arm_reset_client, "left arm reset", timeout_sec)

    def reset_right_arm(self, timeout_sec: float = 10.0) -> bool:
        return self._call_trigger_service(self.right_arm_reset_client, "right arm reset", timeout_sec)

    def calibrate_left_gripper(self, timeout_sec: float = 5.0) -> bool:
        return self._call_trigger_service(self.left_gripper_calibrate_client, "left gripper calibrate", timeout_sec)

    def calibrate_right_gripper(self, timeout_sec: float = 5.0) -> bool:
        return self._call_trigger_service(self.right_gripper_calibrate_client, "right gripper calibrate", timeout_sec)

    def enable_left_gripper(self, timeout_sec: float = 5.0) -> bool:
        return self._call_trigger_service(self.left_gripper_enable_client, "left gripper enable", timeout_sec)

    def enable_right_gripper(self, timeout_sec: float = 5.0) -> bool:
        return self._call_trigger_service(self.right_gripper_enable_client, "right gripper enable", timeout_sec)

    def prepare_grippers(self) -> None:
        gripper_services = [
            (self.left_gripper_calibrate_client, self.calibrate_left_gripper, "left gripper calibrate"),
            (self.right_gripper_calibrate_client, self.calibrate_right_gripper, "right gripper calibrate"),
            (self.left_gripper_enable_client, self.enable_left_gripper, "left gripper enable"),
            (self.right_gripper_enable_client, self.enable_right_gripper, "right gripper enable"),
        ]

        for client, callback, service_name in gripper_services:
            if self._service_available(client):
                callback()
            else:
                self.get_logger().warn(f"{service_name} service unavailable, skipping")

    def initialize_pose(self, prepare_grippers: bool = True, wait_sec: float = 3.5) -> None:
        """初始化位置：调用 h01 自带关节复位服务，把双臂受控复位到固件定义的 Home 位。

        走 /left_arm/joint_reset、/right_arm/joint_reset（Trigger 服务），复位是机器人自身
        运动、双臂约 3s 完成；服务不可用时只打警告、不抛异常。可选顺带标定/使能夹爪。
        """
        self.get_logger().info("[init_pose] 复位双臂到 Home 位 ...")
        ok_left = self.reset_left_arm()
        ok_right = self.reset_right_arm()
        if prepare_grippers:
            self.prepare_grippers()
        # 复位为异步运动，等待其到位（双臂约 3s）
        time.sleep(wait_sec)
        self.get_logger().info(f"[init_pose] 复位完成 (left={ok_left}, right={ok_right})")

    def move_to_ready_pose(self, pose, duration: float = 2.0, rate: float = 30.0, wait_state_sec: float = 5.0) -> None:
        """平滑插值把双臂（+可选腰）移动到给定 ready pose。

        pose 为 16 维 [左臂7,左夹爪,右臂7,右夹爪] 或 20 维（末 4 维为腰 WAIST_1..4）。
        从当前关节角在 duration 秒内线性插值到目标（~rate Hz 逐步下发），避免突跳；双臂与腰同步插值。
        若收不到手臂 joint_state 则整体跳过；pose 含腰但收不到 /waist/joint_state 时只动双臂+夹爪、腰跳过。
        """
        target_left, left_grip, target_right, right_grip, target_waist = split_ready_pose(pose)
        deadline = time.time() + wait_state_sec
        while time.time() < deadline:
            if (self.left_arm_state.is_ready(JointNames.LEFT_ARM)
                    and self.right_arm_state.is_ready(JointNames.RIGHT_ARM)):
                break
            time.sleep(0.1)
        if not (self.left_arm_state.is_ready(JointNames.LEFT_ARM)
                and self.right_arm_state.is_ready(JointNames.RIGHT_ARM)):
            self.get_logger().error(
                "[ready_pose] 未收到手臂 joint_state，跳过 ready pose 初始化（检查真机是否在发关节状态）"
            )
            return

        # 腰：仅当 pose 含腰(20 维) 且已收到 /waist/joint_state 时才插值控腰
        move_waist = target_waist is not None and self.waist_state.is_ready(JointNames.WAIST)
        if target_waist is not None and not move_waist:
            self.get_logger().warn("[ready_pose] 未收到 /waist/joint_state，跳过腰（只动双臂+夹爪）")

        with self.state_lock:
            start_left = self.left_arm_state.positions(JointNames.LEFT_ARM)
            start_right = self.right_arm_state.positions(JointNames.RIGHT_ARM)
            start_waist = self.waist_state.positions(JointNames.WAIST) if move_waist else None

        self.get_logger().info(
            f"[ready_pose] 从当前位姿平滑移动到 ready pose（{duration}s，{'含腰' if move_waist else '不含腰'}）..."
        )
        steps = max(1, int(duration * rate))
        dt = 1.0 / rate
        for i in range(1, steps + 1):
            alpha = i / steps
            left = [s + (t - s) * alpha for s, t in zip(start_left, target_left)]
            right = [s + (t - s) * alpha for s, t in zip(start_right, target_right)]
            self.send_left_arm_command(JointNames.LEFT_ARM, left)
            self.send_right_arm_command(JointNames.RIGHT_ARM, right)
            if move_waist:
                waist = [s + (t - s) * alpha for s, t in zip(start_waist, target_waist)]
                self.send_waist_command(waist)
            time.sleep(dt)
        self.send_left_gripper_command(left_grip)
        self.send_right_gripper_command(right_grip)
        self.get_logger().info("[ready_pose] 到位")

    def _decode_fisheye_yuyv(self, msg):
        data = getattr(msg, "data", None)
        if data is None:
            raise ValueError("FixedYUYVImage missing data field")
        width = getattr(msg, "width", 1920)
        height = getattr(msg, "height", 1536)
        arr = np.frombuffer(data, dtype=np.uint8)
        expected = int(width) * int(height) * 2
        if arr.size < expected:
            raise ValueError(f"YUYV buffer too small: {arr.size} < {expected}")
        if arr.size > expected:
            arr = arr[:expected]
        yuyv = arr.reshape((int(height), int(width), 2))
        bgr = cv2.cvtColor(yuyv, cv2.COLOR_YUV2BGR_YUY2)
        return bgr

    def _ensure_fisheye_video_writer(self, stream_name, frame):
        if stream_name in self.fisheye_video_writers:
            return

        height, width = frame.shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        video_path = self.dump_fisheye_dir / f'{timestamp}_{stream_name}.avi'
        writer = cv2.VideoWriter(str(video_path), fourcc, self.dump_fisheye_fps, (width, height))
        if not writer.isOpened():
            raise RuntimeError(f'failed to open fisheye video writer: {video_path}')

        self.fisheye_video_writers[stream_name] = writer
        self.fisheye_video_paths[stream_name] = video_path
        self.get_logger().info(f'fisheye dump writer ready: {stream_name} -> {video_path}')

    def _dump_fisheye_frames(self, stream_frames):
        if not self.dump_fisheye_videos:
            return

        for stream_name, frame in stream_frames.items():
            self._ensure_fisheye_video_writer(stream_name, frame)
            self.fisheye_video_writers[stream_name].write(frame)

    def close_debug_writers(self):
        for writer in self.fisheye_video_writers.values():
            writer.release()
        if self.fisheye_video_paths:
            for stream_name, video_path in self.fisheye_video_paths.items():
                self.get_logger().info(f'fisheye dump saved: {stream_name} -> {video_path}')
        self.fisheye_video_writers.clear()
        self.fisheye_video_paths.clear()

    def get_frame(self):
        """获取当前帧数据"""
        with self.state_lock:
            # 检查所有数据是否就绪
            data_check = {
                'cam_fisheye_front_image': self.cam_fisheye_front_image,
                'cam_left_wrist_up_image': self.cam_left_wrist_up_image,
                'cam_right_wrist_up_image': self.cam_right_wrist_up_image,
            }
            if self.fisheye_mode == '5fisheye':
                data_check.update({
                    'cam_left_wrist_down_image': self.cam_left_wrist_down_image,
                    'cam_right_wrist_down_image': self.cam_right_wrist_down_image,
                })
            none_items = [k for k, v in data_check.items() if v is None]
            if not self.left_arm_state.is_ready(JointNames.LEFT_ARM):
                none_items.append(f"left_arm_state ({self.sensor_topics['left_arm_state']})")
            if not self.right_arm_state.is_ready(JointNames.RIGHT_ARM):
                none_items.append(f"right_arm_state ({self.sensor_topics['right_arm_state']})")
            if not self.left_gripper_state.is_ready(JointNames.LEFT_GRIPPER):
                none_items.append(f"left_gripper_state ({self.sensor_topics['left_gripper_state']})")
            if not self.right_gripper_state.is_ready(JointNames.RIGHT_GRIPPER):
                none_items.append(f"right_gripper_state ({self.sensor_topics['right_gripper_state']})")
            if none_items:
                missing = []
                for name in none_items:
                    if "(" in name:
                        missing.append(name)
                        continue
                    topic = self.sensor_topics.get(name, "unknown_topic")
                    missing.append(f"{name} ({topic})")
                print(f"[get_frame] 数据未就绪: {', '.join(missing)}")
                return None

            # 解码图像
            if self.robot_type == 'h01':
                # h01: cam_high 使用头部前置鱼眼；手腕仍用鱼眼 YUYV。
                # 3fisheye 顺序: [front, left_up, right_up]；
                # 5fisheye 额外增加 left_down/right_down。
                fisheye_stream_frames = {
                    'front': self._decode_fisheye_yuyv(self.cam_fisheye_front_image),
                    'left_up': self._decode_fisheye_yuyv(self.cam_left_wrist_up_image),
                    'right_up': self._decode_fisheye_yuyv(self.cam_right_wrist_up_image),
                }
                if self.fisheye_mode == '5fisheye':
                    fisheye_stream_frames.update({
                        'left_down': self._decode_fisheye_yuyv(self.cam_left_wrist_down_image),
                        'right_down': self._decode_fisheye_yuyv(self.cam_right_wrist_down_image),
                    })
                imgs = [
                    fisheye_stream_frames['front'],
                    fisheye_stream_frames['left_up'],
                    fisheye_stream_frames['right_up'],
                ]
                if self.fisheye_mode == '5fisheye':
                    imgs.extend([
                        fisheye_stream_frames['left_down'],
                        fisheye_stream_frames['right_down'],
                    ])
            else:
                fisheye_stream_frames = None
                image_msgs = [
                    self.cam_high_image,
                    self.cam_fisheye_front_image,
                    self.cam_left_wrist_up_image,
                    self.cam_right_wrist_up_image,
                ]
                if self.fisheye_mode == '5fisheye':
                    image_msgs.extend([
                        self.cam_left_wrist_down_image,
                        self.cam_right_wrist_down_image,
                    ])
                imgs = [self.bridge.imgmsg_to_cv2(x, "passthrough") for x in image_msgs]

            # 构建状态向量
            qpos_l = self.left_arm_state.positions(JointNames.LEFT_ARM) + self.left_gripper_state.positions(JointNames.LEFT_GRIPPER)
            qpos_r = self.right_arm_state.positions(JointNames.RIGHT_ARM) + self.right_gripper_state.positions(JointNames.RIGHT_GRIPPER)

        if fisheye_stream_frames is not None:
            self._dump_fisheye_frames(fisheye_stream_frames)

        return imgs + [qpos_l, qpos_r]

# ==================== 异步推理 ====================
def async_inference_thread(args, controller, client, trigger_step):
    """
    异步推理线程函数

    获取当前帧 -> 预处理 -> 按模式构建 state（qpose/epose） -> 调用推理服务器
    -> 如为 endpose 动作则 IK 转 qpose -> 存储结果
    """
    global pending_inference_result, pending_inference_lock, inference_trigger_step, _dump_obs_saved

    inference_start = time.time()
    raw_action_shape = None
    normalized_action_shape = None

    try:
        # 获取帧数据
        result = controller.get_frame()
        if not result:
            print(f"[ASYNC_INFER] Step {trigger_step} - 帧同步失败")
            with pending_inference_lock:
                pending_inference_result = None
                inference_trigger_step = None
            return

        # 预处理图像（get_frame returns imgs + [qpos_l, qpos_r]）
        img_msgs = result[:-2]
        imgs = [cv2.cvtColor(
            image_tools.convert_to_uint8(image_tools.resize_with_pad(img, 224, 224)),
            cv2.COLOR_BGR2RGB
        ) for img in img_msgs]

        # get_frame returns imgs + [qpos_l, qpos_r]
        qpos = np.array(result[-2] + result[-1], dtype=np.float32)

        # 按模式构建观测 state：qpose(16D) 或 epose(14D)
        policy_state = build_policy_state(qpos, args.policy_action_space)

        # 模型输入固定为 22 维，并用真实反馈填充头/腰状态。
        policy_state = build_server_state(policy_state, controller)

        # 构建观测
        images = {
            # h01: cam_high 来自 /camera_fisheye_front/image 的头部鱼眼图像
            "cam_high": imgs[0].transpose(2, 0, 1),
            "cam_left_wrist_up": imgs[1].transpose(2, 0, 1),
            "cam_right_wrist_up": imgs[2].transpose(2, 0, 1),
        }
        if args.fisheye_mode == '5fisheye':
            images.update({
                "cam_left_wrist_down": imgs[3].transpose(2, 0, 1),
                "cam_right_wrist_down": imgs[4].transpose(2, 0, 1),
            })

        obs = {
            "state": policy_state,
            "images": images,
            "prompt": args.prompt,
            "base_vel": [0.0, 0.0]
        }

        # 可选：dump 发给 server 的 obs（前 N 帧），用于人工核对图像内容/朝向/颜色 + state/prompt。
        # images 里是 CHW RGB uint8，存 PNG 前转回 HWC 并 RGB->BGR（cv2.imwrite 需 BGR）。
        if args.dump_obs and _dump_obs_saved < args.dump_obs_count:
            dump_dir = Path(args.dump_obs_dir)
            dump_dir.mkdir(parents=True, exist_ok=True)
            for key, chw in images.items():
                hwc_bgr = cv2.cvtColor(np.asarray(chw).transpose(1, 2, 0), cv2.COLOR_RGB2BGR)
                cv2.imwrite(str(dump_dir / f'step{trigger_step}_{key}.png'), hwc_bgr)
            with open(dump_dir / f'step{trigger_step}_obs.txt', 'w') as f:
                f.write(f'prompt: {args.prompt}\n')
                f.write(f'state(len={len(policy_state)}): '
                        f'{np.round(np.asarray(policy_state, dtype=np.float32), 5).tolist()}\n')
                f.write(f'image_keys: {list(images.keys())}\n')
                f.write(f'image_chw_shapes: {[tuple(np.asarray(v).shape) for v in images.values()]}\n')
            _dump_obs_saved += 1
            print(f'[DUMP] step {trigger_step} obs 已存到 {dump_dir} '
                  f'({_dump_obs_saved}/{args.dump_obs_count})，键={list(images.keys())}')

        # 调用推理服务器
        infer_start = time.time()
        infer_result = client.infer(obs)
        raw_actions = infer_result['actions']
        infer_duration = time.time() - infer_start

        if hasattr(raw_actions, "shape"):
            raw_action_shape = tuple(raw_actions.shape)

        # 打印服务器返回的原始动作块（供调试观察模型输出）
        _ra = np.asarray(raw_actions, dtype=np.float32)
        _ra_first = (_ra.reshape(-1, _ra.shape[-1])[0] if _ra.ndim >= 2 else _ra)
        print(f"[INFER] step {trigger_step} server返回 shape={tuple(_ra.shape)} "
              f"首帧={np.round(_ra_first, 4).tolist()} (耗时{infer_duration*1000:.0f}ms)")

        # qpose 按 client 配置截到 16/22 维；endpose 保留 14 维，随后由 IK 转成 16D qpose。
        requested_action_dim = (
            args.action_dim if args.policy_action_space == "qpose" else 14
        )
        actions = slice_actions_to_robot_dim(raw_actions, requested_action_dim)
        normalized_action_shape = tuple(actions.shape)

        # qpose 保留配置的 16/22D；endpose 模式在这里通过 IK 转成 16D。
        actions, ik_duration = convert_policy_actions_to_qpose(actions, qpos, args)

        # 平滑动作（仅 qpose 关节维度；夹爪不平滑）
        if len(actions) > 1:
            actions = actions[:30]
            actions = smoothen_actions(actions)

        total_duration = time.time() - inference_start

        # 存储推理结果
        with pending_inference_lock:
            pending_inference_result = {
                'actions': actions.copy(),
                'trigger_step': trigger_step,
                'action_space': args.policy_action_space,
                'inference_duration': infer_duration,
                'ik_duration': ik_duration,
                'total_duration': total_duration
            }
    except Exception as exc:
        total_duration = time.time() - inference_start
        print(
            f"[ASYNC_INFER][ERROR] Step {trigger_step} mode={args.policy_action_space} "
            f"raw_shape={raw_action_shape} normalized_shape={normalized_action_shape} "
            f"{type(exc).__name__}: {exc} (total={total_duration:.3f}s)"
        )
        with pending_inference_lock:
            pending_inference_result = None
            inference_trigger_step = None
        return

def apply_sliding_window_result(current_buffer, current_index, new_actions,
                                 inference_trigger_remaining, action_execute_horizon):
    """
    应用滑动窗口拼接（固定 horizon 版本）

    Args:
        current_buffer: 当前动作缓冲区
        current_index: 当前执行位置
        new_actions: 新推理的动作
        inference_trigger_remaining: 触发推理时的剩余动作数
        action_execute_horizon: 新动作的执行范围

    Returns:
        拼接并平滑后的新动作序列
    """
    # 计算保留的旧动作数量
    remaining_old_count = max(0, inference_trigger_remaining - int(new_actions.shape[0] * 0.1))

    # 获取保留的旧动作
    if current_buffer is not None and remaining_old_count > 0:
        remaining_old_actions = current_buffer[current_index:current_index + remaining_old_count]
    else:
        remaining_old_actions = np.empty(
            (0, new_actions.shape[1]), dtype=new_actions.dtype
        )

    # 使用 find_nearest_action 确定新动作起始位置。
    # 对齐基准应当是“保留旧动作的最后一帧”或“当前即将执行的动作”，
    # 而不是整个旧缓冲区的最后一帧；否则新序列会朝未来末端姿态对齐，容易产生闪回。
    if len(remaining_old_actions) > 0:
        reference_action = remaining_old_actions[-1]
    elif current_buffer is not None and len(current_buffer) > 0:
        reference_index = min(max(current_index, 0), len(current_buffer) - 1)
        reference_action = current_buffer[reference_index]
    else:
        return None

    search_range = min(len(new_actions), 20)  # 固定搜索范围为前20个动作
    nearest_idx = find_nearest_action(new_actions[:search_range], reference_action)

    # 从最近位置开始取新动作
    new_inference_actions = new_actions[nearest_idx:action_execute_horizon]

    print(f"[SLIDING] nearest_idx: {nearest_idx}, 新动作范围: [{nearest_idx}, {nearest_idx + action_execute_horizon}]")

    # 拼接
    if len(remaining_old_actions) > 0 and len(new_inference_actions) > 0:
        concatenated = np.concatenate([remaining_old_actions, new_inference_actions], axis=0)
    elif len(new_inference_actions) > 0:
        concatenated = new_inference_actions
    else:
        return None

    # 对拼接边界做局部平滑，降低切换瞬间的回跳感。
    if len(remaining_old_actions) > 0 and len(new_inference_actions) > 0:
        concatenated = smooth_action_transition(
            concatenated,
            boundary_index=len(remaining_old_actions),
            blend_radius=3,
        )

    return concatenated

# ==================== 主推理循环 ====================
def model_inference_async(args, controller, client):
    """异步推理主循环"""
    global inference_thread, action_buffer, action_buffer_lock, action_index
    global pending_inference_result, pending_inference_lock, inference_trigger_step

    set_seed(1000)

    # 参数
    inference_trigger_remaining = args.inference_trigger_remaining
    action_execute_horizon = args.action_execute_horizon

    print(f"[MAIN] 异步推理模式启动")
    print(f"  - 触发推理剩余动作数: {inference_trigger_remaining}")
    print(f"  - 动作执行范围: {action_execute_horizon}")
    print(f"  - 发布频率: {args.publish_rate}Hz")
    print(f"  - 动作空间模式: {args.policy_action_space}")
    print(f"  - ROS 发布动作维度: {args.action_dim}")

    # 等待用户确认
    input("Press Enter to start inference...")

    # 初始化状态
    with action_buffer_lock:
        action_buffer = None
        action_index = 0

    with pending_inference_lock:
        pending_inference_result = None
        inference_trigger_step = None

    t = 0
    target_interval = 1.0 / args.publish_rate

    print(f"[MAIN] ========== 开始推理循环 ==========")

    while True:
        step_start = time.time()

        # 1. 初始推理：在拿到第一段动作前反复重试（启动瞬间关节/相机首帧还没到、
        #    或 get_frame 取帧失败时，不至于只试一次就卡死不动）。
        with action_buffer_lock:
            _need_initial = action_buffer is None
        with pending_inference_lock:
            _has_pending = pending_inference_result is not None
        if _need_initial and not _has_pending:
            if inference_thread is None or not inference_thread.is_alive():
                inference_thread = threading.Thread(
                    target=async_inference_thread,
                    args=(args, controller, client, t)
                )
                inference_thread.daemon = True
                inference_thread.start()
                inference_trigger_step = t

        # 2. 检查是否有新推理结果需要应用
        with pending_inference_lock:
            if pending_inference_result is not None:
                with action_buffer_lock:
                    remaining = len(action_buffer) - action_index if action_buffer is not None else 0

                    # 当剩余动作足够少时应用新结果
                    if remaining <= inference_trigger_remaining // 2 or action_buffer is None:
                        infer_ms = pending_inference_result.get('inference_duration', 0.0) * 1000.0
                        ik_ms = pending_inference_result.get('ik_duration', 0.0) * 1000.0
                        total_ms = pending_inference_result.get('total_duration', 0.0) * 1000.0
                        mode_name = pending_inference_result.get('action_space', 'qpose')
                        print(
                            f"[MAIN] Step {t} - 应用推理结果 mode={mode_name}, "
                            f"infer={infer_ms:.1f}ms, ik={ik_ms:.1f}ms, total={total_ms:.1f}ms"
                        )
                        if action_buffer is not None:
                            # 滑动窗口拼接
                            new_buffer = apply_sliding_window_result(
                                action_buffer, action_index,
                                pending_inference_result['actions'],
                                inference_trigger_remaining, action_execute_horizon
                            )
                            if new_buffer is not None:
                                action_buffer = new_buffer
                                action_index = 0
                                print(f"[MAIN] Step {t} - 动作序列更新，新长度: {len(action_buffer)}")
                        else:
                            # 初始推理结果
                            action_buffer = pending_inference_result['actions'][:action_execute_horizon].copy()
                            action_index = 0
                            print(f"[MAIN] Step {t} - 初始动作序列，长度: {len(action_buffer)}")

                        pending_inference_result = None

        # 3. 检查是否需要触发新推理
        with action_buffer_lock:
            if action_buffer is not None:
                remaining = len(action_buffer) - action_index

                # 当剩余动作等于触发阈值时启动新推理
                if remaining == inference_trigger_remaining:
                    if inference_thread is None or not inference_thread.is_alive():
                        inference_thread = threading.Thread(
                            target=async_inference_thread,
                            args=(args, controller, client, t)
                        )
                        inference_thread.daemon = True
                        inference_thread.start()
                        inference_trigger_step = t

        # 4. 执行动作
        with action_buffer_lock:
            if action_buffer is not None and action_index < len(action_buffer):
                action = action_buffer[action_index].copy()

                if action[7] < 0.015:
                    action[7] = 0
                if action[15] < 0.015:
                    action[15] = 0

                # 夹爪偏移：--gripper_offset 加到模型左右夹爪输出上(负值=在当前基础上夹更紧)
                if args.gripper_offset != 0.0:
                    action[7] += args.gripper_offset
                    action[15] += args.gripper_offset

                # 每帧 action 明细打印(调试用)已关闭；需要时取消下面 5 行注释
                # print(f"action: {action}")
                # print(f"action[:7]: {action[:7]}")
                # print(f"action[8:15]: {action[8:15]}")
                # print(f"action[7]: {action[7]}")
                # print(f"action[15]: {action[15]}")

                # 安全开关：默认只推理不下发。22D 模式还会发布头部和腰部命令。
                if args.execute_action:
                    publish_action_to_robot(controller, action, args.action_dim)
                elif t % 30 == 0:
                    print("[DRY-RUN] execute_action=False，未下发动作到机器人（加 --execute_action 开启执行）")

                action_index += 1

                if t % 30 == 0:
                    remaining = len(action_buffer) - action_index
                    print(f"[MAIN] Step {t} - 执行动作, 剩余: {remaining}")

        t += 1

        # 控制频率
        elapsed = time.time() - step_start
        sleep_time = max(0, target_interval - elapsed)
        if sleep_time > 0:
            time.sleep(sleep_time)

def main():
    rclpy.init()

    parser = argparse.ArgumentParser()
    # 基础参数
    parser.add_argument(
        '--max_publish_step',
        type=int,
        default=10000,
        help='已弃用；主循环现在持续运行，按 Ctrl+C 退出',
    )
    parser.add_argument('--publish_rate', type=int, default=20)
    parser.add_argument('--robot_type', default='h01', choices=['h01', 't170d'])
    parser.add_argument('--fisheye_mode', default='5fisheye', choices=['3fisheye', '5fisheye'],
                        help='h01 fisheye camera set: 3fisheye=front/left_up/right_up, 5fisheye adds left_down/right_down')
    parser.add_argument('--prompt', type=str, default='Pick up a plush toy from the top shelf.')
    parser.add_argument(
        '--dump_fisheye_videos',
        action='store_true',
        help='debug: dump 5 h01 fisheye streams to local video files',
    )
    parser.add_argument(
        '--dump_fisheye_dir',
        type=str,
        default='debug/fisheye_videos',
        help='output directory for dumped fisheye videos',
    )
    parser.add_argument(
        '--dump_fisheye_fps',
        type=float,
        default=30.0,
        help='video fps used for dumped fisheye videos',
    )
    # dump 实际发给 server 的 obs（预处理后的 224x224 图 + state + prompt），核对图像内容/朝向/颜色
    parser.add_argument('--dump_obs', action='store_true', default=False,
                        help='把发给 server 的 obs（前 N 帧图像+state+prompt）存盘，人工核对')
    parser.add_argument('--dump_obs_dir', type=str, default='debug/obs_dump',
                        help='--dump_obs 输出目录')
    parser.add_argument('--dump_obs_count', type=int, default=3,
                        help='--dump_obs 最多存多少帧（默认 3）')
    parser.add_argument(
        '--policy_action_space',
        default='qpose',
        choices=['qpose', 'endpose'],
        help=(
            "server 返回动作空间类型: "
            "qpose=(N,16/22)[L7,Lg,R7,Rg,(head2,waist4)], "
            "endpose=(N,14)[L xyzrpy,Lg,R xyzrpy,Rg]"
        ),
    )
    parser.add_argument(
        '--server_state_dim',
        type=int,
        choices=(SERVER_STATE_DIM,),
        default=SERVER_STATE_DIM,
        help='保留该参数以兼容启动脚本，但只接受固定值 22',
    )
    parser.add_argument(
        '--action_dim',
        type=int,
        choices=SUPPORTED_ACTION_DIMS,
        default=ROBOT_ACTION_DIM,
        help='从模型 action 截取并发布到 ROS 的固定维度：16=双臂+夹爪，22=再加头2+腰4',
    )
    parser.add_argument(
        '--h01_urdf_path',
        type=str,
        default='data/H01-EVT1-1103-URDF/H01-EVT1-1119.urdf',
        help='H01 URDF path used by endpose IK (Placo backend)',
    )

    # 异步推理参数
    parser.add_argument('--inference_trigger_remaining', type=int, default=10,
                        help='剩余多少动作时触发新推理')
    parser.add_argument('--action_execute_horizon', type=int, default=30,
                        help='每次推理结果的执行动作数')

    # 推理服务器（ubuntu 上的 gigabrain07 umi_ego h01 websocket server）
    parser.add_argument('--server_host', type=str, default='172.16.100.33')
    parser.add_argument('--server_port', type=int, default=8010)
    # 安全开关：默认只推理、不把动作下发到机械臂/夹爪，避免机械臂乱动；
    # 确认无误后再加 --execute_action 真正执行。
    parser.add_argument('--execute_action', action='store_true', default=False,
                        help='真正把推理动作下发到机器人（默认关闭，仅推理不执行）')
    # 推理前移动双臂到 ready pose（会移动机械臂）。默认开；--no_init_pose 跳过=不动臂（dry-run）。
    parser.add_argument('--init_pose', dest='init_pose', action='store_true', default=True,
                        help='推理前移动双臂到 ready pose（默认开）')
    parser.add_argument('--no_init_pose', dest='init_pose', action='store_false',
                        help='跳过 ready pose（dry-run 不动臂）')
    parser.add_argument('--ready_pose_npy', type=str, default='',
                        help='按任务的 ready pose npy(20维[臂7+爪+臂7+爪+腰4]或16维[臂+爪]);空=用内置 READY_POSE')
    parser.add_argument('--gripper_offset', type=float, default=0.0,
                        help='夹爪偏移(m):加到模型左右夹爪输出上(负值=在当前基础上夹更紧,如 -0.002);0=不偏移')

    args = parser.parse_args()

    # 参数验证
    if args.inference_trigger_remaining >= args.action_execute_horizon:
        raise ValueError(f"inference_trigger_remaining({args.inference_trigger_remaining}) "
                        f"必须小于 action_execute_horizon({args.action_execute_horizon})")
    if args.policy_action_space == 'endpose' and args.robot_type != 'h01':
        raise ValueError("policy_action_space=endpose 当前仅支持 robot_type=h01（需使用 h01 FK/IK）")
    if args.policy_action_space == 'endpose' and args.action_dim != ROBOT_ACTION_DIM:
        raise ValueError("policy_action_space=endpose 只支持 --action_dim 16")
    if args.action_dim == FULL_BODY_ACTION_DIM and args.robot_type != 'h01':
        raise ValueError("--action_dim 22 只支持 robot_type=h01（需要头部和腰部 ROS topic）")
    if args.policy_action_space == 'endpose':
        urdf_path = Path(args.h01_urdf_path)
        if not urdf_path.exists():
            raise FileNotFoundError(f"H01 IK URDF not found: {urdf_path}")
        print(f"[MAIN] endpose IK backend=placo, urdf={urdf_path}")

    # 初始化
    controller = RobotController(
        args.robot_type,
        dump_fisheye_videos=args.dump_fisheye_videos,
        dump_fisheye_dir=args.dump_fisheye_dir,
        dump_fisheye_fps=args.dump_fisheye_fps,
        fisheye_mode=args.fisheye_mode,
    )
    threading.Thread(target=rclpy.spin, args=(controller,), daemon=True).start()

    time.sleep(2.0)
    print(f"连接推理服务器 {args.server_host}:{args.server_port}...")
    from openpi_client import websocket_client_policy

    client = websocket_client_policy.WebsocketClientPolicy(
        host=args.server_host, port=args.server_port
    )

    # 移动双臂到 ready pose（默认开；--no_init_pose 跳过=不动臂）
    if args.init_pose:
        if args.ready_pose_npy:
            ready_pose = np.load(args.ready_pose_npy).astype(np.float64).reshape(-1).tolist()
            src = f'[task npy] {args.ready_pose_npy}'
        else:
            ready_pose = READY_POSE
            src = '[默认 READY_POSE]'
        print(f"[init_pose] 移动双臂到 ready pose（会移动机械臂，2s 平滑插值）{src}...")
        controller.move_to_ready_pose(ready_pose)

    try:
        model_inference_async(args, controller, client)
    except KeyboardInterrupt:
        print("\n中断")
    finally:
        controller.close_debug_writers()
        controller.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
