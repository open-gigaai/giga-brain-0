import threading
import time
from collections import deque
from io import BytesIO
from typing import Any, Dict

import numpy as np
import rospy
import torch
import zmq
from cv_bridge import CvBridge
from einops import rearrange
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from PIL import Image as PILImage
from sensor_msgs.msg import Image, JointState
from std_msgs.msg import Header


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
        """Initialize or reinitialize the socket with current settings."""
        self.socket = self.context.socket(zmq.REQ)
        self.socket.connect(f'tcp://{self.host}:{self.port}')

    def ping(self) -> bool:
        try:
            self.call_endpoint('ping', requires_input=False)
            return True
        except zmq.error.ZMQError:
            self._init_socket()  # Recreate socket for next attempt
            return False

    def kill_server(self):
        """Kill the server."""
        self.call_endpoint('kill', requires_input=False)

    def call_endpoint(self, endpoint: str, data: dict = None, requires_input: bool = True) -> dict:
        """Call an endpoint on the server.

        Args:
            endpoint: The name of the endpoint.
            data: The input data for the endpoint.
            requires_input: Whether the endpoint requires input data.
        """
        request: dict = {'endpoint': endpoint}
        if requires_input:
            request['data'] = data

        self.socket.send(TorchSerializer.to_bytes(request))
        message = self.socket.recv()
        if message == b'ERROR':
            raise RuntimeError('Server error')
        return TorchSerializer.from_bytes(message)

    def __del__(self):
        """Cleanup resources on destruction."""
        self.socket.close()
        self.context.term()


class RobotInferenceClient(BaseInferenceClient):
    """Client for communicating with the RobotInferenceServer."""

    def inference(self, observations: Dict[str, Any]):
        return self.call_endpoint('inference', observations)


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


def get_obs(ros_operator):
    print_flag = True
    while True and not rospy.is_shutdown():
        result = ros_operator.get_frame()
        if not result:
            if print_flag:
                print('frame syn fail')
                print_flag = False
            time.sleep(1 / 30.0)
            continue
        return result


def inference_process_giga_brain_0(client, ros_operator, task_name, use_robot_base):
    (
        img_front,
        img_left,
        img_right,
        img_front_depth,
        img_left_depth,
        img_right_depth,
        puppet_arm_left,
        puppet_arm_right,
        robot_base,
    ) = get_obs(ros_operator)

    img_front = resize_with_pad(img_front, 224, 224).astype(np.float32) / 255.0
    img_left = resize_with_pad(img_left, 224, 224).astype(np.float32) / 255.0
    img_right = resize_with_pad(img_right, 224, 224).astype(np.float32) / 255.0

    # This depth image operation is only applied to the agilex robot.
    if img_front_depth is not None:
        img_front_depth = img_front_depth.copy()[..., None].repeat(3, axis=-1)
        new_img_front_depth = np.zeros([480, 640, 3])
        new_img_front_depth[:400, :, :] = img_front_depth[:, :, :]
        new_img_front_depth = np.clip(new_img_front_depth / 10.0, 0, 255).astype(np.uint8)
        img_front_depth = resize_with_pad(new_img_front_depth, 224, 224).astype(np.float32) / 255.0
    if img_left_depth is not None:
        img_left_depth = img_left_depth.copy()[..., None].repeat(3, axis=-1)
        img_left_depth = np.clip(img_left_depth / 10.0, 0, 255).astype(np.uint8)
        img_left_depth = resize_with_pad(img_left_depth, 224, 224).astype(np.float32) / 255.0
    if img_right_depth is not None:
        img_right_depth = img_right_depth.copy()[..., None].repeat(3, axis=-1)
        img_right_depth = np.clip(img_right_depth / 10.0, 0, 255).astype(np.uint8)
        img_right_depth = resize_with_pad(img_right_depth, 224, 224).astype(np.float32) / 255.0

    obs = make_infer_data(
        img_front,
        img_left,
        img_right,
        img_front_depth,
        img_left_depth,
        img_right_depth,
        task_name,
        np.concatenate((np.array(puppet_arm_left.position), np.array(puppet_arm_right.position)), axis=0),
    )

    obs['qpos'] = np.concatenate((np.array(puppet_arm_left.position), np.array(puppet_arm_right.position)), axis=0)
    obs['qvel'] = np.concatenate((np.array(puppet_arm_left.velocity), np.array(puppet_arm_right.velocity)), axis=0)
    obs['effort'] = np.concatenate((np.array(puppet_arm_left.effort), np.array(puppet_arm_right.effort)), axis=0)
    if use_robot_base:
        obs['base_vel'] = [robot_base.twist.twist.linear.x, robot_base.twist.twist.angular.z]
        obs['qpos'] = np.concatenate((obs['qpos'], obs['base_vel']), axis=0)
    else:
        obs['base_vel'] = [0.0, 0.0]

    actions = client.inference(obs)

    return actions.float().cpu().numpy()


def model_inference_giga_brain_0(
    client,
    ros_operator,
    publish_rate,
    task_name,
    pos_lookahead_step=50,
    max_publish_step=10000,
    chunk_size=50,
    temporal_agg=False,
    state_dim=14,
    use_robot_base=False,
):
    left0 = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.3]
    right0 = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.3]
    left1 = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    right1 = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    ros_operator.puppet_arm_publish_continuous(left0, right0)
    input('Enter any key to continue :')
    ros_operator.puppet_arm_publish_continuous(left1, right1)
    action = None

    while True and not rospy.is_shutdown():
        t = 0
        max_t = 0
        rate = rospy.Rate(publish_rate)
        if temporal_agg:
            all_time_actions = np.zeros([max_publish_step, max_publish_step + chunk_size, state_dim])

        while t < max_publish_step and not rospy.is_shutdown():
            if t >= max_t:
                all_actions = inference_process_giga_brain_0(client, ros_operator, task_name, use_robot_base=use_robot_base)

                max_t = t + pos_lookahead_step
                if temporal_agg:
                    all_time_actions[[t], t : t + chunk_size] = all_actions

            if temporal_agg:
                actions_for_curr_step = all_time_actions[:, t]
                actions_populated = np.all(actions_for_curr_step != 0, axis=1)
                actions_for_curr_step = actions_for_curr_step[actions_populated]
                k = 0.01
                exp_weights = np.exp(-k * np.arange(len(actions_for_curr_step)))
                exp_weights = exp_weights / exp_weights.sum()
                exp_weights = exp_weights[:, np.newaxis]
                raw_action = (actions_for_curr_step * exp_weights).sum(axis=0, keepdims=True)
                action = raw_action[0]
                left_action, right_action = action[:7], action[7:14]
                min_qpos = [-2.618, 0.0, -2.967, -1.745, -1.22, -2.0944]
                max_qpos = [2.618, 3.14, 0.0, 1.745, 1.22, 2.0944]
                left_action[0:6] = np.clip(left_action[0:6], min_qpos, max_qpos)
                right_action[0:6] = np.clip(right_action[0:6], min_qpos, max_qpos)
                ros_operator.puppet_arm_publish(left_action, right_action)
                if use_robot_base:
                    vel_action = action[14:16]
                    ros_operator.robot_base_publish(vel_action)
                t += 1
                rate.sleep()
            else:
                for t_ in range(pos_lookahead_step):
                    action = all_actions[t_]
                    left_action, right_action = action[:7], action[7:14]
                    min_qpos = [-2.618, 0.0, -2.967, -1.745, -1.22, -2.0944]
                    max_qpos = [2.618, 3.14, 0.0, 1.745, 1.22, 2.0944]
                    left_action[0:6] = np.clip(left_action[0:6], min_qpos, max_qpos)
                    right_action[0:6] = np.clip(right_action[0:6], min_qpos, max_qpos)
                    ros_operator.puppet_arm_publish(left_action, right_action)
                    if use_robot_base:
                        vel_action = action[14:16]
                        ros_operator.robot_base_publish(vel_action)
                    t += 1
                    rate.sleep()


class RosOperator:
    def __init__(
        self,
        publish_rate: int = 30,
        arm_steps_length=None,
        use_depth_image: bool = False,
        use_robot_base: bool = False,
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
        joint_state_msg.name = ['joint0', 'joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'joint6']  # 设置关节名称
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
            joint_state_msg.name = ['joint0', 'joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'joint6']  # 设置关节名称
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
            joint_state_msg.name = ['joint0', 'joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'joint6']  # 设置关节名称
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

        while self.img_left_deque[0].header.stamp.to_sec() < frame_time:
            self.img_left_deque.popleft()
        img_left = self.bridge.imgmsg_to_cv2(self.img_left_deque.popleft(), 'passthrough')

        while self.img_right_deque[0].header.stamp.to_sec() < frame_time:
            self.img_right_deque.popleft()
        img_right = self.bridge.imgmsg_to_cv2(self.img_right_deque.popleft(), 'passthrough')

        while self.img_front_deque[0].header.stamp.to_sec() < frame_time:
            self.img_front_deque.popleft()
        img_front = self.bridge.imgmsg_to_cv2(self.img_front_deque.popleft(), 'passthrough')

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


def resize_with_pad(images: np.ndarray, height: int, width: int, method=PILImage.BILINEAR) -> np.ndarray:
    """Replicates tf.image.resize_with_pad for multiple images using PIL.
    Resizes a batch of images to a target height.

    Args:
        images: A batch of images in [..., height, width, channel] format.
        height: The target height of the image.
        width: The target width of the image.
        method: The interpolation method to use. Default is bilinear.

    Returns:
        The resized images in [..., height, width, channel].
    """

    def _resize_with_pad_pil(image: PILImage.Image, height: int, width: int, method: int) -> PILImage.Image:
        cur_width, cur_height = image.size
        if cur_width == width and cur_height == height:
            return image  # No need to resize if the image is already the correct size.

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

    # If the images are already the correct size, return them as is.
    if images.shape[-3:-1] == (height, width):
        return images

    original_shape = images.shape

    images = images.reshape(-1, *original_shape[-3:])
    resized = np.stack([_resize_with_pad_pil(PILImage.fromarray(im), height, width, method=method) for im in images])
    return resized.reshape(*original_shape[:-3], *resized.shape[-3:])


def main():
    publish_rate = 30
    pos_lookahead_step = 30
    max_publish_step = 10000
    chunk_size = 50
    temporal_agg = False
    state_dim = 14
    use_robot_base = False
    use_depth_image = False
    task_name = 'Task name here'

    ros_operator = RosOperator(publish_rate=publish_rate, use_robot_base=use_robot_base, use_depth_image=use_depth_image)
    client = RobotInferenceClient(host='127.0.0.1', port=8080)
    model_inference_giga_brain_0(
        client,
        ros_operator,
        task_name=task_name,
        publish_rate=publish_rate,
        pos_lookahead_step=pos_lookahead_step,
        max_publish_step=max_publish_step,
        chunk_size=chunk_size,
        temporal_agg=temporal_agg,
        state_dim=state_dim,
        use_robot_base=use_robot_base,
    )


if __name__ == '__main__':
    main()
