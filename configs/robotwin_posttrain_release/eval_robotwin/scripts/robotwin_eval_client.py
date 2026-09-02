#!/usr/bin/env python3
# -- coding: UTF-8
"""GigaBrain-0.7 RoboTwin 2.0 仿真评测客户端。

基于 giga-brain-0 的 scripts/inference/inference_agilex_client_unified_smooth.py
改造：保留 ZMQ 通信协议、图像预处理和 observation 打包，把 ROS 收发替换成
RoboTwin 的 env.get_obs() / env.take_action()。

必须放在 RoboTwin 仓库根目录下运行 (import envs / generate_episode_instructions):
    cd /path/to/RoboTwin
    python communication/robotwin_eval_client.py --config task_config/demo_clean.yml ...

协议:
    state  = left_joints(6) + left_gripper(1)
           + right_joints(6) + right_gripper(1)   = 14D
    action = 同上                                  = 14D
"""
import argparse
import datetime
import os
import sys
import time
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, Literal

import imageio
import numpy as np
import torch
import yaml
import zmq
from einops import rearrange
from PIL import Image as PILImage

# RoboTwin 仿真环境依赖：必须从仓库根目录运行
sys.path.append("./")
sys.path.append("./description/utils")
from envs import CONFIGS_PATH  # noqa: E402
from envs.utils.create_actor import UnStableError  # noqa: E402
from generate_episode_instructions import *  # noqa: E402,F403


# ========== ZMQ 通信 ==========

class TorchSerializer:
    @staticmethod
    def to_bytes(data):
        buffer = BytesIO()
        torch.save(data, buffer)
        return buffer.getvalue()

    @staticmethod
    def from_bytes(data):
        return torch.load(BytesIO(data), map_location='cpu')


class BaseInferenceClient:
    # 单次请求的等待上限；服务端单条推理约 7~12s，并发排队时留足余量
    DEFAULT_TIMEOUT_MS = 300000
    # 超时/服务端错误后的重连重试次数
    MAX_RETRIES = 3
    # 每次重试前的退避基数（秒）
    RETRY_BACKOFF_S = 2.0

    def __init__(self, host: str = 'localhost', port: int = 8081, timeout_ms: int = 15000):
        self.context = zmq.Context()
        self.host = host
        self.port = port
        self.timeout_ms = timeout_ms
        self.socket = None
        self._init_socket()

    def _init_socket(self):
        """（重）建 REQ socket。LINGER=0 保证 close 不阻塞。"""
        self._close_socket()
        self.socket = self.context.socket(zmq.REQ)
        self.socket.setsockopt(zmq.LINGER, 0)
        self.socket.connect(f'tcp://{self.host}:{self.port}')

    def _close_socket(self):
        if getattr(self, 'socket', None) is not None:
            try:
                self.socket.close(linger=0)
            except Exception as exc:
                print(f'[CLIENT_WARN] socket.close() 失败: {exc}')
            self.socket = None

    def ping(self) -> bool:
        try:
            self.call_endpoint('ping', requires_input=False, timeout_ms=self.timeout_ms)
            return True
        except Exception as exc:
            print(f'[CLIENT_WARN] ping 失败: {type(exc).__name__}: {exc}')
            self._init_socket()
            return False

    def kill_server(self):
        self.call_endpoint('kill', requires_input=False)

    def _request_once(self, endpoint: str, data: dict = None, requires_input: bool = True,
                      timeout_ms: int = None) -> dict:
        """发送一次请求并等待响应。超时或服务端报错都抛异常，由调用方决定是否重试。"""
        if timeout_ms is None:
            timeout_ms = self.DEFAULT_TIMEOUT_MS

        request: dict = {'endpoint': endpoint}
        if requires_input:
            request['data'] = data

        self.socket.send(TorchSerializer.to_bytes(request))

        if self.socket.poll(timeout_ms) == 0:
            raise TimeoutError(f'Server did not respond within {timeout_ms / 1000:.1f} seconds')

        message = self.socket.recv()
        if message == b'ERROR':
            raise RuntimeError('Server error')
        return TorchSerializer.from_bytes(message)

    def call_endpoint(self, endpoint: str, data: dict = None, requires_input: bool = True,
                      timeout_ms: int = None) -> dict:
        """带重连重试的请求。

        REQ socket 一旦超时，其状态机就停在“等应答”上，后续 send 会失败，
        因此每次重试前必须重建 socket，而不是复用同一个。
        """
        if timeout_ms is None:
            timeout_ms = self.DEFAULT_TIMEOUT_MS

        last_exc = None
        for attempt in range(1, self.MAX_RETRIES + 1):
            try:
                return self._request_once(endpoint, data, requires_input, timeout_ms)
            except (TimeoutError, RuntimeError, zmq.error.ZMQError) as exc:
                last_exc = exc
                print(f'[CLIENT_WARN] {endpoint} 请求失败 (尝试 {attempt}/{self.MAX_RETRIES}): '
                      f'{type(exc).__name__}: {exc}', flush=True)
                # 重建 socket，清掉 REQ 的半开状态
                self._init_socket()
                if attempt < self.MAX_RETRIES:
                    backoff = self.RETRY_BACKOFF_S * attempt
                    print(f'[CLIENT_WARN] {backoff:.1f}s 后重连重试', flush=True)
                    time.sleep(backoff)

        raise RuntimeError(
            f'{endpoint} failed after {self.MAX_RETRIES} attempts; last error: '
            f'{type(last_exc).__name__}: {last_exc}'
        ) from last_exc

    def close(self):
        """显式释放资源。LINGER=0 + close 保证 term() 不会阻塞。"""
        self._close_socket()
        if getattr(self, 'context', None) is not None:
            try:
                self.context.term()
            except Exception as term_exc:
                print(f'[CLIENT_WARN] context.term() 失败: {term_exc}')
            self.context = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


class RobotInferenceClient(BaseInferenceClient):
    def inference(self, observations: Dict[str, Any]):
        return self.call_endpoint('inference', observations)


# ========== 图像预处理 ==========

def resize_with_pad(images: np.ndarray, height: int, width: int,
                    method=PILImage.BILINEAR) -> np.ndarray:
    """Resize images to target size with padding (keeps aspect ratio)."""

    def _resize_with_pad_pil(image: PILImage.Image, height: int, width: int,
                             method: int) -> PILImage.Image:
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
        return zero_image

    if images.shape[-3:-1] == (height, width):
        return images
    original_shape = images.shape
    images = images.reshape(-1, *original_shape[-3:])
    resized = np.stack([
        _resize_with_pad_pil(PILImage.fromarray(im), height, width, method=method)
        for im in images
    ])
    return resized.reshape(*original_shape[:-3], *resized.shape[-3:])


ImageMode = Literal['uint8', 'float_native', 'float_resize224']


def _preprocess_rgb(img: np.ndarray, image_mode: ImageMode) -> np.ndarray:
    """Apply the selected client-side image preprocessing to a single RGB frame.

    float_native 是 RoboTwin 评测的默认值：只做 /255 归一化，把 resize 留给
    server 端的 image_cfg.resize_imgs_with_padding，避免客户端和训练侧
    resize 两次导致的插值差异。
    """
    if image_mode == 'uint8':
        return img.astype(np.uint8)
    if image_mode == 'float_native':
        return img.astype(np.float32) / 255.0
    if image_mode == 'float_resize224':
        return resize_with_pad(img, 224, 224).astype(np.float32) / 255.0
    raise ValueError(f'Unknown image_mode: {image_mode!r}')


def make_infer_data(camera_high, camera_left, camera_right, task_name, qpos):
    """构造 GigaBrain-0.7 模型的输入格式 (与 unified client 一致)。"""
    assert qpos.shape == (14,), f'expected 14D qpos, got {qpos.shape}'

    camera_high_chw = rearrange(camera_high, 'h w c -> c h w')
    camera_left_chw = rearrange(camera_left, 'h w c -> c h w')
    camera_right_chw = rearrange(camera_right, 'h w c -> c h w')

    return {
        'observation.state': torch.from_numpy(qpos).to(torch.float32),
        'observation.images.cam_high': torch.from_numpy(camera_high_chw),
        'observation.images.cam_left_wrist': torch.from_numpy(camera_left_chw),
        'observation.images.cam_right_wrist': torch.from_numpy(camera_right_chw),
        'task': task_name,
    }


# ========== 仿真环境工具函数 ==========

def class_decorator(task_name):
    """Load task environment class."""
    import importlib
    envs_module = importlib.import_module(f'envs.{task_name}')
    try:
        env_class = getattr(envs_module, task_name)
        return env_class()
    except AttributeError as exc:
        raise SystemExit(f'No Task: {task_name}') from exc


def get_camera_config(camera_type):
    camera_config_path = os.path.join('./task_config', '_camera_config.yml')
    assert os.path.isfile(camera_config_path), 'task config file is missing'
    with open(camera_config_path, 'r', encoding='utf-8') as f:
        args = yaml.load(f.read(), Loader=yaml.FullLoader)
    assert camera_type in args, f'camera {camera_type} is not defined'
    return args[camera_type]


def get_embodiment_config(robot_file):
    robot_config_file = os.path.join(robot_file, 'config.yml')
    with open(robot_config_file, 'r', encoding='utf-8') as f:
        return yaml.load(f.read(), Loader=yaml.FullLoader)


# ========== 推理 ==========

def inference_once(client, sim_env, instruction, image_mode: ImageMode):
    """从仿真环境取一帧 observation，请求一次动作 chunk。"""
    observation = sim_env.get_obs()

    img_front = _preprocess_rgb(observation['observation']['head_camera']['rgb'], image_mode)
    img_left = _preprocess_rgb(observation['observation']['left_camera']['rgb'], image_mode)
    img_right = _preprocess_rgb(observation['observation']['right_camera']['rgb'], image_mode)

    # 14 维: left_joints(6) + left_gripper(1) + right_joints(6) + right_gripper(1)
    qpos = observation['joint_action']['vector']

    obs = make_infer_data(img_front, img_left, img_right, instruction, qpos)

    start_time = time.time()
    actions = client.inference(obs)
    print(f'[INFERENCE] model cost time: {time.time() - start_time:.3f}s', flush=True)

    return actions.float().cpu().numpy()


def run_episode(args, client, sim_env, instruction):
    """单回合控制循环：每 pos_lookahead_step 步重新请求一次动作 chunk。"""
    max_publish_step = sim_env.step_lim if sim_env.step_lim is not None else args.max_publish_step
    chunk_size = args.chunk_size

    t = 0
    max_t = 0
    all_actions = None
    succ = False

    if args.temporal_agg:
        all_time_actions = np.zeros(
            [max_publish_step, max_publish_step + chunk_size, args.state_dim]
        )

    while t < max_publish_step and sim_env.take_action_cnt < sim_env.step_lim:
        if t >= max_t:
            all_actions = inference_once(client, sim_env, instruction, args.image_mode)
            max_t = t + args.pos_lookahead_step
            if args.temporal_agg:
                all_time_actions[[t], t:t + chunk_size] = all_actions

        if args.temporal_agg:
            actions_for_curr_step = all_time_actions[:, t]
            actions_populated = np.all(actions_for_curr_step != 0, axis=1)
            actions_for_curr_step = actions_for_curr_step[actions_populated]
            k = 0.01
            exp_weights = np.exp(-k * np.arange(len(actions_for_curr_step)))
            exp_weights = (exp_weights / exp_weights.sum())[:, np.newaxis]
            action = (actions_for_curr_step * exp_weights).sum(axis=0)

            sim_env.take_action(action)
            t += 1
        else:
            for t_ in range(args.pos_lookahead_step):
                if sim_env.take_action_cnt >= sim_env.step_lim:
                    break
                sim_env.take_action(all_actions[t_])
                sim_env.get_obs()
                t += 1

            if sim_env.eval_success:
                succ = True
                break

    return succ


# ========== 参数 ==========

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='GigaBrain-0.7 RoboTwin 2.0 evaluation client (14D state / 14D action).'
    )
    parser.add_argument('--config', type=str, required=True,
                        help='RoboTwin task config (e.g. task_config/demo_clean.yml)')
    parser.add_argument('--task_name', type=str, required=True, help='RoboTwin task name')
    parser.add_argument('--test_num', type=int, default=100,
                        help='Number of evaluated episodes per task')
    parser.add_argument('--seed', type=int, default=0,
                        help='Seed group; start seed = 100000 * (1 + seed)')
    parser.add_argument('--start_seed', type=int, default=None,
                        help='Override the starting seed directly')
    parser.add_argument('--instruction_type', type=str, default='unseen',
                        help='Instruction split used for the language prompt')
    parser.add_argument('--ckpt_setting', type=str, default='demo_clean',
                        help='Label recorded in the result file (demo_clean / demo_randomized)')
    parser.add_argument('--save_dir', type=str, default=None,
                        help='Root directory for evaluation results')
    parser.add_argument('--host', type=str, default='127.0.0.1', help='Policy server host')
    parser.add_argument('--port', type=int, default=8081, help='Policy server ZMQ port')
    parser.add_argument('--image_mode', type=str, default='float_native',
                        choices=['uint8', 'float_native', 'float_resize224'],
                        help='Client-side image preprocessing; keep float_native')
    parser.add_argument('--temporal_agg', action='store_true', default=False,
                        help='Enable ACT-style temporal ensembling of action chunks')
    parser.add_argument('--pos_lookahead_step', type=int, default=50,
                        help='Steps executed per inference call (<= chunk_size)')
    parser.add_argument('--max_publish_step', type=int, default=20000)
    parser.add_argument('--state_dim', type=int, default=14)
    parser.add_argument('--chunk_size', type=int, default=50,
                        help='Action chunk length; must match training action_chunk')
    return parser.parse_args()


def build_task_args(args: argparse.Namespace) -> dict:
    """加载 RoboTwin task config 并补齐 embodiment / camera 信息。"""
    config_path = args.config
    if not os.path.isabs(config_path) and not config_path.startswith('./'):
        config_path = f'./{config_path}'

    with open(config_path, 'r', encoding='utf-8') as f:
        task_args = yaml.load(f.read(), Loader=yaml.FullLoader)

    task_args['task_name'] = args.task_name
    task_args['task_config'] = Path(args.config).stem
    task_args['ckpt_setting'] = args.ckpt_setting

    embodiment_type = task_args.get('embodiment')
    with open(os.path.join(CONFIGS_PATH, '_embodiment_config.yml'), 'r', encoding='utf-8') as f:
        embodiment_types = yaml.load(f.read(), Loader=yaml.FullLoader)

    def embodiment_file(name):
        robot_file = embodiment_types[name]['file_path']
        if robot_file is None:
            raise ValueError(f'No embodiment file for {name}')
        return robot_file

    with open(CONFIGS_PATH + '_camera_config.yml', 'r', encoding='utf-8') as f:
        camera_config = yaml.load(f.read(), Loader=yaml.FullLoader)

    head_camera_type = task_args['camera']['head_camera_type']
    task_args['head_camera_h'] = camera_config[head_camera_type]['h']
    task_args['head_camera_w'] = camera_config[head_camera_type]['w']

    if len(embodiment_type) == 1:
        task_args['left_robot_file'] = embodiment_file(embodiment_type[0])
        task_args['right_robot_file'] = embodiment_file(embodiment_type[0])
        task_args['dual_arm_embodied'] = True
    elif len(embodiment_type) == 3:
        task_args['left_robot_file'] = embodiment_file(embodiment_type[0])
        task_args['right_robot_file'] = embodiment_file(embodiment_type[1])
        task_args['embodiment_dis'] = embodiment_type[2]
        task_args['dual_arm_embodied'] = False
    else:
        raise ValueError('embodiment items should be 1 or 3')

    task_args['left_embodiment_config'] = get_embodiment_config(task_args['left_robot_file'])
    task_args['right_embodiment_config'] = get_embodiment_config(task_args['right_robot_file'])
    return task_args


def main() -> int:
    args = parse_args()
    if args.pos_lookahead_step > args.chunk_size:
        raise ValueError(
            f'pos_lookahead_step ({args.pos_lookahead_step}) must be <= '
            f'chunk_size ({args.chunk_size})'
        )

    task_args = build_task_args(args)

    current_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    save_root = args.save_dir if args.save_dir is not None else 'eval_result'
    save_dir = Path(
        f"{save_root}/{args.task_name}/GigaBrain07Policy/"
        f"{task_args['task_config']}/{current_time}"
    )
    save_dir.mkdir(parents=True, exist_ok=True)

    if task_args['eval_video_log']:
        camera_config = get_camera_config(task_args['camera']['head_camera_type'])
        print(f"[INFO] video enabled, size {camera_config['w']}x{camera_config['h']}")
        task_args['eval_video_save_dir'] = save_dir

    print('=' * 80)
    print('GigaBrain-0.7 RoboTwin evaluation client')
    print('=' * 80)
    print(f'Task          : {args.task_name}')
    print(f'Task config   : {task_args["task_config"]}')
    print(f'Policy server : tcp://{args.host}:{args.port}')
    print(f'Test episodes : {args.test_num}')
    print(f'Image mode    : {args.image_mode}')
    print(f'Lookahead     : {args.pos_lookahead_step}/{args.chunk_size}')
    print('=' * 80, flush=True)

    task_env = class_decorator(args.task_name)

    client = RobotInferenceClient(host=args.host, port=args.port)
    print(f'Checking policy server at {args.host}:{args.port}...')
    while not client.ping():
        print('Waiting for server...')
        time.sleep(2)
    print(f'✓ policy server at {args.host}:{args.port} is ready', flush=True)

    result_file = save_dir / '_result.txt'
    with open(result_file, 'w') as f:
        f.write(f'Timestamp: {current_time}\n\n')
        f.write(f'Instruction Type: {args.instruction_type}\n\n')
        f.write('=' * 60 + '\n')
        f.write('Evaluation Progress:\n')
        f.write('=' * 60 + '\n\n')

    task_env.suc = 0
    task_env.test_num = 0
    now_id = 0
    succ_seed = 0
    now_seed = args.start_seed if args.start_seed is not None else 100000 * (1 + args.seed)
    clear_cache_freq = task_args['clear_cache_freq']
    task_args['eval_mode'] = True

    while succ_seed < args.test_num:
        render_freq = task_args['render_freq']
        task_args['render_freq'] = 0

        # Expert check：只在专家能完成的 seed 上评测策略，保证任务可解。
        try:
            task_env.setup_demo(now_ep_num=now_id, seed=now_seed, is_test=True, **task_args)
            episode_info = task_env.play_once()
            task_env.close_env()
        except UnStableError:
            task_env.close_env()
            now_seed += 1
            task_args['render_freq'] = render_freq
            continue
        except Exception as exc:
            task_env.close_env()
            now_seed += 1
            task_args['render_freq'] = render_freq
            print(f'[WARN] expert check error: {exc}')
            import traceback
            traceback.print_exc()
            continue

        if task_env.plan_success and task_env.check_success():
            succ_seed += 1
        else:
            now_seed += 1
            task_args['render_freq'] = render_freq
            continue

        task_args['render_freq'] = render_freq

        task_env.setup_demo(now_ep_num=now_id, seed=now_seed, is_test=True, **task_args)
        results = generate_episode_descriptions(  # noqa: F405
            args.task_name, [episode_info['info']], args.test_num
        )
        instruction = np.random.choice(results[0][args.instruction_type])

        if task_env.eval_video_path is not None:
            task_env._set_eval_video_ffmpeg(None)

        # 服务端可能崩溃/重启：重连重试都失败时不让整个 task 挂掉，
        # 等服务端回来后继续下一个 seed，保证能跑满 test_num 条。
        try:
            succ = run_episode(args, client, task_env, instruction)
        except Exception as exc:
            print(f'[EVAL_WARN] 本回合推理失败，跳过该 seed: {type(exc).__name__}: {exc}',
                  flush=True)
            import traceback
            traceback.print_exc()

            if task_env.eval_video_path is not None:
                try:
                    task_env._del_eval_video_ffmpeg()
                except Exception as video_exc:
                    print(f'[EVAL_WARN] 丢弃视频缓冲失败: {video_exc}')
            try:
                task_env.close_env()
            except Exception as close_exc:
                print(f'[EVAL_WARN] close_env 失败: {close_exc}')

            # 这个 seed 没有产生有效结果，回退计数并等服务端恢复
            succ_seed -= 1
            now_seed += 1
            print('[EVAL_WARN] 等待服务端恢复...', flush=True)
            while not client.ping():
                print('[EVAL_WARN] 服务端不可用，2s 后重试 ping', flush=True)
                time.sleep(2)
            print('[EVAL_WARN] 服务端已恢复，继续评估', flush=True)
            continue

        if task_env.eval_video_path is not None:
            video_frames = task_env._del_eval_video_ffmpeg()
            if len(video_frames) > 0:
                video_output_path = f'{task_env.eval_video_path}/episode{task_env.test_num}.mp4'
                try:
                    imageio.mimwrite(video_output_path, np.array(video_frames),
                                     fps=10, codec='libx264')
                    print(f'Video saved to `{video_output_path}`, '
                          f'{len(video_frames)} frames')
                except Exception as exc:
                    print(f'Failed to save video: {exc}')

        if succ:
            task_env.suc += 1

        now_id += 1
        task_env.close_env(clear_cache=((succ_seed + 1) % clear_cache_freq == 0))
        if task_env.render_freq:
            task_env.viewer.close()
        task_env.test_num += 1

        rate = round(task_env.suc / task_env.test_num * 100, 1)
        print(
            f'\033[93m{args.task_name}\033[0m | \033[94mGigaBrain-0.7\033[0m | '
            f'\033[92m{task_args["task_config"]}\033[0m | '
            f'\033[91m{task_args["ckpt_setting"]}\033[0m\n'
            f'Success rate: \033[96m{task_env.suc}/{task_env.test_num}\033[0m => '
            f'\033[95m{rate}%\033[0m, current seed: \033[90m{now_seed}\033[0m\n',
            flush=True,
        )

        with open(result_file, 'a') as f:
            f.write(f'\nEpisode {task_env.test_num}:\n')
            f.write(f'  Seed: {now_seed}\n')
            f.write(f'  Instruction: {instruction}\n')
            f.write(f"  Result: {'Success!' if succ else 'Fail!'}\n")
            f.write(f'  Current Success Rate: {task_env.suc}/{task_env.test_num} = {rate}%\n')

        now_seed += 1

    with open(result_file, 'a') as f:
        f.write('\n' + '=' * 60 + '\n')
        f.write('Final Result:\n')
        f.write('=' * 60 + '\n')
        f.write(f'Success Rate: {task_env.suc}/{args.test_num} = '
                f'{task_env.suc / args.test_num * 100:.1f}%\n')
        f.write(f'\nTimestamp: {current_time}\n')
        f.write(f'Instruction Type: {args.instruction_type}\n')

    print(f'\n✓ Results saved to {result_file}')
    client.close()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
