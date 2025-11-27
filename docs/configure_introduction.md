## 1. Data Configuration

Our framework supports passing multiple packaged `LeRobotDataset`s simultaneously and proxies the main `LeRobotDataset` interfaces.

- `delta_info`: Specifies the length of data to read forward or backward from the current timestamp. For example, `'action': 50` indicates reading action data for the current frame and the subsequent 49 frames.
  - If different datasets have different FPS, or if more advanced configuration capabilities (such as reading data from both current and past moments) are required, please review and modify the code implementation to support these features.
- `skip_video_decoding`: Determines whether to skip video data decoding for the current frame when accessing data. This configuration is primarily used to reduce unnecessary CPU computation during `compute norm stats`.

## 2. Experiment Directory and Distributed Configuration

- `runners`: Sets the Trainer class for training.
- `project_dir`: Directory for storing training configs, logs, and checkpoints.
- `gpu_ids`: IDs of the GPUs to be used. If only one GPU is set, `distributed_type` and `fsdp_config` below can be ignored, as distributed training is not used with a single card.
- `distributed_type`: The distributed framework to use, such as DDP, Deepspeed, or FSDP.
- `xxx_config`: Configures the specific settings based on the distributed framework selected above, such as `fsdp_config` or `deepspeed_config`.

## 3. Training Data Processing (DataLoaders) Configuration

- `delta_action_cfg`: Determines whether to use absolute action values or delta action values relative to the state.
  - The `mask` key represents `embodiment_id`. The value indicates whether to use delta values for each degree of freedom (DoF). DoFs exceeding the length of the value list default to `False`.
- `norm_cfg`: Normalization configuration for state and action.
  - The `norm_stats_path` key represents `embodiment_id`, and the value is the path to the normalization statistics.
  - `use_quantiles`: If `True`, uses q01/q99 normalization; if `False`, uses mean/std normalization.
- `image_cfg`: Image processing configuration.
  - `resize_image_with_padding`: Resizes the image while maintaining the aspect ratio and applies padding. (224, 224) specifies the target height and width.
  - `enable_image_aug`: Whether to enable image augmentation. Defaults to `True`.
  - `present_img_keys`: Specifies which images the model uses and their order. If an image key in the `LeRobotDataset` is not listed here, it will not be used for training. If a key listed in `present_img_keys` does not exist in the `LeRobotDataset`, an error will occur and the process will terminate.
  - `enable_depth_img`: Whether to include depth images during processing. If `True`, `depth_img_prefix_name` must also be set. When enabled, if an image has a corresponding depth map, it is concatenated as the fourth channel of the original image. If no depth map exists, a zero-filled tensor is concatenated as the fourth channel.
- `traj_cfg`: Supervision using 2D trajectory.
  - `step_interval`: Sampling interval for the temporal length of the 2D trajectory. If `1`, no sampling is performed.
  - `minmax_value`: Normalization value for the trajectory; the maximum value corresponds to the size after image resizing.
- `prompt_cfg`: Text construction configuration.
  - `tokenizer_model_path`: Path to the text tokenizer.
  - `fast_tokenizer_path`: Path to the action tokenizer.
  - `max_length`: Maximum token length.
  - `discrete_state_input`: Whether to perform text encoding for the state. Defaults to `False` in pi0 and `True` in pi05.
  - `encode_action_input` and `encode_sub_task_input`: Whether to perform autoregression for discrete actions or sub-tasks. These values are ineffective if `sample_ratios` is present, as they will be overridden by the sampler's output.
  - `sample_ratios`: Autoregression targets for each training step. The value represents the sampling probability, and the sum must equal 1.0.
    - `task_only`: Encodes only the task without autoregressive supervision. Commonly used for post-training: `f'Task: {main_task}, State: {state_str};\n'`
    - `task_with_subtask`: Encodes task and subtask without autoregressive supervision. Commonly used for post-training: `f'Task: {main_task}, Subtask:  {sub_task}, State: {state_str};\n'`
    - `task_only_using_subtask_regression`: Encodes only the task, adding subtask autoregressive supervision: `f'Task: {main_task}\nSubtask:  {sub_task}<eos>'`. In this case, flow matching loss is disabled.
    - `task_only_using_fast_regression`: Encodes only the task, adding FAST action autoregressive supervision: `f'Task: {main_task}, State: {state_str};\nAction: {encoded_discrete_actions}|<eos>'`
    - `task_with_subtask_using_fast_regression`: Encodes task and subtask, adding FAST action autoregressive supervision: `f'Task: {main_task}, Subtask:  {sub_task}, State: {state_str};\nAction: {encoded_discrete_actions}|<eos>'`

Data Sampler Configuration:

- Default is `DefaultSampler`. `shuffle` indicates whether to shuffle the data.
- If resampling of different `LeRobotDataset`s is required, `WeightedSampler` can be used.

## 4. Model Configuration

- `pretrained`: If loading a complete model checkpoint, configure this parameter as the model directory. Initialization parameters for the model are also recorded in this path.
- `pretrained_paligemma_path`: If only loading the VLM portion of the weights for pre-training, configure this parameter as the path to the model weights.
- Other configurations: To update or override default parameters in the model definition or parameters recorded in pretrain, specify them here. Refer to `modeling_giga_brain_0.py` for default parameters.

## 5. Training Configuration

- `optimizers`: Optimizer configuration.
- `schedulers`: Learning rate scheduler configuration.
- `train`:
  - `resume`: Whether to resume by default. If `True`, loads the latest model weights from `project_dir`.
  - `max_steps`: Maximum training steps. `max_epochs` can also be used.
  - `with_ema`: Whether to use Exponential Moving Average (EMA).
  - `dynamo_config`: Whether to enable compilation (torch.compile).
  - `activation_checkpointing`: Whether to use activation checkpointing to trade computation for GPU memory.
    - `activation_class_names`: Specifies which Blocks use activation checkpointing. If set to "Gemma2DecoderLayerWithExpert", it enables it for all `Gemma2DecoderLayerWithExpert` layers. If omitted, it is disabled for them. If set to "Gemma2DecoderLayerWithExpert\_\_##\_\_6", it enables activation checkpointing for the first six `Gemma2DecoderLayerWithExpert` layers only.
