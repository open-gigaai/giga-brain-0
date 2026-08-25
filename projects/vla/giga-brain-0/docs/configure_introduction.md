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
  - `mask_unsupervised_action_dims_for_noise` defaults to `False`. When enabled, the configured action-mask length defines the valid action dimensions: padded/invalid dimensions stay zero in the flow-noise input and target, and only valid dimensions contribute to the action loss.
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
  - `fast_token_vocab_mode`: FAST token mapping mode. PaliGemma2 defaults to the legacy `tail` layout: reserve the final 128 vocab ids, then map FAST BPE ids into the remaining tail ids in descending order.
  - `max_length`: Maximum token length.
  - `discrete_state_input`: Whether to encode robot state as text. For the GigaBrain-0.7 `proprio_anchor` mode this must be `False`.
  - `encode_action_input` and `encode_sub_task_input`: Whether to perform autoregression for discrete actions or sub-tasks. These values are ineffective if `sample_ratios` is present, as they will be overridden by the sampler's output.
  - `sample_ratios`: Prompt and autoregressive target mix for each training step. The value represents the sampling probability, and the sum must equal 1.0.
    - `input_task`: Input task only; no language target.
    - `input_subtask`: Input GT subtask only; no language target.
    - `input_task_target_subtask`: Input task; predict subtask. In this case, flow matching loss is disabled.
    - `input_task_target_action`: Input task; predict FAST action tokens.
    - `input_subtask_target_action`: Input GT subtask; predict FAST action tokens. This is the mode where GT subtask is used as prompt and subtask is not predicted.
    - `input_task_target_subtask_action`: Input task; predict subtask and FAST action tokens. FAST token rows are blocked from attending to predicted subtask token columns, so this is parallel supervision rather than action conditioned on the predicted subtask.

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

## 6. Per-Embodiment Diffusion Loss Logging

`GigaBrain07Trainer` 在标准训练损失之外，额外按 `embodiment_id` 分桶记录扩散损失，便于对比联合训练（如 700h 本体 + 700h ego）与单独训练（如 700h 本体）在每个本体上的收敛差异。

- 指标 key：`metric/diff_loss_emb_<id>`，`<id>` 对应 `EmbodimentId` 枚举（`AGILEX=0`、`AGIBOT_G1=1`、`AGIBOT_DEX=2`、`UMI_OMIN=3`、`EGO_DEX=4`、`EGODEX_EEF_HANDBASE=5`、`ROBOCOIN_AGILEX_COBOT_MAGIC=6`、`H01_ROBOT=7`）。
- 桶数取自模型 `config.num_embodiments`；不存在对应样本的本体在该 step 该 rank 上记为 NaN，跨 rank 聚合时 NaN-aware 求平均。
- 这些指标**不参与反向传播**，仅与 `total_loss` 一起进入日志和 `accelerator.log`（即 wandb/tensorboard），不会改变训练动力学。
- 实现位置：`giga_brain_0/giga_brain_0_loss.py:_per_embodiment_diffusion_metrics` 与 `giga_brain_0/giga_brain_0_trainer.py:GigaBrain07Trainer.parse_losses`。
