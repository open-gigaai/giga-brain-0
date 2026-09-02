"""GigaBrain-0.7 RoboTwin 2.0 50-task post-training from the PaliGemma2 200k EMA.

数据来自 RoboTwin 2.0 的 50 个双臂任务 (agilex_cobot_magic / aloha-agilex),
转换成 LeRobot 格式后作为单个 dataset 根目录传入。

训练/评测的域划分 (重要):
    训练只用 demo_clean 采集的数据; 模型没有见过任何 randomized 数据。
    评测同时跑 demo_clean 和 demo_randomized, 后者是跨域泛化,
    成功率低于 clean 属于预期。不要把 randomized 数据混进 DATA_ROOT。

协议 (与评测 server/client 严格对齐):
    state  = left_joints(6) + left_gripper(1)
           + right_joints(6) + right_gripper(1)      = 14D
    action = 同上                                     = 14D

norm_stats 是 14 维；client 发 14 维 state，server 返回 14 维 action。
模型内部 pad 到 max_action_dim=32，由 delta_action_cfg.mask 的 16 位
robot mask 决定哪些维度被监督。
"""

ACTION_CHUNK = 50
ROBOT_TYPE = "agilex_cobot_magic"
# The released/reference RoboTwin checkpoint was trained on embodiment branch 0.
# Keep this explicit because the shared default mapping assigns this exact
# robot_type string to branch 6.
EMBODIMENT_ID = 0

# ========== 用户配置区 ==========
# RoboTwin 2.0 LeRobot 数据集根目录 (含 meta/info.json)
# 只放 50 个任务的 demo_clean 数据，不含 demo_randomized
DATA_ROOT = "/path/to/data/robotwin2_50task_clean_lerobot"
# 训练时用 scripts/compute_norm_stats.py 生成的 norm_stats
NORM_STATS_PATH = "/path/to/assets/robotwin2_50task_14d/giga_norm_stats.json"
# 后训练起点：GigaBrain-0.7 PaliGemma2 proprio_anchor 200k EMA
PRETRAINED_CKPT = "/path/to/checkpoints/gb07_pg2_proprio_anchor_200k/model_ema"
# 训练输出目录 (checkpoint 落在 <PROJECT_DIR>/models/)
PROJECT_DIR = "/path/to/results/gb07_pg2_robotwin_50task_80k"

PALIGEMMA2_TOKENIZER_PATH = (
    "/path/to/huggingface/models--google--paligemma2-3b-pt-224"
)
FAST_TOKENIZER_PATH = (
    "/path/to/huggingface/models--physical-intelligence--fast"
)

AGILEX_DATA_PATHS = [DATA_ROOT]

data_or_config = [
    dict(
        _class_name="LeRobotDataset",
        data_path=data_path,
        delta_info={
            "action": ACTION_CHUNK,
        },
        meta_name="meta",
    )
    for data_path in AGILEX_DATA_PATHS
]


config = dict(
    runners=["gigabrain07.GigaBrain07Trainer"],
    project_dir=PROJECT_DIR,
    launch=dict(
        gpu_ids=[0, 1, 2, 3, 4, 5, 6, 7],
        distributed_type="FSDP",
        fsdp_config=dict(
            fsdp_version="2",
            fsdp_auto_wrap_policy="TRANSFORMER_BASED_WRAP",
            fsdp_transformer_layer_cls_to_wrap="SiglipEncoderLayer,Gemma2DecoderLayerWithExpert",
            fsdp_cpu_ram_efficient_loading="false",
            fsdp_state_dict_type="FULL_STATE_DICT",
        ),
    ),
    dataloaders=dict(
        train=dict(
            data_or_config=data_or_config,
            batch_size_per_gpu=32,
            num_workers=16,
            transform=dict(
                type="GigaBrain07Transform",
                is_train=True,
                # 200k 预训练是 proprio_anchor: state 走 proprio token 通路，
                # 不进文本 prompt (与 ckpt 的 inference_config.json 一致)。
                state_input_mode="proprio_anchor",
                observation_memory_size=1,
                agent_pos_config=dict(state=32),
                robot_type_embodiment_id_overrides={
                    ROBOT_TYPE: EMBODIMENT_ID,
                },
                use_quaternion_to_6d=True,
                delta_action_cfg=dict(
                    selector="robot_type",
                    use_delta_joint_actions=True,
                    mask_unsupervised_action_dims_for_noise=True,
                    # 14 个有效维度 + 2 个 padding；gripper 位 (idx 6/13) 为 False
                    # 表示绝对值而非 delta。
                    mask={
                        ROBOT_TYPE: [
                            True, True, True, True, True, True, False,
                            True, True, True, True, True, True, False,
                            False, False,
                        ],
                    },
                ),
                norm_cfg=dict(
                    selector="data_path",
                    norm_stats_path=[
                        dict(
                            data_paths=AGILEX_DATA_PATHS,
                            path=NORM_STATS_PATH,
                        )
                    ],
                    use_quantiles=True,
                ),
                image_cfg=dict(
                    resize_imgs_with_padding=[224, 224],
                    enable_image_aug=True,
                    present_img_keys=[
                        "observation.images.cam_high",
                        "observation.images.cam_left_wrist",
                        "observation.images.cam_right_wrist",
                    ],
                    enable_depth_img=False,
                    vlm_type="paligemma2",
                ),
                prompt_cfg=dict(
                    tokenizer_model_path=PALIGEMMA2_TOKENIZER_PATH,
                    fast_tokenizer_path=FAST_TOKENIZER_PATH,
                    max_length=120,
                    state_input_mode="proprio_anchor",
                    discrete_state_input=False,
                    discrete_state_input_for_pose_embodiments=False,
                    encoded_action_horizon=ACTION_CHUNK,
                    encode_action_input=False,
                    encode_sub_task_input=False,
                    # 保持 True 以对齐预训练的 prompt 格式。
                    enable_control_mode_token=True,
                    enable_end_effector_token=True,
                    vlm_type="paligemma2",
                    sample_ratios=dict(
                        input_task=1.0,
                        input_subtask=0.0,
                        input_task_target_subtask=0.0,
                        input_task_target_action=0.0,
                        input_subtask_target_action=0.0,
                        input_task_target_subtask_action=0.0,
                    ),
                ),
            ),
            sampler=dict(
                type="DefaultSampler",
                shuffle=True,
            ),
        ),
        test=dict(),
    ),
    models=dict(
        pretrained=PRETRAINED_CKPT,
        vlm_type="paligemma2",
        state_input_mode="proprio_anchor",
        observation_memory_size=1,
        agent_pos_config=dict(state=32),
        enable_knowledge_insulation=False,
        enable_next_token_prediction=False,
        # 动作损失完整回传到 VLM (单次 fused forward)。不设这一项会继承
        # ckpt 里的 0.1，使 VLM 约 90% 梯度隔离。
        action_loss_vlm_gradient_weight=1.0,
    ),
    optimizers=dict(
        type="AdamW",
        betas=(0.9, 0.95),
        lr=2.5e-5,
        eps=1e-8,
        weight_decay=1e-10,
    ),
    schedulers=dict(
        type="WarmupCosineScheduler",
        warmup_steps=1000,
        decay_steps=50000,
        end_value=0.1,
    ),
    train=dict(
        resume=False,
        resume_compile_warmup=True,
        max_steps=80000,
        gradient_accumulation_steps=1,
        llm_loss_weight=0.0,
        mixed_precision="no",
        checkpoint_interval=1000,
        checkpoint_total_limit=5,
        checkpoint_keeps=[
            10000, 20000, 30000, 40000, 50000, 60000, 70000, 80000,
        ],
        checkpoint_safe_serialization=False,
        checkpoint_strict=False,
        log_with="tensorboard",
        log_interval=10,
        with_ema=True,
        dynamo_config=dict(backend="inductor"),
        activation_checkpointing=True,
        activation_class_names=[
            "Gemma2DecoderLayerWithExpert__##__18",
        ],
    ),
)
