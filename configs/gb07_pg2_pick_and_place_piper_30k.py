"""GigaBrain-0.7 pick-and-place post-training from the PaliGemma2 200k EMA."""


PROJECT_ROOT = "/gpfs/users/wangyunmo/experiments"
pick_and_place_data_paths = [
    (
        "/gpfs/dataset/user-private/peng.li/datasets/gigabrain1/"
        "private_datasets/robotics/260411_pick_anything/lerobot/"
        "merged_v2_v30/merged_v2_v30"
    ),
]
NORM_STATS_PATH = (
    "/gpfs/users/wangyunmo/normstats_new2/"
    "norm_stat_giga_robot_1_0_agilex.json"
)
PRETRAINED_PALIGEMMA_PATH = (
    "/gpfs/users/wangyunmo/experiments/"
    "gb1_pg2_ruev_0707_coarse_emb_cycle_proprio_anchor_new_data/"
    "models/checkpoint_epoch_1_step_200000/model_ema/"
    "diffusion_pytorch_model.bin"
)
PALIGEMMA2_TOKENIZER_PATH = (
    "/gpfs/users/wangyunmo/pretrain/models--google--paligemma2-3b-pt-224"
)
FAST_TOKENIZER_PATH = (
    "/gpfs/users/wangyunmo/pretrain/models--physical-intelligence--fast"
)

ACTION_CHUNK = 50
AGENT_POS_CONFIG = {"state": 32}
CAMERA_KEYS = (
    "observation.images.cam_high",
    "observation.images.cam_left_wrist",
    "observation.images.cam_right_wrist",
)
AGILEX_COBOT_MAGIC_DELTA_MASK = [
    True,
    True,
    True,
    True,
    True,
    True,
    False,
    True,
    True,
    True,
    True,
    True,
    True,
    False,
    False,
    False,
]

DATASET_CONFIGS = [
    dict(
        _class_name="LeRobotDataset",
        data_path=data_path,
        delta_info={"action": ACTION_CHUNK},
        meta_name="meta",
        video_backend="pyav",
        torchcodec_decoder_cache_size=128,
        decode_video_keys=CAMERA_KEYS,
    )
    for data_path in pick_and_place_data_paths
]
delta_mask_by_robot_type = {
    "agilex_cobot_magic": list(AGILEX_COBOT_MAGIC_DELTA_MASK),
}
dataset_groups = [
    (
        "pick_and_place",
        list(pick_and_place_data_paths),
        None,
        NORM_STATS_PATH,
    )
]


config = dict(
    runners=["gigabrain07.GigaBrain07Trainer"],
    project_dir=(
        f"{PROJECT_ROOT}/gb1_pg2_pick_and_place_30k_from_200k"
    ),
    launch=dict(
        gpu_ids=[0, 1, 2, 3, 4, 5, 6, 7],
        distributed_type="FSDP",
        fsdp_config=dict(
            fsdp_version="2",
            fsdp_auto_wrap_policy="TRANSFORMER_BASED_WRAP",
            fsdp_transformer_layer_cls_to_wrap=(
                "SiglipEncoderLayer,Gemma2DecoderLayerWithExpert"
            ),
            fsdp_cpu_ram_efficient_loading="false",
            fsdp_state_dict_type="FULL_STATE_DICT",
        ),
    ),
    dataloaders=dict(
        train=dict(
            data_or_config=dict(
                _class_name="WeightedConcatDataset",
                datasets=[DATASET_CONFIGS],
                sampling_weights=[1.0],
                group_names=["robot_pick_and_place"],
            ),
            batch_size_per_gpu=32,
            num_workers=8,
            persistent_workers=False,
            transform=dict(
                type="GigaBrain07Transform",
                is_train=True,
                state_input_mode="proprio_anchor",
                observation_memory_size=1,
                agent_pos_config=dict(AGENT_POS_CONFIG),
                use_quaternion_to_6d=True,
                delta_action_cfg=dict(
                    selector="robot_type",
                    use_delta_joint_actions=True,
                    mask_unsupervised_action_dims_for_noise=True,
                    mask={
                        robot_type: list(mask)
                        for robot_type, mask in delta_mask_by_robot_type.items()
                    },
                ),
                norm_cfg=dict(
                    selector="data_path",
                    norm_stats_path=[
                        dict(
                            data_paths=list(pick_and_place_data_paths),
                            path=NORM_STATS_PATH,
                        )
                    ],
                    use_quantiles=True,
                    enable_clamp=False,
                ),
                image_cfg=dict(
                    resize_imgs_with_padding=[224, 224],
                    enable_image_aug=True,
                    present_img_keys=list(CAMERA_KEYS),
                    enable_depth_img=False,
                    vlm_type="paligemma2",
                ),
                prompt_cfg=dict(
                    tokenizer_model_path=PALIGEMMA2_TOKENIZER_PATH,
                    fast_tokenizer_path=FAST_TOKENIZER_PATH,
                    enable_control_mode_token=True,
                    enable_end_effector_token=True,
                    max_length=120,
                    state_input_mode="proprio_anchor",
                    discrete_state_input=False,
                    discrete_state_input_for_pose_embodiments=False,
                    encode_action_input=False,
                    encode_sub_task_input=False,
                    encoded_action_horizon=50,
                    sample_ratios=dict(
                        input_task=1,
                        input_subtask=0,
                        input_task_target_subtask=0,
                        input_task_target_action=0,
                        input_subtask_target_action=0,
                        input_task_target_subtask_action=0,
                    ),
                    vlm_type="paligemma2",
                ),
            ),
            sampler=dict(
                type='DefaultSampler',
                shuffle=True,
            ),
            # sampler=dict(
            #     type="ShardedListWeightedSampler",
            #     shuffle=True,
            #     shard_mode="group_balanced",
            #     oversample_mode="cycle",
            # ),
            # cache=dict(
            #     type="hf_arrow",
            #     process_scope="local_main",
            #     warmup_mode="sampler_shard",
            #     release_after_open=True,
            #     lerobot_open_cache=dict(max_open_per_worker=32, gc_interval=32),
            # ),
        ),
        test=dict(),
    ),
    models=dict(
        vlm_type="paligemma2",
        pretrained_paligemma_path=PRETRAINED_PALIGEMMA_PATH,
        vlm_hidden_size=2304,
        num_embodiments=8,
        has_action_expert=True,
        proj_width=1024,
        n_action_steps=ACTION_CHUNK,
        flow_action_horizon=50,
        max_action_dim=32,
        num_steps=10,
        enable_knowledge_insulation=False,
        action_loss_vlm_gradient_weight=1.0,
        enable_next_token_prediction=False,
        state_input_mode="proprio_anchor",
        observation_memory_size=1,
        agent_pos_config=dict(AGENT_POS_CONFIG),
        state_hidden_size=2048,
        proj_with_mask=True,
        enable_learnable_traj_token=False,
        encode_action_input=True,
        fast_tokenizer_path=FAST_TOKENIZER_PATH,
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
        decay_steps=25000,
        end_value=0.1,
    ),
    train=dict(
        resume=True,
        resume_compile_warmup=True,
        max_steps=30000,
        gradient_accumulation_steps=1,
        llm_loss_weight=0.0,
        mixed_precision="no",
        checkpoint_interval=5000,
        checkpoint_total_limit=2,
        dynamo_config=dict(backend="inductor"),
        checkpoint_keeps=[10000, 20000, 30000],
        checkpoint_safe_serialization=False,
        checkpoint_strict=False,
        log_with="tensorboard",
        log_interval=50,
        with_ema=True,
        activation_checkpointing=True,
        activation_class_names=[
            # "SiglipEncoderLayer__##__1",
            "Gemma2DecoderLayerWithExpert__##__15",
        ],
        # max_grad_norm=1.0,
        worker_profile=False,
    ),
)
