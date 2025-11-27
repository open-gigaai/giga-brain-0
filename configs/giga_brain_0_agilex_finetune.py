action_chunk = 50
data_paths = [
    './lerobot_data/',
]
data_or_config = []
for data_path in data_paths:
    data_or_config.append(
        dict(
            _class_name='LeRobotDataset',
            data_path=data_path,
            delta_info=dict(
                action=action_chunk,
            ),
            meta_name='meta',
        )
    )
config = dict(
    runners=['giga_brain_0.GigaBrain0Trainer'],
    project_dir='./experiments/vla/giga_brain_0_agilex/finetune_debug',
    launch=dict(
        gpu_ids=[0, 1, 2, 3, 4, 5, 6, 7],
        distributed_type='FSDP',
        fsdp_config=dict(
            fsdp_version='2',
            fsdp_auto_wrap_policy='TRANSFORMER_BASED_WRAP',
            fsdp_transformer_layer_cls_to_wrap='SiglipEncoderLayer,Gemma2DecoderLayerWithExpert',
            fsdp_cpu_ram_efficient_loading='false',
            fsdp_state_dict_type='FULL_STATE_DICT',
        ),
    ),
    dataloaders=dict(
        train=dict(
            data_or_config=data_or_config,
            batch_size_per_gpu=32,
            num_workers=16,
            transform=dict(
                type='GigaBrain0Transform',
                is_train=True,
                delta_action_cfg=dict(
                    use_delta_joint_actions=True,
                    mask={
                        '0': [True, True, True, True, True, True, False, True, True, True, True, True, True, False],
                    },
                ),
                norm_cfg=dict(
                    norm_stats_path={
                        '0': './lerobot_data/meta/agilex_norm_stats.json',
                    },
                    use_quantiles=True,
                ),
                image_cfg=dict(
                    resize_imgs_with_padding=[224, 224],
                    enable_image_aug=True,
                    present_img_keys=['observation.images.cam_high', 'observation.images.cam_left_wrist', 'observation.images.cam_right_wrist'],
                    enable_depth_img=False,
                ),
                prompt_cfg=dict(
                    tokenizer_model_path='google/paligemma-3b-pt-224',
                    fast_tokenizer_path='physical-intelligence/fast',
                    max_length=200,
                    # Can be change following sample_ratios according to your own data.
                    discrete_state_input=True,
                    encode_action_input=False,
                    encode_sub_task_input=False,
                    sample_ratios=dict(
                        task_only=1.0,
                        task_with_subtask=0,
                        task_only_using_subtask_regression=0,
                        task_only_using_fast_regression=0,
                        task_with_subtask_using_fast_regression=0,
                    ),  # 100%
                ),
            ),
            sampler=dict(
                type='DefaultSampler',
                shuffle=True,
            ),
        ),
    ),
    models=dict(
        pretrained='open-gigaai/GigaBrain-0-3.5B-Base',
        enable_knowledge_insulation=False,
        enable_learnable_traj_token=False,
        num_embodiments=3,
    ),
    optimizers=dict(
        type='AdamW',
        betas=(0.9, 0.95),
        lr=2.5e-5,
        eps=1e-8,
        weight_decay=1e-10,
    ),
    schedulers=dict(
        type='WarmupCosineScheduler',
        warmup_steps=1000,
        decay_steps=30000,
        end_value=0.1,  # decay_lr = optimizers.lr * end_value. 1.0 means constant lr
    ),
    train=dict(
        resume=True,
        max_steps=50000,
        gradient_accumulation_steps=1,
        mixed_precision='no',  # Apply non-automatic mixed precision training.
        checkpoint_interval=1000,
        checkpoint_total_limit=5,
        checkpoint_keeps=[5000, 10000, 15000, 20000, 25000, 30000, 35000, 40000, 45000, 50000],
        checkpoint_safe_serialization=False,
        checkpoint_strict=False,
        log_with='tensorboard',
        log_interval=100,
        with_ema=True,
        dynamo_config=dict(backend='inductor'),
        activation_checkpointing=True,
        activation_class_names=[
            'SiglipEncoderLayer',
            'Gemma2DecoderLayerWithExpert',
        ],
    ),
)
