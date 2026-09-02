#!/usr/bin/env bash

# Gigabrain0.7 evaluation profile. This file contains generation settings,
# not expected scores or historical result hashes.

set -euo pipefail

if [ "${1:-}" != "--dump-json" ]; then
    echo "This profile is loaded by MiMo-Embodied/runners/run_eval.sh." >&2
    exit 2
fi

cat <<'JSON'
{
  "schema_version": 1,
  "model_id": "gigabrain0.7",
  "display_name": "Gigabrain0.7",
  "adapter": "gigabrain0.7",
  "defaults": {
    "model_path": "model-repos/Gigabrain0.7",
    "giga_models_dir": "model-repos/gigabrain",
    "tokenizer_model_path": "model-repos/tokenizers/paligemma2-3b-pt-224",
    "fast_tokenizer_path": "model-repos/tokenizers/fast",
    "data_root": "MiMo-Embodied/datasets/public_datasets/VLM",
    "output_root": "MiMo-Embodied/eval_results/gigabrain0.7"
  },
  "model_args": {
    "model_path": "{model_path}",
    "backbone": "paligemma2",
    "dtype": "bfloat16",
    "giga_models_dir": "{giga_models_dir}",
    "tokenizer_model_path": "{tokenizer_model_path}",
    "fast_tokenizer_path": "{fast_tokenizer_path}",
    "image_key": "observation.images.cam_high",
    "paligemma2_weight_format": "policy",
    "paligemma2_policy_force_lang": false
  },
  "model_arg_profiles": {
    "point": {
      "max_new_tokens_cap": 64,
      "paligemma2_policy_low_cpu_mem_usage": true,
      "paligemma2_policy_deterministic": true
    },
    "nonpoint_256": {
      "max_new_tokens_cap": 256,
      "paligemma2_policy_low_cpu_mem_usage": false,
      "paligemma2_policy_deterministic": false
    },
    "nonpoint_64": {
      "max_new_tokens_cap": 64,
      "paligemma2_policy_low_cpu_mem_usage": true,
      "paligemma2_policy_deterministic": false
    }
  },
  "runtime_env_profiles": {
    "point": {
      "PYTHONHASHSEED": "0",
      "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
      "NVIDIA_TF32_OVERRIDE": "0",
      "QWEN_RESIZE_MAX_PIXELS": "50176"
    },
    "nonpoint": {
      "PYTHONHASHSEED": "0",
      "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
      "NVIDIA_TF32_OVERRIDE": null,
      "QWEN_RESIZE_MAX_PIXELS": "50176"
    }
  },
  "tasks": [
    {
      "task": "metavqa_eval_robust",
      "kind": "nonpoint",
      "estimated_rows": 10000,
      "model_arg_profile": "nonpoint_256",
      "runtime_env_profile": "nonpoint",
      "model_args": {"paligemma2_policy_prompt_style": "training_vqa_no_format"},
      "framework_seed": [0, 1234, 1234, 1234],
      "metric_key": "accuracy,robust-choice-extract",
      "score_field": "accuracy",
      "filter_contains": "RobustChoiceFilter"
    },
    {
      "task": "crpe_relation_robust",
      "kind": "nonpoint",
      "estimated_rows": 7576,
      "model_arg_profile": "nonpoint_256",
      "runtime_env_profile": "nonpoint",
      "model_args": {"paligemma2_policy_prompt_style": "training_vqa_no_format"},
      "framework_seed": [0, 1234, 1234, 1234],
      "metric_key": "accuracy,robust-choice-extract",
      "score_field": "accuracy",
      "filter_contains": "RobustChoiceFilter"
    },
    {
      "task": "embspatialbench_robust",
      "kind": "nonpoint",
      "estimated_rows": 3640,
      "model_arg_profile": "nonpoint_256",
      "runtime_env_profile": "nonpoint",
      "model_args": {"paligemma2_policy_prompt_style": "training_vqa_no_format"},
      "framework_seed": [0, 1234, 1234, 1234],
      "metric_key": "accuracy,robust-choice-extract",
      "score_field": "accuracy",
      "filter_contains": "RobustChoiceFilter"
    },
    {
      "task": "cvbench_boxed_robust",
      "kind": "nonpoint",
      "estimated_rows": 2638,
      "model_arg_profile": "nonpoint_256",
      "runtime_env_profile": "nonpoint",
      "model_args": {"paligemma2_policy_prompt_style": "training_vqa_no_format"},
      "framework_seed": [0, 1234, 1234, 1234],
      "metric_key": "accuracy,robust-choice-extract",
      "score_field": "accuracy",
      "filter_contains": "RobustChoiceFilter"
    },
    {
      "task": "part_affordance",
      "kind": "point",
      "estimated_rows": 2000,
      "model_arg_profile": "point",
      "runtime_env_profile": "point",
      "model_args": {"paligemma2_policy_prompt_style": "training_vqa"},
      "framework_seed": [0, 0, 0, 0],
      "generation_kwargs": {
        "max_new_tokens": 64,
        "do_sample": false,
        "temperature": 0.0,
        "top_p": 1.0,
        "point_coordinate_max_decimals": 2,
        "point_stop_after_first": true
      }
    },
    {
      "task": "roborefit",
      "kind": "point",
      "estimated_rows": 2000,
      "model_arg_profile": "point",
      "runtime_env_profile": "point",
      "model_args": {"paligemma2_policy_prompt_style": "training_vqa_point_refit_boundary_interior"},
      "framework_seed": [0, 0, 0, 0],
      "generation_kwargs": {
        "max_new_tokens": 64,
        "do_sample": false,
        "temperature": 0.0,
        "top_p": 1.0,
        "point_coordinate_max_decimals": 2,
        "point_stop_after_first": true
      }
    },
    {
      "task": "erqa_boxed",
      "kind": "nonpoint",
      "estimated_rows": 400,
      "model_arg_profile": "nonpoint_256",
      "runtime_env_profile": "nonpoint",
      "model_args": {"paligemma2_policy_prompt_style": "training_vqa_no_format"},
      "framework_seed": [0, 1234, 1234, 1234],
      "metric_key": "exact_match,flexible-extract",
      "score_field": "exact_match",
      "filter_contains": "MultiChoiceBoxedRegexFilter"
    },
    {
      "task": "roboafford",
      "kind": "point",
      "estimated_rows": 338,
      "model_arg_profile": "point",
      "runtime_env_profile": "point",
      "model_args": {"paligemma2_policy_prompt_style": "training_vqa_point_semantic_single"},
      "framework_seed": [0, 0, 0, 0],
      "generation_kwargs": {
        "max_new_tokens": 64,
        "do_sample": false,
        "temperature": 0.0,
        "top_p": 1.0,
        "point_coordinate_max_decimals": 2,
        "point_stop_after_first": true
      }
    },
    {
      "task": "vabench_point_box",
      "kind": "point",
      "estimated_rows": 300,
      "model_arg_profile": "point",
      "runtime_env_profile": "point",
      "model_args": {"paligemma2_policy_prompt_style": "training_vqa"},
      "framework_seed": [0, 0, 0, 0],
      "generation_kwargs": {
        "max_new_tokens": 64,
        "do_sample": true,
        "temperature": 0.15,
        "top_p": 0.9,
        "sampling_seed": 97,
        "point_coordinate_max_decimals": 8,
        "point_stop_after_first": true
      }
    },
    {
      "task": "sat_robust",
      "kind": "nonpoint",
      "estimated_rows": 150,
      "model_arg_profile": "nonpoint_256",
      "runtime_env_profile": "nonpoint",
      "model_args": {"paligemma2_policy_prompt_style": "training_vqa_no_format"},
      "framework_seed": [0, 1234, 1234, 1234],
      "metric_key": "accuracy,robust-choice-extract",
      "score_field": "accuracy",
      "filter_contains": "RobustChoiceFilter"
    },
    {
      "task": "robospatial-configuration-robust",
      "kind": "nonpoint",
      "estimated_rows": 123,
      "model_arg_profile": "nonpoint_256",
      "runtime_env_profile": "nonpoint",
      "model_args": {"paligemma2_policy_prompt_style": "training_vqa_no_format"},
      "framework_seed": [0, 1234, 1234, 1234],
      "metric_key": "accuracy,robust-yes-no-extract",
      "score_field": "accuracy",
      "filter_contains": "RobustYesNoFilter"
    },
    {
      "task": "robospatial-context",
      "kind": "point",
      "estimated_rows": 122,
      "model_arg_profile": "point",
      "runtime_env_profile": "point",
      "model_args": {"paligemma2_policy_prompt_style": "training_vqa_strict_point_2dp"},
      "framework_seed": [0, 0, 0, 0],
      "generation_kwargs": {
        "max_new_tokens": 64,
        "do_sample": true,
        "temperature": 0.15,
        "top_p": 0.9,
        "sampling_seed": 43,
        "point_coordinate_max_decimals": 2,
        "point_stop_after_first": true
      }
    },
    {
      "task": "robospatial-compatibility-no-format",
      "kind": "nonpoint",
      "estimated_rows": 105,
      "model_arg_profile": "nonpoint_64",
      "runtime_env_profile": "nonpoint",
      "model_args": {"paligemma2_policy_prompt_style": "training_vqa_no_format"},
      "framework_seed": [0, 1234, 1234, 1234],
      "metric_key": "accuracy,none",
      "score_field": "accuracy",
      "filter_contains": null
    },
    {
      "task": "refspatial-bench-location",
      "kind": "point",
      "estimated_rows": 100,
      "model_arg_profile": "point",
      "runtime_env_profile": "point",
      "model_args": {"paligemma2_policy_prompt_style": "training_vqa_point_single_2dp"},
      "framework_seed": [0, 0, 0, 0],
      "generation_kwargs": {
        "max_new_tokens": 64,
        "do_sample": true,
        "temperature": 0.15,
        "top_p": 0.9,
        "sampling_seed": 43,
        "point_coordinate_max_decimals": 2,
        "point_stop_after_first": true
      }
    },
    {
      "task": "refspatial-bench-placement",
      "kind": "point",
      "estimated_rows": 100,
      "model_arg_profile": "point",
      "runtime_env_profile": "point",
      "model_args": {"paligemma2_policy_prompt_style": "training_vqa_point_single_2dp"},
      "framework_seed": [0, 0, 0, 0],
      "generation_kwargs": {
        "max_new_tokens": 64,
        "do_sample": true,
        "temperature": 0.3,
        "top_p": 0.95,
        "sampling_seed": 97,
        "point_coordinate_max_decimals": 2,
        "point_stop_after_first": true
      }
    },
    {
      "task": "where2place_point",
      "kind": "point",
      "estimated_rows": 100,
      "model_arg_profile": "point",
      "runtime_env_profile": "point",
      "model_args": {"paligemma2_policy_prompt_style": "training_vqa_strict_point_2dp"},
      "framework_seed": [0, 0, 0, 0],
      "generation_kwargs": {
        "max_new_tokens": 64,
        "do_sample": true,
        "temperature": 0.15,
        "top_p": 0.9,
        "sampling_seed": 17,
        "point_coordinate_max_decimals": 2,
        "point_stop_after_first": true
      }
    },
    {
      "task": "refspatial-bench-unseen",
      "kind": "point",
      "estimated_rows": 77,
      "model_arg_profile": "point",
      "runtime_env_profile": "point",
      "model_args": {"paligemma2_policy_prompt_style": "training_vqa_point_single_2dp"},
      "framework_seed": [0, 0, 0, 0],
      "generation_kwargs": {
        "max_new_tokens": 64,
        "do_sample": true,
        "temperature": 0.15,
        "top_p": 0.9,
        "sampling_seed": 17,
        "point_coordinate_max_decimals": 2,
        "point_stop_after_first": true
      }
    }
  ],
  "aggregates": [
    {"name": "EmbSpatial", "tasks": ["embspatialbench_robust"]},
    {"name": "ERQA", "tasks": ["erqa_boxed"]},
    {"name": "CVBench", "tasks": ["cvbench_boxed_robust"]},
    {"name": "SAT", "tasks": ["sat_robust"]},
    {"name": "MetaVQA", "tasks": ["metavqa_eval_robust"]},
    {"name": "CRPE", "tasks": ["crpe_relation_robust"]},
    {"name": "RoboSpatial", "tasks": ["robospatial-compatibility-no-format", "robospatial-configuration-robust", "robospatial-context"]},
    {"name": "RefSpatial", "tasks": ["refspatial-bench-location", "refspatial-bench-placement", "refspatial-bench-unseen"]},
    {"name": "RoboAfford", "tasks": ["roboafford"]},
    {"name": "VABench", "tasks": ["vabench_point_box"]},
    {"name": "Where2Place", "tasks": ["where2place_point"]},
    {"name": "PartAffordance", "tasks": ["part_affordance"]},
    {"name": "RoboRefIt", "tasks": ["roborefit"]}
  ]
}
JSON
