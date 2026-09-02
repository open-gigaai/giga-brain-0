# Copyright 2026 Xiaomi Corporation.

"""lmms-eval adapter for the Gigabrain0.7 PaliGemma2 policy checkpoint."""

import os
import sys
import types
from contextlib import nullcontext
from typing import List, Optional, Tuple, Union

import numpy as np
import torch
from accelerate import Accelerator, DistributedType
from loguru import logger as eval_logger
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from lmms_eval.api.instance import Instance
from lmms_eval.api.model import lmms
from lmms_eval.api.registry import register_model
from lmms_eval.models.gigabrain07_prompts import (
    format_affordance_adaptive_single_point_prompt,
    format_refit_boundary_interior_prompt,
    format_single_point_2dp_prompt,
    format_strict_point_2dp_prompt,
    format_training_vqa_prompt,
    strip_benchmark_output_format,
)


def _configure_paligemma2_determinism():
    workspace = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
    if workspace not in {":16:8", ":4096:8"}:
        raise RuntimeError(
            "deterministic PaliGemma2 inference requires "
            "CUBLAS_WORKSPACE_CONFIG=:4096:8 (or :16:8) before process start"
        )
    torch.use_deterministic_algorithms(True, warn_only=False)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    for name in (
        "allow_bf16_reduced_precision_reduction",
        "allow_fp16_reduced_precision_reduction",
    ):
        if hasattr(torch.backends.cuda.matmul, name):
            setattr(torch.backends.cuda.matmul, name, False)
    return {
        "cublas_workspace_config": workspace,
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
        "matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
        "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "sdpa_backend_during_paligemma2_generation": "math",
    }


def _prepare_giga_models_import(giga_models_dir):
    if str(giga_models_dir) not in sys.path:
        sys.path.insert(0, str(giga_models_dir))
    try:
        import xformers.ops  # noqa: F401
    except Exception:
        xformers = sys.modules.get("xformers") or types.ModuleType("xformers")
        operations = types.ModuleType("xformers.ops")
        xformers.ops = operations
        sys.modules["xformers"] = xformers
        sys.modules["xformers.ops"] = operations

    for name in (
        "giga_models.models.diffusion",
        "giga_models.models.wam",
        "giga_models.pipelines.diffusion",
        "giga_models.pipelines.text",
        "giga_models.pipelines.vision",
    ):
        if name not in sys.modules:
            sys.modules[name] = types.ModuleType(name)


def _collate(batch):
    idxs = [item[0] for item in batch]
    prompts = [item[1] for item in batch]
    images = [item[2] for item in batch]
    generate_kwargs = [item[3] for item in batch]
    until = [item[4] for item in batch]
    tasks = [item[5] for item in batch]
    return idxs, prompts, images, generate_kwargs, until, tasks


class Gigabrain07Dataset(Dataset):
    def __init__(self, requests, model):
        self.requests = requests
        self.model = model
        self._warned_gen_kwargs = set()

    def __len__(self):
        return len(self.requests)

    def __getitem__(self, idx):
        contexts, gen_kwargs, doc_to_visual, doc_id, task, split = (
            self.requests[idx].arguments
        )
        gen_kwargs = dict(gen_kwargs)
        max_new_tokens = min(
            int(gen_kwargs.pop("max_new_tokens", 128)),
            self.model.max_new_tokens_cap,
        )
        until = gen_kwargs.pop("until", None)
        if until is None:
            until = []
        elif isinstance(until, str):
            until = [until]
        generate_kwargs = {
            "max_new_tokens": max_new_tokens,
            "do_sample": self.model._as_bool(gen_kwargs.pop("do_sample", False)),
            "temperature": float(gen_kwargs.pop("temperature", 0.0)),
            "top_p": float(gen_kwargs.pop("top_p", 1.0)),
            "repetition_penalty": float(gen_kwargs.pop("repetition_penalty", 1.0)),
            "no_repeat_ngram_size": int(gen_kwargs.pop("no_repeat_ngram_size", 0)),
            "point_coordinate_max_decimals": int(
                gen_kwargs.pop("point_coordinate_max_decimals", 0)
            ),
            "point_stop_after_first": self.model._as_bool(
                gen_kwargs.pop("point_stop_after_first", False)
            ),
            "sampling_seed": (
                None
                if gen_kwargs.get("sampling_seed") in {None, "", "none", "None"}
                else int(gen_kwargs["sampling_seed"])
            ),
            "sampling_key": f"{task}\0{split}\0{doc_id}",
        }
        gen_kwargs.pop("sampling_seed", None)
        unsupported = tuple(sorted(gen_kwargs))
        if unsupported and unsupported not in self._warned_gen_kwargs:
            eval_logger.warning(
                "Ignoring unsupported generation kwargs: {}".format(list(unsupported))
            )
            self._warned_gen_kwargs.add(unsupported)

        visuals = [doc_to_visual(self.model.task_dict[task][split][doc_id])]
        image = None
        if None not in visuals:
            visuals = self.model.flatten(visuals)
            images = [item.convert("RGB") for item in visuals if isinstance(item, Image.Image)]
            if images:
                image = images[0]
                if len(images) > 1:
                    eval_logger.warning(
                        "Gigabrain0.7 uses the first image only; got {} for task={}".format(
                            len(images), task
                        )
                    )
        prompt = self.model._format_prompt(contexts, task)
        return idx, prompt, image, generate_kwargs, until, task


@register_model("gigabrain0.7")
class Gigabrain07VQA(lmms):
    def __init__(
        self,
        model_path: str,
        backbone: str = "paligemma2",
        giga_models_dir: str = "",
        tokenizer_model_path: str = "",
        fast_tokenizer_path: str = "",
        image_key: str = "observation.images.cam_high",
        batch_size: int = 1,
        device: str = "cuda",
        device_map: str = "",
        dtype: Optional[Union[str, torch.dtype]] = "bfloat16",
        max_new_tokens_cap: int = 256,
        paligemma2_weight_format: str = "policy",
        paligemma2_policy_low_cpu_mem_usage: Union[bool, str] = False,
        paligemma2_policy_force_lang: Union[bool, str] = False,
        paligemma2_policy_deterministic: Union[bool, str] = False,
        paligemma2_policy_prompt_style: str = "training_vqa",
        **kwargs,
    ) -> None:
        super().__init__()
        if kwargs:
            raise ValueError("unsupported Gigabrain0.7 model args: {}".format(sorted(kwargs)))
        if backbone.strip().lower() != "paligemma2":
            raise ValueError("Gigabrain0.7 supports only backbone=paligemma2")
        if paligemma2_weight_format.strip().lower() != "policy":
            raise ValueError("Gigabrain0.7 supports only paligemma2_weight_format=policy")
        if not giga_models_dir or not tokenizer_model_path or not fast_tokenizer_path:
            raise ValueError(
                "giga_models_dir, tokenizer_model_path, and fast_tokenizer_path are required"
            )

        accelerator = Accelerator()
        if accelerator.num_processes > 1 and not device_map:
            self._device = torch.device(f"cuda:{accelerator.local_process_index}")
        else:
            self._device = torch.device(device if torch.cuda.is_available() else "cpu")
        if isinstance(dtype, str) and dtype != "auto":
            dtype = getattr(torch, dtype)
        self.dtype = dtype
        self.image_key = image_key
        self.max_new_tokens_cap = int(max_new_tokens_cap)
        self.paligemma2_policy_force_lang = self._as_bool(
            paligemma2_policy_force_lang
        )
        self.paligemma2_policy_deterministic = self._as_bool(
            paligemma2_policy_deterministic
        )
        self.paligemma2_policy_prompt_style = (
            paligemma2_policy_prompt_style.strip().lower()
        )
        supported_prompts = {
            "training_vqa",
            "training_vqa_no_format",
            "training_vqa_strict_point_2dp",
            "training_vqa_point_single_2dp",
            "training_vqa_point_refit_boundary_interior",
            "training_vqa_point_semantic_single",
        }
        if self.paligemma2_policy_prompt_style not in supported_prompts:
            raise ValueError(
                "unsupported Gigabrain0.7 prompt style: {}".format(
                    self.paligemma2_policy_prompt_style
                )
            )

        self.deterministic_runtime_config = None
        if self.paligemma2_policy_deterministic:
            self.deterministic_runtime_config = _configure_paligemma2_determinism()

        _prepare_giga_models_import(giga_models_dir)
        try:
            import giga_models  # noqa: F401
        except ImportError as exc:
            raise ImportError(
                "Could not import giga_models from {}".format(giga_models_dir)
            ) from exc

        from lmms_eval.models import gigabrain07_paligemma2 as pg2

        self._pg2 = pg2
        checkpoint_dir = pg2.resolve_paligemma2_policy_ckpt_dir(model_path)
        if checkpoint_dir is None:
            raise ValueError(
                "No PaliGemma2 Gigabrain0.7 policy checkpoint found under {}".format(
                    model_path
                )
            )
        inference_config = pg2.load_paligemma2_inference_config(checkpoint_dir)
        self.policy = pg2.build_paligemma2_policy_from_checkpoint(
            checkpoint_dir,
            self._device,
            self.dtype,
            low_cpu_mem_usage=self._as_bool(
                paligemma2_policy_low_cpu_mem_usage
            ),
            giga_models_dir=giga_models_dir,
        )
        self.image_transform, self.prompt_transform = (
            pg2.build_paligemma2_transforms_from_inference_config(
                inference_config,
                tokenizer_model_path,
                fast_tokenizer_path,
                fast_token_vocab_mode=getattr(
                    self.policy.config, "fast_token_vocab_mode", None
                ),
            )
        )
        self._tokenizer = self.prompt_transform.paligemma_tokenizer
        self._paligemma2_fast_token_range = pg2.paligemma2_fast_token_range(
            self.prompt_transform
        )
        if self.paligemma2_policy_deterministic:
            self.deterministic_runtime_config = _configure_paligemma2_determinism()

        self.batch_size_per_gpu = int(batch_size)
        if accelerator.num_processes > 1:
            assert accelerator.distributed_type in {
                DistributedType.FSDP,
                DistributedType.MULTI_GPU,
                DistributedType.DEEPSPEED,
            }
            self._rank = accelerator.local_process_index
            self._world_size = accelerator.num_processes
        else:
            self._rank = 0
            self._world_size = 1
        self.accelerator = accelerator

    @staticmethod
    def _as_bool(value):
        if isinstance(value, bool):
            return value
        return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}

    @property
    def config(self):
        return getattr(self.policy, "config", None)

    @property
    def tokenizer(self):
        return self._tokenizer

    @property
    def model(self):
        return self.policy

    @property
    def batch_size(self):
        return self.batch_size_per_gpu

    @property
    def device(self):
        return self._device

    @property
    def rank(self):
        return self._rank

    @property
    def world_size(self):
        return self._world_size

    @staticmethod
    def flatten(values):
        return [item for value in values for item in value]

    def _format_prompt(self, prompt: str, task: str) -> str:
        style = self.paligemma2_policy_prompt_style
        if style == "training_vqa_no_format":
            prompt = strip_benchmark_output_format(prompt)
        elif style != "training_vqa":
            if not self._pg2.is_point_task(task):
                raise ValueError(f"{style} is only valid for point tasks: {task}")
            if style == "training_vqa_strict_point_2dp":
                prompt = format_strict_point_2dp_prompt(prompt)
            elif style == "training_vqa_point_single_2dp":
                prompt = format_single_point_2dp_prompt(prompt)
            elif style == "training_vqa_point_refit_boundary_interior":
                prompt = format_refit_boundary_interior_prompt(prompt)
            elif style == "training_vqa_point_semantic_single":
                if task != "roboafford":
                    raise ValueError(
                        "training_vqa_point_semantic_single is only valid for roboafford"
                    )
                prompt = format_affordance_adaptive_single_point_prompt(prompt)
        return format_training_vqa_prompt(prompt)

    @staticmethod
    def _chw01(pil_image: Image.Image) -> torch.Tensor:
        array = np.asarray(pil_image.convert("RGB"))
        return torch.from_numpy(array).permute(2, 0, 1).float() / 255.0

    def loglikelihood(self, requests: List[Instance]) -> List[Tuple[float, bool]]:
        raise NotImplementedError("Loglikelihood is not implemented for Gigabrain07VQA")

    @torch.no_grad()
    def _answer_one(
        self,
        prompt: str,
        image: Image.Image,
        max_new_tokens: int,
        task: str = "",
        do_sample: bool = False,
        temperature: float = 0.0,
        top_p: float = 1.0,
        repetition_penalty: float = 1.0,
        no_repeat_ngram_size: int = 0,
        point_coordinate_max_decimals: int = 0,
        point_stop_after_first: bool = False,
        sampling_seed: Optional[int] = None,
        sampling_key: str = "",
    ) -> str:
        image_tensor = self._chw01(image).to(self._device)
        suppress_token_ranges = None
        if self.paligemma2_policy_force_lang and not self._pg2.is_point_task(task):
            fast_range = self._paligemma2_fast_token_range
            suppress_token_ranges = [fast_range] if fast_range is not None else None
        attention_context = nullcontext()
        if self.paligemma2_policy_deterministic:
            from torch.nn.attention import SDPBackend, sdpa_kernel

            attention_context = sdpa_kernel([SDPBackend.MATH])
        with attention_context:
            return self._pg2.predict_vqa_paligemma2(
                self.policy,
                self.image_transform,
                self.prompt_transform,
                {self.image_key: image_tensor},
                prompt,
                self._device,
                max_new_tokens=max_new_tokens,
                suppress_token_ranges=suppress_token_ranges,
                do_sample=do_sample,
                temperature=temperature,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                no_repeat_ngram_size=no_repeat_ngram_size,
                point_coordinate_max_decimals=(
                    point_coordinate_max_decimals
                    if self._pg2.is_point_task(task)
                    else 0
                ),
                point_stop_after_first=(
                    point_stop_after_first if self._pg2.is_point_task(task) else False
                ),
                sampling_seed=sampling_seed,
                sampling_key=sampling_key,
            )

    def generate_until(self, requests) -> List[str]:
        responses = [None] * len(requests)
        progress = tqdm(
            total=len(requests),
            disable=(self.rank != 0),
            desc="Model Responding",
        )
        dataset = Gigabrain07Dataset(requests, self)
        dataloader = DataLoader(
            dataset,
            batch_size=1,
            shuffle=False,
            num_workers=0,
            collate_fn=_collate,
        )
        for idxs, prompts, images, generation, stop_terms, tasks in dataloader:
            for idx, prompt, image, kwargs, until, task in zip(
                idxs, prompts, images, generation, stop_terms, tasks
            ):
                if image is None:
                    raise ValueError(
                        "Gigabrain07VQA requires image input for MiMo embodied tasks"
                    )
                answer = self._answer_one(
                    prompt,
                    image,
                    int(kwargs["max_new_tokens"]),
                    task=task,
                    do_sample=bool(kwargs["do_sample"]),
                    temperature=float(kwargs["temperature"]),
                    top_p=float(kwargs["top_p"]),
                    repetition_penalty=float(kwargs["repetition_penalty"]),
                    no_repeat_ngram_size=int(kwargs["no_repeat_ngram_size"]),
                    point_coordinate_max_decimals=int(
                        kwargs["point_coordinate_max_decimals"]
                    ),
                    point_stop_after_first=bool(kwargs["point_stop_after_first"]),
                    sampling_seed=kwargs["sampling_seed"],
                    sampling_key=kwargs["sampling_key"],
                )
                for term in until:
                    if term:
                        answer = answer.split(term)[0]
                responses[idx] = answer
                self.cache_hook.add_partial(
                    "generate_until", (prompt, kwargs), answer
                )
            progress.update(len(idxs))
        progress.close()
        return responses

    def generate_until_multi_round(self, requests) -> List[str]:
        raise NotImplementedError(
            "Multi-round generation is not implemented for Gigabrain07VQA"
        )
