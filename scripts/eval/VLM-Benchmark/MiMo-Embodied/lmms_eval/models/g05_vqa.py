# Copyright 2026 Xiaomi Corporation.

"""G05 VQA adapter for MiMo-Embodied lmms_eval."""

import os
import re
import sys
from pathlib import Path
from typing import List

from accelerate import Accelerator
from loguru import logger as eval_logger
from PIL import Image
from tqdm import tqdm

from lmms_eval.api.instance import Instance
from lmms_eval.api.model import lmms
from lmms_eval.api.registry import register_model
from lmms_eval.models.model_utils.qwen.vision_process import smart_resize


BENCHMARK_ROOT = os.environ.get(
    "GIGA_BENCHMARK_ROOT", str(Path(__file__).resolve().parents[2])
)
if BENCHMARK_ROOT not in sys.path:
    sys.path.insert(0, BENCHMARK_ROOT)

from g05_adapter import G05VQAAdapter, extract_yes_no, format_mimo_prompt  # noqa: E402


G05_POINT_TASKS = {
    "part_affordance",
    "refspatial-bench-location",
    "refspatial-bench-placement",
    "refspatial-bench-unseen",
    "roboafford",
    "roborefit",
    "robospatial-context",
    "vabench_point_box",
    "where2place_point",
}

G05_RESIZE_MAX_PIXELS = 50176


def _coordinate_target_size(image_size):
    resize_max_pixels = int(
        os.getenv("QWEN_RESIZE_MAX_PIXELS", str(G05_RESIZE_MAX_PIXELS))
        or G05_RESIZE_MAX_PIXELS
    )
    width, height = image_size
    if resize_max_pixels <= 0:
        resize_max_pixels = G05_RESIZE_MAX_PIXELS
    resized_height, resized_width = smart_resize(
        height=int(height), width=int(width), max_pixels=resize_max_pixels
    )
    return resized_width, resized_height


def _format_point_coordinates(points) -> str:
    return "[" + ", ".join(f"({x}, {y})" for x, y in points) + "]"


def _normalize_point_answer(answer: str, image_size) -> str:
    pairs = re.findall(
        r"[\[\(]\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*[\]\)]",
        answer,
    )
    if not pairs:
        return answer
    values = [(float(x), float(y)) for x, y in pairs]

    if all(0.0 <= x <= 1.0 and 0.0 <= y <= 1.0 for x, y in values):
        points = values
    else:
        width, height = _coordinate_target_size(image_size)
        points = [[x / width, y / height] for x, y in values]
    return _format_point_coordinates(points)


@register_model("g05_vqa")
class G05VQA(lmms):
    def __init__(
        self,
        model_root: str = "../model-repos/g05/G05-local",
        variant: str = "base",
        repo_path: str = "../model-repos/GalaxeaVLA-main",
        batch_size: int = 1,
        device: str = "cuda",
        max_new_tokens: int = 512,
        temperature: float = 0.0,
        top_p: float = 1.0,
        top_k: int = 1,
        **kwargs,
    ) -> None:
        super().__init__()
        if kwargs:
            eval_logger.warning(f"Ignoring unsupported G05VQA model args: {sorted(kwargs)}")

        accelerator = Accelerator()
        if accelerator.num_processes > 1:
            device = f"cuda:{accelerator.local_process_index}"

        self._rank = accelerator.local_process_index
        self._world_size = accelerator.num_processes
        self.batch_size_per_gpu = int(batch_size)
        if self.batch_size_per_gpu != 1:
            eval_logger.warning("G05VQA currently runs one sample at a time; batch_size is ignored.")
        self.accelerator = accelerator
        self.adapter = G05VQAAdapter(
            model_root=model_root,
            variant=variant,
            repo_path=repo_path,
            device=device,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
        )

    @property
    def config(self):
        return self.adapter.cfg

    @property
    def tokenizer(self):
        return self.adapter.policy.processor.tokenizer

    @property
    def model(self):
        return self.adapter.policy

    @property
    def batch_size(self):
        return self.batch_size_per_gpu

    @property
    def device(self):
        return self.adapter.device

    @property
    def rank(self):
        return self._rank

    @property
    def world_size(self):
        return self._world_size

    def flatten(self, input):
        new_list = []
        for i in input:
            for j in i:
                new_list.append(j)
        return new_list

    def loglikelihood(self, requests: List[Instance]):
        raise NotImplementedError("Loglikelihood is not implemented for G05VQA")

    def generate_until(self, requests) -> List[str]:
        res = [None] * len(requests)
        pbar = tqdm(total=len(requests), disable=(self.rank != 0), desc="Model Responding")

        for idx, request in enumerate(requests):
            contexts, gen_kwargs, doc_to_visual, doc_id, task, split = request.arguments
            gen_kwargs = dict(gen_kwargs)
            requested_max_new_tokens = int(
                gen_kwargs.pop("max_new_tokens", self.adapter.max_new_tokens)
            )
            max_new_tokens = min(requested_max_new_tokens, self.adapter.max_new_tokens)
            until = gen_kwargs.pop("until", None)
            if until is None:
                until = []
            elif isinstance(until, str):
                until = [until]

            visuals = [doc_to_visual(self.task_dict[task][split][doc_id])]
            image_input = None
            primary_image = None
            if None not in visuals:
                visuals = self.flatten(visuals)
                images = [visual.convert("RGB") for visual in visuals if isinstance(visual, Image.Image)]
                if images:
                    primary_image = images[0]
                    image_input = images if len(images) > 1 else primary_image
            if image_input is None:
                raise ValueError(f"G05VQA requires image input for task={task}, doc_id={doc_id}")

            prompt = format_mimo_prompt(contexts)
            answer = self.adapter.generate(image_input, prompt, max_new_tokens=max_new_tokens)
            if "answer yes or no" in contexts.lower() or "yes/no" in contexts.lower():
                answer = extract_yes_no(answer)
            elif task in G05_POINT_TASKS:
                answer = _normalize_point_answer(answer, primary_image.size)
            for term in until:
                if term:
                    answer = answer.split(term)[0]
            res[idx] = answer
            self.cache_hook.add_partial("generate_until", (contexts, gen_kwargs), answer)
            pbar.update(1)

        pbar.close()
        return res

    def generate_until_multi_round(self, requests) -> List[str]:
        raise NotImplementedError("Multi-round generation is not implemented for G05VQA")
