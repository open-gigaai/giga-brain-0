"""MiMo-Embodied lmms_eval backend for the VLM inside Hy-VLA checkpoints."""

import json
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

from hy_vla_adapter import (  # noqa: E402
    HyVLAVQAAdapter,
    extract_answer_content,
    extract_choice,
    extract_yes_no,
    format_mimo_prompt,
)


POINT_TASKS = {
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
DEFAULT_RESIZE_MAX_PIXELS = 50176


def _flatten_visuals(value):
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        flattened = []
        for item in value:
            flattened.extend(_flatten_visuals(item))
        return flattened
    return [value]


def _point_label(prompt: str) -> str:
    for pattern in (
        r'(?is)based\s+on\s+the\s+description:\s*"([^"]+)"\s*,\s*locate',
        r'(?is)description:\s*"([^"]+)"',
    ):
        match = re.search(pattern, prompt)
        if match:
            return match.group(1).strip()
    return prompt.strip()


def _coordinate_size(image_size):
    max_pixels = int(
        os.getenv("QWEN_RESIZE_MAX_PIXELS", str(DEFAULT_RESIZE_MAX_PIXELS))
        or DEFAULT_RESIZE_MAX_PIXELS
    )
    if max_pixels <= 0:
        max_pixels = DEFAULT_RESIZE_MAX_PIXELS
    width, height = image_size
    resized_height, resized_width = smart_resize(
        height=height,
        width=width,
        max_pixels=max_pixels,
    )
    return resized_width, resized_height


def _normalize_points(answer: str, image_size, label: str) -> str:
    content = extract_answer_content(answer)
    matches = re.findall(
        r"(?:<point>\s*)?[\[\(]\s*(-?\d+(?:\.\d+)?)\s*,\s*"
        r"(-?\d+(?:\.\d+)?)\s*[\]\)]\s*(?:</point>)?",
        content,
        flags=re.IGNORECASE,
    )
    if not matches:
        return content

    values = [(float(x), float(y)) for x, y in matches]
    if all(0 <= x <= 1 and 0 <= y <= 1 for x, y in values):
        points = values
    elif all(0 <= x <= 1000 and 0 <= y <= 1000 for x, y in values):
        points = [[x / 1000, y / 1000] for x, y in values]
    else:
        width, height = _coordinate_size(image_size)
        points = [[x / width, y / height] for x, y in values]
    return json.dumps(
        [{"points": point, "label": label} for point in points],
        ensure_ascii=False,
    )


@register_model("hy_vla_vqa")
class HyVLAVQA(lmms):
    def __init__(
        self,
        checkpoint_path: str = "../model-repos/hy-vla/Hy-Embodied-0.5-VLA-RoboTwin",
        repo_path: str = "../model-repos/Hy-Embodied-0.5-VLA-main",
        batch_size: int = 1,
        device: str = "cuda",
        max_new_tokens: int = 128,
        temperature: float = 0.0,
        top_p: float = 1.0,
        thinking: bool = False,
        image_mode: str = "all",
        **kwargs,
    ) -> None:
        super().__init__()
        if kwargs:
            eval_logger.warning(f"Ignoring unsupported HyVLAVQA args: {sorted(kwargs)}")

        accelerator = Accelerator()
        if accelerator.num_processes > 1:
            device = f"cuda:{accelerator.local_process_index}"
        self.accelerator = accelerator
        self._rank = accelerator.local_process_index
        self._world_size = accelerator.num_processes
        self.batch_size_per_gpu = int(batch_size)
        if self.batch_size_per_gpu != 1:
            eval_logger.warning("HyVLAVQA runs one sample at a time; batch_size is ignored")

        self.adapter = HyVLAVQAAdapter(
            checkpoint_path=checkpoint_path,
            repo_path=repo_path,
            device=device,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            thinking=thinking,
            image_mode=image_mode,
        )

    @property
    def config(self):
        return self.adapter.config

    @property
    def tokenizer(self):
        return self.adapter.processor.tokenizer

    @property
    def model(self):
        return self.adapter.model

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

    def loglikelihood(self, requests: List[Instance]):
        raise NotImplementedError("loglikelihood is not implemented for HyVLAVQA")

    def generate_until(self, requests) -> List[str]:
        responses = [None] * len(requests)
        progress = tqdm(
            total=len(requests),
            disable=self.rank != 0,
            desc="Model Responding",
        )

        for index, request in enumerate(requests):
            contexts, gen_kwargs, doc_to_visual, doc_id, task, split = request.arguments
            gen_kwargs = dict(gen_kwargs)
            requested_tokens = int(
                gen_kwargs.pop("max_new_tokens", self.adapter.max_new_tokens)
            )
            max_new_tokens = min(requested_tokens, self.adapter.max_new_tokens)
            until = gen_kwargs.pop("until", [])
            if isinstance(until, str):
                until = [until]

            document = self.task_dict[task][split][doc_id]
            visual_output = doc_to_visual(document)
            images = [
                item.convert("RGB")
                for item in _flatten_visuals(visual_output)
                if isinstance(item, Image.Image)
            ]
            if not images:
                raise ValueError(
                    f"HyVLAVQA requires image input for task={task}, doc_id={doc_id}"
                )

            prompt, prompt_kind, choices = format_mimo_prompt(contexts)
            raw_answer = self.adapter.generate(
                images,
                prompt,
                max_new_tokens=max_new_tokens,
            )
            eval_logger.debug(
                f"HyVLAVQA raw response task={task} doc_id={doc_id}: {raw_answer!r}"
            )

            if prompt_kind == "yes_no":
                answer = extract_yes_no(raw_answer)
            elif task in POINT_TASKS or prompt_kind == "point":
                answer = _normalize_points(raw_answer, images[0].size, _point_label(contexts))
            elif prompt_kind == "choice":
                answer = extract_choice(raw_answer, choices=choices)
            else:
                answer = extract_answer_content(raw_answer)
            for term in until:
                if term:
                    answer = answer.split(term)[0]

            responses[index] = answer
            self.cache_hook.add_partial(
                "generate_until",
                (contexts, gen_kwargs),
                answer,
            )
            progress.update(1)

        progress.close()
        return responses

    def generate_until_multi_round(self, requests) -> List[str]:
        raise NotImplementedError("Multi-round generation is not implemented for HyVLAVQA")
