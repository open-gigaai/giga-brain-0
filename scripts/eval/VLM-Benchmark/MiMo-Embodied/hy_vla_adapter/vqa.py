"""Standalone VQA inference adapter for Hy-Embodied-0.5-VLA checkpoints."""

from __future__ import annotations

import gc
import os
import re
import sys
from pathlib import Path
from typing import Optional, Sequence

import torch
from PIL import Image


DEFAULT_HY_VLA_REPO = "../model-repos/Hy-Embodied-0.5-VLA-main"
DEFAULT_HY_VLA_CHECKPOINT = "../model-repos/hy-vla/Hy-Embodied-0.5-VLA-RoboTwin"
CHOICE_LETTERS = "ABCDEFGH"


def _ensure_repo_importable(repo_path: str) -> None:
    path = str(Path(repo_path).resolve())
    if path not in sys.path:
        sys.path.insert(0, path)


def _as_rgb_images(images) -> list[Image.Image]:
    if isinstance(images, (Image.Image, str, os.PathLike)):
        images = [images]
    if not isinstance(images, Sequence) or not images:
        raise ValueError("HyVLAVQAAdapter requires at least one image")

    converted = []
    for image in images:
        if isinstance(image, Image.Image):
            converted.append(image.convert("RGB"))
        elif isinstance(image, (str, os.PathLike)):
            converted.append(Image.open(image).convert("RGB"))
        else:
            raise TypeError(f"Unsupported image input: {type(image)!r}")
    return converted


def _choice_letters(prompt: str) -> list[str]:
    found = []
    seen = set()
    for pattern in (
        r"(?m)^\s*[\(\[]\s*([A-H])\s*[\)\]]",
        r"(?m)^\s*([A-H])\s*[\).:：]",
        r"(?<![A-Za-z])([A-H])\s*[\).:：]\s+",
    ):
        for match in re.finditer(pattern, prompt, flags=re.IGNORECASE):
            letter = match.group(1).upper()
            if letter not in seen:
                found.append(letter)
                seen.add(letter)
    return found


def extract_answer_content(text: str) -> str:
    if not text:
        return ""
    text = str(text).strip()
    answer_blocks = re.findall(r"(?is)<answer>\s*(.*?)\s*</answer>", text)
    if answer_blocks:
        return answer_blocks[-1].strip()
    text = re.sub(r"(?is)<think>.*?</think>", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def extract_choice(text: str, choices: str = CHOICE_LETTERS) -> str:
    answer = extract_answer_content(text)
    if not answer:
        return ""
    allowed = re.escape(choices.upper())
    for pattern in (
        rf"(?i)(?:final\s+answer|answer|option|choice|答案|选项)\s*"
        rf"(?:is|:|：)?\s*[\(\[]?\s*([{allowed}])\b",
        rf"(?i)^\s*[\(\[]?\s*([{allowed}])\s*[\)\].,，。:]?",
        rf"(?i)\b([{allowed}])\b",
    ):
        matches = re.findall(pattern, answer)
        if matches:
            return matches[-1].upper()
    return ""


def extract_yes_no(text: str) -> str:
    answer = extract_answer_content(text).lower()
    if re.search(r"\byes\b", answer):
        return "yes"
    if re.search(r"\bno\b", answer):
        return "no"
    choice = extract_choice(answer, choices="AB")
    if choice == "A":
        return "yes"
    if choice == "B":
        return "no"
    return ""


def _strip_benchmark_instructions(prompt: str) -> str:
    prompt = prompt.replace("<image>", "").strip()
    prompt = re.sub(
        r"(?is)\bput\s+your\s+final\s+answer\s+in\s+\$?\\boxed\{\}\$?\.?",
        "",
        prompt,
    )
    prompt = re.sub(
        r"(?is)\byour\s+answer\s+should\s+be\s+formatted\s+as\s+"
        r"\$?\\boxed\{.*?\}\$?\.?",
        "",
        prompt,
    )
    return re.sub(r"\n{3,}", "\n\n", prompt).strip()


def _is_multiple_choice(prompt: str) -> bool:
    return len(_choice_letters(prompt)) >= 2 and bool(
        re.search(r"(?is)\b(?:choices?|options?|select\s+from)\b", prompt)
        or re.search(r"(?m)^\s*(?:[\(\[]?[A-H][\)\].:：])\s+", prompt)
    )


def _is_yes_no(prompt: str) -> bool:
    return bool(
        re.search(r"(?is)\banswer\s+yes\s+or\s+no\b|\byes\s*/\s*no\b", prompt)
    )


def _is_point_request(prompt: str) -> bool:
    return bool(
        re.search(r"(?is)\blocate\s+points?\s+matching\s+the\s+description\b", prompt)
        or re.search(r"(?is)\blist\s+of\s+tuples\b", prompt)
    )


def _point_description(prompt: str) -> str:
    for pattern in (
        r'(?is)based\s+on\s+the\s+description:\s*"([^"]+)"',
        r"(?is)based\s+on\s+the\s+description:\s*'([^']+)'",
    ):
        match = re.search(pattern, prompt)
        if match:
            return match.group(1).strip()
    return prompt.strip()


def format_mimo_prompt(prompt: str) -> tuple[str, str, str]:
    """Return ``(formatted_prompt, kind, allowed_choices)``."""
    prompt = _strip_benchmark_instructions(prompt)
    if _is_multiple_choice(prompt):
        letters = "".join(_choice_letters(prompt)) or "ABCD"
        prompt = re.sub(
            r"(?is)\byou\s+can\s+only\s+answer\s+one\s+letter\s+from\s+"
            r"[A-H](?:,\s*[A-H])*(?:,\s*or\s*[A-H])?\.?",
            "",
            prompt,
        )
        prompt = re.sub(r"\n{3,}", "\n\n", prompt).strip()
        return f"{prompt}\nAnswer with only the option letter.", "choice", letters
    if _is_yes_no(prompt):
        prompt = re.sub(r"(?is)\banswer\s+yes\s+or\s+no\.?", "", prompt)
        prompt = re.sub(r"(?is)\byes\s*/\s*no\.?", "", prompt)
        prompt = re.sub(r"\n{3,}", "\n\n", prompt).strip()
        return (
            f"{prompt}\nSelect one answer:\n(A) yes\n(B) no\n"
            "Answer with only the option letter.",
            "yes_no",
            "AB",
        )
    if _is_point_request(prompt):
        description = _point_description(prompt)
        return (
            f'Identify one or more points matching this description: "{description}". '
            "Return only a list in this format: "
            "[<point>(x1, y1)</point>, <point>(x2, y2)</point>]. "
            "Use integer coordinates from 0 to 1000.",
            "point",
            "",
        )
    return prompt, "text", ""


class HyVLAVQAAdapter:
    """Extract and run the VLM tower contained in a Hy-VLA checkpoint."""

    def __init__(
        self,
        checkpoint_path: str = DEFAULT_HY_VLA_CHECKPOINT,
        repo_path: str = DEFAULT_HY_VLA_REPO,
        device: str = "cuda",
        max_new_tokens: int = 128,
        temperature: float = 0.0,
        top_p: float = 1.0,
        thinking: bool = False,
        image_mode: str = "all",
    ) -> None:
        if image_mode not in {"all", "first"}:
            raise ValueError("image_mode must be 'all' or 'first'")

        checkpoint = Path(checkpoint_path)
        required = (
            checkpoint / "config.json",
            checkpoint / "model.safetensors",
            checkpoint / "tokenizer.json",
            checkpoint / "preprocessor_config.json",
        )
        missing = [str(path) for path in required if not path.is_file()]
        if missing:
            raise FileNotFoundError(f"Incomplete Hy-VLA checkpoint; missing: {missing}")

        _ensure_repo_importable(repo_path)
        from hy_vla import HyVLA, HyVLAConfig
        from hy_vla.hunyuan_vl_mot import HunYuanVLMoTProcessor

        self.checkpoint_path = str(checkpoint.resolve())
        self.repo_path = str(Path(repo_path).resolve())
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.max_new_tokens = int(max_new_tokens)
        self.temperature = float(temperature)
        self.top_p = float(top_p)
        self.thinking = bool(thinking)
        self.image_mode = image_mode

        print(f"[hy-vla-vqa] loading processor: {self.checkpoint_path}", flush=True)
        print(f"[hy-vla-vqa] loading Hy-VLA checkpoint on CPU: {self.checkpoint_path}", flush=True)
        self.config = HyVLAConfig.from_pretrained(
            self.checkpoint_path,
            local_files_only=True,
        )
        policy = HyVLA.from_pretrained(
            self.checkpoint_path,
            config=self.config,
            local_files_only=True,
            map_location="cpu",
            strict=False,
        )
        policy.enable_video_encoder_if_needed()

        from transformers import AutoImageProcessor

        tokenizer = policy.language_tokenizer
        token_attrs = {
            "vision_start_token": "<｜hy_place▁holder▁no▁666｜>",
            "vision_end_token": "<｜hy_place▁holder▁no▁667｜>",
            "image_newline_token": "<｜hy_place▁holder▁no▁668｜>",
            "image_token": "<｜hy_place▁holder▁no▁669｜>",
            "video_token": "<｜hy_place▁holder▁no▁670｜>",
        }
        for name, token in token_attrs.items():
            setattr(tokenizer, name, token)
            setattr(tokenizer, f"{name}_id", tokenizer.convert_tokens_to_ids(token))
        tokenizer.padding_side = "left"
        image_processor = AutoImageProcessor.from_pretrained(
            self.checkpoint_path,
            local_files_only=True,
        )
        try:
            from transformers.models.qwen2_vl.video_processing_qwen2_vl import (
                Qwen2VLVideoProcessor,
            )

            video_processor = Qwen2VLVideoProcessor()
        except (ImportError, TypeError):
            from transformers.video_processing_utils import BaseVideoProcessor

            video_processor = BaseVideoProcessor()
        chat_template = checkpoint / "chat_template.jinja"
        template = chat_template.read_text(encoding="utf-8") if chat_template.is_file() else None
        self.processor = HunYuanVLMoTProcessor(
            image_processor=image_processor,
            tokenizer=tokenizer,
            video_processor=video_processor,
            chat_template=template,
        )

        self.model = policy.model.dual_tower.vlm
        policy.model.dual_tower.vlm = None
        del policy
        gc.collect()

        try:
            self.model.tie_weights()
        except (AttributeError, NotImplementedError):
            pass
        self.model.to(device=self.device, dtype=torch.bfloat16).eval()
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
        print(
            f"[hy-vla-vqa] VLM ready: {type(self.model).__name__} on {self.device}",
            flush=True,
        )

    def _messages(self, images: list[Image.Image], prompt: str):
        content = [{"type": "image", "image": image} for image in images]
        content.append({"type": "text", "text": prompt})
        return [{"role": "user", "content": content}]

    def generate(
        self,
        images,
        prompt: str,
        max_new_tokens: Optional[int] = None,
    ) -> str:
        pil_images = _as_rgb_images(images)
        if self.image_mode == "first":
            pil_images = pil_images[:1]

        template_kwargs = {
            "tokenize": True,
            "add_generation_prompt": True,
            "return_dict": True,
            "return_tensors": "pt",
            "enable_thinking": self.thinking,
        }
        try:
            inputs = self.processor.apply_chat_template(
                self._messages(pil_images, prompt),
                **template_kwargs,
            )
        except TypeError:
            template_kwargs.pop("enable_thinking")
            inputs = self.processor.apply_chat_template(
                self._messages(pil_images, prompt),
                **template_kwargs,
            )
        inputs = inputs.to(self.device)

        generation_kwargs = {
            "max_new_tokens": int(max_new_tokens or self.max_new_tokens),
            "use_cache": True,
            "do_sample": self.temperature > 0,
        }
        choices = _choice_letters(prompt)
        if choices:
            allowed_ids = []
            for letter in choices:
                for choice_text in (letter, f" {letter}"):
                    token_ids = self.processor.tokenizer.encode(
                        choice_text, add_special_tokens=False
                    )
                    if len(token_ids) == 1:
                        allowed_ids.append(token_ids[0])
            allowed_ids = sorted(set(allowed_ids))
            eos_token_id = self.processor.tokenizer.eos_token_id
            prompt_length = inputs["input_ids"].shape[1]

            def allowed_tokens(_batch_id, input_ids):
                return allowed_ids if input_ids.shape[-1] == prompt_length else [eos_token_id]

            generation_kwargs.update(
                max_new_tokens=2,
                prefix_allowed_tokens_fn=allowed_tokens,
            )
        if self.temperature > 0:
            generation_kwargs.update(
                temperature=self.temperature,
                top_p=self.top_p,
            )

        autocast_enabled = self.device.type == "cuda"
        with torch.inference_mode(), torch.autocast(
            device_type=self.device.type,
            dtype=torch.bfloat16,
            enabled=autocast_enabled,
        ):
            generated = self.model.generate(**inputs, **generation_kwargs)

        prompt_length = inputs["input_ids"].shape[1]
        output_ids = generated[:, prompt_length:]
        return self.processor.batch_decode(
            output_ids,
            skip_special_tokens=True,
        )[0].strip()
