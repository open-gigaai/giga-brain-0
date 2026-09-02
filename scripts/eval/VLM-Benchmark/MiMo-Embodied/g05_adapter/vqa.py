"""Shared G05 VQA adapter for MiMo-Embodied."""

from __future__ import annotations

import os
import re
import sys
import logging
from contextlib import contextmanager
from pathlib import Path
from typing import Optional

import torch
from PIL import Image
from torchvision import transforms


logger = logging.getLogger(__name__)

DEFAULT_G05_REPO = "../model-repos/GalaxeaVLA-main"
DEFAULT_G05_ROOT = "../model-repos/g05/G05-local"

G05_VARIANTS = {
    "base": ("g05-base", "checkpoints/model_state_dict.pt"),
    "droid": ("g05-droid", "checkpoints/model_state_dict.pt"),
    "libero": ("g05-libero", "model.pt"),
    "robotwin20": ("g05-robotwin20", "checkpoints/model_state_dict.pt"),
    "so101": ("g05-so101", "checkpoints/model_state_dict.pt"),
}


def _ensure_g05_importable(repo_path: str) -> None:
    src_path = os.path.join(repo_path, "src")
    for path in (src_path, repo_path):
        if path and path not in sys.path:
            sys.path.insert(0, path)


@contextmanager
def _working_directory(path: str):
    previous = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous)


def _resolve_g05_root(model_root: str) -> Path:
    root = Path(model_root)
    if (root / "g05-base").is_dir():
        return root
    if (root / "G05-local" / "g05-base").is_dir():
        return root / "G05-local"
    raise FileNotFoundError(
        f"Could not find g05-base under {root}. Expected OpenGalaxea/G05 layout."
    )


def _register_repo_oc_load(omega_conf, repo_path: str) -> None:
    repo_root = Path(repo_path)
    legacy_paths = {
        "configs/data/_mixtures/robotwin.yaml": "configs/data/robotwin.yaml",
    }

    def load_from_repo(path: str, key: Optional[str] = None):
        load_path = Path(path)
        if not load_path.is_absolute():
            load_path = repo_root / load_path
        if not load_path.exists() and path in legacy_paths:
            load_path = repo_root / legacy_paths[path]
        cfg = omega_conf.load(load_path)
        if key in (None, ""):
            return cfg
        return omega_conf.select(cfg, key)

    omega_conf.register_new_resolver("oc.load", load_from_repo, replace=True)


def _to_rgb_image(image) -> Image.Image:
    if isinstance(image, Image.Image):
        return image.convert("RGB")
    if isinstance(image, (str, os.PathLike)):
        return Image.open(image).convert("RGB")
    raise TypeError(f"Unsupported image input: {type(image)!r}")


def _first_image(image) -> Image.Image:
    if isinstance(image, (list, tuple)):
        if not image:
            raise ValueError("G05VQA received an empty image list")
        return _to_rgb_image(image[0])
    return _to_rgb_image(image)


def _as_token_id(value) -> Optional[int]:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _summarize_token_ids(token_ids: list[int]) -> str:
    if not token_ids:
        return "[]"
    ranges = []
    start = prev = int(token_ids[0])
    for token_id in token_ids[1:]:
        token_id = int(token_id)
        if token_id == prev + 1:
            prev = token_id
            continue
        ranges.append((start, prev))
        start = prev = token_id
    ranges.append((start, prev))
    parts = [str(a) if a == b else f"{a}-{b}" for a, b in ranges]
    return f"{len(token_ids)} ids: " + ", ".join(parts[:8]) + (" ..." if len(parts) > 8 else "")


DEFAULT_CHOICE_LETTERS = "ABCDEFGH"


def _choice_letters_from_prompt(prompt: str) -> list[str]:
    letters: list[str] = []
    seen = set()
    for pattern in (
        r"(?m)^\s*[\(\[]\s*([A-H])\s*[\)\]]",
        r"(?m)^\s*([A-H])\s*[\).:：]",
        r"(?<![A-Za-z])([A-H])\s*[\).:：]\s+",
    ):
        for match in re.finditer(pattern, prompt, flags=re.IGNORECASE):
            letter = match.group(1).upper()
            if letter not in seen:
                letters.append(letter)
                seen.add(letter)
    return letters if len(letters) >= 2 else list("ABCD")


def extract_choice(text: str, choices: str = DEFAULT_CHOICE_LETTERS) -> str:
    if not text:
        return ""
    text = re.sub(r"<think>.*?</think>", " ", str(text), flags=re.DOTALL | re.IGNORECASE)
    choice_class = re.escape(choices.upper())
    for pattern in (
        rf"(?i)(?:final\s+answer|answer|option|choice|答案|选项)\s*(?:is|:|：)?\s*[\(\[]?\s*([{choice_class}])\b",
        rf"(?i)^\s*[\(\[]?\s*([{choice_class}])\s*[\)\].,，。:]?",
        rf"(?i)\b([{choice_class}])\b",
    ):
        matches = re.findall(pattern, text)
        if matches:
            return matches[-1].upper()
    return ""


def extract_yes_no(text: str) -> str:
    if not text:
        return ""
    normalized = str(text).strip().lower()
    if re.search(r"\byes\b", normalized):
        return "yes"
    if re.search(r"\bno\b", normalized):
        return "no"
    choice = extract_choice(normalized, choices="AB")
    if choice == "A":
        return "yes"
    if choice == "B":
        return "no"
    return ""


def _strip_common_prompt_boilerplate(prompt: str) -> str:
    prompt = prompt.strip()
    prompt = prompt.replace("<image>", "").strip()
    prompt = re.sub(
        r"(?is)\bput\s+your\s+final\s+answer\s+in\s+\$?\\boxed\{\}\$?\.?",
        "",
        prompt,
    )
    prompt = re.sub(
        r"(?is)\byour\s+answer\s+should\s+be\s+formatted\s+as\s+\$?\\boxed\{.*?\}\$?\.?",
        "",
        prompt,
    )
    return re.sub(r"\n{3,}", "\n\n", prompt).strip()


def _has_explicit_choices(prompt: str) -> bool:
    return len(_choice_letters_from_prompt(prompt)) >= 2 and bool(
        re.search(r"(?is)\b(?:choices?|options?|select\s+from)\b", prompt)
        or re.search(r"(?m)^\s*(?:[\(\[]?[A-H][\)\].:：])\s+", prompt)
    )


def _is_yes_no_prompt(prompt: str) -> bool:
    return bool(re.search(r"(?is)\banswer\s+yes\s+or\s+no\b|\byes\s*/\s*no\b", prompt))


def _is_point_prompt(prompt: str) -> bool:
    return bool(
        re.search(r"(?is)\blocate\s+points?\s+matching\s+the\s+description\b", prompt)
        or re.search(r"(?is)\blist\s+of\s+tuples\b", prompt)
    )


def _description_from_point_prompt(prompt: str) -> str:
    match = re.search(r'(?is)based\s+on\s+the\s+description:\s*"([^"]+)"', prompt)
    if match:
        return match.group(1).strip()
    match = re.search(r"(?is)based\s+on\s+the\s+description:\s*'([^']+)'", prompt)
    if match:
        return match.group(1).strip()
    return prompt.strip()


def format_mcq_prompt(prompt: str) -> str:
    prompt = _strip_common_prompt_boilerplate(prompt)
    prompt = re.sub(
        r"(?is)\byou\s+can\s+only\s+answer\s+one\s+letter\s+from\s+[A-H](?:,\s*[A-H])*(?:,\s*or\s*[A-H])?\.?",
        "",
        prompt,
    )
    prompt = re.sub(
        r"(?is)\breturn\s+only\s+one\s+final\s+(?:option\s+)?letter\s*:?\s*[A-H](?:,\s*[A-H])*(?:,\s*or\s*[A-H])?\.?",
        "",
        prompt,
    )
    prompt = re.sub(r"\n{3,}", "\n\n", prompt).strip()
    return f"{prompt}\nAnswer with only the option letter."


def format_yes_no_as_mcq_prompt(prompt: str) -> str:
    prompt = _strip_common_prompt_boilerplate(prompt)
    prompt = re.sub(r"(?is)\banswer\s+yes\s+or\s+no\.?", "", prompt)
    prompt = re.sub(r"(?is)\byes\s*/\s*no\.?", "", prompt)
    prompt = re.sub(r"\n{3,}", "\n\n", prompt).strip()
    return f"{prompt}\nSelect from the following choices.\n(A) yes\n(B) no\nAnswer with only the option letter."


def format_point_prompt(prompt: str) -> str:
    description = _description_from_point_prompt(_strip_common_prompt_boilerplate(prompt))
    return (
        f'Identify one or more points matching this description: "{description}". '
        "Your answer should be formatted as a list of tuples, i.e. [(x1, y1), (x2, y2), ...]."
    )


def format_mimo_prompt(prompt: str) -> str:
    prompt = _strip_common_prompt_boilerplate(prompt)
    if _has_explicit_choices(prompt):
        return format_mcq_prompt(prompt)
    if _is_yes_no_prompt(prompt):
        return format_yes_no_as_mcq_prompt(prompt)
    if _is_point_prompt(prompt):
        return format_point_prompt(prompt)
    return prompt


class G05VQAAdapter:
    def __init__(
        self,
        model_root: str = DEFAULT_G05_ROOT,
        variant: str = "base",
        repo_path: str = DEFAULT_G05_REPO,
        device: str = "cuda",
        max_new_tokens: int = 512,
        temperature: float = 0.0,
        top_p: float = 1.0,
        top_k: int = 1,
        use_tmp_cache: bool = True,
    ) -> None:
        _ensure_g05_importable(repo_path)

        from omegaconf import OmegaConf

        from g05.utils.checkpoint.checkpoint_utils import load_model_from_checkpoint
        from g05.utils.checkpoint.ckpt_utils import load_config_from_run_dir
        from g05.utils.config.config_resolvers import register_default_resolvers

        register_default_resolvers()
        _register_repo_oc_load(OmegaConf, repo_path)

        variant = variant.removeprefix("g05-").lower()
        if variant not in G05_VARIANTS:
            raise ValueError(
                f"Unknown G05 variant {variant!r}; expected one of {sorted(G05_VARIANTS)}"
            )
        run_dir_name, checkpoint_relpath = G05_VARIANTS[variant]

        env_cached_root = os.environ.get("G05_LOCAL_ROOT")
        if env_cached_root:
            root = _resolve_g05_root(env_cached_root)
        elif use_tmp_cache and (Path("/tmp/g05-local") / run_dir_name / checkpoint_relpath).is_file():
            root = Path("/tmp/g05-local")
        else:
            root = _resolve_g05_root(model_root)

        self.root = root
        self.variant = variant
        self.run_dir = root / run_dir_name
        self.ckpt = self.run_dir / checkpoint_relpath
        self.processor_dir = root / "qwen3_5_2b_base_processor"
        self.action_tokenizer = root / "action_tokenizer.pt"
        config_path = self.run_dir / ".hydra" / "config.yaml"
        for required_path in (config_path, self.ckpt, self.processor_dir, self.action_tokenizer):
            if not required_path.exists():
                raise FileNotFoundError(
                    f"Missing file required by G05 variant {variant!r}: {required_path}"
                )
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.max_new_tokens = int(max_new_tokens)
        self.temperature = float(temperature)
        self.top_p = float(top_p)
        self.top_k = int(top_k)

        overrides = [
            f"model.model_arch.hf_processor_path={self.processor_dir}",
            f"model.tokenizer.vq_config.ckpt_dir={self.action_tokenizer}",
            f"model.model_arch.AT_CONFIG.ckpt_dir={self.action_tokenizer}",
            f"tokenizer.vq_config.ckpt_dir={self.action_tokenizer}",
            "model.model_weights_to_bf16=false",
        ]
        # Exported fine-tune configs contain oc.load references relative to the
        # GalaxeaVLA repository (for example configs/data/parts_meta/*.yaml).
        with _working_directory(repo_path):
            cfg = load_config_from_run_dir(self.run_dir, str(self.ckpt), overrides)
        self.cfg = cfg
        self.camera_size_config = OmegaConf.to_container(
            cfg.model.model_arch.camera_size_config, resolve=True
        )

        print(f"[g05] variant: {self.variant}", flush=True)
        print(f"[g05] loading checkpoint: {self.ckpt}", flush=True)
        self.policy = load_model_from_checkpoint(
            cfg.model.model_arch,
            cfg.ckpt_path,
            device=str(self.device),
            extra_prefixes=["normalizer."],
            use_meta_device=True,
            eval_mode=True,
        )
        self.policy.apply_fp32_params()
        if hasattr(self.policy, "action_tokenizer"):
            self.policy.action_tokenizer.to(str(self.device))
        self.policy.eval()
        print("[g05] model ready", flush=True)

    def _image_to_tensor(self, image: Image.Image) -> torch.Tensor:
        h, w = self.camera_size_config.get("exterior", [256, 256])
        tensor = transforms.Compose(
            [
                transforms.Resize((int(h), int(w))),
                transforms.PILToTensor(),
            ]
        )(image)
        tensor = tensor.to(dtype=torch.float32) / 255.0
        return tensor.to(self.device)

    def _images_to_tensor(self, image) -> torch.Tensor:
        if isinstance(image, (list, tuple)):
            if not image:
                raise ValueError("G05VQA received an empty image list")
            images = [_to_rgb_image(img) for img in image]
        else:
            images = [_to_rgb_image(image)]
        tensors = [self._image_to_tensor(img) for img in images]
        return torch.stack(tensors, dim=0).unsqueeze(0)

    def _empty_retry_ignore_token_ids(self) -> list[int]:
        ids: set[int] = set()
        model_config = getattr(self.policy, "model_config", None)
        processor = getattr(self.policy, "processor", None)
        tokenizer = getattr(processor, "tokenizer", None)
        for source, attrs in (
            (model_config, ("eos_token_id", "pad_token_id")),
            (processor, ("eos_token_id", "pad_token_id")),
            (tokenizer, ("eos_token_id", "pad_token_id")),
        ):
            for attr in attrs:
                token_id = _as_token_id(getattr(source, attr, None))
                if token_id is not None:
                    ids.add(token_id)
        action_tokenizer = getattr(self.policy, "action_tokenizer", None)
        for attr in ("action_token_begin_idx", "action_token_end_idx"):
            if _as_token_id(getattr(action_tokenizer, attr, None)) is None:
                break
        else:
            begin = int(getattr(action_tokenizer, "action_token_begin_idx"))
            end = int(getattr(action_tokenizer, "action_token_end_idx"))
            if 0 <= begin < end and end - begin <= 100000:
                ids.update(range(begin, end))
        return sorted(ids)

    def _infer_vqa_once(
        self,
        pixel_values: torch.Tensor,
        prompt: str,
        max_new_tokens: int,
        **generation_overrides,
    ) -> str:
        with torch.no_grad(), torch.autocast(
            "cuda", dtype=torch.bfloat16, enabled=self.device.type == "cuda"
        ):
            answers = self.policy.infer_vqa(
                prompt,
                pixel_values,
                num_tokens_to_generate=max_new_tokens,
                temperature=self.temperature,
                top_p=self.top_p,
                top_k=self.top_k,
                **generation_overrides,
            )
        if not answers:
            return ""
        return str(answers[0]).strip()

    def generate(self, image, prompt: str, max_new_tokens: Optional[int] = None) -> str:
        pixel_values = self._images_to_tensor(image)
        token_budget = int(max_new_tokens or self.max_new_tokens)
        answer = self._infer_vqa_once(pixel_values, prompt, token_budget)
        if answer or os.getenv("G05_VQA_RETRY_EMPTY", "1").lower() in {"0", "false", "no"}:
            return answer

        ignore_token_ids = self._empty_retry_ignore_token_ids()
        if not ignore_token_ids:
            return answer

        retry_budget = max(
            1,
            min(token_budget, int(os.getenv("G05_VQA_EMPTY_RETRY_MAX_NEW_TOKENS", "64"))),
        )
        logger.warning(
            "G05VQA produced an empty answer; retrying with ignore_token_ids=%s, max_new_tokens=%s",
            _summarize_token_ids(ignore_token_ids),
            retry_budget,
        )
        retry_answer = self._infer_vqa_once(
            pixel_values,
            prompt,
            retry_budget,
            ignore_token_ids=ignore_token_ids,
        )
        if not retry_answer:
            logger.warning("G05VQA empty-answer retry also produced an empty answer")
        return retry_answer

    def answer_choice(self, image, prompt: str) -> dict:
        formatted = format_mcq_prompt(prompt)
        raw = self.generate(image, formatted)
        choices = "".join(_choice_letters_from_prompt(formatted))
        choice = extract_choice(raw, choices=choices)
        if not choice and raw:
            first = raw.strip()[:1].upper()
            choice = first if first in choices else ""
        return {"answer": choice, "raw_answer": raw}
