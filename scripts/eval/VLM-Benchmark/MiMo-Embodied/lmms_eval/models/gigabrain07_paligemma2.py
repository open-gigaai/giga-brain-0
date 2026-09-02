# Copyright 2026 Xiaomi Corporation.

"""PaliGemma2 policy loading and decoding helpers for Gigabrain0.7."""

import copy
import hashlib
import inspect
import json
import os
import re
import sys
import types

import torch
import torch.nn as nn

from lmms_eval.models.gigabrain07_prompts import ensure_prompt_fits

_POINT_TASKS = {
    "where2place_point",
    "roboafford",
    "part_affordance",
    "roborefit",
    "vabench_point_box",
    "refspatial-bench-location",
    "refspatial-bench-placement",
    "refspatial-bench-unseen",
    "robospatial-context",
}


def _ensure_package_stub(name, path):
    if name in sys.modules:
        return
    module = types.ModuleType(name)
    module.__path__ = [path]
    sys.modules[name] = module


def _ensure_torch_dynamo_stub():
    if "torch._dynamo" in sys.modules:
        return
    module = types.ModuleType("torch._dynamo")

    def allow_in_graph(obj):
        return obj

    def is_compiling():
        return False

    module.allow_in_graph = allow_in_graph
    module.is_compiling = is_compiling
    sys.modules["torch._dynamo"] = module
    if "torch" in sys.modules:
        setattr(sys.modules["torch"], "_dynamo", module)


def _prepare_lightweight_giga_models_import(giga_models_dir=None):
    giga_models_dir = (
        giga_models_dir
        or os.environ.get("GIGA_MODELS_DIR")
    )
    if not giga_models_dir:
        raise ValueError("giga_models_dir is required")
    package_root = os.path.join(giga_models_dir, "giga_models")
    if not os.path.isdir(package_root):
        return
    if giga_models_dir not in sys.path:
        sys.path.insert(0, giga_models_dir)
    _ensure_torch_dynamo_stub()
    _ensure_package_stub("giga_models", package_root)
    _ensure_package_stub("giga_models.models", os.path.join(package_root, "models"))
    _ensure_package_stub("giga_models.models.vla", os.path.join(package_root, "models", "vla"))
    _ensure_package_stub(
        "giga_models.models.vla.giga_brain_0",
        os.path.join(package_root, "models", "vla", "giga_brain_0"),
    )
    _ensure_package_stub("giga_models.utils", os.path.join(package_root, "utils"))
    _ensure_package_stub("giga_models.pipelines", os.path.join(package_root, "pipelines"))
    _ensure_package_stub("giga_models.pipelines.vla", os.path.join(package_root, "pipelines", "vla"))
    _ensure_package_stub(
        "giga_models.pipelines.vla.giga_brain_0",
        os.path.join(package_root, "pipelines", "vla", "giga_brain_0"),
    )


def _import_gigabrain07_policy(giga_models_dir=None):
    _prepare_lightweight_giga_models_import(giga_models_dir)
    from giga_models.models.vla.giga_brain_0.modeling_giga_brain_0 import GigaBrain0Policy

    return GigaBrain0Policy


def is_point_task(task):
    return str(task) in _POINT_TASKS


def _load_json(path):
    with open(path) as f:
        return json.load(f)


def _candidate_ckpt_dirs(model_path):
    candidates = [model_path]
    candidates.extend(os.path.join(model_path, subdir) for subdir in ("model_ema", "model"))
    seen = set()
    for candidate in candidates:
        candidate = os.path.normpath(candidate)
        if candidate in seen:
            continue
        seen.add(candidate)
        yield candidate


def resolve_paligemma2_policy_ckpt_dir(model_path):
    """Return a GigaBrain0Policy Paligemma2 checkpoint dir, if model_path points to one."""
    for ckpt_dir in _candidate_ckpt_dirs(model_path):
        cfg_path = os.path.join(ckpt_dir, "config.json")
        if not os.path.isfile(cfg_path):
            continue
        cfg = _load_json(cfg_path)
        vlm_type = str(cfg.get("vlm_type", "")).lower()
        class_name = str(cfg.get("_class_name", "")).lower()
        has_policy_weights = any(
            os.path.isfile(os.path.join(ckpt_dir, filename))
            for filename in ("diffusion_pytorch_model.bin", "diffusion_pytorch_model.safetensors")
        )
        if class_name == "gigabrain0policy" and vlm_type in {"paligemma", "paligemma2"} and has_policy_weights:
            return ckpt_dir
    return None


def load_paligemma2_inference_config(ckpt_dir):
    print(f"[gigabrain0.7] loading inference_config from {ckpt_dir}", flush=True)
    from giga_models.pipelines.vla.giga_brain_0.giga_brain_0_utils import load_inference_config

    cfg = load_inference_config(ckpt_dir)
    print("[gigabrain0.7] loaded inference_config", flush=True)
    return cfg


def build_paligemma2_policy_from_checkpoint(
    ckpt_dir, device, dtype=torch.bfloat16, low_cpu_mem_usage=False, giga_models_dir=None
):
    """Load a trained Paligemma2 GigaBrain0Policy checkpoint directly."""
    GigaBrain0Policy = _import_gigabrain07_policy(giga_models_dir)

    print(
        f"[gigabrain0.7] from_pretrained start: {ckpt_dir}, "
        f"low_cpu_mem_usage={low_cpu_mem_usage}",
        flush=True,
    )
    policy = GigaBrain0Policy.from_pretrained(
        ckpt_dir,
        low_cpu_mem_usage=low_cpu_mem_usage,
        torch_dtype=dtype if dtype != "auto" else None,
    )
    print("[gigabrain0.7] from_pretrained done", flush=True)
    if dtype is not None and dtype != "auto":
        print(f"[gigabrain0.7] converting dtype={dtype}", flush=True)
        policy.to(dtype=dtype)
    print(f"[gigabrain0.7] moving policy to device={device}", flush=True)
    nn.Module.to(policy, device)
    policy.eval()
    print("[gigabrain0.7] policy ready", flush=True)
    return policy


def _constructor_kwargs(cls, cfg):
    valid_keys = set(inspect.signature(cls.__init__).parameters)
    valid_keys.discard("self")
    return {key: value for key, value in cfg.items() if key in valid_keys}


def build_paligemma2_transforms_from_inference_config(
    inference_cfg,
    tokenizer_model_path,
    fast_tokenizer_path,
    fast_token_vocab_mode=None,
):
    """Build eval transforms from the checkpoint sidecar, overriding only local paths."""
    from giga_models.pipelines.vla.giga_brain_0.giga_brain_0_utils import (
        ImageTransform,
        PromptTokenizerTransform,
    )

    print("[gigabrain0.7] building transforms from inference_config", flush=True)
    image_cfg = copy.deepcopy(inference_cfg.get("image_cfg") or {})
    image_cfg["is_train"] = False
    image_cfg["enable_image_aug"] = False
    image_cfg.setdefault("vlm_type", "paligemma2")
    image_transform = ImageTransform(**_constructor_kwargs(ImageTransform, image_cfg))

    prompt_cfg = copy.deepcopy(inference_cfg.get("prompt_cfg") or {})
    prompt_cfg["is_train"] = False
    prompt_cfg["tokenizer_model_path"] = tokenizer_model_path
    prompt_cfg["fast_tokenizer_path"] = fast_tokenizer_path
    prompt_cfg.setdefault("max_length", 300)
    prompt_cfg.setdefault("vlm_type", "paligemma2")
    if fast_token_vocab_mode is not None:
        prompt_cfg["fast_token_vocab_mode"] = fast_token_vocab_mode
    prompt_transform = PromptTokenizerTransform(
        **_constructor_kwargs(PromptTokenizerTransform, prompt_cfg)
    )
    print("[gigabrain0.7] transforms ready", flush=True)
    return image_transform, prompt_transform


def paligemma2_fast_token_range(prompt_transform):
    """Return the legacy tail-mode PaliGemma FAST-token id range.

    Tail mode remaps FAST ids into a slice of the existing tokenizer vocab:
    ``vocab_size - 1 - fast_skip_tokens - fast_id``. That slice overlaps the
    visual loc-token area, so callers should suppress it only for non-point QA.
    """
    if getattr(prompt_transform, "fast_token_vocab_mode", None) != "tail":
        return None
    tokenizer = prompt_transform.paligemma_tokenizer
    vocab_size = tokenizer.vocab_size
    text_token_length = getattr(prompt_transform, "text_token_length", None)
    if text_token_length is not None:
        vocab_size = min(vocab_size, int(text_token_length))
    fast_skip_tokens = int(getattr(prompt_transform, "fast_skip_tokens", 128))
    fast_vocab_size = int(getattr(prompt_transform, "fast_vocab_size", 0) or 0)
    if fast_vocab_size <= 0 and getattr(prompt_transform, "encode_action_input", False):
        fast_vocab_size = 2048
    if fast_vocab_size <= 0:
        return None
    max_token_id = int(vocab_size) - 1 - fast_skip_tokens
    min_token_id = max_token_id - fast_vocab_size + 1
    if min_token_id < 0 or max_token_id < min_token_id:
        return None
    return min_token_id, max_token_id + 1


def _tokenize_prompt(tokenizer, max_length, prompt_text, device):
    """Left-pad the plain prompt to max_length (matches pipeline._tokenize_autoregressive_prompt)."""
    out = tokenizer(prompt_text, add_special_tokens=True, return_tensors="pt", truncation=False)
    ids = out["input_ids"].squeeze(0)
    mask = out["attention_mask"].squeeze(0)
    ensure_prompt_fits(ids.shape[0], max_length)
    padded = tokenizer.pad(
        {"input_ids": ids.tolist(), "attention_mask": mask.tolist()},
        padding="max_length", padding_side="left", max_length=max_length, return_tensors="pt",
    )
    lang_tokens = padded["input_ids"].squeeze(0).to(dtype=torch.int32, device=device)
    lang_masks = padded["attention_mask"].squeeze(0).to(dtype=torch.bool, device=device)
    return lang_tokens, lang_masks


def _vqa_state_generation_kwargs(policy, batch_size, device):
    """Match training's state contract for language-only VQA samples."""
    config = getattr(policy, "config", None)
    if getattr(config, "state_input_mode", "prompt") != "proprio_anchor":
        return {}
    return {
        "proprioception_present": torch.zeros(
            batch_size, dtype=torch.bool, device=device
        )
    }


def _sampling_generator(sampling_seed, sampling_key, device):
    if sampling_seed is None:
        return None
    payload = f"{int(sampling_seed)}\0{sampling_key}".encode("utf-8")
    derived_seed = int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")
    derived_seed &= (1 << 63) - 1
    generator = torch.Generator(device=device)
    generator.manual_seed(derived_seed)
    return generator


def _select_next_token(
    step_logits,
    do_sample=False,
    temperature=0.0,
    top_p=1.0,
    generator=None,
):
    if not do_sample:
        return int(torch.argmax(step_logits).item())
    if temperature <= 0:
        raise ValueError("temperature must be > 0 when do_sample=true")
    if not 0 < top_p <= 1:
        raise ValueError("top_p must be in (0, 1]")

    probabilities = torch.softmax(step_logits / temperature, dim=-1)
    if top_p < 1:
        sorted_probabilities, sorted_indices = torch.sort(
            probabilities, descending=True
        )
        remove = sorted_probabilities.cumsum(dim=-1) > top_p
        remove[1:] = remove[:-1].clone()
        remove[0] = False
        sorted_probabilities[remove] = 0
        sorted_probabilities /= sorted_probabilities.sum()
        sampled_index = torch.multinomial(
            sorted_probabilities, num_samples=1, generator=generator
        )
        return int(sorted_indices[sampled_index].item())
    return int(
        torch.multinomial(probabilities, num_samples=1, generator=generator).item()
    )


def _apply_repetition_penalty(step_logits, generated_tokens, penalty=1.0):
    """Apply the standard decoder-only repetition penalty in-place on a clone."""
    penalty = float(penalty)
    if penalty <= 0:
        raise ValueError("repetition_penalty must be > 0")
    if penalty == 1.0 or not generated_tokens:
        return step_logits

    adjusted = step_logits.clone()
    token_ids = torch.tensor(
        sorted(set(generated_tokens)), dtype=torch.long, device=adjusted.device
    )
    scores = adjusted[token_ids]
    adjusted[token_ids] = torch.where(scores < 0, scores * penalty, scores / penalty)
    return adjusted


def _no_repeat_ngram_banned_tokens(generated_tokens, ngram_size):
    """Return tokens that would recreate an n-gram in the generated suffix."""
    ngram_size = int(ngram_size)
    if ngram_size < 0:
        raise ValueError("no_repeat_ngram_size must be >= 0")
    if ngram_size == 0 or len(generated_tokens) + 1 < ngram_size:
        return set()
    if ngram_size == 1:
        return set(generated_tokens)

    prefix = tuple(generated_tokens[-(ngram_size - 1):])
    banned = set()
    for start in range(len(generated_tokens) - ngram_size + 1):
        if tuple(generated_tokens[start:start + ngram_size - 1]) == prefix:
            banned.add(generated_tokens[start + ngram_size - 1])
    return banned


_COORDINATE_NUMBER_PATTERN = r"-?\d+(?:\.(?P<{fraction}>\d+))?"
_POINT_VALUE_FIELD_RE = re.compile(
    r"(?:"
    r'\"\s*(?:points?|labels?|current_point|current_description|current_bbox|location|items|'
    r'<loc\d+>(?:_point(?:_[a-z0-9]+)*)?)\"?\s*:'
    r"|<points?>"
    r")",
    re.IGNORECASE,
)
_POINT_FIRST_COORDINATE_RE = re.compile(
    rf"^\s*{_COORDINATE_NUMBER_PATTERN.format(fraction='x_fraction')}\s*$"
)
_POINT_SECOND_COORDINATE_RE = re.compile(
    rf"^\s*{_COORDINATE_NUMBER_PATTERN.format(fraction='x_fraction')}\s*,\s*"
    rf"{_COORDINATE_NUMBER_PATTERN.format(fraction='y_fraction')}\s*$"
)
_POINT_MISSING_COMMA_SECOND_COORDINATE_RE = re.compile(
    rf"^\s*{_COORDINATE_NUMBER_PATTERN.format(fraction='x_fraction')}\s+"
    rf"{_COORDINATE_NUMBER_PATTERN.format(fraction='y_fraction')}\s*$"
)
_POINT_THIRD_COORDINATE_RE = re.compile(
    rf"^\s*{_COORDINATE_NUMBER_PATTERN.format(fraction='x_fraction')}\s*,\s*"
    rf"{_COORDINATE_NUMBER_PATTERN.format(fraction='y_fraction')}\s*,\s*"
    rf"{_COORDINATE_NUMBER_PATTERN.format(fraction='z_fraction')}\s*$"
)
_POINT_FOURTH_COORDINATE_RE = re.compile(
    rf"^\s*{_COORDINATE_NUMBER_PATTERN.format(fraction='x_fraction')}\s*,\s*"
    rf"{_COORDINATE_NUMBER_PATTERN.format(fraction='y_fraction')}\s*,\s*"
    rf"{_COORDINATE_NUMBER_PATTERN.format(fraction='z_fraction')}\s*,\s*"
    rf"{_COORDINATE_NUMBER_PATTERN.format(fraction='w_fraction')}\s*$"
)
_POINT_COMPLETE_COORDINATE_RE = re.compile(
    r"[\[\(]\s*-?\d+(?:\.\d+)?\s*,\s*-?\d+(?:\.\d+)?\s*[\]\)]"
)
_POINT_COMPLETE_BOX_RE = re.compile(
    r"(?:"
    r"\[\s*-?\d+(?:\.\d+)?\s*,\s*-?\d+(?:\.\d+)?\s*,\s*"
    r"-?\d+(?:\.\d+)?\s*,\s*-?\d+(?:\.\d+)?\s*\]"
    r"|\(\s*-?\d+(?:\.\d+)?\s*,\s*-?\d+(?:\.\d+)?\s*,\s*"
    r"-?\d+(?:\.\d+)?\s*,\s*-?\d+(?:\.\d+)?\s*\)"
    r")"
)


def _active_point_value(text):
    """Return an explicit point value, excluding ordinary prose and string fields.

    ``label`` is accepted only when its value starts with an array/tuple. This
    covers malformed point outputs that place coordinates under the label key
    without treating the normal textual label as coordinates.
    """
    text = str(text)
    fields = list(_POINT_VALUE_FIELD_RE.finditer(text))
    if fields:
        value = text[fields[-1].end():]
        if not re.match(r"^\s*[\[\(]", value):
            return ""
    elif re.match(
        r"^\s*(?:(?:<answer>)?\s*<points?>\s*)?[\[\(]",
        text,
        flags=re.IGNORECASE,
    ):
        value = text
    else:
        return ""
    return value


def _active_point_segment(text):
    """Return the unfinished innermost point segment, excluding ordinary prose."""
    segment, _ = _active_point_state(text)
    return segment


def _active_point_state(text):
    value = _active_point_value(text)
    if not value:
        return "", ""

    square_open = value.rfind("[")
    round_open = value.rfind("(")
    last_open = max(square_open, round_open)
    if last_open < 0:
        return "", ""
    return value[last_open + 1:], value[last_open]


def _has_complete_point(text):
    return bool(_POINT_COMPLETE_COORDINATE_RE.search(str(text)))


def _has_complete_box(text):
    return bool(_POINT_COMPLETE_BOX_RE.search(str(text)))


def _complete_point_counts(text):
    text = str(text)
    return (
        len(_POINT_COMPLETE_COORDINATE_RE.findall(text)),
        len(_POINT_COMPLETE_BOX_RE.findall(text)),
    )


def _point_coordinate_delimiter_action(text, max_decimals=0):
    """Choose a grammar delimiter once an unfinished coordinate reaches its precision cap."""
    max_decimals = int(max_decimals)
    if max_decimals < 0:
        raise ValueError("point_coordinate_max_decimals must be >= 0")
    if max_decimals == 0:
        return None

    segment, opener = _active_point_state(text)
    if not segment:
        return None
    fourth = _POINT_FOURTH_COORDINATE_RE.fullmatch(segment)
    if fourth:
        fraction = fourth.group("w_fraction") or ""
        if len(fraction) >= max_decimals:
            return ")" if opener == "(" else "]"
        return None
    third = _POINT_THIRD_COORDINATE_RE.fullmatch(segment)
    if third:
        fraction = third.group("z_fraction") or ""
        return "," if len(fraction) >= max_decimals else None
    second = _POINT_SECOND_COORDINATE_RE.fullmatch(segment)
    if second:
        fraction = second.group("y_fraction") or ""
        if len(fraction) >= max_decimals:
            return ")" if opener == "(" else "]"
        return None
    missing_comma_second = _POINT_MISSING_COMMA_SECOND_COORDINATE_RE.fullmatch(
        segment
    )
    if missing_comma_second:
        fraction = missing_comma_second.group("y_fraction") or ""
        if len(fraction) >= max_decimals:
            return ")" if opener == "(" else "]"
        return None
    first = _POINT_FIRST_COORDINATE_RE.fullmatch(segment)
    if first:
        fraction = first.group("x_fraction") or ""
        return "," if len(fraction) >= max_decimals else None
    return None


def _single_point_arity_delimiter_action(
    text, next_token_text, end_of_sequence=False
):
    """Close a point before the model starts a third coordinate."""
    segment, opener = _active_point_state(text)
    if not segment or not _POINT_SECOND_COORDINATE_RE.fullmatch(segment):
        return None
    if not end_of_sequence and not re.match(
        r"^\s*[,\]\)]", str(next_token_text)
    ):
        return None
    return ")" if opener == "(" else "]"


def _encode_forced_delimiter(tokenizer, delimiter):
    encoded = tokenizer(delimiter, add_special_tokens=False)
    token_ids = encoded["input_ids"]
    if token_ids and isinstance(token_ids[0], (list, tuple)):
        token_ids = token_ids[0]
    token_ids = [int(token_id) for token_id in token_ids]
    if len(token_ids) != 1:
        raise ValueError(
            f"Forced delimiter {delimiter!r} must encode to exactly one token; "
            f"got {token_ids}"
        )
    return token_ids


def _token_is_decimal_digit(tokenizer, token_id):
    decoded = tokenizer.decode([int(token_id)], skip_special_tokens=True)
    return bool(re.fullmatch(r"\d", decoded))


@torch.no_grad()
def predict_vqa_paligemma2(policy, image_transform, prompt_transform, images_dict, prompt_text,
                           device, max_new_tokens=64, suppress_token_ranges=None,
                           do_sample=False, temperature=0.0, top_p=1.0,
                           repetition_penalty=1.0, no_repeat_ngram_size=0,
                           point_coordinate_max_decimals=0,
                           point_stop_after_first=False,
                           sampling_seed=None, sampling_key=""):
    """Single-image VQA, greedy by default with optional temperature sampling."""
    tokenizer = prompt_transform.paligemma_tokenizer
    for key in images_dict:
        images_dict[key] = images_dict[key].to(device)
    images, img_masks, _ = image_transform(images_dict)
    images = [im[None, ...] for im in images]
    img_masks = [m[None, ...] for m in img_masks]
    lang_tokens, lang_masks = _tokenize_prompt(tokenizer, prompt_transform.max_length, prompt_text, device)
    lang_tokens = lang_tokens[None, ...]
    lang_masks = lang_masks[None, ...]

    state_kwargs = _vqa_state_generation_kwargs(policy, lang_tokens.shape[0], device)
    next_logits, state = policy.init_lang_generation(
        images, img_masks, lang_tokens, lang_masks, **state_kwargs
    )
    eos_raw = tokenizer.eos_token_id
    eos_ids = set(eos_raw) if isinstance(eos_raw, (list, tuple)) else {eos_raw}
    out_tokens = []
    sampling_generator = _sampling_generator(
        sampling_seed, sampling_key, next_logits.device
    )
    recovered_completion_counts = None
    for _ in range(max_new_tokens):
        delimiter = None
        if (
            point_coordinate_max_decimals
            or point_stop_after_first
            or recovered_completion_counts is not None
        ):
            decoded = tokenizer.decode(out_tokens, skip_special_tokens=True)
            complete_counts = _complete_point_counts(decoded)
            if point_stop_after_first and complete_counts[0]:
                break
            if recovered_completion_counts is not None and any(
                current > previous
                for current, previous in zip(
                    complete_counts, recovered_completion_counts
                )
            ):
                break
        if point_coordinate_max_decimals:
            delimiter = _point_coordinate_delimiter_action(
                decoded, point_coordinate_max_decimals
            )
        step_logits = next_logits[0].float()
        step_logits = _apply_repetition_penalty(
            step_logits, out_tokens, repetition_penalty
        )
        banned_tokens = _no_repeat_ngram_banned_tokens(
            out_tokens, no_repeat_ngram_size
        )
        if banned_tokens:
            step_logits = step_logits.clone()
            step_logits[list(banned_tokens)] = -torch.inf
        if suppress_token_ranges:
            step_logits = step_logits.clone()
            for start, end in suppress_token_ranges:
                step_logits[int(start): int(end)] = -torch.inf
        tok = _select_next_token(
            step_logits,
            do_sample=do_sample,
            temperature=temperature,
            top_p=top_p,
            generator=sampling_generator,
        )
        arity_delimiter = None
        if point_stop_after_first:
            arity_delimiter = _single_point_arity_delimiter_action(
                decoded,
                tokenizer.decode([int(tok)], skip_special_tokens=True),
                end_of_sequence=tok in eos_ids,
            )
        forced_delimiter = None
        if arity_delimiter is not None:
            forced_delimiter = arity_delimiter
        elif delimiter is not None and _token_is_decimal_digit(tokenizer, tok):
            if recovered_completion_counts is None:
                recovered_completion_counts = _complete_point_counts(decoded)
            forced_delimiter = delimiter
        if forced_delimiter is not None:
            tok = _encode_forced_delimiter(tokenizer, forced_delimiter)[0]
            if not 0 <= tok < next_logits.shape[-1]:
                raise ValueError(
                    f"Forced delimiter token {tok} is outside the "
                    f"language logits vocabulary ({next_logits.shape[-1]})"
                )
        if tok in eos_ids:
            break
        out_tokens.append(tok)
        inp = torch.tensor([[tok]], dtype=torch.long, device=device)
        next_logits, state = policy.next_lang_logits(state, inp)
    return tokenizer.decode(out_tokens, skip_special_tokens=True).strip()
