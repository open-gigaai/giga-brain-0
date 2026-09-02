# Copyright (C) 2026 Xiaomi Corporation.
import argparse
import json
import re
from collections import defaultdict
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

from PIL import Image
from tqdm import tqdm


OPTION_TOKEN_RE = re.compile(r"^[A-Z]+$")
NUMBER_RE = re.compile(r"[-+]?(?:\d*\.\d+|\d+)")


def add_model_args(parser: argparse.ArgumentParser, default_output: str) -> None:
    parser.add_argument("--model-path", default="XiaomiRobotics/Xiaomi-Robotics-0-Pretrain")
    parser.add_argument("--output", default=default_output)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--max-length", type=int, default=None, help="Optional tokenizer max length for prompt encoding.")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument(
        "--attn-implementation",
        default="flash_attention_2",
        choices=["flash_attention_2", "sdpa", "eager"],
    )
    parser.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    parser.add_argument("--trust-remote-code", action=argparse.BooleanOptionalAction, default=True)


def torch_dtype(name: str):
    import torch

    return {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[name]


def patch_transformers_rope_default() -> None:
    import torch
    from transformers import modeling_rope_utils

    rope_init_functions = modeling_rope_utils.ROPE_INIT_FUNCTIONS
    if "default" in rope_init_functions:
        return

    for alias in ("base", "original", "standard"):
        if alias in rope_init_functions:
            rope_init_functions["default"] = rope_init_functions[alias]
            print(f"Patched Transformers RoPE registry: default -> {alias}")
            return

    def compute_default_rope_parameters(config=None, device=None, seq_len=None, **rope_kwargs):
        if config is not None:
            base = getattr(config, "rope_theta", 10000.0)
            partial_rotary_factor = getattr(config, "partial_rotary_factor", 1.0)
            head_dim = getattr(config, "head_dim", None)
            if head_dim is None:
                head_dim = config.hidden_size // config.num_attention_heads
            dim = int(head_dim * partial_rotary_factor)
        else:
            base = rope_kwargs.get("base", 10000.0)
            dim = rope_kwargs["dim"]

        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32, device=device) / dim))
        return inv_freq, 1.0

    rope_init_functions["default"] = compute_default_rope_parameters


def unwrap_vlm(model):
    return getattr(model, "vlm", model)


def load_model_and_processor(args):
    import torch
    from transformers import AutoModel, AutoProcessor

    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    patch_transformers_rope_default()
    model_kwargs = {
        "trust_remote_code": args.trust_remote_code,
        "attn_implementation": args.attn_implementation,
        "dtype": torch_dtype(args.dtype),
    }
    try:
        model = AutoModel.from_pretrained(args.model_path, **model_kwargs)
    except ValueError as exc:
        message = str(exc)
        unsupported_sdpa = (
            args.attn_implementation != "eager"
            and "scaled_dot_product_attention" in message
            and "attn_implementation=\"eager\"" in message
        )
        if not unsupported_sdpa:
            raise
        print(
            f"Model does not support attn_implementation={args.attn_implementation!r}; "
            "retrying with attn_implementation='eager'."
        )
        model_kwargs["attn_implementation"] = "eager"
        model = AutoModel.from_pretrained(args.model_path, **model_kwargs)
    model = unwrap_vlm(model).to(device).eval()
    processor = AutoProcessor.from_pretrained(args.model_path, trust_remote_code=args.trust_remote_code, use_fast=False)
    return model, processor, device


def image_from_bytes(data: bytes) -> Image.Image:
    return Image.open(BytesIO(data)).convert("RGB")


def image_from_field(field: Dict[str, Any], data_root: Optional[Path] = None) -> Image.Image:
    if isinstance(field, dict) and field.get("bytes") is not None:
        return image_from_bytes(field["bytes"])
    if isinstance(field, dict) and field.get("path"):
        path = Path(field["path"])
        if data_root is not None and not path.is_absolute():
            path = data_root / path
        return Image.open(path).convert("RGB")
    raise ValueError(f"Unsupported image field: {field}")


def mask_from_field(field: Any, data_root: Optional[Path] = None) -> Optional[Image.Image]:
    if field is None:
        return None
    if isinstance(field, dict) and field.get("bytes") is not None:
        return Image.open(BytesIO(field["bytes"])).convert("L")
    if isinstance(field, dict) and field.get("path"):
        path = Path(field["path"])
        if data_root is not None and not path.is_absolute():
            path = data_root / path
        return Image.open(path).convert("L")
    return None


def build_vlm_prompt(question: str, num_images: int = 1, answer_instruction: str = "Answer directly and concisely.") -> str:
    image_blocks = []
    for image_idx in range(num_images):
        if num_images == 1:
            image_blocks.append("<|vision_start|><|image_pad|><|vision_end|>")
        else:
            image_blocks.append(f"Image {image_idx + 1}:\n<|vision_start|><|image_pad|><|vision_end|>")
    return (
        "<|im_start|>user\n"
        + "\n".join(image_blocks)
        + "\n"
        + question.strip()
        + "\n"
        + answer_instruction.strip()
        + "<|im_end|>\n"
        + "<|im_start|>assistant\n"
    )


def normalize_answer(text: str) -> str:
    text = str(text).strip()
    text = re.sub(r"^<\|im_start\|>assistant\s*", "", text)
    text = re.sub(r"<\|im_end\|>.*$", "", text, flags=re.DOTALL)
    text = text.strip()
    text = text.splitlines()[0].strip() if text else ""
    return text.strip(" .,:;\"'`").lower()


def extract_options(text: str, max_letter: str = "K") -> Set[str]:
    text = normalize_answer(text).upper()
    tokens = re.split(r"[^A-Z]+", text)
    allowed = {chr(i) for i in range(ord("A"), ord(max_letter.upper()) + 1)}
    options: Set[str] = set()
    for token in tokens:
        if OPTION_TOKEN_RE.match(token) and all(ch in allowed for ch in token):
            options.update(token)
    return options


def canonical_options(options: Iterable[str]) -> str:
    return "".join(sorted(set(options)))


def choice_accuracy(prediction: str, answer: str, max_letter: str = "K") -> bool:
    return extract_options(prediction, max_letter=max_letter) == extract_options(answer, max_letter=max_letter)


def index_to_letter(index: int) -> str:
    return chr(ord("A") + int(index))


def parse_numbers(text: str) -> List[float]:
    return [float(x) for x in NUMBER_RE.findall(str(text))]


def parse_points(text: str, image_size: Optional[Tuple[int, int]] = None) -> List[Tuple[float, float]]:
    nums = parse_numbers(text)
    points = []
    width, height = image_size or (1, 1)
    for i in range(0, len(nums) - 1, 2):
        x, y = nums[i], nums[i + 1]
        if x > 1.0 or y > 1.0:
            x = x / max(width, 1)
            y = y / max(height, 1)
        points.append((x, y))
    return points


def point_mask_score(points: Sequence[Tuple[float, float]], mask: Optional[Image.Image]) -> Dict[str, Any]:
    if mask is None:
        return {"point_count": len(points), "hit_count": 0, "hit_rate": 0.0, "success": False}
    mask = mask.convert("L")
    width, height = mask.size
    hit_count = 0
    for x, y in points:
        px = min(max(int(round(x * (width - 1))), 0), width - 1)
        py = min(max(int(round(y * (height - 1))), 0), height - 1)
        if mask.getpixel((px, py)) > 0:
            hit_count += 1
    return {
        "point_count": len(points),
        "hit_count": hit_count,
        "hit_rate": hit_count / len(points) if points else 0.0,
        "success": hit_count > 0,
    }


def parse_bbox(text: str, image_size: Optional[Tuple[int, int]] = None) -> Optional[List[float]]:
    nums = parse_numbers(text)
    if len(nums) < 4:
        return None
    x1, y1, x2, y2 = nums[:4]
    width, height = image_size or (1, 1)
    if max(abs(x1), abs(y1), abs(x2), abs(y2)) <= 1.0:
        x1, x2 = x1 * width, x2 * width
        y1, y2 = y1 * height, y2 * height
    x1, x2 = sorted([x1, x2])
    y1, y2 = sorted([y1, y2])
    return [x1, y1, x2, y2]


def bbox_iou(pred: Optional[Sequence[float]], target: Sequence[float]) -> float:
    if pred is None:
        return 0.0
    px1, py1, px2, py2 = pred
    tx1, ty1, tx2, ty2 = target
    ix1, iy1 = max(px1, tx1), max(py1, ty1)
    ix2, iy2 = min(px2, tx2), min(py2, ty2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    p_area = max(0.0, px2 - px1) * max(0.0, py2 - py1)
    t_area = max(0.0, tx2 - tx1) * max(0.0, ty2 - ty1)
    denom = p_area + t_area - inter
    return inter / denom if denom > 0 else 0.0


def point_in_bbox(point: Tuple[float, float], bbox: Sequence[float], image_size: Tuple[int, int]) -> bool:
    width, height = image_size
    x, y = point
    if x <= 1.0 and y <= 1.0:
        x *= width
        y *= height
    x1, y1, x2, y2 = bbox
    x1, x2 = sorted([x1, x2])
    y1, y2 = sorted([y1, y2])
    return x1 <= x <= x2 and y1 <= y <= y2


def bbox_metrics(predictions: List[Dict[str, Any]], iou_threshold: float = 0.5, group_keys: Sequence[str] = ()) -> Dict[str, Any]:
    total = len(predictions)
    ious = [float(row.get("iou", 0.0)) for row in predictions]
    correct = sum(int(iou >= iou_threshold) for iou in ious)
    rows = [{**row, "correct": float(row.get("iou", 0.0)) >= iou_threshold} for row in predictions]
    metrics = {
        "num_examples": total,
        "correct": correct,
        "accuracy": correct / total if total else 0.0,
        "mean_iou": sum(ious) / total if total else 0.0,
        "iou_threshold": iou_threshold,
    }
    metrics.update(grouped_accuracy(rows, group_keys))
    return metrics


def generate_predictions(model, processor, device, records: List[Dict[str, Any]], args, desc: str) -> List[str]:
    import torch

    decoded: List[str] = []
    with torch.inference_mode():
        for start in tqdm(range(0, len(records), args.batch_size), desc=desc):
            batch = records[start : start + args.batch_size]
            prompts = [item["prompt"] for item in batch]
            flat_images = [image for item in batch for image in item["images"]]
            processor_kwargs = {
                "text": prompts,
                "images": flat_images,
                "videos": None,
                "padding": True,
                "return_tensors": "pt",
            }
            if args.max_length is not None:
                processor_kwargs.update({"truncation": True, "max_length": args.max_length})
            inputs = processor(**processor_kwargs).to(device)
            generated = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                use_cache=True,
            )
            generated = generated[:, inputs["input_ids"].shape[1] :]
            decoded.extend(processor.batch_decode(generated, skip_special_tokens=True, clean_up_tokenization_spaces=False))
    return decoded


def grouped_accuracy(rows: List[Dict[str, Any]], keys: Sequence[str]) -> Dict[str, Any]:
    metrics = {}
    for key in keys:
        buckets: Dict[str, Dict[str, int]] = defaultdict(lambda: {"total": 0, "correct": 0})
        for row in rows:
            bucket = buckets[str(row.get(key, "unknown"))]
            bucket["total"] += 1
            bucket["correct"] += int(bool(row.get("correct", False)))
        metrics[f"by_{key}"] = {
            name: {
                "num_examples": values["total"],
                "correct": values["correct"],
                "accuracy": values["correct"] / values["total"] if values["total"] else 0.0,
            }
            for name, values in sorted(buckets.items())
        }
    return metrics


def write_outputs(predictions: List[Dict[str, Any]], output: str, metrics: Dict[str, Any]) -> None:
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for row in predictions:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    metrics_path = output_path.with_suffix(".metrics.json")
    metrics.update({"output": str(output_path)})
    metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Predictions: {output_path}")
    print(f"Metrics: {metrics_path}")


def exact_metrics(predictions: List[Dict[str, Any]], group_keys: Sequence[str] = ()) -> Dict[str, Any]:
    total = len(predictions)
    correct = sum(int(bool(row.get("correct", False))) for row in predictions)
    metrics = {
        "num_examples": total,
        "correct": correct,
        "accuracy": correct / total if total else 0.0,
    }
    metrics.update(grouped_accuracy(predictions, group_keys))
    return metrics


def point_metrics(predictions: List[Dict[str, Any]], group_keys: Sequence[str] = ()) -> Dict[str, Any]:
    total = len(predictions)
    success = sum(int(bool(row.get("success", False))) for row in predictions)
    hit_rates = [float(row.get("hit_rate", 0.0)) for row in predictions]
    metrics = {
        "num_examples": total,
        "success_count": success,
        "success_rate": success / total if total else 0.0,
        "mean_point_hit_rate": sum(hit_rates) / total if total else 0.0,
    }
    metrics.update(grouped_accuracy([{**row, "correct": row.get("success", False)} for row in predictions], group_keys))
    return metrics
