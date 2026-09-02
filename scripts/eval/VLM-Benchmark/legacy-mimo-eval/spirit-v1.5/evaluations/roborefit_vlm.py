import argparse
import io
import json
import re
import sys
import time
from dataclasses import dataclass
from dataclasses import fields
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
from PIL import Image
from safetensors.torch import load_file as safe_load_file
from transformers import AutoProcessor

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from model import SpiritVLAConfig
from model import SpiritVLAPolicy


DEFAULT_CKPT_PATH = "model-repos/spirit-v1.5"
DEFAULT_DATASET_PATH = (
    "datasets/public_datasets/VLM/benchmarks/roborefit/"
    "llava_json/roborefit_test_llava.jsonl"
)
DEFAULT_SUFFIX = (
    "Please output the normalized position coordinates of the target object, "
    "arranged from top left to bottom right, such as [x1, y1, x2, y2]."
)


@dataclass
class RoboRefItSample:
    sample_id: int
    question_id: int
    ref_exp: str
    image_path: str | None
    image: Image.Image
    bbox: list[float]
    normalized_bbox: list[float]
    original_image_size: tuple[int, int]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate the Spirit-v1.5 Qwen3-VL backbone on RoboRefIt.",
    )
    parser.add_argument("--ckpt-path", default=DEFAULT_CKPT_PATH, help="Spirit-v1.5 checkpoint directory.")
    parser.add_argument("--dataset-path", default=DEFAULT_DATASET_PATH, help="RoboRefIt dataset directory.")
    parser.add_argument(
        "--processor-path",
        default=None,
        help="Tokenizer/processor path. Defaults to the backbone in ckpt config.",
    )
    parser.add_argument(
        "--backbone-path",
        default=None,
        help="Override the backbone path used to construct Qwen3-VL before loading Spirit weights.",
    )
    parser.add_argument("--output-dir", default="outputs/roborefit_vlm", help="Directory for JSONL and summary outputs.")
    parser.add_argument("--split", default="test", help="Dataset split name when loading through datasets.")
    parser.add_argument("--max-samples", type=int, default=None, help="Optional cap for quick smoke tests.")
    parser.add_argument("--batch-size", type=int, default=1, help="Generation batch size.")
    parser.add_argument("--device", default="cuda", help="Device for inference, e.g. cuda, cuda:0, or cpu.")
    parser.add_argument(
        "--dtype",
        choices=("auto", "float32", "float16", "bfloat16"),
        default="bfloat16",
        help="Dtype used after loading the model.",
    )
    parser.add_argument("--attn-implementation", default=None, help="Override attention implementation from config.")
    parser.add_argument("--max-new-tokens", type=int, default=32, help="Maximum generated answer tokens.")
    parser.add_argument("--temperature", type=float, default=0.0, help="Generation temperature. 0 means greedy.")
    parser.add_argument("--top-p", type=float, default=1.0, help="Nucleus sampling value when temperature > 0.")
    parser.add_argument("--num-beams", type=int, default=1, help="Beam count for generation.")
    parser.add_argument(
        "--system-prompt",
        default="You are a helpful visual grounding assistant.",
        help="System prompt used for bbox generation.",
    )
    parser.add_argument(
        "--question-suffix",
        default=DEFAULT_SUFFIX,
        help="Instruction appended to each referring expression.",
    )
    parser.add_argument(
        "--coordinate-scale",
        choices=("auto", "pixels", "normalized", "qwen_1000"),
        default="auto",
        help="Expected coordinate scale in model output.",
    )
    parser.add_argument(
        "--max-image-pixels",
        type=int,
        default=262144,
        help="Resize each image to at most this many pixels before processing. Use 0 to disable resizing.",
    )
    parser.add_argument("--seed", type=int, default=0, help="Torch random seed.")
    parser.add_argument("--resume", action="store_true", help="Skip samples already present in predictions.jsonl.")
    parser.add_argument("--trust-remote-code", action="store_true", help="Pass trust_remote_code=True to AutoProcessor.")
    return parser.parse_args()


def load_config(ckpt_path: Path) -> dict[str, Any]:
    config_path = ckpt_path / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"config.json not found in checkpoint directory: {ckpt_path}")
    return json.loads(config_path.read_text())


def torch_dtype(name: str) -> torch.dtype | None:
    if name == "auto":
        return None
    return {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }[name]


def build_spirit_config(raw_config: dict[str, Any], device: str) -> SpiritVLAConfig:
    valid_keys = {field.name for field in fields(SpiritVLAConfig)}
    filtered = {key: value for key, value in raw_config.items() if key in valid_keys}
    filtered["device"] = device
    return SpiritVLAConfig(**filtered)


def load_policy(ckpt_path: Path, raw_config: dict[str, Any], device: torch.device) -> SpiritVLAPolicy:
    config = build_spirit_config(raw_config, str(device))
    model = SpiritVLAPolicy(config)
    weight_path = ckpt_path / "model.safetensors"
    if not weight_path.exists():
        raise FileNotFoundError(f"model.safetensors not found in checkpoint directory: {ckpt_path}")
    load_device = str(device) if device.type == "cuda" else "cpu"
    state_dict = safe_load_file(str(weight_path), device=load_device)
    model.load_state_dict(state_dict, strict=True)
    return model


def decode_image(image_obj: Any) -> Image.Image:
    if isinstance(image_obj, Image.Image):
        return image_obj.convert("RGB")
    if isinstance(image_obj, dict):
        if image_obj.get("bytes") is not None:
            return Image.open(io.BytesIO(image_obj["bytes"])).convert("RGB")
        if image_obj.get("path"):
            return Image.open(image_obj["path"]).convert("RGB")
    if isinstance(image_obj, (str, Path)):
        return Image.open(image_obj).convert("RGB")
    raise TypeError(f"Unsupported RoboRefIt image object: {type(image_obj)!r}")


def resize_image(image: Image.Image, max_image_pixels: int) -> Image.Image:
    if max_image_pixels <= 0:
        return image
    width, height = image.size
    pixels = width * height
    if pixels <= max_image_pixels:
        return image
    scale = (max_image_pixels / pixels) ** 0.5
    new_size = (max(1, int(width * scale)), max(1, int(height * scale)))
    return image.resize(new_size, Image.Resampling.LANCZOS)


def scale_bbox(bbox: list[float], original_size: tuple[int, int], resized_size: tuple[int, int]) -> list[float]:
    if original_size == resized_size:
        return bbox
    original_width, original_height = original_size
    resized_width, resized_height = resized_size
    scale_x = resized_width / original_width
    scale_y = resized_height / original_height
    x1, y1, x2, y2 = bbox
    return [x1 * scale_x, y1 * scale_y, x2 * scale_x, y2 * scale_y]


def parse_normalized_bbox(text: str) -> list[float] | None:
    number = r"-?\d+(?:\.\d+)?"
    bracket_match = re.search(
        rf"[\[\(]\s*({number})\s*,\s*({number})\s*,\s*({number})\s*,\s*({number})\s*[\]\)]",
        text,
    )
    if bracket_match:
        values = [float(value) for value in bracket_match.groups()]
    else:
        values = [float(value) for value in re.findall(number, text)[-4:]]
    if len(values) != 4:
        return None
    return values


def normalized_to_pixels(bbox: list[float], size: tuple[int, int]) -> list[float]:
    width, height = size
    x1, y1, x2, y2 = bbox
    return [x1 * width, y1 * height, x2 * width, y2 * height]


def clean_llava_question(text: str) -> str:
    question = text.replace("<image>", "").strip()
    lines = [line.strip() for line in question.splitlines() if line.strip()]
    if len(lines) > 1 and "normalized position coordinates" in lines[-1].lower():
        lines = lines[:-1]
    return "\n".join(lines).strip()


def resolve_llava_jsonl_path(dataset_path: Path, split: str) -> Path:
    if dataset_path.is_file():
        return dataset_path
    candidates = [
        dataset_path / f"roborefit_{split}_llava.jsonl",
        dataset_path / f"{split}_llava.jsonl",
        dataset_path / f"{split}.jsonl",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        "Could not find a RoboRefIt LLaVA jsonl file. Tried: "
        + ", ".join(str(candidate) for candidate in candidates)
    )


def load_roborefit_llava(
    dataset_path: Path,
    split: str,
    max_samples: int | None,
    max_image_pixels: int,
) -> list[RoboRefItSample]:
    jsonl_path = resolve_llava_jsonl_path(dataset_path, split)
    samples: list[RoboRefItSample] = []
    with jsonl_path.open("r", encoding="utf-8") as handle:
        for idx, line in enumerate(handle):
            if max_samples is not None and len(samples) >= max_samples:
                break
            if not line.strip():
                continue
            row = json.loads(line)
            conversations = row.get("conversations") or []
            if len(conversations) < 2:
                raise ValueError(f"Missing LLaVA conversations in {jsonl_path} line {idx + 1}")

            image_path = str(row["image"])
            image = decode_image(image_path)
            original_size = image.size
            resized_image = resize_image(image, max_image_pixels)

            normalized_bbox = parse_normalized_bbox(str(conversations[1].get("value", "")))
            if normalized_bbox is None:
                raise ValueError(f"Could not parse normalized bbox in {jsonl_path} line {idx + 1}")
            bbox = normalized_to_pixels(normalized_bbox, original_size)

            samples.append(
                RoboRefItSample(
                    sample_id=len(samples),
                    question_id=int(row.get("id", idx)),
                    ref_exp=clean_llava_question(str(conversations[0].get("value", ""))),
                    image_path=image_path,
                    image=resized_image,
                    bbox=scale_bbox(bbox, original_size, resized_image.size),
                    normalized_bbox=normalized_bbox,
                    original_image_size=original_size,
                )
            )
    return samples


def load_roborefit(
    dataset_path: Path,
    split: str,
    max_samples: int | None,
    max_image_pixels: int,
) -> list[RoboRefItSample]:
    if dataset_path.suffix == ".jsonl" or (dataset_path.is_dir() and not (dataset_path / "data").exists()):
        return load_roborefit_llava(dataset_path, split, max_samples, max_image_pixels)

    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise ImportError(
            "RoboRefIt is stored as parquet with an image column. Install datasets/pyarrow "
            "or run `pip install datasets pyarrow` before evaluation."
        ) from exc

    data_files = sorted((dataset_path / "data").glob(f"{split}-*.parquet"))
    if not data_files:
        raise FileNotFoundError(f"No parquet files found for split '{split}' under {dataset_path / 'data'}")

    dataset = load_dataset(
        "parquet",
        data_files={split: [str(path) for path in data_files]},
        split=split,
    )
    if max_samples is not None:
        dataset = dataset.select(range(min(max_samples, len(dataset))))

    samples: list[RoboRefItSample] = []
    for idx, row in enumerate(dataset):
        image = decode_image(row["image"])
        original_size = image.size
        resized_image = resize_image(image, max_image_pixels)
        bbox = [float(value) for value in row["bbox"]]
        samples.append(
            RoboRefItSample(
                sample_id=idx,
                question_id=int(row["id"]),
                ref_exp=str(row["ref_exp"]).strip(),
                image_path=None,
                image=resized_image,
                bbox=scale_bbox(bbox, original_size, resized_image.size),
                normalized_bbox=[float(value) for value in row["normalized_bbox"]],
                original_image_size=original_size,
            )
        )
    return samples


def chunked(items: list[RoboRefItSample], size: int) -> Iterable[list[RoboRefItSample]]:
    for start in range(0, len(items), size):
        yield items[start : start + size]


def build_messages(sample: RoboRefItSample, system_prompt: str, question_suffix: str) -> list[dict[str, Any]]:
    question = sample.ref_exp.strip()
    if question_suffix:
        question = f"{question}\n{question_suffix.strip()}"
    return [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": [
                {"type": "image", "image": sample.image},
                {"type": "text", "text": question},
            ],
        },
    ]


def prepare_inputs(
    processor: Any,
    samples: list[RoboRefItSample],
    system_prompt: str,
    question_suffix: str,
    device: torch.device,
    dtype: torch.dtype | None,
) -> dict[str, torch.Tensor]:
    texts = [
        processor.apply_chat_template(
            build_messages(sample, system_prompt, question_suffix),
            tokenize=False,
            add_generation_prompt=True,
        )
        for sample in samples
    ]
    images = [sample.image for sample in samples]
    inputs = processor(text=texts, images=images, padding=True, return_tensors="pt")
    moved_inputs = {}
    for key, value in inputs.items():
        if not torch.is_tensor(value):
            moved_inputs[key] = value
        elif dtype is not None and torch.is_floating_point(value):
            moved_inputs[key] = value.to(device=device, dtype=dtype)
        else:
            moved_inputs[key] = value.to(device=device)
    return moved_inputs


def decode_answers(
    processor: Any,
    inputs: dict[str, torch.Tensor],
    generated_ids: torch.Tensor,
) -> list[str]:
    input_lengths = inputs["input_ids"].shape[1]
    generated_trimmed = generated_ids[:, input_lengths:]
    return processor.batch_decode(
        generated_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )


def infer_coordinate_scale(values: list[float]) -> str:
    if all(0.0 <= value <= 1.0 for value in values):
        return "normalized"
    if all(0.0 <= value <= 1000.0 for value in values):
        return "qwen_1000"
    return "pixels"


def parse_bbox(text: str, width: int, height: int, coordinate_scale: str) -> list[float] | None:
    number = r"-?\d+(?:\.\d+)?"
    bracket_match = re.search(
        rf"[\[\(]\s*({number})\s*,\s*({number})\s*,\s*({number})\s*,\s*({number})\s*[\]\)]",
        text,
    )
    if bracket_match:
        values = [float(value) for value in bracket_match.groups()]
    else:
        values = [float(value) for value in re.findall(number, text)[:4]]
    if len(values) != 4:
        return None

    scale = infer_coordinate_scale(values) if coordinate_scale == "auto" else coordinate_scale
    x1, y1, x2, y2 = values
    if scale == "normalized":
        x1 *= width
        x2 *= width
        y1 *= height
        y2 *= height
    elif scale == "qwen_1000":
        x1 = x1 / 1000.0 * width
        x2 = x2 / 1000.0 * width
        y1 = y1 / 1000.0 * height
        y2 = y2 / 1000.0 * height
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1

    x1 = min(max(x1, 0.0), width - 1.0)
    y1 = min(max(y1, 0.0), height - 1.0)
    x2 = min(max(x2, 0.0), width - 1.0)
    y2 = min(max(y2, 0.0), height - 1.0)
    if x2 <= x1 or y2 <= y1:
        return None
    return [x1, y1, x2, y2]


def bbox_iou(pred: list[float], target: list[float]) -> float:
    px1, py1, px2, py2 = pred
    tx1, ty1, tx2, ty2 = target
    ix1 = max(px1, tx1)
    iy1 = max(py1, ty1)
    ix2 = min(px2, tx2)
    iy2 = min(py2, ty2)
    iw = max(ix2 - ix1, 0.0)
    ih = max(iy2 - iy1, 0.0)
    inter = iw * ih
    pred_area = max(px2 - px1, 0.0) * max(py2 - py1, 0.0)
    target_area = max(tx2 - tx1, 0.0) * max(ty2 - ty1, 0.0)
    union = pred_area + target_area - inter
    return inter / union if union > 0 else 0.0


def load_seen_ids(predictions_path: Path) -> set[int]:
    if not predictions_path.exists():
        return set()
    seen: set[int] = set()
    with predictions_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                seen.add(int(json.loads(line)["sample_id"]))
    return seen


def summarize_records(records: list[dict[str, Any]]) -> dict[str, float | int]:
    num_samples = len(records)
    parsed = [record for record in records if record["parsed"]]
    mean_iou = float(np.mean([record["iou"] for record in records])) if records else 0.0
    mean_iou_parsed = float(np.mean([record["iou"] for record in parsed])) if parsed else 0.0
    return {
        "num_samples": num_samples,
        "parsed_samples": len(parsed),
        "parse_rate": len(parsed) / max(num_samples, 1),
        "mean_iou": mean_iou,
        "mean_iou_parsed": mean_iou_parsed,
        "acc_iou_0_25": sum(record["iou"] >= 0.25 for record in records) / max(num_samples, 1),
        "acc_iou_0_5": sum(record["iou"] >= 0.5 for record in records) / max(num_samples, 1),
        "acc_iou_0_75": sum(record["iou"] >= 0.75 for record in records) / max(num_samples, 1),
    }


def write_summary(output_dir: Path, summary: dict[str, Any]) -> None:
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)

    ckpt_path = Path(args.ckpt_path).expanduser().resolve()
    dataset_path = Path(args.dataset_path).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    cfg = load_config(ckpt_path)
    if args.backbone_path:
        cfg["backbone"] = args.backbone_path
    if args.attn_implementation:
        cfg["attention_implementation"] = args.attn_implementation

    processor_path = args.processor_path or cfg.get("backbone")
    if processor_path is None:
        raise ValueError("Could not infer processor path from checkpoint config. Pass --processor-path explicitly.")

    device = torch.device(args.device if torch.cuda.is_available() or not args.device.startswith("cuda") else "cpu")
    if device.type == "cpu" and args.device.startswith("cuda"):
        print("CUDA requested but unavailable; falling back to CPU.", flush=True)

    print(f"Loading Spirit checkpoint: {ckpt_path}", flush=True)
    policy = load_policy(ckpt_path, cfg, device)
    policy.eval()
    dtype = torch_dtype(args.dtype)
    if dtype is not None:
        policy = policy.to(dtype=dtype)
    policy = policy.to(device)
    model = policy.qwen

    print(f"Loading processor: {processor_path}", flush=True)
    processor = AutoProcessor.from_pretrained(processor_path, trust_remote_code=args.trust_remote_code)

    samples = load_roborefit(dataset_path, args.split, args.max_samples, args.max_image_pixels)
    predictions_path = output_dir / "predictions.jsonl"
    seen_ids = load_seen_ids(predictions_path) if args.resume else set()
    samples_to_run = [sample for sample in samples if sample.sample_id not in seen_ids]
    print(
        f"Loaded {len(samples)} RoboRefIt samples; evaluating {len(samples_to_run)} "
        f"(resume skipped {len(seen_ids)}).",
        flush=True,
    )

    start_time = time.time()
    new_records: list[dict[str, Any]] = []
    mode = "a" if args.resume else "w"
    with predictions_path.open(mode, encoding="utf-8") as handle:
        for batch_index, batch in enumerate(chunked(samples_to_run, args.batch_size), start=1):
            inputs = prepare_inputs(processor, batch, args.system_prompt, args.question_suffix, device, dtype)
            generate_kwargs = {
                "max_new_tokens": args.max_new_tokens,
                "num_beams": args.num_beams,
                "do_sample": args.temperature > 0,
            }
            if args.temperature > 0:
                generate_kwargs.update({"temperature": args.temperature, "top_p": args.top_p})

            with torch.inference_mode():
                generated_ids = model.generate(**inputs, **generate_kwargs)

            predictions = decode_answers(processor, inputs, generated_ids)
            del inputs, generated_ids
            if device.type == "cuda":
                torch.cuda.empty_cache()
            for sample, prediction in zip(batch, predictions, strict=True):
                width, height = sample.image.size
                pred_bbox = parse_bbox(prediction, width, height, args.coordinate_scale)
                parsed = pred_bbox is not None
                iou = bbox_iou(pred_bbox, sample.bbox) if pred_bbox is not None else 0.0
                record = {
                    "sample_id": sample.sample_id,
                    "question_id": sample.question_id,
                    "ref_exp": sample.ref_exp,
                    "image_path": sample.image_path,
                    "bbox": sample.bbox,
                    "normalized_bbox": sample.normalized_bbox,
                    "original_image_size": list(sample.original_image_size),
                    "prediction": prediction.strip(),
                    "pred_bbox": pred_bbox,
                    "parsed": parsed,
                    "iou": iou,
                    "acc_iou_0_25": iou >= 0.25,
                    "acc_iou_0_5": iou >= 0.5,
                    "acc_iou_0_75": iou >= 0.75,
                    "image_size": [width, height],
                }
                new_records.append(record)
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                handle.flush()

            metrics = summarize_records(new_records)
            print(
                f"[{batch_index}] processed={metrics['num_samples']}/{len(samples_to_run)} "
                f"mean_iou={metrics['mean_iou']:.4f} acc@0.5={metrics['acc_iou_0_5']:.4f}",
                flush=True,
            )

    if args.resume and seen_ids:
        all_records = [
            json.loads(line)
            for line in predictions_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    else:
        all_records = new_records

    summary = {
        "dataset_path": str(dataset_path),
        "ckpt_path": str(ckpt_path),
        "processor_path": processor_path,
        "split": args.split,
        **summarize_records(all_records),
        "elapsed_seconds": time.time() - start_time,
        "predictions_path": str(predictions_path),
        "coordinate_scale": args.coordinate_scale,
        "max_image_pixels": args.max_image_pixels,
        "generation": {
            "max_new_tokens": args.max_new_tokens,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "num_beams": args.num_beams,
        },
    }
    write_summary(output_dir, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
