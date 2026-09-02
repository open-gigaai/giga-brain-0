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
    "datasets/public_datasets/VLM/benchmarks/vabench-point-bbox/"
    "llava_json/vabench_point_bbox_llava.jsonl"
)


@dataclass
class VABenchPointBBoxSample:
    sample_id: int
    question_id: int
    problem: str
    image_path: str | None
    image: Image.Image
    bbox: list[float]
    normalized_bbox: list[float]
    original_image_size: tuple[int, int]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate the Spirit-v1.5 Qwen3-VL backbone on VABench point-bbox.",
    )
    parser.add_argument("--ckpt-path", default=DEFAULT_CKPT_PATH, help="Spirit-v1.5 checkpoint directory.")
    parser.add_argument("--dataset-path", default=DEFAULT_DATASET_PATH, help="VABench point-bbox dataset directory.")
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
    parser.add_argument("--output-dir", default="outputs/vabench_point_bbox_vlm", help="Directory for outputs.")
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
    parser.add_argument("--max-new-tokens", type=int, default=256, help="Maximum generated answer tokens.")
    parser.add_argument("--temperature", type=float, default=0.0, help="Generation temperature. 0 means greedy.")
    parser.add_argument("--top-p", type=float, default=1.0, help="Nucleus sampling value when temperature > 0.")
    parser.add_argument("--num-beams", type=int, default=1, help="Beam count for generation.")
    parser.add_argument(
        "--system-prompt",
        default="You are a helpful robotic visual affordance assistant.",
        help="System prompt used for point generation.",
    )
    parser.add_argument(
        "--question-suffix",
        default="",
        help="Optional suffix appended to each VABench problem.",
    )
    parser.add_argument(
        "--coordinate-scale",
        choices=("auto", "pixels", "normalized", "qwen_1000"),
        default="auto",
        help="Expected coordinate scale in model output.",
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
    raise TypeError(f"Unsupported VABench image object: {type(image_obj)!r}")


def parse_normalized_bbox(text: str) -> list[float] | None:
    number = r"-?\d+(?:\.\d+)?"
    values = [float(value) for value in re.findall(number, text)]
    if len(values) < 4:
        return None
    return values[-4:]


def normalized_bbox_to_pixels(normalized_bbox: list[float], image_size: tuple[int, int]) -> list[float]:
    width, height = image_size
    x1, y1, x2, y2 = normalized_bbox
    return [x1 * width, y1 * height, x2 * width, y2 * height]


def resolve_llava_jsonl_path(dataset_path: Path, split: str) -> Path:
    if dataset_path.is_file():
        return dataset_path
    candidates = [
        dataset_path / "vabench_point_bbox_llava.jsonl",
        dataset_path / f"vabench_point_bbox_{split}_llava.jsonl",
        dataset_path / f"{split}_llava.jsonl",
        dataset_path / f"{split}.jsonl",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        "Could not find a VABench point-bbox LLaVA jsonl file. Tried: "
        + ", ".join(str(candidate) for candidate in candidates)
    )


def load_vabench_llava(
    dataset_path: Path,
    split: str,
    max_samples: int | None,
) -> list[VABenchPointBBoxSample]:
    jsonl_path = resolve_llava_jsonl_path(dataset_path, split)
    samples: list[VABenchPointBBoxSample] = []
    with jsonl_path.open("r", encoding="utf-8") as handle:
        for line_index, line in enumerate(handle, start=1):
            if max_samples is not None and len(samples) >= max_samples:
                break
            if not line.strip():
                continue
            row = json.loads(line)
            conversations = row.get("conversations") or []
            if len(conversations) < 2:
                raise ValueError(f"Missing LLaVA conversations in {jsonl_path} line {line_index}")

            image_path = str(row["image"])
            image = decode_image(image_path)
            normalized_bbox = parse_normalized_bbox(str(conversations[1].get("value", "")))
            if normalized_bbox is None:
                raise ValueError(f"Could not parse normalized bbox in {jsonl_path} line {line_index}")

            samples.append(
                VABenchPointBBoxSample(
                    sample_id=len(samples),
                    question_id=int(row.get("id", len(samples))),
                    problem=str(conversations[0].get("value", "")),
                    image_path=image_path,
                    image=image,
                    bbox=normalized_bbox_to_pixels(normalized_bbox, image.size),
                    normalized_bbox=normalized_bbox,
                    original_image_size=image.size,
                )
            )
    return samples


def load_vabench(dataset_path: Path, split: str, max_samples: int | None) -> list[VABenchPointBBoxSample]:
    if dataset_path.suffix == ".jsonl" or (dataset_path.is_dir() and not (dataset_path / "data").exists()):
        return load_vabench_llava(dataset_path, split, max_samples)

    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise ImportError(
            "VABench point-bbox is stored as parquet with an image column. Install datasets/pyarrow "
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

    samples: list[VABenchPointBBoxSample] = []
    for idx, row in enumerate(dataset):
        image = decode_image(row["image"])
        samples.append(
            VABenchPointBBoxSample(
                sample_id=idx,
                question_id=int(row["idx"]),
                problem=str(row["problem"]),
                image_path=None,
                image=image,
                bbox=[float(value) for value in row["bbox"]],
                normalized_bbox=[float(value) for value in row["normalized_bbox"]],
                original_image_size=image.size,
            )
        )
    return samples


def clean_problem(problem: str) -> str:
    return problem.replace("<image>", "").strip()


def chunked(items: list[VABenchPointBBoxSample], size: int) -> Iterable[list[VABenchPointBBoxSample]]:
    for start in range(0, len(items), size):
        yield items[start : start + size]


def build_messages(sample: VABenchPointBBoxSample, system_prompt: str, question_suffix: str) -> list[dict[str, Any]]:
    question = clean_problem(sample.problem)
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
    samples: list[VABenchPointBBoxSample],
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


def extract_coordinate_source(text: str) -> str:
    for tag in ("point", "bbox", "box", "answer"):
        section = re.search(rf"<{tag}>(.*?)</{tag}>", text, flags=re.IGNORECASE | re.DOTALL)
        if section:
            return section.group(1)
    return text


def extract_number_pairs(text: str) -> list[tuple[float, float]]:
    source = extract_coordinate_source(text)
    number = r"-?\d+(?:\.\d+)?"
    return [(float(x_raw), float(y_raw)) for x_raw, y_raw in re.findall(rf"[\[\(]?\s*({number})\s*,\s*({number})\s*[\]\)]?", source)]


def infer_coordinate_scale(values: list[float]) -> str:
    if all(0.0 <= value <= 1.0 for value in values):
        return "normalized"
    if all(0.0 <= value <= 1000.0 for value in values):
        return "qwen_1000"
    return "pixels"


def convert_coordinate(value: float, size: int, coordinate_scale: str) -> float:
    if coordinate_scale == "normalized":
        return value * size
    if coordinate_scale == "qwen_1000":
        return value / 1000.0 * size
    return value


def parse_points(text: str, width: int, height: int, coordinate_scale: str) -> list[tuple[int, int]]:
    pairs = extract_number_pairs(text)
    flat_values = [value for pair in pairs for value in pair]
    scale = infer_coordinate_scale(flat_values) if coordinate_scale == "auto" and flat_values else coordinate_scale
    points: list[tuple[int, int]] = []
    for x_raw, y_raw in pairs:
        x = convert_coordinate(x_raw, width, scale)
        y = convert_coordinate(y_raw, height, scale)
        x_i = int(round(x))
        y_i = int(round(y))
        if 0 <= x_i < width and 0 <= y_i < height:
            points.append((x_i, y_i))
    return points


def point_in_bbox(point: tuple[int, int], bbox: list[float]) -> bool:
    x, y = point
    x1, y1, x2, y2 = bbox
    return x1 <= x <= x2 and y1 <= y <= y2


def points_to_bbox(points: list[tuple[int, int]]) -> list[float] | None:
    if not points:
        return None
    xs = [point[0] for point in points]
    ys = [point[1] for point in points]
    return [float(min(xs)), float(min(ys)), float(max(xs)), float(max(ys))]


def parse_bbox(text: str, width: int, height: int, coordinate_scale: str) -> tuple[list[float] | None, str | None, list[float] | None]:
    pairs = extract_number_pairs(text)
    if len(pairs) >= 2:
        values = [pairs[0][0], pairs[0][1], pairs[1][0], pairs[1][1]]
    else:
        number = r"-?\d+(?:\.\d+)?"
        values = [float(value) for value in re.findall(number, extract_coordinate_source(text))[:4]]
    if len(values) != 4:
        return None, None, None

    scale = infer_coordinate_scale(values) if coordinate_scale == "auto" else coordinate_scale
    x1, y1, x2, y2 = values
    x1 = convert_coordinate(x1, width, scale)
    x2 = convert_coordinate(x2, width, scale)
    y1 = convert_coordinate(y1, height, scale)
    y2 = convert_coordinate(y2, height, scale)
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    x1 = min(max(x1, 0.0), width - 1.0)
    y1 = min(max(y1, 0.0), height - 1.0)
    x2 = min(max(x2, 0.0), width - 1.0)
    y2 = min(max(y2, 0.0), height - 1.0)
    if x2 <= x1 or y2 <= y1:
        return values, scale, None
    return values, scale, [x1, y1, x2, y2]


def bbox_iou(pred: list[float] | None, target: list[float]) -> float:
    if pred is None:
        return 0.0
    px1, py1, px2, py2 = pred
    tx1, ty1, tx2, ty2 = target
    ix1 = max(px1, tx1)
    iy1 = max(py1, ty1)
    ix2 = min(px2, tx2)
    iy2 = min(py2, ty2)
    inter = max(ix2 - ix1, 0.0) * max(iy2 - iy1, 0.0)
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


def summarize_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    num_samples = len(records)
    parsed_samples = sum(bool(record.get("parsed_bbox", record.get("pred_bbox") is not None)) for record in records)
    parsed_point_samples = sum(record["num_points"] > 0 for record in records)
    point_hits = sum(int(record["point_hits"]) for record in records)
    num_points = sum(int(record["num_points"]) for record in records)
    sample_successes = sum(int(record["sample_success"]) for record in records)
    mean_iou = float(np.mean([record.get("iou", 0.0) for record in records])) if records else 0.0
    parsed_records = [record for record in records if bool(record.get("parsed_bbox", record.get("pred_bbox") is not None))]
    mean_iou_parsed = float(np.mean([record.get("iou", 0.0) for record in parsed_records])) if parsed_records else 0.0
    mean_point_bbox_iou = float(np.mean([record["points_bbox_iou"] for record in records])) if records else 0.0
    scale_counts: dict[str, int] = {}
    for record in records:
        scale = record.get("coordinate_scale_inferred") or "unparsed"
        scale_counts[scale] = scale_counts.get(scale, 0) + 1
    return {
        "num_samples": num_samples,
        "parsed_samples": parsed_samples,
        "parse_rate": parsed_samples / max(num_samples, 1),
        "coordinate_scale_counts": scale_counts,
        "parsed_point_samples": parsed_point_samples,
        "point_parse_rate": parsed_point_samples / max(num_samples, 1),
        "mean_iou": mean_iou,
        "mean_iou_parsed": mean_iou_parsed,
        "acc_iou_0_25": sum(record.get("iou", 0.0) >= 0.25 for record in records) / max(num_samples, 1),
        "acc_iou_0_5": sum(record.get("iou", 0.0) >= 0.5 for record in records) / max(num_samples, 1),
        "acc_iou_0_75": sum(record.get("iou", 0.0) >= 0.75 for record in records) / max(num_samples, 1),
        "point_hits": point_hits,
        "num_points": num_points,
        "point_accuracy": point_hits / max(num_points, 1),
        "successful_samples": sample_successes,
        "sample_success_rate": sample_successes / max(num_samples, 1),
        "mean_points_bbox_iou": mean_point_bbox_iou,
        "acc_points_bbox_iou_0_25": sum(record["points_bbox_iou"] >= 0.25 for record in records) / max(num_samples, 1),
        "acc_points_bbox_iou_0_5": sum(record["points_bbox_iou"] >= 0.5 for record in records) / max(num_samples, 1),
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

    samples = load_vabench(dataset_path, args.split, args.max_samples)
    predictions_path = output_dir / "predictions.jsonl"
    seen_ids = load_seen_ids(predictions_path) if args.resume else set()
    samples_to_run = [sample for sample in samples if sample.sample_id not in seen_ids]
    print(
        f"Loaded {len(samples)} VABench point-bbox samples; evaluating {len(samples_to_run)} "
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
            for sample, prediction in zip(batch, predictions, strict=True):
                width, height = sample.image.size
                raw_bbox_values, coordinate_scale, pred_bbox = parse_bbox(
                    prediction,
                    width,
                    height,
                    args.coordinate_scale,
                )
                parsed_bbox = pred_bbox is not None
                iou = bbox_iou(pred_bbox, sample.bbox)
                points = parse_points(prediction, width, height, args.coordinate_scale)
                hits = sum(int(point_in_bbox(point, sample.bbox)) for point in points)
                points_bbox = points_to_bbox(points)
                points_bbox_iou = bbox_iou(points_bbox, sample.bbox)
                record = {
                    "sample_id": sample.sample_id,
                    "question_id": sample.question_id,
                    "problem": clean_problem(sample.problem),
                    "image_path": sample.image_path,
                    "bbox": sample.bbox,
                    "normalized_bbox": sample.normalized_bbox,
                    "original_image_size": list(sample.original_image_size),
                    "prediction": prediction.strip(),
                    "raw_bbox_values": raw_bbox_values,
                    "coordinate_scale_inferred": coordinate_scale,
                    "pred_bbox": pred_bbox,
                    "parsed_bbox": parsed_bbox,
                    "iou": iou,
                    "acc_iou_0_25": iou >= 0.25,
                    "acc_iou_0_5": iou >= 0.5,
                    "acc_iou_0_75": iou >= 0.75,
                    "points": points,
                    "point_hits": hits,
                    "num_points": len(points),
                    "point_accuracy": hits / max(len(points), 1),
                    "sample_success": hits > 0,
                    "points_bbox": points_bbox,
                    "points_bbox_iou": points_bbox_iou,
                    "image_size": [width, height],
                }
                new_records.append(record)
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                handle.flush()

            metrics = summarize_records(new_records)
            print(
                f"[{batch_index}] processed={metrics['num_samples']}/{len(samples_to_run)} "
                f"mean_iou={metrics['mean_iou']:.4f} "
                f"acc@0.5={metrics['acc_iou_0_5']:.4f} "
                f"point_accuracy={metrics['point_accuracy']:.4f}",
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
