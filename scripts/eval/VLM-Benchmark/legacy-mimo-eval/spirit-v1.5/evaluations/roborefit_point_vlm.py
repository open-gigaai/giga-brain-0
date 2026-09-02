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
DEFAULT_DATASET_PATH = "datasets/public_datasets/VLM/vqa/roborefit"
PROMPT_MIMO_EMBODIED = (
    'Based on the description: "{ref_exp}", locate points matching the description. '
    'Output a JSON in the format [{{"points": [...], "label": "{{the_whole_description}}"}}, ...].'
)


@dataclass
class RoboRefItPointSample:
    sample_id: int
    question_id: int
    ref_exp: str
    image: Image.Image
    bbox: list[float]
    normalized_bbox: list[float]
    original_image_size: tuple[int, int]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate Spirit-v1.5 on RoboRefIt using the point-in-bbox metric.",
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
    parser.add_argument("--output-dir", default="outputs/roborefit_point_vlm", help="Directory for outputs.")
    parser.add_argument("--split", default="test", help="Dataset split name.")
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
    parser.add_argument("--max-new-tokens", type=int, default=32768, help="Maximum generated answer tokens.")
    parser.add_argument("--temperature", type=float, default=0.0, help="Generation temperature. 0 means greedy.")
    parser.add_argument("--top-p", type=float, default=1.0, help="Nucleus sampling value when temperature > 0.")
    parser.add_argument("--num-beams", type=int, default=1, help="Beam count for generation.")
    parser.add_argument(
        "--system-prompt",
        default="You are a helpful visual grounding assistant.",
        help="System prompt used for point generation.",
    )
    parser.add_argument(
        "--coordinate-scale",
        choices=("pixels", "normalized", "qwen_1000", "auto"),
        default="pixels",
        help="Coordinate scale used by predicted points. Default matches the referenced RoboRefIt utils.",
    )
    parser.add_argument(
        "--max-image-pixels",
        type=int,
        default=262144,
        help="Resize each image to at most this many pixels before inference. Use 0 to disable.",
    )
    parser.add_argument(
        "--strict-after-think",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Match the referenced utils: ignore output unless it contains </think>.",
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


def load_roborefit(dataset_path: Path, split: str, max_samples: int | None, max_image_pixels: int) -> list[RoboRefItPointSample]:
    data_files = sorted((dataset_path / "data").glob(f"{split}-*.parquet"))
    if not data_files:
        raise FileNotFoundError(f"No parquet files found for split '{split}' under {dataset_path / 'data'}")

    try:
        from datasets import load_dataset
        dataset = load_dataset(
            "parquet",
            data_files={split: [str(path) for path in data_files]},
            split=split,
        )
        if max_samples is not None:
            dataset = dataset.select(range(min(max_samples, len(dataset))))
        rows = enumerate(dataset)
    except ImportError:
        try:
            import pandas as pd
        except ImportError as exc:
            raise ImportError(
                "RoboRefIt is stored as parquet. Install datasets or pandas/pyarrow before evaluation."
            ) from exc

        loaded_rows: list[dict[str, Any]] = []
        for parquet_path in data_files:
            frame = pd.read_parquet(parquet_path)
            loaded_rows.extend(frame.to_dict("records"))
            if max_samples is not None and len(loaded_rows) >= max_samples:
                loaded_rows = loaded_rows[:max_samples]
                break
        rows = enumerate(loaded_rows)

    samples: list[RoboRefItPointSample] = []
    for idx, row in rows:
        image = decode_image(row["image"])
        original_size = image.size
        resized_image = resize_image(image, max_image_pixels)
        bbox = [float(value) for value in row["bbox"]]
        samples.append(
            RoboRefItPointSample(
                sample_id=idx,
                question_id=int(row["id"]),
                ref_exp=str(row["ref_exp"]).strip(),
                image=resized_image,
                bbox=scale_bbox(bbox, original_size, resized_image.size),
                normalized_bbox=[float(value) for value in row["normalized_bbox"]],
                original_image_size=original_size,
            )
        )
    return samples


def build_messages(sample: RoboRefItPointSample, system_prompt: str) -> list[dict[str, Any]]:
    question = PROMPT_MIMO_EMBODIED.format(ref_exp=sample.ref_exp)
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
    samples: list[RoboRefItPointSample],
    system_prompt: str,
    device: torch.device,
    dtype: torch.dtype | None,
) -> dict[str, torch.Tensor]:
    texts = [
        processor.apply_chat_template(
            build_messages(sample, system_prompt),
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


def decode_answers(processor: Any, inputs: dict[str, torch.Tensor], generated_ids: torch.Tensor) -> list[str]:
    input_lengths = inputs["input_ids"].shape[1]
    generated_trimmed = generated_ids[:, input_lengths:]
    return processor.batch_decode(
        generated_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )


def extract_after_think_content(text: str, strict: bool) -> str:
    last_think_end = text.rfind("</think>")
    if last_think_end != -1:
        return text[last_think_end + len("</think>") :].strip()
    return "" if strict else text


def coordinate_source(text: str) -> str:
    for tag in ("point", "points", "answer"):
        match = re.search(rf"<{tag}>(.*?)</{tag}>", text, flags=re.IGNORECASE | re.DOTALL)
        if match:
            return match.group(1)
    return text


def collect_points_from_json(obj: Any) -> list[list[float]]:
    points: list[list[float]] = []
    if isinstance(obj, dict):
        if isinstance(obj.get("points"), list):
            points.extend(collect_points_from_json(obj["points"]))
        else:
            for value in obj.values():
                points.extend(collect_points_from_json(value))
    elif isinstance(obj, list):
        if len(obj) == 2 and all(isinstance(value, (int, float)) for value in obj):
            points.append([float(obj[0]), float(obj[1])])
        else:
            for value in obj:
                points.extend(collect_points_from_json(value))
    return points


def raw_point_pairs(text: str) -> list[list[float]]:
    source = coordinate_source(text)
    try:
        parsed = json.loads(source)
        points = collect_points_from_json(parsed)
        if points:
            return points
    except json.JSONDecodeError:
        pass

    number = r"-?\d+(?:\.\d+)?"
    pairs = re.findall(rf"[\[\(]\s*({number})\s*,\s*({number})\s*[\]\)]", source)
    return [[float(x), float(y)] for x, y in pairs]


def infer_coordinate_scale(points: list[list[float]]) -> str:
    values = [value for point in points for value in point]
    if values and all(0.0 <= value <= 1.0 for value in values):
        return "normalized"
    if values and all(0.0 <= value <= 1000.0 for value in values):
        return "qwen_1000"
    return "pixels"


def convert_points(points: list[list[float]], width: int, height: int, coordinate_scale: str) -> list[list[int]]:
    scale = infer_coordinate_scale(points) if coordinate_scale == "auto" else coordinate_scale
    converted: list[list[int]] = []
    for x_raw, y_raw in points:
        x = x_raw
        y = y_raw
        if scale == "normalized":
            x *= width
            y *= height
        elif scale == "qwen_1000":
            x = x / 1000.0 * width
            y = y / 1000.0 * height
        x_i = int(round(x))
        y_i = int(round(y))
        if 0 <= x_i < width and 0 <= y_i < height:
            converted.append([x_i, y_i])
    return converted


def parse_points(text: str, width: int, height: int, coordinate_scale: str, strict_after_think: bool) -> tuple[str, list[list[float]], list[list[int]], str]:
    filtered = extract_after_think_content(text, strict=strict_after_think)
    if not filtered:
        return filtered, [], [], "unparsed"
    raw_points = raw_point_pairs(filtered)
    if not raw_points:
        return filtered, [], [], "unparsed"
    scale = infer_coordinate_scale(raw_points) if coordinate_scale == "auto" else coordinate_scale
    return filtered, raw_points, convert_points(raw_points, width, height, scale), scale


def check_points_in_bbox(points: list[list[int]], bbox: list[float]) -> tuple[int, int]:
    x1, y1, x2, y2 = bbox
    hits = 0
    for point in points:
        if len(point) == 2:
            x, y = point
            if x1 <= x <= x2 and y1 <= y <= y2:
                hits += 1
    return hits, len(points)


def load_seen_ids(predictions_path: Path) -> set[int]:
    if not predictions_path.exists():
        return set()
    seen: set[int] = set()
    with predictions_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                seen.add(int(json.loads(line)["sample_id"]))
    return seen


def chunked(items: list[RoboRefItPointSample], size: int) -> Iterable[list[RoboRefItPointSample]]:
    for start in range(0, len(items), size):
        yield items[start : start + size]


def summarize_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    num_samples = len(records)
    accuracies = [float(record["accuracy"]) for record in records]
    parsed_samples = sum(record["num_points"] > 0 for record in records)
    point_hits = sum(int(record["point_hits"]) for record in records)
    num_points = sum(int(record["num_points"]) for record in records)
    scale_counts: dict[str, int] = {}
    for record in records:
        scale = str(record.get("coordinate_scale_inferred", "unparsed"))
        scale_counts[scale] = scale_counts.get(scale, 0) + 1
    return {
        "num_samples": num_samples,
        "parsed_samples": parsed_samples,
        "parse_rate": parsed_samples / max(num_samples, 1),
        "accuracy": float(np.mean(accuracies)) if accuracies else 0.0,
        "point_hits": point_hits,
        "num_points": num_points,
        "point_accuracy": point_hits / max(num_points, 1),
        "successful_samples": sum(record["point_hits"] > 0 for record in records),
        "sample_success_rate": sum(record["point_hits"] > 0 for record in records) / max(num_samples, 1),
        "coordinate_scale_counts": scale_counts,
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
            inputs = prepare_inputs(processor, batch, args.system_prompt, device, dtype)
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
                filtered_prediction, raw_points, points, inferred_scale = parse_points(
                    prediction,
                    width,
                    height,
                    args.coordinate_scale,
                    args.strict_after_think,
                )
                point_hits, num_points = check_points_in_bbox(points, sample.bbox)
                accuracy = point_hits / num_points if num_points > 0 else 0.0
                record = {
                    "sample_id": sample.sample_id,
                    "question_id": sample.question_id,
                    "ref_exp": sample.ref_exp,
                    "bbox": sample.bbox,
                    "normalized_bbox": sample.normalized_bbox,
                    "original_image_size": list(sample.original_image_size),
                    "image_size": [width, height],
                    "prompt": PROMPT_MIMO_EMBODIED.format(ref_exp=sample.ref_exp),
                    "prediction": prediction.strip(),
                    "filtered_prediction": filtered_prediction,
                    "raw_points": raw_points,
                    "points": points,
                    "coordinate_scale_inferred": inferred_scale,
                    "point_hits": point_hits,
                    "num_points": num_points,
                    "accuracy": accuracy,
                    "sample_success": point_hits > 0,
                }
                new_records.append(record)
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                handle.flush()

            metrics = summarize_records(new_records)
            print(
                f"[{batch_index}] processed={metrics['num_samples']}/{len(samples_to_run)} "
                f"accuracy={metrics['accuracy']:.4f} point_accuracy={metrics['point_accuracy']:.4f}",
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
        "prompt_template": PROMPT_MIMO_EMBODIED,
        "coordinate_scale": args.coordinate_scale,
        "strict_after_think": args.strict_after_think,
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
