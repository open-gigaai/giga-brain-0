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
DEFAULT_DATASET_PATH = "datasets/public_datasets/VLM/vqa/benchmarks/Where2Place"
DEFAULT_SUFFIX = (
    "Return 3 candidate points inside the requested free space. "
    "Use image pixel coordinates with origin at the top-left. "
    "Answer only as JSON: [[x1, y1], [x2, y2], [x3, y3]]."
)


@dataclass
class Where2PlaceSample:
    sample_id: int
    question_id: str
    question_type: str
    question: str
    image: Image.Image
    mask: Image.Image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate the Spirit-v1.5 Qwen3-VL backbone on Where2Place.",
    )
    parser.add_argument("--ckpt-path", default=DEFAULT_CKPT_PATH, help="Spirit-v1.5 checkpoint directory.")
    parser.add_argument("--dataset-path", default=DEFAULT_DATASET_PATH, help="Where2Place dataset directory.")
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
    parser.add_argument("--output-dir", default="outputs/where2place_vlm", help="Directory for JSONL and summary outputs.")
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
    parser.add_argument("--max-new-tokens", type=int, default=96, help="Maximum generated answer tokens.")
    parser.add_argument("--temperature", type=float, default=0.0, help="Generation temperature. 0 means greedy.")
    parser.add_argument("--top-p", type=float, default=1.0, help="Nucleus sampling value when temperature > 0.")
    parser.add_argument("--num-beams", type=int, default=1, help="Beam count for generation.")
    parser.add_argument(
        "--system-prompt",
        default="You are a helpful visual spatial reasoning assistant.",
        help="System prompt used for coordinate generation.",
    )
    parser.add_argument(
        "--question-suffix",
        default=DEFAULT_SUFFIX,
        help="Instruction appended to each Where2Place question.",
    )
    parser.add_argument("--mask-threshold", type=int, default=127, help="Mask threshold for valid placement region.")
    parser.add_argument(
        "--coordinate-scale",
        choices=("pixels", "normalized"),
        default="pixels",
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


def decode_image(image_obj: Any, mode: str) -> Image.Image:
    if isinstance(image_obj, Image.Image):
        return image_obj.convert(mode)
    if isinstance(image_obj, dict):
        if image_obj.get("bytes") is not None:
            return Image.open(io.BytesIO(image_obj["bytes"])).convert(mode)
        if image_obj.get("path"):
            return Image.open(image_obj["path"]).convert(mode)
    if isinstance(image_obj, (str, Path)):
        return Image.open(image_obj).convert(mode)
    raise TypeError(f"Unsupported Where2Place image object: {type(image_obj)!r}")


def load_where2place(dataset_path: Path, split: str, max_samples: int | None) -> list[Where2PlaceSample]:
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise ImportError(
            "Where2Place is stored as parquet with image columns. Install datasets/pyarrow "
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

    samples: list[Where2PlaceSample] = []
    for idx, row in enumerate(dataset):
        samples.append(
            Where2PlaceSample(
                sample_id=idx,
                question_id=str(row["question_id"]),
                question_type=str(row["question_type"]),
                question=str(row["question"]),
                image=decode_image(row["image"], "RGB"),
                mask=decode_image(row["mask"], "L"),
            )
        )
    return samples


def chunked(items: list[Where2PlaceSample], size: int) -> Iterable[list[Where2PlaceSample]]:
    for start in range(0, len(items), size):
        yield items[start : start + size]


def build_messages(sample: Where2PlaceSample, system_prompt: str, question_suffix: str) -> list[dict[str, Any]]:
    question = sample.question.strip()
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
    samples: list[Where2PlaceSample],
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


def parse_points(text: str, width: int, height: int, coordinate_scale: str) -> list[tuple[int, int]]:
    number = r"-?\d+(?:\.\d+)?"
    pairs = re.findall(rf"[\[\(]?\s*({number})\s*,\s*({number})\s*[\]\)]?", text)
    points: list[tuple[int, int]] = []
    for x_raw, y_raw in pairs:
        x = float(x_raw)
        y = float(y_raw)
        if coordinate_scale == "normalized" or (0.0 <= x <= 1.0 and 0.0 <= y <= 1.0):
            x *= width - 1
            y *= height - 1
        x_i = int(round(x))
        y_i = int(round(y))
        if 0 <= x_i < width and 0 <= y_i < height:
            points.append((x_i, y_i))
    return points


def score_points(points: list[tuple[int, int]], mask: Image.Image, threshold: int) -> tuple[int, int, float, bool]:
    mask_arr = np.array(mask.convert("L"))
    hits = 0
    for x, y in points:
        hits += int(mask_arr[y, x] > threshold)
    total = len(points)
    hit_rate = hits / max(total, 1)
    return hits, total, hit_rate, hits > 0


def load_seen_ids(predictions_path: Path) -> set[int]:
    if not predictions_path.exists():
        return set()
    seen: set[int] = set()
    with predictions_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                seen.add(int(json.loads(line)["sample_id"]))
    return seen


def build_type_metrics(records: list[dict[str, Any]]) -> dict[str, dict[str, float | int]]:
    metrics: dict[str, dict[str, float | int]] = {}
    for record in records:
        key = record["question_type"]
        bucket = metrics.setdefault(
            key,
            {
                "num_samples": 0,
                "parsed_samples": 0,
                "point_hits": 0,
                "num_points": 0,
                "successful_samples": 0,
                "point_accuracy": 0.0,
                "parse_rate": 0.0,
                "sample_success_rate": 0.0,
            },
        )
        bucket["num_samples"] += 1
        bucket["parsed_samples"] += int(record["num_points"] > 0)
        bucket["point_hits"] += int(record["point_hits"])
        bucket["num_points"] += int(record["num_points"])
        bucket["successful_samples"] += int(record["sample_success"])
    for bucket in metrics.values():
        bucket["point_accuracy"] = bucket["point_hits"] / max(bucket["num_points"], 1)
        bucket["parse_rate"] = bucket["parsed_samples"] / max(bucket["num_samples"], 1)
        bucket["sample_success_rate"] = bucket["successful_samples"] / max(bucket["num_samples"], 1)
    return metrics


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

    samples = load_where2place(dataset_path, args.split, args.max_samples)
    predictions_path = output_dir / "predictions.jsonl"
    seen_ids = load_seen_ids(predictions_path) if args.resume else set()
    samples_to_run = [sample for sample in samples if sample.sample_id not in seen_ids]
    print(
        f"Loaded {len(samples)} Where2Place samples; evaluating {len(samples_to_run)} "
        f"(resume skipped {len(seen_ids)}).",
        flush=True,
    )

    start_time = time.time()
    new_records: list[dict[str, Any]] = []
    point_hits = 0
    num_points = 0
    sample_successes = 0
    parsed_samples = 0
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
                points = parse_points(prediction, width, height, args.coordinate_scale)
                hits, total, hit_rate, sample_success = score_points(points, sample.mask, args.mask_threshold)
                point_hits += hits
                num_points += total
                sample_successes += int(sample_success)
                parsed_samples += int(total > 0)
                record = {
                    "sample_id": sample.sample_id,
                    "question_id": sample.question_id,
                    "question_type": sample.question_type,
                    "question": sample.question,
                    "prediction": prediction.strip(),
                    "points": points,
                    "point_hits": hits,
                    "num_points": total,
                    "point_accuracy": hit_rate,
                    "sample_success": sample_success,
                    "image_size": [width, height],
                    "mask_threshold": args.mask_threshold,
                }
                new_records.append(record)
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                handle.flush()

            processed = len(new_records)
            print(
                f"[{batch_index}] processed={processed}/{len(samples_to_run)} "
                f"point_accuracy={point_hits / max(num_points, 1):.4f} "
                f"sample_success_rate={sample_successes / max(processed, 1):.4f}",
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

    num_samples = len(all_records)
    point_hits = sum(int(record["point_hits"]) for record in all_records)
    num_points = sum(int(record["num_points"]) for record in all_records)
    sample_successes = sum(int(record["sample_success"]) for record in all_records)
    parsed_samples = sum(int(record["num_points"] > 0) for record in all_records)
    elapsed = time.time() - start_time
    summary = {
        "dataset_path": str(dataset_path),
        "ckpt_path": str(ckpt_path),
        "processor_path": processor_path,
        "split": args.split,
        "num_samples": num_samples,
        "parsed_samples": parsed_samples,
        "parse_rate": parsed_samples / max(num_samples, 1),
        "point_hits": point_hits,
        "num_points": num_points,
        "point_accuracy": point_hits / max(num_points, 1),
        "successful_samples": sample_successes,
        "sample_success_rate": sample_successes / max(num_samples, 1),
        "by_question_type": build_type_metrics(all_records),
        "elapsed_seconds": elapsed,
        "predictions_path": str(predictions_path),
        "mask_threshold": args.mask_threshold,
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
