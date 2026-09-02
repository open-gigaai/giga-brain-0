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
import pandas as pd
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
DEFAULT_DATASET_PATH = "datasets/public_datasets/VLM/vqa/RoboSpatial-Home"
CONTEXT_SUFFIX = (
    "Return several candidate points inside the requested empty space. "
    "Use normalized coordinates between 0 and 1. "
    "Answer only as a list of tuples: [(x1, y1), (x2, y2), ...]."
)
BINARY_SUFFIX = "Answer only yes or no."


@dataclass
class RoboSpatialHomeSample:
    sample_id: int
    category: str
    question: str
    answer: str
    image: Image.Image
    depth_image: Image.Image
    mask: Image.Image | None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate the Spirit-v1.5 Qwen3-VL backbone on RoboSpatial-Home.",
    )
    parser.add_argument("--ckpt-path", default=DEFAULT_CKPT_PATH, help="Spirit-v1.5 checkpoint directory.")
    parser.add_argument("--dataset-path", default=DEFAULT_DATASET_PATH, help="RoboSpatial-Home dataset directory.")
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
    parser.add_argument("--output-dir", default="outputs/robospatial_home_vlm", help="Directory for outputs.")
    parser.add_argument(
        "--category",
        choices=("all", "configuration", "compatibility", "context"),
        default="all",
        help="RoboSpatial-Home category to evaluate.",
    )
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
        default="You are a helpful robotic spatial reasoning assistant.",
        help="System prompt used for VQA generation.",
    )
    parser.add_argument("--binary-suffix", default=BINARY_SUFFIX, help="Instruction appended to yes/no questions.")
    parser.add_argument("--context-suffix", default=CONTEXT_SUFFIX, help="Instruction appended to context questions.")
    parser.add_argument(
        "--use-depth",
        action="store_true",
        help="Pass the depth image as a second image after the RGB image.",
    )
    parser.add_argument("--mask-threshold", type=int, default=127, help="Mask threshold for valid context regions.")
    parser.add_argument(
        "--coordinate-scale",
        choices=("normalized", "pixels"),
        default="normalized",
        help="Expected coordinate scale in context model outputs.",
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


def decode_image(image_obj: Any, mode: str) -> Image.Image | None:
    if image_obj is None or (isinstance(image_obj, float) and pd.isna(image_obj)):
        return None
    if isinstance(image_obj, Image.Image):
        return image_obj.convert(mode)
    if isinstance(image_obj, dict):
        if image_obj.get("bytes") is not None:
            return Image.open(io.BytesIO(image_obj["bytes"])).convert(mode)
        if image_obj.get("path"):
            return Image.open(image_obj["path"]).convert(mode)
    if isinstance(image_obj, (bytes, bytearray)):
        return Image.open(io.BytesIO(image_obj)).convert(mode)
    if isinstance(image_obj, (str, Path)):
        return Image.open(image_obj).convert(mode)
    raise TypeError(f"Unsupported RoboSpatial-Home image object: {type(image_obj)!r}")


def category_files(dataset_path: Path, category: str) -> list[Path]:
    data_dir = dataset_path / "data"
    categories = ["configuration", "compatibility", "context"] if category == "all" else [category]
    paths = [data_dir / f"{name}-00000-of-00001.parquet" for name in categories]
    missing = [path for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing RoboSpatial-Home parquet files: {missing}")
    return paths


def load_robospatial_home(dataset_path: Path, category: str, max_samples: int | None) -> list[RoboSpatialHomeSample]:
    frames = [pd.read_parquet(path) for path in category_files(dataset_path, category)]
    df = pd.concat(frames, ignore_index=True)
    if max_samples is not None:
        df = df.head(max_samples)

    samples: list[RoboSpatialHomeSample] = []
    for sample_id, row in enumerate(df.itertuples(index=False)):
        image = decode_image(row.img, "RGB")
        depth_image = decode_image(row.depth_image, "RGB")
        mask = decode_image(row.mask, "L")
        if image is None or depth_image is None:
            raise ValueError(f"Missing RGB or depth image for sample_id={sample_id}")
        samples.append(
            RoboSpatialHomeSample(
                sample_id=sample_id,
                category=str(row.category),
                question=str(row.question),
                answer=str(row.answer),
                image=image,
                depth_image=depth_image,
                mask=mask,
            )
        )
    return samples


def chunked(items: list[RoboSpatialHomeSample], size: int) -> Iterable[list[RoboSpatialHomeSample]]:
    for start in range(0, len(items), size):
        yield items[start : start + size]


def build_question(sample: RoboSpatialHomeSample, binary_suffix: str, context_suffix: str) -> str:
    question = sample.question.strip()
    if sample.category == "context":
        suffix = context_suffix
    else:
        suffix = binary_suffix
    if suffix:
        question = f"{question}\n{suffix.strip()}"
    return question


def build_messages(
    sample: RoboSpatialHomeSample,
    system_prompt: str,
    binary_suffix: str,
    context_suffix: str,
    use_depth: bool,
) -> list[dict[str, Any]]:
    content: list[dict[str, Any]] = [{"type": "image", "image": sample.image}]
    if use_depth:
        content.append({"type": "image", "image": sample.depth_image})
    content.append({"type": "text", "text": build_question(sample, binary_suffix, context_suffix)})
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": content},
    ]


def prepare_inputs(
    processor: Any,
    samples: list[RoboSpatialHomeSample],
    system_prompt: str,
    binary_suffix: str,
    context_suffix: str,
    use_depth: bool,
    device: torch.device,
    dtype: torch.dtype | None,
) -> dict[str, torch.Tensor]:
    texts = [
        processor.apply_chat_template(
            build_messages(sample, system_prompt, binary_suffix, context_suffix, use_depth),
            tokenize=False,
            add_generation_prompt=True,
        )
        for sample in samples
    ]
    if use_depth:
        images: list[Any] = [[sample.image, sample.depth_image] for sample in samples]
        if len(samples) == 1:
            images = images[0]
    else:
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


def normalize_yes_no(text: str) -> str:
    text_norm = re.sub(r"[^a-z]+", " ", text.lower()).strip()
    tokens = text_norm.split()
    if not tokens:
        return ""
    if tokens[0] in {"yes", "no"}:
        return tokens[0]
    if "yes" in tokens and "no" not in tokens:
        return "yes"
    if "no" in tokens and "yes" not in tokens:
        return "no"
    return ""


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


def score_points(points: list[tuple[int, int]], mask: Image.Image | None, threshold: int) -> tuple[int, int, float, bool]:
    if mask is None:
        return 0, len(points), 0.0, False
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


def category_metrics(records: list[dict[str, Any]]) -> dict[str, dict[str, float | int]]:
    metrics: dict[str, dict[str, float | int]] = {}
    for record in records:
        category = record["category"]
        bucket = metrics.setdefault(
            category,
            {
                "num_samples": 0,
                "num_correct": 0,
                "parsed_samples": 0,
                "point_hits": 0,
                "num_points": 0,
                "successful_samples": 0,
                "accuracy": 0.0,
                "parse_rate": 0.0,
                "point_accuracy": 0.0,
                "sample_success_rate": 0.0,
            },
        )
        bucket["num_samples"] += 1
        if category == "context":
            bucket["parsed_samples"] += int(record["num_points"] > 0)
            bucket["point_hits"] += int(record["point_hits"])
            bucket["num_points"] += int(record["num_points"])
            bucket["successful_samples"] += int(record["sample_success"])
        else:
            bucket["parsed_samples"] += int(bool(record["normalized_prediction"]))
            bucket["num_correct"] += int(record["correct"])
    for category, bucket in metrics.items():
        if category == "context":
            bucket["parse_rate"] = bucket["parsed_samples"] / max(bucket["num_samples"], 1)
            bucket["point_accuracy"] = bucket["point_hits"] / max(bucket["num_points"], 1)
            bucket["sample_success_rate"] = bucket["successful_samples"] / max(bucket["num_samples"], 1)
        else:
            bucket["accuracy"] = bucket["num_correct"] / max(bucket["num_samples"], 1)
            bucket["parse_rate"] = bucket["parsed_samples"] / max(bucket["num_samples"], 1)
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

    samples = load_robospatial_home(dataset_path, args.category, args.max_samples)
    predictions_path = output_dir / "predictions.jsonl"
    seen_ids = load_seen_ids(predictions_path) if args.resume else set()
    samples_to_run = [sample for sample in samples if sample.sample_id not in seen_ids]
    print(
        f"Loaded {len(samples)} RoboSpatial-Home samples; evaluating {len(samples_to_run)} "
        f"(resume skipped {len(seen_ids)}).",
        flush=True,
    )

    start_time = time.time()
    new_records: list[dict[str, Any]] = []
    binary_correct = 0
    binary_total = 0
    point_hits = 0
    num_points = 0
    mode = "a" if args.resume else "w"
    with predictions_path.open(mode, encoding="utf-8") as handle:
        for batch_index, batch in enumerate(chunked(samples_to_run, args.batch_size), start=1):
            inputs = prepare_inputs(
                processor,
                batch,
                args.system_prompt,
                args.binary_suffix,
                args.context_suffix,
                args.use_depth,
                device,
                dtype,
            )
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
                if sample.category == "context":
                    points = parse_points(prediction, width, height, args.coordinate_scale)
                    hits, total, hit_rate, sample_success = score_points(points, sample.mask, args.mask_threshold)
                    point_hits += hits
                    num_points += total
                    record = {
                        "sample_id": sample.sample_id,
                        "category": sample.category,
                        "question": sample.question,
                        "answer": sample.answer,
                        "prediction": prediction.strip(),
                        "normalized_answer": "",
                        "normalized_prediction": "",
                        "correct": False,
                        "points": points,
                        "point_hits": hits,
                        "num_points": total,
                        "point_accuracy": hit_rate,
                        "sample_success": sample_success,
                        "image_size": [width, height],
                        "mask_threshold": args.mask_threshold,
                    }
                else:
                    pred_norm = normalize_yes_no(prediction)
                    ans_norm = normalize_yes_no(sample.answer)
                    is_correct = pred_norm == ans_norm and bool(pred_norm)
                    binary_correct += int(is_correct)
                    binary_total += 1
                    record = {
                        "sample_id": sample.sample_id,
                        "category": sample.category,
                        "question": sample.question,
                        "answer": sample.answer,
                        "prediction": prediction.strip(),
                        "normalized_answer": ans_norm,
                        "normalized_prediction": pred_norm,
                        "correct": is_correct,
                        "points": [],
                        "point_hits": 0,
                        "num_points": 0,
                        "point_accuracy": 0.0,
                        "sample_success": False,
                        "image_size": [width, height],
                        "mask_threshold": args.mask_threshold,
                    }
                new_records.append(record)
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                handle.flush()

            processed = len(new_records)
            print(
                f"[{batch_index}] processed={processed}/{len(samples_to_run)} "
                f"binary_accuracy={binary_correct / max(binary_total, 1):.4f} "
                f"context_point_accuracy={point_hits / max(num_points, 1):.4f}",
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

    binary_records = [record for record in all_records if record["category"] != "context"]
    context_records = [record for record in all_records if record["category"] == "context"]
    binary_correct = sum(int(record["correct"]) for record in binary_records)
    binary_parsed = sum(int(bool(record["normalized_prediction"])) for record in binary_records)
    point_hits = sum(int(record["point_hits"]) for record in context_records)
    num_points = sum(int(record["num_points"]) for record in context_records)
    context_parsed = sum(int(record["num_points"] > 0) for record in context_records)
    context_success = sum(int(record["sample_success"]) for record in context_records)

    summary = {
        "dataset_path": str(dataset_path),
        "ckpt_path": str(ckpt_path),
        "processor_path": processor_path,
        "category": args.category,
        "num_samples": len(all_records),
        "binary_num_samples": len(binary_records),
        "binary_accuracy": binary_correct / max(len(binary_records), 1),
        "binary_parse_rate": binary_parsed / max(len(binary_records), 1),
        "context_num_samples": len(context_records),
        "context_parse_rate": context_parsed / max(len(context_records), 1),
        "context_point_hits": point_hits,
        "context_num_points": num_points,
        "context_point_accuracy": point_hits / max(num_points, 1),
        "context_successful_samples": context_success,
        "context_sample_success_rate": context_success / max(len(context_records), 1),
        "by_category": category_metrics(all_records),
        "elapsed_seconds": time.time() - start_time,
        "predictions_path": str(predictions_path),
        "mask_threshold": args.mask_threshold,
        "coordinate_scale": args.coordinate_scale,
        "use_depth": args.use_depth,
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
