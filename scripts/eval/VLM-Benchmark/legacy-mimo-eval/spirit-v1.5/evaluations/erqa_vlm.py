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
DEFAULT_DATASET_PATH = "datasets/public_datasets/VLM/vqa/benchmarks/ERQA"


@dataclass
class ERQASample:
    sample_id: int
    question_id: str
    question_type: str
    question: str
    answer: str
    visual_indices: list[int]
    selected_indices: list[int]
    images: list[Image.Image]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate the Spirit-v1.5 Qwen3-VL backbone on ERQA.",
    )
    parser.add_argument("--ckpt-path", default=DEFAULT_CKPT_PATH, help="Spirit-v1.5 checkpoint directory.")
    parser.add_argument("--dataset-path", default=DEFAULT_DATASET_PATH, help="ERQA dataset directory.")
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
    parser.add_argument("--output-dir", default="outputs/erqa_vlm", help="Directory for JSONL and summary outputs.")
    parser.add_argument("--split", default="test", help="Dataset split name when loading through datasets.")
    parser.add_argument("--max-samples", type=int, default=None, help="Optional cap for quick smoke tests.")
    parser.add_argument("--num-shards", type=int, default=1, help="Total number of data shards for parallel evaluation.")
    parser.add_argument("--shard-index", type=int, default=0, help="Shard index for this process, in [0, num_shards).")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Generation batch size. Keep 1 if processor batching with variable image counts fails.",
    )
    parser.add_argument("--device", default="cuda", help="Device for inference, e.g. cuda, cuda:0, or cpu.")
    parser.add_argument(
        "--dtype",
        choices=("auto", "float32", "float16", "bfloat16"),
        default="bfloat16",
        help="Dtype used after loading the model.",
    )
    parser.add_argument("--attn-implementation", default=None, help="Override attention implementation from config.")
    parser.add_argument("--max-new-tokens", type=int, default=16, help="Maximum generated answer tokens.")
    parser.add_argument("--temperature", type=float, default=0.0, help="Generation temperature. 0 means greedy.")
    parser.add_argument("--top-p", type=float, default=1.0, help="Nucleus sampling value when temperature > 0.")
    parser.add_argument("--num-beams", type=int, default=1, help="Beam count for generation.")
    parser.add_argument(
        "--system-prompt",
        default="You are a helpful visual question answering assistant.",
        help="System prompt used for VQA generation.",
    )
    parser.add_argument(
        "--question-suffix",
        default="",
        help="Optional suffix appended to each ERQA question.",
    )
    parser.add_argument(
        "--use-visual-indices",
        action="store_true",
        default=True,
        help="Use only de-duplicated images referenced by visual_indices when the list is non-empty. Enabled by default.",
    )
    parser.add_argument(
        "--use-all-images",
        dest="use_visual_indices",
        action="store_false",
        help="Ignore visual_indices and pass all images from each ERQA row.",
    )
    parser.add_argument(
        "--max-images-per-sample",
        type=int,
        default=1,
        help="Maximum images passed to the VLM for one ERQA sample. Use 0 to disable the cap.",
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
    raise TypeError(f"Unsupported ERQA image object: {type(image_obj)!r}")


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


def select_images(
    images: list[Image.Image],
    visual_indices: list[int],
    use_visual_indices: bool,
    max_images_per_sample: int,
    max_image_pixels: int,
) -> tuple[list[Image.Image], list[int]]:
    selected_indices: list[int]
    if not use_visual_indices or not visual_indices:
        selected_indices = list(range(len(images)))
    else:
        selected_indices = []
        seen = set()
        for idx in visual_indices:
            if 0 <= idx < len(images) and idx not in seen:
                selected_indices.append(idx)
                seen.add(idx)
        if not selected_indices:
            selected_indices = list(range(len(images)))

    if max_images_per_sample > 0:
        selected_indices = selected_indices[:max_images_per_sample]

    selected = [resize_image(images[idx], max_image_pixels) for idx in selected_indices]
    fallback = [resize_image(image, max_image_pixels) for image in images[: max_images_per_sample or None]]
    return selected or fallback, selected_indices


def load_erqa(
    dataset_path: Path,
    split: str,
    max_samples: int | None,
    use_visual_indices: bool,
    max_images_per_sample: int,
    max_image_pixels: int,
) -> list[ERQASample]:
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise ImportError(
            "ERQA is stored as parquet with an image sequence column. Install datasets/pyarrow "
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

    samples: list[ERQASample] = []
    for idx, row in enumerate(dataset):
        visual_indices = [int(value) for value in row.get("visual_indices", [])]
        images = [decode_image(image_obj) for image_obj in row["images"]]
        selected_images, selected_indices = select_images(
            images,
            visual_indices,
            use_visual_indices,
            max_images_per_sample,
            max_image_pixels,
        )
        samples.append(
            ERQASample(
                sample_id=idx,
                question_id=str(row["question_id"]),
                question_type=str(row["question_type"]),
                question=str(row["question"]),
                answer=str(row["answer"]),
                visual_indices=visual_indices,
                selected_indices=selected_indices,
                images=selected_images,
            )
        )
    return samples


def chunked(items: list[ERQASample], size: int) -> Iterable[list[ERQASample]]:
    for start in range(0, len(items), size):
        yield items[start : start + size]


def shard_samples(samples: list[ERQASample], num_shards: int, shard_index: int) -> list[ERQASample]:
    if num_shards < 1:
        raise ValueError("--num-shards must be >= 1")
    if shard_index < 0 or shard_index >= num_shards:
        raise ValueError("--shard-index must be in [0, num_shards)")
    if num_shards == 1:
        return samples
    return [sample for offset, sample in enumerate(samples) if offset % num_shards == shard_index]


def build_messages(sample: ERQASample, system_prompt: str, question_suffix: str) -> list[dict[str, Any]]:
    question = sample.question.strip()
    if question_suffix:
        question = f"{question}\n{question_suffix.strip()}"

    content: list[dict[str, Any]] = []
    for image in sample.images:
        content.append({"type": "image", "image": image})
    content.append({"type": "text", "text": question})
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": content},
    ]


def prepare_inputs(
    processor: Any,
    samples: list[ERQASample],
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
    images: list[Any]
    if len(samples) == 1:
        images = samples[0].images
    else:
        images = [sample.images for sample in samples]
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


def normalize_text(text: str) -> str:
    text = text.strip().lower()
    text = re.sub(r"^[\"'`]+|[\"'`.。,:;!?]+$", "", text)
    text = re.sub(r"\b(the|a|an)\b", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def extract_choice(text: str) -> str | None:
    patterns = [
        r"^\s*[\(\[]?\s*([A-Da-d])\s*[\)\].:：]?",
        r"\b(?:answer|option|choice)\s*(?:is|:)?\s*[\(\[]?\s*([A-Da-d])\b",
        r"\b([A-Da-d])\b",
    ]
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            return match.group(1).upper()
    return None


def score_prediction(prediction: str, answer: str) -> tuple[bool, str, str]:
    pred_norm = extract_choice(prediction) or normalize_text(prediction).upper()
    ans_norm = answer.strip().upper()
    return pred_norm == ans_norm, pred_norm, ans_norm


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
        bucket = metrics.setdefault(key, {"num_samples": 0, "num_correct": 0, "accuracy": 0.0})
        bucket["num_samples"] += 1
        bucket["num_correct"] += int(record["correct"])
    for bucket in metrics.values():
        bucket["accuracy"] = bucket["num_correct"] / max(bucket["num_samples"], 1)
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

    all_samples = load_erqa(
        dataset_path,
        args.split,
        args.max_samples,
        args.use_visual_indices,
        args.max_images_per_sample,
        args.max_image_pixels,
    )
    samples = shard_samples(all_samples, args.num_shards, args.shard_index)
    predictions_path = output_dir / "predictions.jsonl"
    seen_ids = load_seen_ids(predictions_path) if args.resume else set()
    samples_to_run = [sample for sample in samples if sample.sample_id not in seen_ids]
    print(
        f"Loaded {len(all_samples)} ERQA samples; shard {args.shard_index}/{args.num_shards} "
        f"has {len(samples)} samples; evaluating {len(samples_to_run)} "
        f"(resume skipped {len(seen_ids)}).",
        flush=True,
    )

    start_time = time.time()
    new_records: list[dict[str, Any]] = []
    total = 0
    correct = 0
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
                is_correct, pred_norm, ans_norm = score_prediction(prediction, sample.answer)
                total += 1
                correct += int(is_correct)
                record = {
                    "sample_id": sample.sample_id,
                    "question_id": sample.question_id,
                    "question_type": sample.question_type,
                    "question": sample.question,
                    "answer": sample.answer,
                    "prediction": prediction.strip(),
                    "normalized_answer": ans_norm,
                    "normalized_prediction": pred_norm,
                    "correct": is_correct,
                    "visual_indices": sample.visual_indices,
                    "selected_indices": sample.selected_indices,
                    "num_images": len(sample.images),
                }
                new_records.append(record)
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                handle.flush()

            running_accuracy = correct / max(total, 1)
            print(
                f"[{batch_index}] processed={total}/{len(samples_to_run)} "
                f"running_accuracy={running_accuracy:.4f}",
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

    total = len(all_records)
    correct = sum(int(record["correct"]) for record in all_records)
    elapsed = time.time() - start_time
    summary = {
        "dataset_path": str(dataset_path),
        "ckpt_path": str(ckpt_path),
        "processor_path": processor_path,
        "split": args.split,
        "num_total_loaded": len(all_samples),
        "num_shards": args.num_shards,
        "shard_index": args.shard_index,
        "num_samples": total,
        "num_correct": correct,
        "accuracy": correct / max(total, 1),
        "by_question_type": build_type_metrics(all_records),
        "elapsed_seconds": elapsed,
        "predictions_path": str(predictions_path),
        "use_visual_indices": args.use_visual_indices,
        "max_images_per_sample": args.max_images_per_sample,
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
