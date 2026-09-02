import argparse
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
DEFAULT_DATASET_PATH = "datasets/public_datasets/VLM/vqa/CRPE"
DEFAULT_SUFFIX = "Answer directly with only the option letter and nothing else."
CHOICE_LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


@dataclass
class CRPESample:
    sample_id: int
    group_id: str
    rotation_index: int
    question_id: int
    split: str
    category: str
    image_path: str
    question: str
    answer: str
    image: Image.Image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate the Spirit-v1.5 Qwen3-VL backbone on CRPE.",
    )
    parser.add_argument("--ckpt-path", default=DEFAULT_CKPT_PATH, help="Spirit-v1.5 checkpoint directory.")
    parser.add_argument("--dataset-path", default=DEFAULT_DATASET_PATH, help="CRPE dataset directory.")
    parser.add_argument(
        "--image-root",
        default=None,
        help="Optional root for relative CRPE image paths, e.g. a directory containing coco/val2017.",
    )
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
    parser.add_argument("--output-dir", default="outputs/crpe_vlm", help="Directory for outputs.")
    parser.add_argument(
        "--split",
        choices=("all", "exist", "relation"),
        default="all",
        help="CRPE split to evaluate.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Optional cap on original circular groups. Each group has four rotated prompts.",
    )
    parser.add_argument("--batch-size", type=int, default=1, help="Generation batch size.")
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
    parser.add_argument("--question-suffix", default=DEFAULT_SUFFIX, help="Optional suffix appended to each question.")
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


def crpe_files(dataset_path: Path, split: str) -> list[tuple[str, Path]]:
    files = []
    if split in {"all", "exist"}:
        files.append(("exist", dataset_path / "crpe_exist.jsonl"))
    if split in {"all", "relation"}:
        files.append(("relation", dataset_path / "crpe_relation.jsonl"))
    missing = [path for _, path in files if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing CRPE jsonl files: {missing}")
    return files


def candidate_roots(dataset_path: Path, image_root: str | None) -> list[Path]:
    roots: list[Path] = []
    if image_root:
        roots.append(Path(image_root).expanduser().resolve())
    roots.extend(
        [
            dataset_path,
            dataset_path.parent,
            Path("datasets/public_datasets"),
            Path("datasets/public_datasets/VLM"),
            Path("datasets/public_datasets/COCO"),
            Path("datasets/public_datasets/coco"),
            Path("datasets/public_datasets/coco2017"),
            Path("datasets/public_datasets/MSCOCO"),
            Path("datasets/public_datasets/vision/COCO"),
        ]
    )
    deduped = []
    seen = set()
    for root in roots:
        if root not in seen:
            deduped.append(root)
            seen.add(root)
    return deduped


def resolve_image_path(image_path: str, roots: list[Path]) -> Path:
    path = Path(image_path)
    if path.is_absolute() and path.exists():
        return path
    candidates = []
    for root in roots:
        candidates.append(root / image_path)
        if image_path.startswith("coco/"):
            candidates.append(root / image_path.removeprefix("coco/"))
            candidates.append(root / "COCO" / image_path.removeprefix("coco/"))
        if image_path.startswith("abnormal_images/"):
            candidates.append(root / "CRPE" / image_path)
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        f"CRPE image not found for {image_path!r}. Pass --image-root pointing to a directory containing coco/val2017."
    )


def circular_group_id(split: str, question_id: int, num_choices: int = 4) -> str:
    return f"{split}:{question_id // num_choices}"


def rotation_index(question_id: int, num_choices: int = 4) -> int:
    return question_id % num_choices


def load_crpe(
    dataset_path: Path,
    image_root: str | None,
    split: str,
    max_samples: int | None,
) -> list[CRPESample]:
    roots = candidate_roots(dataset_path, image_root)
    samples: list[CRPESample] = []
    seen_groups: set[str] = set()
    for split_name, path in crpe_files(dataset_path, split):
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                question_id = int(row["question_id"])
                group_id = circular_group_id(split_name, question_id)
                if group_id not in seen_groups:
                    if max_samples is not None and len(seen_groups) >= max_samples:
                        return samples
                    seen_groups.add(group_id)
                image_path = str(row["image"])
                samples.append(
                    CRPESample(
                        sample_id=len(samples),
                        group_id=group_id,
                        rotation_index=rotation_index(question_id),
                        question_id=question_id,
                        split=split_name,
                        category=str(row["category"]),
                        image_path=image_path,
                        question=str(row["text"]),
                        answer=str(row["correct_option"]).upper(),
                        image=Image.open(resolve_image_path(image_path, roots)).convert("RGB"),
                    )
                )
    return samples


def chunked(items: list[CRPESample], size: int) -> Iterable[list[CRPESample]]:
    for start in range(0, len(items), size):
        yield items[start : start + size]


def build_messages(sample: CRPESample, system_prompt: str, question_suffix: str) -> list[dict[str, Any]]:
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
    samples: list[CRPESample],
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


def extract_choice(text: str) -> str | None:
    patterns = [
        r"\b(?:answer|option|choice)\s*(?:is|:)?\s*[\(\[]?\s*([A-Da-d])\b",
        r"\(([A-Da-d])\)",
        r"^\s*[\(\[]?\s*([A-Da-d])\s*[\)\].:：]?(?:\s|$)",
        r"(?<![A-Za-z])([A-Da-d])(?![A-Za-z])",
    ]
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            return match.group(1).upper()
    return None


def group_metrics(records: list[dict[str, Any]], key: str) -> dict[str, dict[str, float | int]]:
    metrics: dict[str, dict[str, float | int]] = {}
    for record in records:
        group = str(record[key])
        bucket = metrics.setdefault(group, {"num_prompts": 0, "num_correct": 0, "accuracy": 0.0})
        bucket["num_prompts"] += 1
        bucket["num_correct"] += int(record["correct"])
    for bucket in metrics.values():
        bucket["accuracy"] = bucket["num_correct"] / max(bucket["num_prompts"], 1)
    return metrics


def circular_metrics(records: list[dict[str, Any]]) -> dict[str, float | int]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        grouped.setdefault(record["group_id"], []).append(record)
    num_groups = len(grouped)
    if num_groups == 0:
        return {"num_groups": 0, "circular_correct": 0, "circular_accuracy": 0.0, "mean_rotation_accuracy": 0.0}

    circular_correct = 0
    rotation_accuracies = []
    for group_records in grouped.values():
        num_correct = sum(int(record["correct"]) for record in group_records)
        circular_correct += int(num_correct == len(group_records))
        rotation_accuracies.append(num_correct / max(len(group_records), 1))
    return {
        "num_groups": num_groups,
        "circular_correct": circular_correct,
        "circular_accuracy": circular_correct / num_groups,
        "mean_rotation_accuracy": sum(rotation_accuracies) / num_groups,
    }


def load_seen_ids(predictions_path: Path) -> set[int]:
    if not predictions_path.exists():
        return set()
    seen: set[int] = set()
    with predictions_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                seen.add(int(json.loads(line)["sample_id"]))
    return seen


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

    samples = load_crpe(dataset_path, args.image_root, args.split, args.max_samples)
    predictions_path = output_dir / "predictions.jsonl"
    seen_ids = load_seen_ids(predictions_path) if args.resume else set()
    samples_to_run = [sample for sample in samples if sample.sample_id not in seen_ids]
    print(
        f"Loaded {len(samples)} CRPE prompts; evaluating {len(samples_to_run)} "
        f"(resume skipped {len(seen_ids)}).",
        flush=True,
    )

    start_time = time.time()
    new_records: list[dict[str, Any]] = []
    correct = 0
    parseable = 0
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
                pred_norm = extract_choice(prediction) or ""
                is_correct = pred_norm == sample.answer
                correct += int(is_correct)
                parseable += int(bool(pred_norm))
                record = {
                    "sample_id": sample.sample_id,
                    "group_id": sample.group_id,
                    "rotation_index": sample.rotation_index,
                    "question_id": sample.question_id,
                    "split": sample.split,
                    "category": sample.category,
                    "image_path": sample.image_path,
                    "question": sample.question,
                    "answer": sample.answer,
                    "prediction": prediction.strip(),
                    "normalized_prediction": pred_norm,
                    "correct": is_correct,
                }
                new_records.append(record)
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                handle.flush()

            processed = len(new_records)
            print(
                f"[{batch_index}] processed={processed}/{len(samples_to_run)} "
                f"running_accuracy={correct / max(processed, 1):.4f} "
                f"parse_rate={parseable / max(processed, 1):.4f}",
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

    num_prompts = len(all_records)
    num_correct = sum(int(record["correct"]) for record in all_records)
    num_parseable = sum(int(bool(record["normalized_prediction"])) for record in all_records)
    summary = {
        "dataset_path": str(dataset_path),
        "image_root": args.image_root,
        "ckpt_path": str(ckpt_path),
        "processor_path": processor_path,
        "split": args.split,
        "num_prompts": num_prompts,
        "num_correct": num_correct,
        "accuracy": num_correct / max(num_prompts, 1),
        "parse_rate": num_parseable / max(num_prompts, 1),
        "circular": circular_metrics(all_records),
        "by_split": group_metrics(all_records, "split"),
        "by_category": group_metrics(all_records, "category"),
        "elapsed_seconds": time.time() - start_time,
        "predictions_path": str(predictions_path),
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
