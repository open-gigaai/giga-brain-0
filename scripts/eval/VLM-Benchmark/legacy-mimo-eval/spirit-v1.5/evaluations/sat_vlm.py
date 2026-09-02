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
DEFAULT_DATASET_PATH = "datasets/public_datasets/VLM/vqa/benchmarks/SAT"
DEFAULT_SUFFIX = "Please answer directly with only the letter of the correct option and nothing else."
CHOICE_LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


@dataclass
class SATSample:
    sample_id: int
    eval_id: str
    rotation_index: int
    question_type: str
    question: str
    answer_options: list[str]
    answer_index: int
    correct_answer: str
    images: list[Image.Image]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate the Spirit-v1.5 Qwen3-VL backbone on SAT.",
    )
    parser.add_argument("--ckpt-path", default=DEFAULT_CKPT_PATH, help="Spirit-v1.5 checkpoint directory.")
    parser.add_argument("--dataset-path", default=DEFAULT_DATASET_PATH, help="SAT dataset directory.")
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
    parser.add_argument("--output-dir", default="outputs/sat_vlm", help="Directory for outputs.")
    parser.add_argument(
        "--split",
        choices=("test", "val", "train", "static"),
        default="test",
        help="SAT split to evaluate.",
    )
    parser.add_argument(
        "--circular-eval",
        dest="circular_eval",
        action="store_true",
        default=True,
        help="Evaluate every cyclic answer-option rotation. Enabled by default.",
    )
    parser.add_argument(
        "--no-circular-eval",
        dest="circular_eval",
        action="store_false",
        help="Evaluate only the answer order stored in the parquet file.",
    )
    parser.add_argument("--max-samples", type=int, default=None, help="Optional cap for quick smoke tests.")
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
    parser.add_argument(
        "--max-image-pixels",
        type=int,
        default=262144,
        help="Resize each image to at most this many pixels before processing. Use 0 to disable resizing.",
    )
    parser.add_argument("--temperature", type=float, default=0.0, help="Generation temperature. 0 means greedy.")
    parser.add_argument("--top-p", type=float, default=1.0, help="Nucleus sampling value when temperature > 0.")
    parser.add_argument("--num-beams", type=int, default=1, help="Beam count for generation.")
    parser.add_argument(
        "--system-prompt",
        default="You are a helpful visual question answering assistant.",
        help="System prompt used for VQA generation.",
    )
    parser.add_argument("--question-suffix", default=DEFAULT_SUFFIX, help="Optional suffix appended to each prompt.")
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
    if isinstance(image_obj, (bytes, bytearray)):
        return Image.open(io.BytesIO(image_obj)).convert("RGB")
    if isinstance(image_obj, (str, Path)):
        return Image.open(image_obj).convert("RGB")
    raise TypeError(f"Unsupported SAT image object: {type(image_obj)!r}")


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


def split_path(dataset_path: Path, split: str) -> Path:
    path = dataset_path / f"SAT_{split}.parquet"
    if not path.exists():
        raise FileNotFoundError(f"SAT split parquet not found: {path}")
    return path


def normalize_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return " ".join(text.split())


def rotate_options(options: list[str], shift: int) -> list[str]:
    shift = shift % len(options)
    return options[shift:] + options[:shift]


def make_rotations(options: list[str], correct_answer: str, circular_eval: bool) -> list[tuple[int, list[str], int]]:
    shifts = range(len(options)) if circular_eval else range(1)
    rotations = []
    normalized_correct = normalize_text(correct_answer)
    for shift in shifts:
        rotated = rotate_options(options, shift)
        normalized_options = [normalize_text(option) for option in rotated]
        if normalized_correct not in normalized_options:
            raise ValueError(f"Correct answer not found in answer options: {correct_answer!r}")
        rotations.append((shift, rotated, normalized_options.index(normalized_correct)))
    return rotations


def load_sat(
    dataset_path: Path,
    split: str,
    max_samples: int | None,
    circular_eval: bool,
    max_image_pixels: int,
) -> list[SATSample]:
    df = pd.read_parquet(split_path(dataset_path, split))
    if max_samples is not None:
        df = df.head(max_samples)

    samples: list[SATSample] = []
    for sample_id, row in enumerate(df.itertuples(index=False)):
        images = [resize_image(decode_image(image_obj), max_image_pixels) for image_obj in row.image_bytes]
        answers = [str(answer) for answer in row.answers]
        correct_answer = str(row.correct_answer)
        for shift, rotated_answers, answer_index in make_rotations(answers, correct_answer, circular_eval):
            samples.append(
                SATSample(
                    sample_id=sample_id,
                    eval_id=f"{sample_id}:rot{shift}",
                    rotation_index=shift,
                    question_type=str(row.question_type),
                    question=str(row.question),
                    answer_options=rotated_answers,
                    answer_index=answer_index,
                    correct_answer=correct_answer,
                    images=images,
                )
            )
    return samples


def chunked(items: list[SATSample], size: int) -> Iterable[list[SATSample]]:
    for start in range(0, len(items), size):
        yield items[start : start + size]


def format_options(options: list[str]) -> str:
    return "\n".join(f"{CHOICE_LETTERS[index]}. {option}" for index, option in enumerate(options))


def build_messages(sample: SATSample, system_prompt: str, question_suffix: str) -> list[dict[str, Any]]:
    question = f"{sample.question.strip()}\nOptions:\n{format_options(sample.answer_options)}"
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
    samples: list[SATSample],
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


def answer_letter(sample: SATSample) -> str:
    return CHOICE_LETTERS[sample.answer_index]


def extract_choice(text: str, options: list[str]) -> str | None:
    max_letter = CHOICE_LETTERS[len(options) - 1]
    patterns = [
        rf"\b(?:answer|option|choice)\s*(?:is|:)?\s*[\(\[]?\s*([A-{max_letter}a-{max_letter.lower()}])\b",
        rf"\(([A-{max_letter}a-{max_letter.lower()}])\)",
        rf"^\s*[\(\[]?\s*([A-{max_letter}a-{max_letter.lower()}])\s*[\)\].:：]?(?:\s|$)",
        rf"(?<![A-Za-z])([A-{max_letter}a-{max_letter.lower()}])(?![A-Za-z])",
    ]
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            return match.group(1).upper()

    normalized_prediction = normalize_text(text)
    for index, option in enumerate(options):
        normalized_option = normalize_text(option)
        if normalized_option and normalized_option in normalized_prediction:
            return CHOICE_LETTERS[index]
    return None


def build_group_metrics(records: list[dict[str, Any]], key: str) -> dict[str, dict[str, float | int]]:
    metrics: dict[str, dict[str, float | int]] = {}
    for record in records:
        group = record[key]
        bucket = metrics.setdefault(group, {"num_samples": 0, "num_correct": 0, "accuracy": 0.0})
        bucket["num_samples"] += 1
        bucket["num_correct"] += int(record["correct"])
    for bucket in metrics.values():
        bucket["accuracy"] = bucket["num_correct"] / max(bucket["num_samples"], 1)
    return metrics


def circular_metrics(records: list[dict[str, Any]]) -> dict[str, float | int]:
    grouped: dict[int, list[dict[str, Any]]] = {}
    for record in records:
        grouped.setdefault(int(record["sample_id"]), []).append(record)

    num_samples = len(grouped)
    if num_samples == 0:
        return {
            "num_original_samples": 0,
            "circular_correct": 0,
            "circular_accuracy": 0.0,
            "mean_rotation_accuracy": 0.0,
        }

    circular_correct = 0
    rotation_accuracies = []
    for sample_records in grouped.values():
        num_correct = sum(int(record["correct"]) for record in sample_records)
        circular_correct += int(num_correct == len(sample_records))
        rotation_accuracies.append(num_correct / max(len(sample_records), 1))

    return {
        "num_original_samples": num_samples,
        "circular_correct": circular_correct,
        "circular_accuracy": circular_correct / num_samples,
        "mean_rotation_accuracy": sum(rotation_accuracies) / num_samples,
    }


def load_seen_ids(predictions_path: Path) -> set[str]:
    if not predictions_path.exists():
        return set()
    seen: set[str] = set()
    with predictions_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                seen.add(str(json.loads(line)["eval_id"]))
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

    samples = load_sat(dataset_path, args.split, args.max_samples, args.circular_eval, args.max_image_pixels)
    predictions_path = output_dir / "predictions.jsonl"
    seen_ids = load_seen_ids(predictions_path) if args.resume else set()
    samples_to_run = [sample for sample in samples if sample.eval_id not in seen_ids]
    print(
        f"Loaded {len(samples)} SAT eval prompts; evaluating {len(samples_to_run)} "
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
            del inputs, generated_ids
            if device.type == "cuda":
                torch.cuda.empty_cache()
            for sample, prediction in zip(batch, predictions, strict=True):
                pred_norm = extract_choice(prediction, sample.answer_options) or ""
                ans_norm = answer_letter(sample)
                is_correct = pred_norm == ans_norm
                correct += int(is_correct)
                parseable += int(bool(pred_norm))
                record = {
                    "eval_id": sample.eval_id,
                    "sample_id": sample.sample_id,
                    "rotation_index": sample.rotation_index,
                    "question_type": sample.question_type,
                    "question": sample.question,
                    "answer_options": sample.answer_options,
                    "answer_index": sample.answer_index,
                    "correct_answer": sample.correct_answer,
                    "prediction": prediction.strip(),
                    "normalized_answer": ans_norm,
                    "normalized_prediction": pred_norm,
                    "correct": is_correct,
                    "num_images": len(sample.images),
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

    num_samples = len(all_records)
    num_correct = sum(int(record["correct"]) for record in all_records)
    num_parseable = sum(int(bool(record["normalized_prediction"])) for record in all_records)
    summary = {
        "dataset_path": str(dataset_path),
        "ckpt_path": str(ckpt_path),
        "processor_path": processor_path,
        "split": args.split,
        "circular_eval": args.circular_eval,
        "num_eval_prompts": num_samples,
        "num_correct": num_correct,
        "accuracy": num_correct / max(num_samples, 1),
        "parse_rate": num_parseable / max(num_samples, 1),
        "circular": circular_metrics(all_records),
        "by_question_type": build_group_metrics(all_records, "question_type"),
        "by_num_images": build_group_metrics(all_records, "num_images"),
        "elapsed_seconds": time.time() - start_time,
        "predictions_path": str(predictions_path),
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
