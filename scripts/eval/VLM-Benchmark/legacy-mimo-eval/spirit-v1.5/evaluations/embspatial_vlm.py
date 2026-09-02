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
DEFAULT_DATASET_PATH = "datasets/public_datasets/VLM/vqa/EmbSpatial-Bench"
DEFAULT_SUFFIX = "Please answer directly with only the letter of the correct option and nothing else."
CHOICE_LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


@dataclass
class EmbSpatialSample:
    sample_id: int
    question_id: str
    data_source: str
    scene_id: str
    relation: str
    question: str
    answer_options: list[str]
    answer_index: int
    image: Image.Image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate the Spirit-v1.5 Qwen3-VL backbone on EmbSpatial-Bench.",
    )
    parser.add_argument("--ckpt-path", default=DEFAULT_CKPT_PATH, help="Spirit-v1.5 checkpoint directory.")
    parser.add_argument("--dataset-path", default=DEFAULT_DATASET_PATH, help="EmbSpatial-Bench dataset directory.")
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
    parser.add_argument("--output-dir", default="outputs/embspatial_vlm", help="Directory for outputs.")
    parser.add_argument("--split", default="test", help="Parquet split prefix under dataset_path/data.")
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
    parser.add_argument("--max-new-tokens", type=int, default=16, help="Maximum generated answer tokens.")
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
    if isinstance(image_obj, (str, Path)):
        return Image.open(image_obj).convert("RGB")
    raise TypeError(f"Unsupported EmbSpatial-Bench image object: {type(image_obj)!r}")


def parquet_files(dataset_path: Path, split: str) -> list[Path]:
    data_dir = dataset_path / "data"
    paths = sorted(data_dir.glob(f"{split}-*.parquet"))
    if not paths:
        raise FileNotFoundError(f"No EmbSpatial-Bench parquet files found for split '{split}' under {data_dir}")
    return paths


def load_embspatial(dataset_path: Path, split: str, max_samples: int | None) -> list[EmbSpatialSample]:
    frames = [pd.read_parquet(path) for path in parquet_files(dataset_path, split)]
    df = pd.concat(frames, ignore_index=True)
    if max_samples is not None:
        df = df.head(max_samples)

    samples: list[EmbSpatialSample] = []
    for sample_id, row in enumerate(df.itertuples(index=False)):
        answer_options = [str(option) for option in row.answer_options]
        answer_index = int(row.answer)
        if answer_index < 0 or answer_index >= len(answer_options):
            raise ValueError(f"Invalid answer index {answer_index} for question_id={row.question_id}")
        samples.append(
            EmbSpatialSample(
                sample_id=sample_id,
                question_id=str(row.question_id),
                data_source=str(row.data_source),
                scene_id=str(row.scene_id),
                relation=str(row.relation),
                question=str(row.question),
                answer_options=answer_options,
                answer_index=answer_index,
                image=decode_image(row.image),
            )
        )
    return samples


def chunked(items: list[EmbSpatialSample], size: int) -> Iterable[list[EmbSpatialSample]]:
    for start in range(0, len(items), size):
        yield items[start : start + size]


def format_options(options: list[str]) -> str:
    return "\n".join(f"{CHOICE_LETTERS[index]}. {option}" for index, option in enumerate(options))


def build_messages(sample: EmbSpatialSample, system_prompt: str, question_suffix: str) -> list[dict[str, Any]]:
    question = f"{sample.question.strip()}\nOptions:\n{format_options(sample.answer_options)}"
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
    samples: list[EmbSpatialSample],
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


def normalize_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return " ".join(text.split())


def answer_letter(sample: EmbSpatialSample) -> str:
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

    samples = load_embspatial(dataset_path, args.split, args.max_samples)
    predictions_path = output_dir / "predictions.jsonl"
    seen_ids = load_seen_ids(predictions_path) if args.resume else set()
    samples_to_run = [sample for sample in samples if sample.sample_id not in seen_ids]
    print(
        f"Loaded {len(samples)} EmbSpatial-Bench samples; evaluating {len(samples_to_run)} "
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
                pred_norm = extract_choice(prediction, sample.answer_options) or ""
                ans_norm = answer_letter(sample)
                is_correct = pred_norm == ans_norm
                correct += int(is_correct)
                parseable += int(bool(pred_norm))
                record = {
                    "sample_id": sample.sample_id,
                    "question_id": sample.question_id,
                    "data_source": sample.data_source,
                    "scene_id": sample.scene_id,
                    "relation": sample.relation,
                    "question": sample.question,
                    "answer_options": sample.answer_options,
                    "answer_index": sample.answer_index,
                    "answer": sample.answer_options[sample.answer_index],
                    "prediction": prediction.strip(),
                    "normalized_answer": ans_norm,
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

    num_samples = len(all_records)
    num_correct = sum(int(record["correct"]) for record in all_records)
    num_parseable = sum(int(bool(record["normalized_prediction"])) for record in all_records)
    summary = {
        "dataset_path": str(dataset_path),
        "ckpt_path": str(ckpt_path),
        "processor_path": processor_path,
        "split": args.split,
        "num_samples": num_samples,
        "num_correct": num_correct,
        "accuracy": num_correct / max(num_samples, 1),
        "parse_rate": num_parseable / max(num_samples, 1),
        "by_relation": build_group_metrics(all_records, "relation"),
        "by_data_source": build_group_metrics(all_records, "data_source"),
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
