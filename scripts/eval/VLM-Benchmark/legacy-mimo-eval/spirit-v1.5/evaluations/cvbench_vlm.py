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
DEFAULT_DATASET_PATH = "datasets/public_datasets/VLM/vqa/CV-Bench"
DEFAULT_SUFFIX = "Please answer directly with only the letter of the correct option and nothing else."


@dataclass
class CVBenchSample:
    sample_id: int
    question_id: int
    type: str
    task: str
    question: str
    choices: list[str]
    answer: str
    prompt: str
    filename: str
    source: str
    source_dataset: str
    image: Image.Image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate the Spirit-v1.5 Qwen3-VL backbone on CV-Bench.",
    )
    parser.add_argument("--ckpt-path", default=DEFAULT_CKPT_PATH, help="Spirit-v1.5 checkpoint directory.")
    parser.add_argument("--dataset-path", default=DEFAULT_DATASET_PATH, help="CV-Bench dataset directory.")
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
    parser.add_argument("--output-dir", default="outputs/cvbench_vlm", help="Directory for outputs.")
    parser.add_argument("--subset", choices=("all", "2d", "3d"), default="all", help="CV-Bench subset to evaluate.")
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
    raise TypeError(f"Unsupported CV-Bench image object: {type(image_obj)!r}")


def parquet_files(dataset_path: Path, subset: str) -> list[Path]:
    if subset == "2d":
        return [dataset_path / "test_2d.parquet"]
    if subset == "3d":
        return [dataset_path / "test_3d.parquet"]
    return [dataset_path / "test_2d.parquet", dataset_path / "test_3d.parquet"]


def load_cvbench(dataset_path: Path, subset: str, max_samples: int | None) -> list[CVBenchSample]:
    frames = []
    for path in parquet_files(dataset_path, subset):
        if not path.exists():
            raise FileNotFoundError(f"CV-Bench parquet not found: {path}")
        frames.append(pd.read_parquet(path))
    df = pd.concat(frames, ignore_index=True)
    if max_samples is not None:
        df = df.head(max_samples)

    samples: list[CVBenchSample] = []
    for sample_id, row in enumerate(df.itertuples(index=False)):
        samples.append(
            CVBenchSample(
                sample_id=sample_id,
                question_id=int(row.idx),
                type=str(row.type),
                task=str(row.task),
                question=str(row.question),
                choices=[str(choice) for choice in row.choices],
                answer=str(row.answer),
                prompt=str(row.prompt),
                filename=str(row.filename),
                source=str(row.source),
                source_dataset=str(row.source_dataset),
                image=decode_image(row.image),
            )
        )
    return samples


def chunked(items: list[CVBenchSample], size: int) -> Iterable[list[CVBenchSample]]:
    for start in range(0, len(items), size):
        yield items[start : start + size]


def build_messages(sample: CVBenchSample, system_prompt: str, question_suffix: str) -> list[dict[str, Any]]:
    question = sample.prompt.strip()
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
    samples: list[CVBenchSample],
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


def normalize_answer(answer: str) -> str:
    match = re.search(r"\(([A-F])\)", answer.upper())
    if match:
        return match.group(1)
    match = re.search(r"(?<![A-Z])([A-F])(?![A-Z])", answer.upper())
    return match.group(0) if match else answer.strip().upper()


def extract_choice(text: str) -> str | None:
    patterns = [
        r"\b(?:answer|option|choice)\s*(?:is|:)?\s*[\(\[]?\s*([A-Fa-f])\b",
        r"\(([A-Fa-f])\)",
        r"^\s*[\(\[]?\s*([A-Fa-f])\s*[\)\].:：]?(?:\s|$)",
        r"(?<![A-Za-z])([A-Fa-f])(?![A-Za-z])",
    ]
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            return match.group(1).upper()
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


def source_accuracy(records: list[dict[str, Any]], source: str) -> float | None:
    selected = [record for record in records if record["source"] == source]
    if not selected:
        return None
    return sum(int(record["correct"]) for record in selected) / len(selected)


def cvbench_accuracy(records: list[dict[str, Any]]) -> float | None:
    acc_ade = source_accuracy(records, "ADE20K")
    acc_coco = source_accuracy(records, "COCO")
    acc_omni = source_accuracy(records, "Omni3D")
    if acc_ade is None or acc_coco is None or acc_omni is None:
        return None
    acc_2d = (acc_ade + acc_coco) / 2
    return ((acc_2d + acc_omni) / 2)


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

    samples = load_cvbench(dataset_path, args.subset, args.max_samples)
    predictions_path = output_dir / "predictions.jsonl"
    seen_ids = load_seen_ids(predictions_path) if args.resume else set()
    samples_to_run = [sample for sample in samples if sample.sample_id not in seen_ids]
    print(
        f"Loaded {len(samples)} CV-Bench samples; evaluating {len(samples_to_run)} "
        f"(resume skipped {len(seen_ids)}).",
        flush=True,
    )

    start_time = time.time()
    new_records: list[dict[str, Any]] = []
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
            for sample, prediction in zip(batch, predictions, strict=True):
                pred_norm = extract_choice(prediction) or ""
                ans_norm = normalize_answer(sample.answer)
                is_correct = pred_norm == ans_norm
                correct += int(is_correct)
                record = {
                    "sample_id": sample.sample_id,
                    "question_id": sample.question_id,
                    "type": sample.type,
                    "task": sample.task,
                    "source": sample.source,
                    "source_dataset": sample.source_dataset,
                    "filename": sample.filename,
                    "question": sample.question,
                    "choices": sample.choices,
                    "answer": sample.answer,
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
                f"running_accuracy={correct / max(processed, 1):.4f}",
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
    by_source = build_group_metrics(all_records, "source")
    by_type = build_group_metrics(all_records, "type")
    summary = {
        "dataset_path": str(dataset_path),
        "ckpt_path": str(ckpt_path),
        "processor_path": processor_path,
        "subset": args.subset,
        "num_samples": num_samples,
        "num_correct": num_correct,
        "accuracy": num_correct / max(num_samples, 1),
        "cv_bench_accuracy": cvbench_accuracy(all_records),
        "by_type": by_type,
        "by_task": build_group_metrics(all_records, "task"),
        "by_source": by_source,
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
