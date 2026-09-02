# Copyright (C) 2026 Xiaomi Corporation.
import argparse
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from PIL import Image
from tqdm import tqdm
import sys
sys.path.insert(0, '.')
from eval_vlm.main import (
    load_auto_model_with_eager_fallback,
    normalize_answer,
    patch_transformers_rope_default,
    torch_dtype,
    unwrap_vlm,
)


OPTION_TOKEN_RE = re.compile(r"^[A-K]+$", re.IGNORECASE)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate XR-0 VLM capability on EO-Bench.")
    parser.add_argument(
        "--model-path",
        default="XiaomiRobotics/Xiaomi-Robotics-0-Pretrain",
        help="Hugging Face model id or local checkpoint directory.",
    )
    parser.add_argument(
        "--data-root",
        default="datasets/public_datasets/VLM/vqa/benchmarks/EO-Bench",
        help="EO-Bench dataset root containing data/*.parquet and images/.",
    )
    parser.add_argument("--output", default="eval_vlm/results/eobench_predictions.jsonl")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--limit", type=int, default=None, help="Optional debug limit.")
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument(
        "--attn-implementation",
        default="flash_attention_2",
        choices=["flash_attention_2", "sdpa", "eager"],
        help="Use sdpa/eager if flash-attn is unavailable.",
    )
    parser.add_argument(
        "--dtype",
        default="bfloat16",
        choices=["bfloat16", "float16", "float32"],
    )
    parser.add_argument(
        "--trust-remote-code",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    return parser.parse_args()


def read_eobench(data_root: Path, limit: Optional[int] = None) -> List[Dict[str, Any]]:
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise RuntimeError("pyarrow is required to read EO-Bench parquet files: pip install pyarrow") from exc

    parquet_files = sorted((data_root / "data").glob("*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"No parquet files found under {data_root / 'data'}")

    rows: List[Dict[str, Any]] = []
    for parquet_file in parquet_files:
        table = pq.read_table(parquet_file)
        for row in table.to_pylist():
            image_paths = list(row["image_paths"])
            images = []
            for image_path in image_paths:
                path = data_root / image_path
                if not path.exists():
                    raise FileNotFoundError(f"Missing EO-Bench image: {path}")
                images.append(Image.open(path).convert("RGB"))

            rows.append(
                {
                    "id": str(row["id"]),
                    "question": row["question"],
                    "question_type": row["question_type"],
                    "answer": row["answer"],
                    "image_paths": image_paths,
                    "images": images,
                    "num_images": int(row.get("num_images", len(images))),
                }
            )
            if limit is not None and len(rows) >= limit:
                return rows
    return rows


def build_prompt(question: str, num_images: int) -> str:
    image_blocks = []
    for image_idx in range(num_images):
        if num_images == 1:
            image_blocks.append("<|vision_start|><|image_pad|><|vision_end|>")
        else:
            image_blocks.append(f"Image {image_idx + 1}:\n<|vision_start|><|image_pad|><|vision_end|>")

    return (
        "<|im_start|>user\n"
        + "\n".join(image_blocks)
        + "\n"
        + question.strip()
        + "\nAnswer directly with only the option letter or letters and nothing else.<|im_end|>\n"
        + "<|im_start|>assistant\n"
    )


def extract_options(text: str) -> Set[str]:
    text = normalize_answer(text).upper()
    if not text:
        return set()

    # Keep compact answers like "CI" while avoiding letters embedded in words like "ANSWER".
    tokens = re.split(r"[^A-Z]+", text)
    options: Set[str] = set()
    for token in tokens:
        if OPTION_TOKEN_RE.match(token):
            options.update(token)
    return options


def canonical_options(options: Set[str]) -> str:
    return "".join(sorted(options))


def score_prediction(prediction: str, answer: str) -> bool:
    return extract_options(prediction) == extract_options(answer)


def summarize_metrics(predictions: List[Dict[str, Any]]) -> Dict[str, Any]:
    total = len(predictions)
    correct = sum(int(item["correct"]) for item in predictions)
    by_type: Dict[str, Dict[str, int]] = defaultdict(lambda: {"total": 0, "correct": 0})

    for item in predictions:
        bucket = by_type[item["question_type"]]
        bucket["total"] += 1
        bucket["correct"] += int(item["correct"])

    return {
        "num_examples": total,
        "correct": correct,
        "accuracy": correct / total if total else 0.0,
        "by_question_type": {
            question_type: {
                "num_examples": values["total"],
                "correct": values["correct"],
                "accuracy": values["correct"] / values["total"] if values["total"] else 0.0,
            }
            for question_type, values in sorted(by_type.items())
        },
    }


def main() -> None:
    import torch
    from transformers import AutoModel, AutoProcessor

    args = parse_args()
    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")

    records = read_eobench(Path(args.data_root), args.limit)
    print(f"Loaded {len(records)} EO-Bench examples from {args.data_root}")

    patch_transformers_rope_default()
    model = load_auto_model_with_eager_fallback(
        AutoModel,
        args.model_path,
        trust_remote_code=args.trust_remote_code,
        attn_implementation=args.attn_implementation,
        dtype=torch_dtype(args.dtype),
    )
    model = unwrap_vlm(model).to(device).eval()
    processor = AutoProcessor.from_pretrained(
        args.model_path,
        trust_remote_code=args.trust_remote_code,
        use_fast=False,
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    predictions: List[Dict[str, Any]] = []
    with torch.inference_mode():
        for start in tqdm(range(0, len(records), args.batch_size), desc="EO-Bench"):
            batch = records[start : start + args.batch_size]
            prompts = [build_prompt(item["question"], item["num_images"]) for item in batch]
            flat_images = [image for item in batch for image in item["images"]]

            inputs = processor(
                text=prompts,
                images=flat_images,
                videos=None,
                padding=True,
                return_tensors="pt",
            ).to(device)

            generated = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                use_cache=True,
            )
            generated = generated[:, inputs["input_ids"].shape[1] :]
            decoded = processor.batch_decode(generated, skip_special_tokens=True, clean_up_tokenization_spaces=False)

            for item, prediction in zip(batch, decoded):
                pred_options = extract_options(prediction)
                answer_options = extract_options(item["answer"])
                is_correct = pred_options == answer_options
                predictions.append(
                    {
                        "id": item["id"],
                        "question_type": item["question_type"],
                        "question": item["question"],
                        "answer": item["answer"],
                        "answer_options": canonical_options(answer_options),
                        "prediction": prediction.strip(),
                        "prediction_options": canonical_options(pred_options),
                        "normalized_prediction": normalize_answer(prediction),
                        "correct": is_correct,
                        "num_images": item["num_images"],
                        "image_paths": item["image_paths"],
                    }
                )

    with output_path.open("w", encoding="utf-8") as f:
        for row in predictions:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    metrics = summarize_metrics(predictions)
    metrics.update(
        {
            "model_path": args.model_path,
            "data_root": args.data_root,
            "output": str(output_path),
        }
    )
    metrics_path = output_path.with_suffix(".metrics.json")
    metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(f"Accuracy: {metrics['accuracy']:.4f} ({metrics['correct']}/{metrics['num_examples']})")
    print(f"Predictions: {output_path}")
    print(f"Metrics: {metrics_path}")


if __name__ == "__main__":
    main()
