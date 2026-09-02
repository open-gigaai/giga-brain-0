# Copyright (C) 2026 Xiaomi Corporation.
import argparse
import json
import re
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, Optional

from PIL import Image
from tqdm import tqdm


CHOICE_RE = re.compile(r"\b([A-E])\b", re.IGNORECASE)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate XR-0 VLM capability on RealWorldQA.")
    parser.add_argument(
        "--model-path",
        default="XiaomiRobotics/Xiaomi-Robotics-0-Pretrain",
        help="Hugging Face model id or local checkpoint directory.",
    )
    parser.add_argument(
        "--data-root",
        default="datasets/public_datasets/VLM/vqa/benchmarks/RealWorldQA",
        help="RealWorldQA dataset root containing data/*.parquet.",
    )
    parser.add_argument("--output", default="eval_vlm/results/realworldqa_predictions.jsonl")
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


def torch_dtype(name: str):
    import torch

    return {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[name]


def patch_transformers_rope_default() -> None:
    """Make older/newer Transformers RoPE registries accept Qwen's `default` key."""
    import torch
    from transformers import modeling_rope_utils

    rope_init_functions = modeling_rope_utils.ROPE_INIT_FUNCTIONS
    if "default" in rope_init_functions:
        return

    for alias in ("base", "original", "standard"):
        if alias in rope_init_functions:
            rope_init_functions["default"] = rope_init_functions[alias]
            print(f"Patched Transformers RoPE registry: default -> {alias}")
            return

    def compute_default_rope_parameters(config=None, device=None, seq_len=None, **rope_kwargs):
        if config is not None:
            base = getattr(config, "rope_theta", 10000.0)
            partial_rotary_factor = getattr(config, "partial_rotary_factor", 1.0)
            head_dim = getattr(config, "head_dim", None)
            if head_dim is None:
                head_dim = config.hidden_size // config.num_attention_heads
            dim = int(head_dim * partial_rotary_factor)
        else:
            base = rope_kwargs.get("base", 10000.0)
            dim = rope_kwargs["dim"]

        inv_freq = 1.0 / (
            base ** (torch.arange(0, dim, 2, dtype=torch.float32, device=device) / dim)
        )
        return inv_freq, 1.0

    rope_init_functions["default"] = compute_default_rope_parameters
    print(
        "Patched Transformers RoPE registry with local `default` implementation; "
        f"available keys are now: {sorted(rope_init_functions.keys())}"
    )


def read_realworldqa(data_root: Path, limit: Optional[int] = None) -> List[Dict[str, Any]]:
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise RuntimeError("pyarrow is required to read RealWorldQA parquet files: pip install pyarrow") from exc

    parquet_files = sorted((data_root / "data").glob("*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"No parquet files found under {data_root / 'data'}")

    rows: List[Dict[str, Any]] = []
    for parquet_file in parquet_files:
        table = pq.read_table(parquet_file)
        for row_idx, row in enumerate(table.to_pylist()):
            image_field = row["image"]
            image_bytes = image_field.get("bytes") if isinstance(image_field, dict) else None
            if image_bytes is None:
                image_path = data_root / row["image_path"]
                image = Image.open(image_path).convert("RGB")
            else:
                image = Image.open(BytesIO(image_bytes)).convert("RGB")

            rows.append(
                {
                    "id": f"{parquet_file.stem}:{row_idx}",
                    "image_path": row.get("image_path"),
                    "image": image,
                    "question": row["question"],
                    "answer": row["answer"],
                }
            )
            if limit is not None and len(rows) >= limit:
                return rows
    return rows


def build_prompt(question: str) -> str:
    return (
        "<|im_start|>user\n"
        "<|vision_start|><|image_pad|><|vision_end|>\n"
        f"{question.strip()}\n"
        "Answer directly and keep the answer as short as possible.<|im_end|>\n"
        "<|im_start|>assistant\n"
    )


def normalize_answer(text: str) -> str:
    text = text.strip()
    text = re.sub(r"^<\|im_start\|>assistant\s*", "", text)
    text = re.sub(r"<\|im_end\|>.*$", "", text, flags=re.DOTALL)
    text = text.strip()
    text = text.splitlines()[0].strip() if text else ""
    text = text.strip(" .,:;\"'`")
    return text.lower()


def extract_choice(text: str) -> Optional[str]:
    match = CHOICE_RE.search(text.strip())
    return match.group(1).upper() if match else None


def score_prediction(prediction: str, answer: str) -> bool:
    pred_choice = extract_choice(prediction)
    answer_choice = extract_choice(answer)
    if answer_choice is not None:
        return pred_choice == answer_choice
    return normalize_answer(prediction) == normalize_answer(answer)


def unwrap_vlm(model):
    # XR-0 VLA checkpoints expose the underlying Qwen3-VL model as `.vlm`.
    return getattr(model, "vlm", model)


def load_auto_model_with_eager_fallback(AutoModel, model_path: str, **model_kwargs):
    try:
        return AutoModel.from_pretrained(model_path, **model_kwargs)
    except ValueError as exc:
        message = str(exc)
        unsupported_sdpa = (
            model_kwargs.get("attn_implementation") != "eager"
            and "scaled_dot_product_attention" in message
            and "attn_implementation=\"eager\"" in message
        )
        if not unsupported_sdpa:
            raise
        print(
            f"Model does not support attn_implementation={model_kwargs.get('attn_implementation')!r}; "
            "retrying with attn_implementation='eager'."
        )
        model_kwargs = dict(model_kwargs)
        model_kwargs["attn_implementation"] = "eager"
        return AutoModel.from_pretrained(model_path, **model_kwargs)


def main() -> None:
    import torch
    from transformers import AutoModel, AutoProcessor

    args = parse_args()
    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")

    records = read_realworldqa(Path(args.data_root), args.limit)
    print(f"Loaded {len(records)} RealWorldQA examples from {args.data_root}")

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

    correct = 0
    predictions: List[Dict[str, Any]] = []
    with torch.inference_mode():
        for start in tqdm(range(0, len(records), args.batch_size), desc="RealWorldQA"):
            batch = records[start : start + args.batch_size]
            prompts = [build_prompt(item["question"]) for item in batch]
            images = [item["image"] for item in batch]
            inputs = processor(
                text=prompts,
                images=images,
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
                is_correct = score_prediction(prediction, item["answer"])
                correct += int(is_correct)
                predictions.append(
                    {
                        "id": item["id"],
                        "image_path": item["image_path"],
                        "question": item["question"],
                        "answer": item["answer"],
                        "prediction": prediction.strip(),
                        "normalized_prediction": normalize_answer(prediction),
                        "correct": is_correct,
                    }
                )

    accuracy = correct / len(records) if records else 0.0
    with output_path.open("w", encoding="utf-8") as f:
        for row in predictions:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    metrics_path = output_path.with_suffix(".metrics.json")
    metrics_path.write_text(
        json.dumps(
            {
                "model_path": args.model_path,
                "data_root": args.data_root,
                "num_examples": len(records),
                "correct": correct,
                "accuracy": accuracy,
                "output": str(output_path),
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    print(f"Accuracy: {accuracy:.4f} ({correct}/{len(records)})")
    print(f"Predictions: {output_path}")
    print(f"Metrics: {metrics_path}")


if __name__ == "__main__":
    main()
