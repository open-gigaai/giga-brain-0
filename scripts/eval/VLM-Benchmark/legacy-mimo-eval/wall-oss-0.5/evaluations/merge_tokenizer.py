#!/usr/bin/env python3
"""Merge Wall-X action tokens into a Qwen2.5-VL processor tokenizer."""

from __future__ import annotations

import argparse
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create a Wall-X processor directory by adding FAST action tokens "
            "to a Qwen2.5-VL processor tokenizer."
        )
    )
    parser.add_argument(
        "--processor-path",
        required=True,
        help="Base Qwen2.5-VL processor directory or Hugging Face repo id.",
    )
    parser.add_argument(
        "--action-tokenizer-path",
        required=True,
        help="FAST/action tokenizer processor directory or Hugging Face repo id.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory where the merged processor will be written.",
    )
    parser.add_argument(
        "--use-fast",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use the fast tokenizer implementation when loading the base processor.",
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Allow custom code when loading the action tokenizer processor.",
    )
    return parser.parse_args()


def _resolve_action_vocab_size(action_processor) -> int:
    vocab_size = getattr(action_processor, "vocab_size", None)
    if vocab_size is None and hasattr(action_processor, "tokenizer"):
        vocab_size = getattr(action_processor.tokenizer, "vocab_size", None)
    if vocab_size is None:
        raise AttributeError(
            "Could not determine action tokenizer vocab size from the loaded processor."
        )
    return int(vocab_size)


def merge_tokenizer(
    *,
    processor_path: str,
    action_tokenizer_path: str,
    output_dir: str,
    use_fast: bool,
    trust_remote_code: bool,
) -> None:
    from transformers import AutoProcessor

    processor = AutoProcessor.from_pretrained(processor_path, use_fast=use_fast)
    processor.tokenizer.padding_side = "left"

    action_processor = AutoProcessor.from_pretrained(
        action_tokenizer_path,
        trust_remote_code=trust_remote_code,
    )
    action_vocab_size = _resolve_action_vocab_size(action_processor)

    new_tokens = ["<|propri|>", "<|action|>"]
    new_tokens += [f"<|action_token_{i}|>" for i in range(action_vocab_size)]
    num_added_tokens = processor.tokenizer.add_tokens(new_tokens)

    begin_idx_token = "<|action_token_0|>"
    token_id = processor.tokenizer.convert_tokens_to_ids(begin_idx_token)
    processor.tokenizer.init_kwargs["action_token_start_index"] = token_id
    processor.tokenizer.init_kwargs["action_token_vocab_size"] = action_vocab_size

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    processor.save_pretrained(output_path)

    print(f"Saved merged processor to {output_path}")
    print(f"Added {num_added_tokens} tokenizer tokens")
    print(f"action_token_start_index={token_id}")
    print(f"action_token_vocab_size={action_vocab_size}")


def main() -> None:
    args = parse_args()
    merge_tokenizer(
        processor_path=args.processor_path,
        action_tokenizer_path=args.action_tokenizer_path,
        output_dir=args.output_dir,
        use_fast=args.use_fast,
        trust_remote_code=args.trust_remote_code,
    )


if __name__ == "__main__":
    main()
