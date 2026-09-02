#!/usr/bin/env python
"""
Custom script to merge FSDP sharded checkpoints with compatibility handling.
Works around the StorageMeta compatibility issue between PyTorch versions.
"""

import os
import sys
from pathlib import Path
from typing import Dict

import torch
from safetensors.torch import save_file


def patch_metadata_loader():
    """Patch the metadata loader to handle missing StorageMeta class."""
    import torch.distributed.checkpoint.metadata as metadata_module

    # Create a dummy StorageMeta class if it doesn't exist
    if not hasattr(metadata_module, "StorageMeta"):
        print("[INFO] Creating StorageMeta compatibility shim")

        class StorageMeta:
            """Compatibility shim for old StorageMeta class."""

            def __init__(self, *args, **kwargs):
                # Store all args as attributes
                self.args = args
                self.kwargs = kwargs

        # Inject the class into the module
        metadata_module.StorageMeta = StorageMeta

        # Also make it available for unpickling
        sys.modules["torch.distributed.checkpoint.metadata"].StorageMeta = StorageMeta


def load_sharded_checkpoint(checkpoint_dir: str) -> Dict[str, torch.Tensor]:
    """
    Load a sharded FSDP checkpoint by manually reading all shard files.

    Args:
        checkpoint_dir: Path to directory containing .distcp files

    Returns:
        Dictionary of merged model state
    """
    import torch.distributed.checkpoint as dist_cp
    import torch.distributed.checkpoint.format_utils as dist_cp_format_utils

    print(f"[INFO] Loading checkpoint from {checkpoint_dir}")

    # Apply the compatibility patch
    patch_metadata_loader()

    # Try to load using the standard approach
    try:
        state_dict = {}
        storage_reader = dist_cp.FileSystemReader(checkpoint_dir)

        dist_cp_format_utils._load_state_dict(
            state_dict,
            storage_reader=storage_reader,
            planner=dist_cp_format_utils._EmptyStateDictLoadPlanner(),
            no_dist=True,
        )

        print(f"[INFO] Successfully loaded state dict with {len(state_dict)} keys")
        return state_dict

    except AttributeError as e:
        if "StorageMeta" in str(e):
            raise RuntimeError(
                "Unable to load this DCP checkpoint because its metadata uses "
                "StorageMeta from a different PyTorch version. The previous "
                "manual .distcp fallback was removed because FSDP shards cannot "
                "be reconstructed by directly loading shard files and overwriting "
                "duplicate keys. Please run this script with a PyTorch version "
                "compatible with the checkpoint writer, or re-save the checkpoint "
                "with the current PyTorch version."
            ) from e
        raise


def save_merged_checkpoint(
    state_dict: Dict[str, torch.Tensor],
    output_path: str,
    safe_serialization: bool = True,
):
    """
    Save the merged checkpoint to disk.

    Args:
        state_dict: Model state dictionary
        output_path: Directory to save the merged checkpoint
        safe_serialization: If True, save as .safetensors, else as .bin
    """
    output_dir = Path(output_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Handle nested state dict structure (e.g., {model: {...}})
    if len(state_dict.keys()) == 1 and all(
        isinstance(v, dict) for v in state_dict.values()
    ):
        print("[INFO] Unwrapping nested state dict")
        state_dict = state_dict[list(state_dict.keys())[0]]

    # Prepare tensors for saving
    save_dict = {}
    for key, value in state_dict.items():
        if isinstance(value, torch.Tensor):
            # Convert to CPU and contiguous
            save_dict[key] = value.cpu().contiguous()
        else:
            print(f"[WARNING] Skipping non-tensor key: {key} (type: {type(value)})")

    if safe_serialization:
        output_file = output_dir / "model.safetensors"
        print(f"[INFO] Saving merged checkpoint to {output_file}")
        save_file(save_dict, output_file)
    else:
        output_file = output_dir / "pytorch_model.bin"
        print(f"[INFO] Saving merged checkpoint to {output_file}")
        torch.save(save_dict, output_file)

    print("[SUCCESS] Checkpoint saved successfully!")
    print(f"[INFO] Saved {len(save_dict)} tensors")

    # Print size info
    total_params = sum(v.numel() for v in save_dict.values())
    total_size_gb = sum(v.numel() * v.element_size() for v in save_dict.values()) / (
        1024**3
    )
    print(f"[INFO] Total parameters: {total_params:,}")
    print(f"[INFO] Total size: {total_size_gb:.2f} GB")

    return output_file


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Merge FSDP sharded checkpoints with compatibility handling"
    )
    parser.add_argument(
        "checkpoint_dir",
        type=str,
        help="Directory containing sharded FSDP checkpoint files (*.distcp)",
    )
    parser.add_argument(
        "output_path", type=str, help="Output directory for merged checkpoint"
    )
    parser.add_argument(
        "--unsafe-serialization",
        action="store_true",
        help="Save as .bin instead of .safetensors",
    )

    args = parser.parse_args()

    # Validate input
    if not os.path.exists(args.checkpoint_dir):
        print(f"[ERROR] Checkpoint directory not found: {args.checkpoint_dir}")
        sys.exit(1)

    try:
        # Load the sharded checkpoint
        state_dict = load_sharded_checkpoint(args.checkpoint_dir)

        # Save the merged checkpoint
        safe_serialization = not args.unsafe_serialization
        output_file = save_merged_checkpoint(
            state_dict, args.output_path, safe_serialization
        )

        print("\n[COMPLETE] Checkpoint merging successful!")
        print(f"[COMPLETE] Output: {output_file}")

    except Exception as e:
        print(f"\n[ERROR] Failed to merge checkpoint: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
