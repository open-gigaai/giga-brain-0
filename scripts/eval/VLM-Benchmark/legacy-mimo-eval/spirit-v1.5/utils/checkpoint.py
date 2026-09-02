# ==============================================================================
# Attribution
# ------------------------------------------------------------------------------
# Released by Spirit AI Team.
# ==============================================================================

import torch
import shutil
from pathlib import Path
from safetensors.torch import save_file, load_file
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import FullStateDictConfig, StateDictType


def save_model(
    model,
    step: int,
    output_dir: str,
    rank: int,
):
    output_path = Path(output_dir)
    if isinstance(model, FSDP):
        with FSDP.state_dict_type(
            model,
            StateDictType.FULL_STATE_DICT,
            FullStateDictConfig(offload_to_cpu=True, rank0_only=True),
        ):
            state_dict = model.state_dict()
    else:
        state_dict = model.state_dict()

    if rank == 0:
        output_path.mkdir(parents=True, exist_ok=True)
        state_dict = {k: v.detach().cpu().clone() for k, v in state_dict.items()}

        save_path = output_path / f"model_step_{step}.safetensors"
        latest_path = output_path / "model.safetensors"

        save_file(state_dict, save_path)

        if latest_path.exists() or latest_path.is_symlink():
            latest_path.unlink()
        try:
            latest_path.hardlink_to(save_path)
        except OSError:
            shutil.copy2(save_path, latest_path)

        print(f"Saved model to {save_path}")
        print(f"Updated latest inference checkpoint: {latest_path}")
