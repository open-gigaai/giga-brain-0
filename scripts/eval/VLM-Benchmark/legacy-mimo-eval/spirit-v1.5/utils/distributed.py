# ==============================================================================
# Attribution
# ------------------------------------------------------------------------------
# Released by Spirit AI Team.
# ==============================================================================

import functools
import os
import torch
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import MixedPrecision
from torch.distributed.fsdp import BackwardPrefetch, ShardingStrategy
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy

try:
    from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLTextRMSNorm
except Exception:
    Qwen3VLTextRMSNorm = None

def setup_distributed():
    if "RANK" in os.environ:
        dist.init_process_group(backend="nccl")
        local_rank = int(os.environ["LOCAL_RANK"])
        global_rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        torch.cuda.set_device(local_rank)

        mesh = init_device_mesh("cuda", (world_size,))

        print(f"Rank {global_rank}/{world_size} initialized")
    else:
        local_rank = 0
        global_rank = 0
        world_size = 1
        mesh = None
        torch.cuda.set_device(0)

        print("Single GPU mode")

    return local_rank, global_rank, world_size, mesh


def apply_fsdp(model, mesh):
    if mesh is None:
        return model

    ignored_modules = []

    embed_tokens = None
    lm_head = None
    visual = None
    for name, module in model.named_modules():
        if name.endswith("embed_tokens"):
            embed_tokens = module
        elif name.endswith("lm_head"):
            lm_head = module
        elif name.endswith("qwen.model.visual"):
            visual = module
    if embed_tokens is not None and lm_head is not None:
        ignored_modules.extend([embed_tokens, lm_head])
    if visual is not None:
        ignored_modules.append(visual)

    modules_to_ignore = (
        torch.nn.LayerNorm,
        torch.nn.Dropout,
        torch.nn.Identity,
        torch.nn.GELU,
        torch.nn.SiLU,
    )
    for module in model.modules():
        if isinstance(module, modules_to_ignore):
            ignored_modules.append(module)
        elif Qwen3VLTextRMSNorm is not None and isinstance(module, Qwen3VLTextRMSNorm):
            ignored_modules.append(module)

    unique_ignored_modules = []
    seen = set()
    for module in ignored_modules:
        if id(module) not in seen:
            unique_ignored_modules.append(module)
            seen.add(id(module))

    wrap_classes = [torch.nn.Linear, torch.nn.Embedding, torch.nn.LayerNorm]
    if Qwen3VLTextRMSNorm is not None:
        wrap_classes.append(Qwen3VLTextRMSNorm)
    auto_wrap_policy = functools.partial(
        transformer_auto_wrap_policy,
        transformer_layer_cls=wrap_classes,
    )

    mp_policy = MixedPrecision(
        param_dtype=torch.float32,
        reduce_dtype=torch.float32,
        buffer_dtype=torch.float32,
        cast_forward_inputs=True,
        cast_root_forward_inputs=True,
    )

    model = FSDP(
        model,
        sharding_strategy=ShardingStrategy.FULL_SHARD,
        ignored_modules=unique_ignored_modules,
        device_id=torch.cuda.current_device(),
        auto_wrap_policy=auto_wrap_policy,
        mixed_precision=mp_policy,
        forward_prefetch=True,
        backward_prefetch=BackwardPrefetch.BACKWARD_PRE,
        limit_all_gathers=True,
        use_orig_params=True,
        sync_module_states=True,
    )

    print(f"Applied FSDP to model (ignored_modules={len(unique_ignored_modules)})")
    return model


def cleanup():
    if dist.is_initialized():
        dist.destroy_process_group()
