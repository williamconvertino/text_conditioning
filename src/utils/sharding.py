"""FSDP (torch FSDP2, `fully_shard`) helpers shared by pretraining
(src/training/trainer.py) and fine-tuning (src/finetune/loop.py), for models
too large for DDP's full per-GPU replica of params + grads + AdamW state
(the 3B/8B configs in configs/scale/; README "Scaling Up").

Each Transformer layer is its own FSDP unit; everything else (embeddings,
final norm, lm_head, the text encoder, the null vector, a fine-tuning prefix)
lives in the root unit, so the root module's forward() must be the entry
point for every forward pass -- calling a submodule directly would find the
root's parameters still sharded.

Parameters stay fp32 (no MixedPrecisionPolicy): compute precision comes from
the same torch.autocast the DDP path uses, so a run's numbers don't depend on
which of the two paths trained it.

Checkpoints are always written as full (unsharded) state dicts with the same
keys as a DDP run's, so load_run(), evaluation, benchmarking and fine-tuning
load an FSDP-trained run exactly like any other.
"""

import torch
from torch.distributed.checkpoint.state_dict import (
    StateDictOptions, get_model_state_dict, get_optimizer_state_dict, set_model_state_dict,
    set_optimizer_state_dict)
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import fully_shard
from torch.distributed.tensor import DTensor

from src.models.modeling_llama import DecoderLayer

_FULL = StateDictOptions(full_state_dict=True, cpu_offload=True)


def shard_model(model, device, world_size):
    """Shard `model` in place (each DecoderLayer, then the root) across every
    rank, moving it from CPU to `device`. Returns the same module."""
    mesh = init_device_mesh(device.type, (world_size,))
    for module in model.modules():
        if isinstance(module, DecoderLayer):
            fully_shard(module, mesh=mesh)
    fully_shard(model, mesh=mesh)
    return model


def full_model_state(model):
    """Full state dict on CPU (rank 0 only; other ranks get {}). Every rank
    must call this."""
    return get_model_state_dict(model, options=_FULL)


def full_optimizer_state(model, optimizer):
    """Full, parameter-name-keyed optimizer state on CPU (rank 0 only). Every
    rank must call this."""
    return get_optimizer_state_dict(model, optimizer, options=_FULL)


def load_full_model_state(model, state):
    """Load a full state dict (present on every rank) into a sharded model."""
    set_model_state_dict(model, state, options=StateDictOptions(full_state_dict=True))


def load_full_optimizer_state(model, optimizer, state):
    set_optimizer_state_dict(model, optimizer, state, options=StateDictOptions(full_state_dict=True))


def local(t):
    """This rank's shard of a (possibly DTensor) tensor."""
    return t.to_local() if isinstance(t, DTensor) else t


def full_tensor(t):
    """The whole (possibly sharded) tensor, on every rank. Every rank must call
    this for a DTensor."""
    return t.full_tensor() if isinstance(t, DTensor) else t


def scalar(t):
    """float() of a possibly-DTensor scalar, e.g. clip_grad_norm_'s result."""
    return float(full_tensor(t)) if isinstance(t, torch.Tensor) else float(t)
