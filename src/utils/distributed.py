"""Minimal torch.distributed helpers shared by training and evaluation."""

import os

import torch
import torch.distributed as dist


def ddp_setup():
    """Initialize a process group if launched under torchrun, else single-process.

    Returns (rank, world_size, local_rank, device, is_ddp).
    """
    if "RANK" in os.environ and int(os.environ.get("WORLD_SIZE", "1")) > 1:
        dist.init_process_group("nccl" if torch.cuda.is_available() else "gloo")
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        local_rank = int(os.environ["LOCAL_RANK"])
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
            device = torch.device(f"cuda:{local_rank}")
        else:
            device = torch.device("cpu")
        return rank, world_size, local_rank, device, True
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    return 0, 1, 0, device, False


def ddp_cleanup(is_ddp):
    if is_ddp:
        dist.destroy_process_group()


def all_reduce_mean(value, device, is_ddp):
    """Reduce a python float across ranks (mean). No-op if not distributed."""
    if not is_ddp:
        return value
    t = torch.tensor([value], device=device, dtype=torch.float64)
    dist.all_reduce(t, op=dist.ReduceOp.SUM)
    return (t / dist.get_world_size()).item()


def all_reduce_sum(value, device, is_ddp):
    """Sum a python number across ranks. No-op if not distributed."""
    if not is_ddp:
        return value
    t = torch.tensor([value], device=device, dtype=torch.float64)
    dist.all_reduce(t, op=dist.ReduceOp.SUM)
    return t.item()


def broadcast_scalar(value, device, is_ddp, src=0):
    if not is_ddp:
        return value
    t = torch.tensor([value], device=device, dtype=torch.float64)
    dist.broadcast(t, src=src)
    return t.item()


def all_reduce_max(value, device, is_ddp):
    """Max of a python number across ranks. No-op if not distributed."""
    if not is_ddp:
        return value
    t = torch.tensor([value], device=device, dtype=torch.float64)
    dist.all_reduce(t, op=dist.ReduceOp.MAX)
    return t.item()


def all_gather_object(obj, is_ddp):
    """[obj from rank 0, rank 1, ...] (just [obj] if not distributed)."""
    if not is_ddp:
        return [obj]
    out = [None] * dist.get_world_size()
    dist.all_gather_object(out, obj)
    return out
