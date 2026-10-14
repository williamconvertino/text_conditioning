"""Shared training loop for all experiments (baseline / meco / text-conditioned).
Which variant a run trains is entirely determined by its experiment config (see
configs/experiment/*.yaml and src/config.py), not by separate training scripts.

Resume is automatic: rerunning the same command picks up ckpt_last.pt. The data
cursor is derived from the step count, so resumption replays the exact same
stream (src/data/shards.py's TokenizedDocStream).

A "text" run's `text_conditioning_source` picks what the encoder sees: the
document itself ("self", the default), other documents from the same URL
domain ("domain_random" / "domain_mean", README "Domain-Conditioned Training",
src/data/domain_index.py), or the document's URL metadata ("url", README
"URL-Encoded Conditioning", src/data/url_conditioning.py).

Multi-GPU runs use DDP by default. `fsdp: true` (configs/scale/3b.yaml,
8b.yaml) shards params, grads and optimizer state across ranks instead
(src/utils/sharding.py), and `gradient_checkpointing: true` recomputes layer
activations in backward -- together what the 3B/8B models need to fit
(README "Scaling Up"). Checkpoints are identical either way.

`length_bucketing: true` (the scale configs) groups each step's documents
into length-sorted, trimmed micro-batches instead of padding every document
to block_size (src/data/collator.py's bucketed_step_batches): the same
documents per step, ~3x less compute. The step's loss is then the exact
token-weighted mean over its documents, and tokens_seen is the exact global
count.

`packing: true` (configs/scale/1.5b.yaml) instead concatenates documents into
fixed block_size rows (src/data/packing.py's PackedBatcher): no padding at
all, attention restricted to each document segment, and MeCo's prefix / the
URL encoder's slot repeated at every segment start. A step is a fixed number
of rows (global_batch_seqs), so tokens_seen counts row positions (identical
across variants at a given step), the loss is the exact token-weighted mean
over the global step, and the per-rank data cursor is saved in ckpt_last.pt.
Only baseline, MeCo and URL-encoded runs pack. `compile: true` torch.compiles
the decoder layers and loss.

A TRAIN_STOP_AT environment variable (unix time, set by
slurm/scale/train.slurm from the job's time limit) makes the run validate,
write ckpt_last.pt and exit cleanly once that time passes, so a job cut off by
its time limit loses no work.
"""

import math
import os
import time

import torch
from transformers import AutoTokenizer

from src.config import load_config, save_config
from src.data.collator import (VARIANTS, bucketed_step_batches, count_loss_tokens, train_batches,
                                load_val_batches)
from src.data.domain_index import SOURCES, DomainIndex, DomainPartnerSampler
from src.data.packing import PackedBatcher, Prefetcher, pin_batch
from src.data.url_conditioning import build_url_source
from src.data.shards import load_manifest, split_shards, rank_shards, TokenizedDocStream
from src.models.configuration_llama import LlamaConfig
from src.models.modeling_llama import LlamaForCausalLM
from src.models.text_encoder import TextEncoderConfig, TextEncoder
from src.models.conditioning import TextConditioner
from src.models.conditioned_lm import ConditionedLM
from src.training.checkpointing import save_checkpoint, load_checkpoint, save_eval_checkpoint
from src.training.schedules import build_schedule, sample_keep_mask
from src.eval.perplexity import evaluate_loss as evaluate_loss_full, evaluate_loss_sums
from src.utils.distributed import (ddp_setup, ddp_cleanup, all_gather_object, all_reduce_max, all_reduce_mean,
                                   all_reduce_sum, broadcast_scalar)
from src.utils.logging import MetricLogger
from src.utils import sharding


def lr_at(step, cfg):
    warmup, max_steps = cfg["warmup_steps"], cfg["max_steps"]
    lr, min_lr = cfg["lr"], cfg["min_lr"]
    if step < warmup:
        return lr * (step + 1) / max(1, warmup)
    if step >= max_steps:
        return min_lr
    prog = (step - warmup) / max(1, max_steps - warmup)
    return min_lr + 0.5 * (lr - min_lr) * (1 + math.cos(math.pi * prog))


def make_optimizer(model, cfg, fused=None):
    decay, no_decay = [], []
    for _, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (decay if p.dim() >= 2 else no_decay).append(p)
    groups = [{"params": decay, "weight_decay": cfg["weight_decay"]},
              {"params": no_decay, "weight_decay": 0.0}]
    fused = torch.cuda.is_available() if fused is None else fused
    return torch.optim.AdamW(groups, lr=cfg["lr"], betas=(0.9, 0.95), eps=1e-8, fused=fused)


def count_parameters(module):
    return sum(p.numel() for p in module.parameters())


def build_model(cfg, tokenizer):
    """Construct the ConditionedLM for a resolved experiment config. Shared by
    training (this module) and evaluation (scripts/evaluate.py), so a run's
    architecture is always rebuilt identically from its resolved_config.yaml."""
    lm_cfg = LlamaConfig(
        vocab_size=len(tokenizer),
        hidden_size=cfg["hidden_size"], intermediate_size=cfg["intermediate_size"],
        num_hidden_layers=cfg["num_hidden_layers"], num_attention_heads=cfg["num_attention_heads"],
        num_key_value_heads=cfg["num_key_value_heads"],
        max_position_embeddings=cfg["max_position_embeddings"], rms_norm_eps=cfg["rms_norm_eps"],
        rope_theta=cfg["rope_theta"], initializer_range=cfg["initializer_range"],
        tie_word_embeddings=cfg.get("tie_word_embeddings", False),
        pad_token_id=tokenizer.pad_token_id, bos_token_id=(tokenizer.bos_token_id or 0),
        eos_token_id=tokenizer.eos_token_id,
        pinned_condition=bool(cfg.get("strong_conditioning", False)))
    if lm_cfg.pinned_condition and cfg["variant"] != "text":
        raise SystemExit("strong_conditioning needs variant: text (a conditioning vector to pin)")
    lm = LlamaForCausalLM(lm_cfg)
    lm.model.gradient_checkpointing = bool(cfg.get("gradient_checkpointing", False))

    variant = cfg["variant"]
    conditioner = None
    if variant == "text":
        enc_cfg_dict = load_config(cfg["text_encoder_config"])
        enc_cfg = TextEncoderConfig(
            vocab_size=len(tokenizer), hidden_size=enc_cfg_dict["hidden_size"],
            intermediate_size=enc_cfg_dict["intermediate_size"],
            num_hidden_layers=enc_cfg_dict["num_hidden_layers"],
            num_attention_heads=enc_cfg_dict["num_attention_heads"],
            num_key_value_heads=enc_cfg_dict["num_key_value_heads"],
            max_position_embeddings=enc_cfg_dict["max_position_embeddings"],
            rms_norm_eps=enc_cfg_dict["rms_norm_eps"], rope_theta=enc_cfg_dict["rope_theta"],
            initializer_range=enc_cfg_dict["initializer_range"], output_dim=cfg["hidden_size"],
            pooling=enc_cfg_dict.get("pooling", "cls"), pad_token_id=tokenizer.pad_token_id)
        encoder = TextEncoder(enc_cfg)
        conditioner = TextConditioner(encoder, cfg["hidden_size"])

    return ConditionedLM(lm, variant, conditioner), lm_cfg


# metrics.jsonl column prefix for each non-self text_conditioning_source's
# training-condition val metric (src/analysis/figures.py's CONDITIONED_PREFIXES
# reads these back).
SOURCE_METRIC_PREFIX = {"domain_random": "domain_conditioned", "domain_mean": "domain_conditioned",
                        "url": "url_conditioned"}


def build_condition_sources(cfg, tokenizer, train_stems, val_stems):
    """(train_source, val_source) supplying the encoder's input for a "text"
    run whose text_conditioning_source isn't "self", or (None, None) for
    self-conditioning / other variants (see src/data/collator.py).

    domain_random / domain_mean: training partners come from the precomputed
      training-split index (scripts/build_domain_index.py); validation
      partners come from the validation shards only, indexed in memory here,
      mirroring src/eval/conditioning.py's same-domain/domain-mean modes, which
      also never condition on training documents.
    url: the document's own URL metadata, identical for train and val."""
    source = cfg.get("text_conditioning_source", "self")
    if source not in SOURCES:
        raise ValueError(f"unknown text_conditioning_source {source!r}; expected one of {SOURCES}")
    if cfg["variant"] != "text" or source == "self":
        return None, None
    if source == "url":
        url_source = build_url_source(cfg, tokenizer)
        return url_source, url_source
    dc = cfg.get("domain_conditioning", {})
    kwargs = {"n_support": dc.get("n_support", 8), "min_domain_docs": dc.get("min_domain_docs", 2),
              "seed": cfg["seed"], "tok_dir": cfg["tok_dir"]}
    train_index = DomainIndex.load(dc["index_dir"], expected_stems=train_stems)
    val_index = DomainIndex.build(cfg["tok_dir"], val_stems)
    return (DomainPartnerSampler(train_index, source, **kwargs),
            DomainPartnerSampler(val_index, source, **kwargs))


def check_manifest(cfg, tokenizer):
    manifest = load_manifest(cfg["tok_dir"])
    if manifest["tokenizer"] != cfg["tokenizer"]:
        raise SystemExit(
            f"train config tokenizer={cfg['tokenizer']!r} does not match the tokenizer "
            f"{cfg['tok_dir']} was pretokenized with ({manifest['tokenizer']!r}). Point "
            f"--tokenizer at the same model, or re-run src/data/tokenize_fineweb.py "
            f"into a fresh tok_dir.")
    if len(tokenizer) != manifest["vocab_size"]:
        raise SystemExit(f"tokenizer vocab_size {len(tokenizer)} != manifest vocab_size "
                         f"{manifest['vocab_size']} for {cfg['tok_dir']}")


def packing_mode(cfg):
    """src/data/packing.py mode for a `packing: true` run."""
    if cfg["variant"] in ("none", "meco"):
        return cfg["variant"]
    if cfg["variant"] == "text" and cfg.get("text_conditioning_source", "self") == "url":
        return "url"
    raise SystemExit("packing supports baseline (none), meco and URL-encoded (text + "
                     "text_conditioning_source: url) runs only")


def trim_batches(batches):
    """Unpacked val batches trimmed to their longest document (same loss: the
    cut positions are padding, after every real token causally)."""
    for b in batches:
        b["input_ids"] = b["input_ids"][:, :max(1, int(b["lengths"].max()))]
    return batches


def sharded_evaluate_loss(model, batches, keep_mask_value, rank, world, device, is_ddp):
    """evaluate_loss over every batch, each rank scoring only batches[rank::world]
    (the result is identical on every rank). Runs eager: the unpacked val
    batches vary in length, which would make compiled layers recompile."""
    with torch.compiler.set_stance("force_eager"):
        nll, n = evaluate_loss_sums(model, batches[rank::world], keep_mask_value)
    nll, n = all_reduce_sum(nll, device, is_ddp), all_reduce_sum(n, device, is_ddp)
    loss = nll / max(1, n)
    return loss, math.exp(min(loss, 20.0))


def run(cfg, run_dir):
    rank, world, local_rank, device, is_ddp = ddp_setup()
    is_main = rank == 0
    torch.manual_seed(cfg["seed"] + rank)

    packing = bool(cfg.get("packing", False))
    global_key = "global_batch_seqs" if packing else "global_batch_docs"
    if cfg.get(global_key):
        # fixed documents (packed: rows) per optimizer step whatever the GPU
        # count (the scale configs, which may run on 1-4 nodes): grad_accum
        # follows world size
        per_micro = cfg["local_batch"] * world
        if cfg[global_key] % per_micro:
            raise SystemExit(f"{global_key} {cfg[global_key]} is not a multiple of "
                             f"local_batch x world_size = {per_micro}")
        cfg["grad_accum"] = cfg[global_key] // per_micro

    if is_main:
        os.makedirs(run_dir, exist_ok=True)
        save_config(cfg, os.path.join(run_dir, "resolved_config.yaml"))
    logger = MetricLogger(run_dir)
    last_path = os.path.join(run_dir, "ckpt_last.pt")
    best_path = os.path.join(run_dir, "ckpt_best.pt")

    tokenizer = AutoTokenizer.from_pretrained(cfg["tokenizer"])
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    check_manifest(cfg, tokenizer)

    variant = cfg["variant"]
    assert variant in VARIANTS

    train_stems, val_stems = split_shards(cfg["tok_dir"], cfg["n_val_shards"])
    my_stems = rank_shards(train_stems, rank, world)
    if is_main:
        print(f"[{cfg['name']}] train shards: {len(train_stems)}  val shards: {len(val_stems)}  "
              f"world_size: {world}", flush=True)

    train_source, val_source = build_condition_sources(cfg, tokenizer, train_stems, val_stems)
    cond_source_name = cfg.get("text_conditioning_source", "self")
    if is_main and isinstance(train_source, DomainPartnerSampler):
        min_docs = train_source.min_domain_docs
        print(f"[{cfg['name']}] {cond_source_name} conditioning "
              f"(up to {train_source.n_partners} partner doc(s)); docs with an in-domain partner: "
              f"train {100 * train_source.index.coverage(min_docs):.1f}%  "
              f"val {100 * val_source.index.coverage(min_docs):.1f}%", flush=True)
    elif is_main and train_source is not None:
        print(f"[{cfg['name']}] url conditioning: encoder sees the document's "
              f"{train_source.field} (<= {train_source.encoder_max_tokens} tokens)", flush=True)

    fsdp = bool(cfg.get("fsdp", False)) and is_ddp
    if fsdp:
        # every rank builds the identical initialization, then keeps only its
        # shard (DDP instead broadcasts rank 0's weights when wrapping)
        torch.manual_seed(cfg["seed"])
    model, _ = build_model(cfg, tokenizer)
    if is_main:
        n_lm = count_parameters(model.lm)
        n_enc = count_parameters(model.conditioner.encoder) if model.conditioner is not None else 0
        print(f"[{cfg['name']}] backbone params: {n_lm/1e6:.1f}M  encoder params: {n_enc/1e6:.1f}M  "
              f"total: {(n_lm+n_enc)/1e6:.1f}M  ({'fsdp' if fsdp else 'ddp' if is_ddp else 'single'}"
              f"{', gradient checkpointing' if cfg.get('gradient_checkpointing') else ''})", flush=True)

    if cfg.get("compile", False) and device.type == "cuda":
        model.compile_for_training()

    raw_model = model
    if fsdp:
        torch.manual_seed(cfg["seed"] + rank)
        model = raw_model = sharding.shard_model(model, device, world)
    else:
        model = model.to(device)
        if is_ddp:
            ddp_kwargs = {"device_ids": [local_rank]} if device.type == "cuda" else {}
            # grads live in DDP's all-reduce buckets instead of a second copy
            # (saves one model's worth of fp32 memory -- ~13GB at 3B)
            model = torch.nn.parallel.DistributedDataParallel(model, gradient_as_bucket_view=True,
                                                              **ddp_kwargs)
            raw_model = model.module

    opt = make_optimizer(raw_model, cfg, fused=False if fsdp else None)

    def full_states(with_optimizer=True):
        """(model state, optimizer state) to checkpoint. Under FSDP every rank
        must call this (it gathers the shards); only rank 0 gets the tensors."""
        if not fsdp:
            return raw_model, (opt if with_optimizer else None)
        return (sharding.full_model_state(raw_model),
                sharding.full_optimizer_state(raw_model, opt) if with_optimizer else None)

    step, best_val, tokens_seen = 0, float("inf"), 0
    data_cursor, loss_tokens_seen = None, 0
    if os.path.exists(last_path) and not cfg.get("no_resume", False):
        if fsdp:
            ckpt = torch.load(last_path, map_location="cpu", mmap=True, weights_only=False)
            sharding.load_full_model_state(raw_model, ckpt["model"])
            sharding.load_full_optimizer_state(raw_model, opt, ckpt["optimizer"])
            step, best_val, tokens_seen = ckpt["step"], ckpt["best_val"], ckpt["tokens_seen"]
            data_cursor, loss_tokens_seen = ckpt.get("data_cursor"), ckpt.get("loss_tokens_seen", 0)
            del ckpt
        else:
            step, best_val, tokens_seen = load_checkpoint(last_path, raw_model, opt, map_location=device)
            if packing:
                ckpt = torch.load(last_path, map_location="cpu", mmap=True, weights_only=False)
                data_cursor, loss_tokens_seen = ckpt.get("data_cursor"), ckpt.get("loss_tokens_seen", 0)
                del ckpt
        if packing:
            if data_cursor is None or len(data_cursor) != world:
                raise SystemExit(f"{last_path} has no data cursor for {world} ranks; a packed run must "
                                 f"resume with the GPU count it started with")
            data_cursor = data_cursor[rank]
        if is_main:
            print(f"resumed from {last_path} at step {step}", flush=True)

    # Standing, never-overwritten snapshots for evaluation (distinct from
    # ckpt_last.pt above, which is resume-only and gets overwritten on every
    # validation step). One is written every ckpt_save_interval_tokens tokens
    # actually consumed, so the cadence is correct regardless of
    # world_size/batch/block_size -- unlike a fixed step count, which would
    # only hit ~250M tokens for the world_size it was computed against.
    ckpt_token_interval = cfg.get("ckpt_save_interval_tokens", 250_000_000)
    eval_ckpt_dir = os.path.join(run_dir, "checkpoints")
    if is_main:
        os.makedirs(eval_ckpt_dir, exist_ok=True)
    next_ckpt_tokens = ((tokens_seen // ckpt_token_interval) + 1) * ckpt_token_interval

    encoder_max_tokens = cfg.get("encoder_max_tokens", 512)
    if not packing:
        skip_docs = step * cfg["local_batch"] * cfg["grad_accum"]
        stream = TokenizedDocStream(cfg["tok_dir"], my_stems, cfg["seed"], skip_docs=skip_docs)

    max_steps = cfg["max_steps"]
    # "text" applies its schedule post-hoc, in embedding space (see the training
    # loop below); "meco"'s domain prefix is ordinary tokens, so its schedule has
    # to be applied at collation time instead (keep_mask_fn below) -- see
    # src/data/collator.py's train_batches/collate_lm_batch docstrings. Domain-
    # conditioned / URL-encoded "text" runs also apply it at collation time, so
    # a dropped example's conditioning input is never fetched or encoded.
    collate_time_schedule = variant == "meco" or train_source is not None
    schedule = None
    if variant in ("text", "meco"):
        schedule = build_schedule(cfg.get("conditioning_schedule", {"kind": "constant", "p_condition": 1.0}))

    def collate_keep_mask_fn(n=None):
        p = schedule(step, max_steps)
        return sample_keep_mask(p, cfg["local_batch"] if n is None else n, "cpu").tolist()

    bucketing = bool(cfg.get("length_bucketing", False)) and not packing
    exact_tokens = fsdp or bucketing
    rows_per_step = cfg["local_batch"] * cfg["grad_accum"]
    if packing:
        mode = packing_mode(cfg)
        packer = PackedBatcher(
            cfg["tok_dir"], my_stems, tokenizer, mode, cfg["block_size"], rows_per_step=rows_per_step,
            micro_rows=cfg["local_batch"], seed=cfg["seed"], rank=rank,
            meco_max_tokens=cfg["meco_max_tokens"],
            url_field=train_source.field if mode == "url" else "url",
            url_max_tokens=train_source.encoder_max_tokens if mode == "url" else 0,
            p_condition_fn=(lambda: schedule(step, max_steps)) if schedule is not None else None,
            cursor=data_cursor)
        step_batches = Prefetcher(packer, depth=cfg.get("prefetch_steps", 4),
                                  transform=(lambda x: ([pin_batch(b) for b in x[0]], x[1]))
                                  if device.type == "cuda" else None)
        if is_main:
            print(f"[{cfg['name']}] packing ({mode}): {rows_per_step * world} x {cfg['block_size']} tokens "
                  f"per step ({rows_per_step * world * cfg['block_size'] / 1e6:.2f}M), "
                  f"micro-batch {cfg['local_batch']} rows, grad_accum {cfg['grad_accum']}"
                  f"{', compiled' if cfg.get('compile') and device.type == 'cuda' else ''}", flush=True)
    elif bucketing:
        step_batches = bucketed_step_batches(
            stream, tokenizer, variant, cfg["block_size"], cfg["meco_max_tokens"], encoder_max_tokens,
            docs_per_step=cfg["local_batch"] * cfg["grad_accum"],
            tokens_per_micro=cfg["local_batch"] * cfg["block_size"],
            n_micro_fn=lambda k: int(all_reduce_max(k, device, is_ddp)),
            keep_mask_fn=collate_keep_mask_fn if collate_time_schedule else None,
            condition_source=train_source)
    else:
        batches = train_batches(stream, tokenizer, variant, cfg["block_size"], cfg["meco_max_tokens"],
                                encoder_max_tokens, cfg["local_batch"],
                                keep_mask_fn=collate_keep_mask_fn if collate_time_schedule else None,
                                condition_source=train_source)
    stop_at = float(os.environ.get("TRAIN_STOP_AT") or "inf")
    val_batches = load_val_batches(cfg["tok_dir"], val_stems, tokenizer, variant, cfg["block_size"],
                                   cfg["meco_max_tokens"], encoder_max_tokens, cfg["val_docs"],
                                   cfg["val_batch_size"])
    # meco also needs an "unconditioned" (no domain prefix) val set to measure
    # what its backbone does without metadata, using the same trained weights.
    # Always the two extremes (fully conditioned / fully unconditioned),
    # regardless of the training schedule's current p_condition -- same
    # convention "text" uses (keep_mask_value True/False below).
    val_batches_uncond = None
    if variant == "meco":
        val_batches_uncond = load_val_batches(cfg["tok_dir"], val_stems, tokenizer, "none",
                                              cfg["block_size"], cfg["meco_max_tokens"],
                                              encoder_max_tokens, cfg["val_docs"], cfg["val_batch_size"])
    # Domain-conditioned / URL-encoded "text" runs: the same val docs,
    # conditioned the way training conditions them (null when there's no
    # conditioning input). val_batches above stay self-conditioned; domain runs
    # also log self-conditioning from them (their encoder reads document text,
    # so it's comparable with the original text_conditioned runs), URL runs
    # don't (their encoder never sees document text).
    val_batches_source = None
    log_self = cond_source_name in ("domain_random", "domain_mean")
    if val_source is not None:
        val_batches_source = load_val_batches(cfg["tok_dir"], val_stems, tokenizer, variant,
                                              cfg["block_size"], cfg["meco_max_tokens"],
                                              encoder_max_tokens, cfg["val_docs"], cfg["val_batch_size"],
                                              condition_source=val_source)

    if packing:
        for vb in (val_batches, val_batches_uncond, val_batches_source):
            if vb is not None:
                trim_batches(vb)

    dtype_map = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}
    dtype = dtype_map[cfg.get("dtype", "bfloat16")]
    if device.type == "cuda":
        autocast = torch.autocast("cuda", dtype=dtype)
    else:
        autocast = torch.autocast("cpu", dtype=torch.bfloat16, enabled=(dtype == torch.bfloat16))

    if packing:     # each rank scores 1/world of the val set
        def evaluate_loss(m, batches, keep_mask_value):
            return sharded_evaluate_loss(m, batches, keep_mask_value, rank, world, device, is_ddp)
    else:
        evaluate_loss = evaluate_loss_full

    model.train()
    t0 = time.time()
    while step < max_steps:
        lr = lr_at(step, cfg)
        for pg in opt.param_groups:
            pg["lr"] = lr

        opt.zero_grad(set_to_none=True)
        loss_accum, tok_accum = 0.0, 0
        if packing:
            micro_batches, data_cursor = next(step_batches)
            # exact token-weighted mean over the global step (DDP averages
            # gradients over ranks, hence the factor world)
            global_tokens = max(1.0, all_reduce_mean(sum(b["n_loss_tokens"] for b in micro_batches),
                                                     device, is_ddp) * world)
            loss_sum = torch.zeros((), device=device)
            for micro, batch in enumerate(micro_batches):
                if is_ddp and not fsdp:
                    model.require_backward_grad_sync = (micro == len(micro_batches) - 1)
                with autocast:
                    loss, _, _ = model(batch)
                (loss * (batch["n_loss_tokens"] * world / global_tokens)).backward()
                loss_sum += loss.detach() * batch["n_loss_tokens"]
            train_loss = all_reduce_mean(loss_sum.item(), device, is_ddp) * world / global_tokens
            tok_accum = rows_per_step * cfg["block_size"]
            loss_tokens_seen += round(global_tokens)
        else:
            if bucketing:
                micro_batches = next(step_batches)
                step_tokens = max(1, sum(count_loss_tokens(b, variant) for b in micro_batches))
            else:
                micro_batches = (next(batches) for _ in range(cfg["grad_accum"]))
            n_micro = len(micro_batches) if bucketing else cfg["grad_accum"]
            for micro, batch in enumerate(micro_batches):
                keep_mask = None
                if variant == "text" and not collate_time_schedule:
                    p = schedule(step, max_steps)
                    keep_mask = sample_keep_mask(p, batch["input_ids"].shape[0], device)
                if is_ddp and not fsdp:
                    model.require_backward_grad_sync = (micro == n_micro - 1)
                with autocast:
                    loss, n_tok, _ = model(batch, keep_mask=keep_mask)
                if bucketing:
                    # token-weighted mean over the whole step, whatever the split
                    (loss * (n_tok / step_tokens)).backward()
                    loss_accum += loss.item() * n_tok.item()
                else:
                    (loss / cfg["grad_accum"]).backward()
                    loss_accum += loss.item()
                tok_accum += n_tok.item()
            train_loss = loss_accum / max(1, tok_accum) if bucketing else loss_accum / cfg["grad_accum"]

        gnorm = sharding.scalar(torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"]))
        opt.step()
        step += 1
        if exact_tokens:
            # exact global count, identical on every rank (under FSDP the
            # snapshot below is a collective, so all must agree on when)
            tokens_seen += round(all_reduce_mean(tok_accum, device, is_ddp) * world)
        else:
            tokens_seen += tok_accum * world

        if tokens_seen >= next_ckpt_tokens and (is_main or fsdp):
            snap_path = os.path.join(
                eval_ckpt_dir, f"ckpt_step{step:07d}_tok{tokens_seen/1e6:.0f}M.pt")
            model_state, _ = full_states(with_optimizer=False)
            if is_main:
                save_eval_checkpoint(snap_path, model_state, step, best_val, tokens_seen, cfg)
                print(f"[{cfg['name']}] wrote eval checkpoint {snap_path}", flush=True)
            del model_state
            while tokens_seen >= next_ckpt_tokens:
                next_ckpt_tokens += ckpt_token_interval

        if is_main and step % cfg["log_interval"] == 0:
            dt = time.time() - t0
            t0 = time.time()
            rec = {"step": step, "split": "train", "loss": train_loss,
                   "lr": lr, "tokens": tokens_seen, "grad_norm": gnorm,
                   "sec_per_step": dt / cfg["log_interval"]}
            if device.type == "cuda":
                rec["max_mem_gb"] = torch.cuda.max_memory_allocated(device) / 2**30
            if variant in ("text", "meco"):
                rec["p_condition"] = schedule(step, max_steps)
            if packing:
                rec["loss_tokens"] = loss_tokens_seen
            logger.log(rec)
            print(f"step {step:>6}  loss {rec['loss']:.4f}  lr {lr:.2e}  "
                  f"tok {tokens_seen/1e9:.3f}B  |g| {gnorm:.2f}  "
                  f"{rec['sec_per_step']:.2f}s/it"
                  + (f"  mem {rec['max_mem_gb']:.1f}G" if "max_mem_gb" in rec else ""), flush=True)

        # past the job's deadline: validate + write ckpt_last.pt now, then exit
        # (rank 0's clock decides, so every rank stops at the same step)
        stop = stop_at != float("inf") and bool(broadcast_scalar(float(time.time() > stop_at), device, is_ddp))
        if step % cfg["val_interval"] == 0 or step == max_steps or stop:
            self_loss = None
            if variant == "text" and val_batches_source is not None:
                # headline val loss (and ckpt_best.pt selection) = the
                # training-time condition (in-domain partners / the URL)
                val_loss, val_ppl = evaluate_loss(raw_model, val_batches_source, keep_mask_value=True)
                if log_self:
                    self_loss, _ = evaluate_loss(raw_model, val_batches, keep_mask_value=True)
                uncond_loss, uncond_ppl = evaluate_loss(raw_model, val_batches, keep_mask_value=False)
            elif variant == "text":
                val_loss, val_ppl = evaluate_loss(raw_model, val_batches, keep_mask_value=True)
                uncond_loss, uncond_ppl = evaluate_loss(raw_model, val_batches, keep_mask_value=False)
            elif variant == "meco":
                val_loss, val_ppl = evaluate_loss(raw_model, val_batches, keep_mask_value=None)
                uncond_loss, uncond_ppl = evaluate_loss(raw_model, val_batches_uncond, keep_mask_value=None)
            else:
                val_loss, val_ppl = evaluate_loss(raw_model, val_batches, keep_mask_value=None)
                uncond_loss, uncond_ppl = None, None

            val_loss = all_reduce_mean(val_loss, device, is_ddp)
            rec = {"step": step, "split": "val", "loss": val_loss,
                   "ppl": math.exp(min(val_loss, 20.0)), "tokens": tokens_seen}
            if uncond_loss is not None:
                uncond_loss = all_reduce_mean(uncond_loss, device, is_ddp)
                uncond_ppl = math.exp(min(uncond_loss, 20.0))
                rec["unconditioned_loss"] = uncond_loss
                rec["unconditioned_ppl"] = uncond_ppl
                if variant == "text" and val_batches_source is not None:
                    prefix = SOURCE_METRIC_PREFIX[cond_source_name]
                    rec[f"{prefix}_loss"] = val_loss
                    rec[f"{prefix}_ppl"] = rec["ppl"]
                    if self_loss is not None:
                        self_loss = all_reduce_mean(self_loss, device, is_ddp)
                        rec["self_conditioned_loss"] = self_loss
                        rec["self_conditioned_ppl"] = math.exp(min(self_loss, 20.0))
                elif variant == "text":
                    rec["self_conditioned_loss"] = val_loss
                    rec["self_conditioned_ppl"] = rec["ppl"]
                elif variant == "meco":
                    rec["meco_conditioned_loss"] = val_loss
                    rec["meco_conditioned_ppl"] = rec["ppl"]

            # ckpt_last.pt (the resume checkpoint) is saved on every validation
            # step, not on its own interval -- that keeps the two in lockstep so
            # a resumed run never has to re-derive a data cursor past the last
            # validation it actually logged, and bounds lost work on interruption
            # to at most val_interval steps. val_loss is already all-reduced and
            # best_val broadcast, so every rank takes the same branch (needed
            # under FSDP, where gathering the state is a collective).
            improved = val_loss < best_val
            if improved:
                best_val = val_loss
            if is_main:
                logger.log(rec)
                logger.export_csv()
                print(f"  [val] step {step}  loss {val_loss:.4f}  ppl {rec['ppl']:.2f}", flush=True)
            extra = None
            if packing:     # every rank's stream position (a collective)
                extra = {"data_cursor": [tuple(c) for c in all_gather_object(tuple(data_cursor), is_ddp)],
                         "loss_tokens_seen": loss_tokens_seen}
            if cfg.get("save_checkpoints", True) and (is_main or fsdp):   # false: throughput tests only
                model_state, opt_state = full_states()
                if is_main:
                    if improved and cfg.get("save_best_checkpoint", True):
                        save_checkpoint(best_path, model_state, opt_state, step, best_val, tokens_seen, cfg,
                                        extra)
                    save_checkpoint(last_path, model_state, opt_state, step, best_val, tokens_seen, cfg, extra)
                del model_state, opt_state
            best_val = broadcast_scalar(best_val, device, is_ddp)
            if stop:
                if is_main:
                    print(f"[{cfg['name']}] reached TRAIN_STOP_AT at step {step}; saved ckpt_last.pt, "
                          f"exiting (rerun to resume)", flush=True)
                break

    if is_main:
        print(f"[{cfg['name']}] done. best val loss {best_val:.4f}", flush=True)
        logger.export_csv()
    ddp_cleanup(is_ddp)
