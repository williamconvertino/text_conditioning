"""Fine-tuning loop (scripts/finetune/run_finetune.py, README "Fine-Tuning").

Objective (`objective`):
  choice_ce  cross-entropy over a doc's choices, scored by their
             log-likelihoods (length-normalized for acc_norm tasks) -- a
             smooth surrogate for the accuracy lm-eval reports, and the same
             loss the steering experiments optimize vectors with
             (src/steering/scoring.py's doc_loss_and_correct). Loglikelihood
             tasks (LAMBADA) use the target's per-token NLL.
  lm         ordinary supervised fine-tuning: per-token NLL of the correct
             answer given the question.

Evaluation: step 0 (before any update) and every `eval_every` steps score
the select split and the curve subset of eval; the curve is what
src/finetune/report.py plots against step. The iterate with the best mean
select accuracy (step 0 included, so fine-tuning can never be selected into
something worse than the starting point on select) is kept in CPU memory and
scored on the full eval split at the end, alongside the final iterate and the
step-0 model. Every accuracy comes from lm-eval's own task.process_results
applied to FinetuneModel's fp32 scores (the same numbers CtrlTokensLM
produces), so they are directly comparable with the benchmark tables.

Distributed: under torchrun, training batches are split across ranks (DDP,
or FSDP with `fsdp: true` for full fine-tuning of the 8B model) and every
evaluation is sharded by document, then gathered.
"""

import json
import math
import os
import time
from contextlib import nullcontext

import numpy as np
import torch
import torch.distributed as dist

from src.finetune.data import DocSampler, doc_batches, make_batch
from src.steering.scoring import doc_loss_and_correct
from src.utils import sharding

DTYPES = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}


class Dist:
    def __init__(self, rank, world, device, is_ddp, fsdp):
        self.rank, self.world, self.device, self.is_ddp, self.fsdp = rank, world, device, is_ddp, fsdp
        self.is_main = rank == 0

    def max_int(self, x):
        if not self.is_ddp:
            return x
        t = torch.tensor([x], device=self.device)
        dist.all_reduce(t, op=dist.ReduceOp.MAX)
        return int(t.item())

    def mean(self, x):
        if not self.is_ddp:
            return x
        t = torch.tensor([x], device=self.device, dtype=torch.float64)
        dist.all_reduce(t)
        return t.item() / self.world

    def gather(self, obj):
        if not self.is_ddp:
            return [obj]
        out = [None] * self.world
        dist.all_gather_object(out, obj)
        return out


def autocast_ctx(device, dtype):
    if dtype == torch.float32:
        return nullcontext()
    return torch.autocast(device.type, dtype=dtype)


def lr_factor(step, cfg):
    """Linear warmup, then cosine decay to min_lr_frac of the peak."""
    warmup, total = cfg["warmup_steps"], cfg["steps"]
    if step < warmup:
        return (step + 1) / max(1, warmup)
    prog = min(1.0, (step - warmup) / max(1, total - warmup))
    lo = cfg.get("min_lr_frac", 0.1)
    return lo + 0.5 * (1 - lo) * (1 + math.cos(math.pi * prog))


def make_optimizer(ft, mode, cfg):
    params = ft.trainable_parameters()
    if mode == "peft":
        # Adam's step is ~lr per coordinate, so scale it to the vector being
        # learned: page-URL vectors have ~1.4 RMS per coordinate, token
        # embeddings ~100x less (src/steering's lr 0.02 ~= 0.015 x 1.4).
        with torch.no_grad():
            rms = float(sharding.full_tensor(ft.soft_prompt.detach()).float().pow(2).mean().sqrt())
        lr = cfg["peft_lr_relative"] * max(rms, 1e-6)
        return torch.optim.Adam(params, lr=lr), lr
    decay = [p for p in params if p.dim() >= 2]
    no_decay = [p for p in params if p.dim() < 2]
    groups = [{"params": decay, "weight_decay": cfg["weight_decay"]},
              {"params": no_decay, "weight_decay": 0.0}]
    return torch.optim.AdamW(groups, lr=cfg["lr"], betas=(0.9, 0.95), eps=1e-8), cfg["lr"]


@torch.no_grad()
def score(ft_call, td, split, doc_ids, prefix_ids, pad_id, D, cfg):
    """Accuracy of the current model on td's `split` docs doc_ids, sharded
    across ranks by document. Returns (acc, stderr, {doc_id: value})."""
    prep, docs = td.prep[split], td.docs[split]
    mine = doc_ids[D.rank::D.world]
    batches = doc_batches(prep, mine, cfg["eval_batch_seqs"])
    n_batches = D.max_int(len(batches)) if D.fsdp else len(batches)
    per_doc = {}
    with autocast_ctx(D.device, DTYPES[cfg["eval_dtype"]]):
        for i in range(n_batches):
            if i >= len(batches):
                # FSDP forwards are collectives: every rank runs the same count
                ids, mask = make_batch(prep, [0], prefix_ids, pad_id)
                ft_call(ids.to(D.device), mask.to(D.device))
                continue
            rows = prep.doc_rows(batches[i])
            ids, mask = make_batch(prep, rows, prefix_ids, pad_id)
            sums, greedy = ft_call(ids.to(D.device), mask.to(D.device))
            sums, greedy = sums.float().cpu().tolist(), greedy.cpu().tolist()
            pos = 0
            for d in batches[i]:
                n = len(prep.doc_rows([d]))
                res = list(zip(sums[pos:pos + n], greedy[pos:pos + n]))
                per_doc[d] = float(td.task.process_results(docs[d], res)[td.metric])
                pos += n
    merged = {}
    for part in D.gather(per_doc):
        merged.update(part)
    vals = [merged[d] for d in doc_ids]
    n = len(vals)
    acc = sum(vals) / max(1, n)
    stderr = float(np.std(vals, ddof=1) / math.sqrt(n)) if n > 1 else None
    return acc, stderr, merged


def subset_acc(per_doc, ids):
    vals = [per_doc[d] for d in ids]
    return sum(vals) / max(1, len(vals))


def train_loss(ft_call, td, doc_ids, prefix_ids, pad_id, device, objective):
    prep = td.prep["fit"]
    if objective == "choice_ce":
        rows = prep.doc_rows(doc_ids)
        ids, mask = make_batch(prep, rows, prefix_ids, pad_id)
        sums, greedy = ft_call(ids.to(device), mask.to(device))
        loss, correct = doc_loss_and_correct(prep, doc_ids, rows, sums, greedy)
        return loss, correct
    if objective == "lm":
        rows = td.gold_rows(prep, doc_ids)
        ids, mask = make_batch(prep, rows, prefix_ids, pad_id)
        sums, _ = ft_call(ids.to(device), mask.to(device))
        n_tok = torch.tensor([prep.cont_lens[r] for r in rows], device=sums.device, dtype=sums.dtype)
        return (-sums / n_tok).mean(), []
    raise ValueError(f"objective must be choice_ce | lm, got {objective!r}")


def run_finetune(ft, wrapped, tasks, mode, cfg, D, out_dir, pad_id, info, log=print):
    """ft: the FinetuneModel (FSDP-sharded in place, or unwrapped); wrapped:
    what training calls (DDP wrapper, or ft itself). tasks: [TaskData] (one
    task, or several for a joint "all" run: each micro-batch cycles through
    them). Writes out_dir/{metrics.jsonl, result.json, prefix.pt}."""
    prefix_ids = ft.prefix_ids
    eval_call = ft                      # no DDP sync needed to score
    opt, peak_lr = make_optimizer(ft, mode, cfg)
    trainable = ft.trainable_parameters()

    per_rank = max(1, cfg["batch_docs"] // D.world)
    micro = min(cfg["micro_docs"], per_rank)
    n_micro = math.ceil(per_rank / micro)
    samplers = [DocSampler(td.prep["fit"].n_docs, f"{cfg['seed']}-{td.name}", D.rank) for td in tasks]
    train_dtype = DTYPES[cfg["train_dtype"]]

    metrics_path = os.path.join(out_dir, "metrics.jsonl")
    if D.is_main:
        os.makedirs(out_dir, exist_ok=True)
        open(metrics_path, "w").close()

    def log_rec(rec):
        if D.is_main:
            with open(metrics_path, "a") as f:
                f.write(json.dumps(rec) + "\n")

    def evaluate(step, full=False, record=True):
        """{task: {select, curve[, eval, eval_stderr, per_doc]}} at `step`."""
        ft.eval()
        out = {}
        for td in tasks:
            sel, _, _ = score(eval_call, td, "select", list(range(td.prep["select"].n_docs)), prefix_ids,
                              pad_id, D, cfg)
            ids = list(range(td.prep["eval"].n_docs)) if full else td.curve_ids
            acc, stderr, per_doc = score(eval_call, td, "eval", ids, prefix_ids, pad_id, D, cfg)
            rec = {"select": sel, "curve": subset_acc(per_doc, td.curve_ids)}
            if full:
                rec.update(eval=acc, eval_stderr=stderr, n_eval=len(ids), per_doc=[per_doc[d] for d in ids])
            out[td.name] = rec
            if record:
                log_rec({"step": step, "split": "eval", "task": td.name, "select_acc": sel,
                         "eval_acc": rec["curve"], "n_curve": len(td.curve_ids)})
        ft.train()
        mean_sel = sum(r["select"] for r in out.values()) / len(out)
        log(f"  [eval] step {step:>5}  select {mean_sel:.4f}  "
            + "  ".join(f"{t} {r['curve']:.4f}" for t, r in out.items()))
        return out, mean_sel

    def snapshot():
        return [sharding.local(p.detach()).to("cpu", copy=True) for p in trainable]

    def restore(snap):
        with torch.no_grad():
            for p, s in zip(trainable, snap):
                sharding.local(p.data).copy_(s)

    t0 = time.time()
    base_eval, best_sel = evaluate(0, full=True)
    best_step, best_snap = 0, None
    history = [{"step": 0, **{t: {"select": r["select"], "curve": r["curve"]} for t, r in base_eval.items()}}]

    ft.train()
    for step in range(1, cfg["steps"] + 1):
        for g in opt.param_groups:
            g["lr"] = peak_lr * lr_factor(step - 1, cfg)
        opt.zero_grad(set_to_none=True)
        tot, correct = 0.0, []
        for m in range(n_micro):
            k = ((step - 1) * n_micro + m) % len(tasks)
            ids = samplers[k].take(min(micro, per_rank - m * micro))
            sync = nullcontext()
            if D.is_ddp and not D.fsdp and m < n_micro - 1:
                sync = wrapped.no_sync()
            with sync:
                with autocast_ctx(D.device, train_dtype):
                    loss, c = train_loss(wrapped, tasks[k], ids, prefix_ids, pad_id, D.device, cfg["objective"])
                (loss / n_micro).backward()
            tot += loss.item() / n_micro
            correct += c
        gnorm = sharding.scalar(torch.nn.utils.clip_grad_norm_(trainable, cfg["grad_clip"]))
        opt.step()

        if step % cfg["log_every"] == 0:
            loss_mean = D.mean(tot)
            rec = {"step": step, "split": "train", "loss": loss_mean, "lr": opt.param_groups[0]["lr"],
                   "grad_norm": gnorm}
            if correct:
                rec["fit_batch_acc"] = D.mean(sum(correct) / len(correct))
            log_rec(rec)
            log(f"step {step:>5}  loss {loss_mean:.4f}  lr {rec['lr']:.2e}  |g| {gnorm:.3f}  "
                f"({time.time() - t0:.0f}s)")

        last = step == cfg["steps"]
        if last or step % cfg["eval_every"] == 0:
            res, mean_sel = evaluate(step, full=last)
            history.append({"step": step, **{t: {"select": r["select"], "curve": r["curve"]}
                                             for t, r in res.items()}})
            if mean_sel > best_sel:
                best_sel, best_step = mean_sel, step
                best_snap = None if last else snapshot()
            if last:
                final_eval = res

    if best_step == cfg["steps"]:
        best_eval = final_eval
    elif best_step == 0:
        best_eval = base_eval
    else:
        restore(best_snap)              # the model is left at the kept iterate
        best_eval, _ = evaluate(best_step, full=True, record=False)

    # the kept (best-on-select) iterate's learned conditioning, for peft
    prefix = None
    if ft.soft_prompt is not None:
        prefix = sharding.full_tensor(ft.soft_prompt.detach()).float().cpu()

    def summary(e):
        return {t: {k: v for k, v in r.items() if k != "per_doc"} for t, r in e.items()}

    result = {
        "mode": mode, "objective": cfg["objective"], "tasks": [td.name for td in tasks],
        "splits": {td.name: td.splits.info for td in tasks}, "metric": {td.name: td.metric for td in tasks},
        "n_curve": {td.name: len(td.curve_ids) for td in tasks},
        "conditioning": info, "peak_lr": peak_lr, "config": cfg,
        "base": summary(base_eval), "final": summary(final_eval), "best": summary(best_eval),
        "best_step": best_step, "best_select": best_sel, "history": history,
        "seconds": time.time() - t0,
    }
    if D.is_main:
        with open(os.path.join(out_dir, "result.json.tmp"), "w") as f:
            json.dump(result, f, indent=1, default=float)
        os.replace(os.path.join(out_dir, "result.json.tmp"), os.path.join(out_dir, "result.json"))
        torch.save({"per_doc": {k: {t: r["per_doc"] for t, r in e.items()}
                                for k, e in (("base", base_eval), ("final", final_eval), ("best", best_eval))},
                    "prefix": prefix, "conditioning": info}, os.path.join(out_dir, "prefix.pt"))
    return result
