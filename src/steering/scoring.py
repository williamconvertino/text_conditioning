"""Scoring task documents under a conditioning vector, two ways:

* score_docs: the *reported* numbers. It runs src/benchmarks/lm_wrapper.py's
  CtrlTokensLM (same tokenization, truncation, fp32 forward) and lm-eval's own
  task.process_results, so on a task's full eval split it reproduces exactly
  what run_benchmarks.py reports -- just on any subset of docs, and without
  going through simple_evaluate.

* PreparedDocs + logprobs: the fast, differentiable path (bf16 autocast)
  used to optimize a vector and to rank candidates. It is only ever used to
  make selection decisions on fit/select docs; every number reported for an
  eval split comes from score_docs.
"""

import math

import numpy as np
import torch
import torch.nn.functional as F

from src.benchmarks.lm_wrapper import CtrlTokensLM
from src.steering.task_data import Req, doc_requests, gold_index


def score_docs(lm, splits, docs):
    """lm: a CtrlTokensLM. Returns {"acc": mean headline metric, "stderr",
    "n", "per_doc": [0/1 ...]}."""
    task = splits.task
    reqs, owners = [], []
    for i, doc in enumerate(docs):
        for args in doc_requests(task, doc, i):
            reqs.append(Req(args))
            owners.append(i)
    lls = lm.loglikelihood(reqs)
    per_doc = [[] for _ in docs]
    for i, ll in zip(owners, lls):
        per_doc[i].append(ll)
    vals = [float(task.process_results(doc, res)[splits.metric]) for doc, res in zip(docs, per_doc)]
    n = len(vals)
    mean = sum(vals) / max(1, n)
    stderr = float(np.std(vals, ddof=1) / math.sqrt(n)) if n > 1 else None
    return {"acc": mean, "stderr": stderr, "n": n, "per_doc": vals}


def make_lm(model, tokenizer, cfg, spec, batch_size=32):
    return CtrlTokensLM(model, tokenizer, cfg["variant"], spec, block_size=cfg["block_size"],
                        meco_max_tokens=cfg.get("meco_max_tokens", 48), batch_size=batch_size)


class PreparedDocs:
    """Token sequences for every request of `docs`, built with CtrlTokensLM's
    own _build_example (no MeCo prefix: only used for vector-conditioned runs)."""

    def __init__(self, builder_lm, splits, docs):
        self.splits = splits
        self.seqs, self.cont_lens, self.norm, self.doc_of = [], [], [], []
        self.gold = []
        for i, doc in enumerate(docs):
            for context, continuation in doc_requests(splits.task, doc, i):
                seq, clen = builder_lm._build_example(context, continuation)
                self.seqs.append(seq)
                self.cont_lens.append(clen)
                self.norm.append(max(1, len(continuation)))
                self.doc_of.append(i)
            self.gold.append(gold_index(splits.task, doc))
        self.n_docs = len(docs)
        self.doc_of = torch.tensor(self.doc_of)
        self.norm = torch.tensor(self.norm, dtype=torch.float32)
        self.first = torch.searchsorted(self.doc_of, torch.arange(self.n_docs))
        self.mc = splits.task.OUTPUT_TYPE == "multiple_choice"
        self.use_norm = splits.metric == "acc_norm"

    def doc_rows(self, doc_ids):
        """Request indices belonging to doc_ids (in order)."""
        rows = []
        for d in doc_ids:
            lo = int(self.first[d])
            hi = int(self.first[d + 1]) if d + 1 < self.n_docs else len(self.seqs)
            rows.extend(range(lo, hi))
        return rows


def logprobs(model, z, seqs, cont_lens, pad_id, device):
    """Differentiable sum log p(continuation) and greedy-match flag per
    sequence, conditioned on z ([H], or [B, H]). Mirrors
    CtrlTokensLM._forward_batch."""
    B, T = len(seqs), max(len(s) for s in seqs)
    input_ids = torch.full((B, T), pad_id, dtype=torch.long)
    mask = torch.zeros((B, T), dtype=torch.bool)
    for i, (s, c) in enumerate(zip(seqs, cont_lens)):
        input_ids[i, :len(s)] = torch.tensor(s)
        mask[i, len(s) - c:len(s)] = True
    input_ids, mask = input_ids.to(device), mask.to(device)
    lm = model.lm
    with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
        tok = lm.get_input_embeddings()(input_ids)
        zz = z.to(device=device, dtype=tok.dtype)
        zz = zz.unsqueeze(0).expand(B, -1) if zz.dim() == 1 else zz
        out = lm(inputs_embeds=torch.cat([zz.unsqueeze(1), tok], dim=1))
    targets = torch.cat([torch.zeros(B, 1, dtype=torch.long, device=device), input_ids], 1)[:, 1:]
    full_mask = torch.cat([torch.zeros(B, 1, dtype=torch.bool, device=device), mask], 1)[:, 1:]
    # log-softmax only at scored (continuation) positions: a few per sequence,
    # instead of [B, T, vocab] in fp32.
    rows, cols = full_mask.nonzero(as_tuple=True)
    logp = F.log_softmax(out.logits[:, :-1, :][rows, cols].float(), -1)
    tgt = targets[rows, cols]
    tok_lp = logp.gather(-1, tgt.unsqueeze(-1)).squeeze(-1)
    sums = torch.zeros(B, device=device, dtype=tok_lp.dtype).index_add(0, rows, tok_lp)
    miss = torch.zeros(B, device=device, dtype=torch.long).index_add(0, rows, (logp.argmax(-1) != tgt).long())
    return sums, miss == 0


def doc_scores(model, z, prep, doc_ids, pad_id, device, max_seqs=64):
    """Per-request (sum logp, greedy) for doc_ids, chunked to <= max_seqs
    sequences per forward. Keeps the graph (call under no_grad to skip it)."""
    rows = prep.doc_rows(doc_ids)
    sums, greedy = [], []
    for s in range(0, len(rows), max_seqs):
        chunk = rows[s:s + max_seqs]
        a, b = logprobs(model, z, [prep.seqs[r] for r in chunk], [prep.cont_lens[r] for r in chunk],
                        pad_id, device)
        sums.append(a)
        greedy.append(b)
    return rows, torch.cat(sums), torch.cat(greedy)


def doc_loss_and_correct(prep, doc_ids, rows, sums, greedy):
    """Mean task loss over doc_ids and a per-doc correctness list. Multiple
    choice: cross-entropy over the choices' (length-normalized for acc_norm)
    log-likelihoods -- a smooth surrogate of the accuracy lm-eval reports.
    Loglikelihood tasks (lambada): per-token NLL of the target; correct =
    greedy match."""
    losses, correct = [], []
    pos = 0
    for d in doc_ids:
        n = len(prep.doc_rows([d]))
        s = sums[pos:pos + n]
        idx = rows[pos:pos + n]
        if prep.mc:
            scores = s / prep.norm[idx].to(s.device) if prep.use_norm else s
            gold = prep.gold[d]
            losses.append(F.cross_entropy(scores.unsqueeze(0), torch.tensor([gold], device=s.device)))
            correct.append(int(scores.argmax()) == gold)
        else:
            losses.append(-s[0] / prep.cont_lens[idx[0]])
            correct.append(bool(greedy[pos]))
        pos += n
    return torch.stack(losses).mean(), correct


@torch.no_grad()
def fast_accuracy(model, z, prep, pad_id, device, max_seqs=128):
    """bf16 selection-time accuracy of z on all of prep's docs."""
    doc_ids = list(range(prep.n_docs))
    correct = []
    for s in range(0, len(doc_ids), 64):
        ids = doc_ids[s:s + 64]
        rows, sums, greedy = doc_scores(model, z, prep, ids, pad_id, device, max_seqs)
        correct += doc_loss_and_correct(prep, ids, rows, sums, greedy)[1]
    return sum(correct) / max(1, len(correct))
