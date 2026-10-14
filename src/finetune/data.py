"""Benchmark-task data for fine-tuning (src/finetune/loop.py).

Splits are src/steering/task_data.py's, so fine-tuning, the steering
experiments and the benchmark tables share their definitions: fit (here all
of the task's training data, n_fit null), select (model selection -- the
reported checkpoint is the best one on select), eval (lm-eval's own eval
split). The accuracy-vs-step curves score a fixed random subset of eval
(`curve_eval_docs`); the final numbers use all of it.

Requests are tokenized once, with src/benchmarks/lm_wrapper.py's own
_build_example (the tokenization and truncation every benchmark number uses),
into src/steering/scoring.py's PreparedDocs.
"""

import random

import torch

from src.benchmarks.conditioning import resolve_none
from src.benchmarks.lm_wrapper import CtrlTokensLM
from src.steering.scoring import PreparedDocs


class TaskData:
    """One task's docs and prepared requests for every split."""

    def __init__(self, splits, tokenizer, block_size, prefix_ids, curve_eval_docs, seed):
        self.name = splits.name
        self.splits = splits
        self.task = splits.task
        self.metric = splits.metric
        # Token prefixes (MeCo's domain) share the block with the request, as
        # in CtrlTokensLM; embedding-space prefix vectors are extra positions.
        builder = CtrlTokensLM(None, tokenizer, "none", resolve_none(), block_size=block_size - len(prefix_ids),
                               device=torch.device("cpu"))
        self.docs = {"fit": splits.fit, "select": splits.select, "eval": splits.eval}
        self.prep = {k: PreparedDocs(builder, splits, docs) for k, docs in self.docs.items()}
        n_eval = len(splits.eval)
        order = list(range(n_eval))
        random.Random(f"{seed}-{self.name}-curve").shuffle(order)
        self.curve_ids = sorted(order[:curve_eval_docs]) if curve_eval_docs else list(range(n_eval))

    def gold_rows(self, prep, doc_ids):
        """The request holding each doc's correct answer (the target itself
        for loglikelihood tasks)."""
        rows = []
        for d in doc_ids:
            g = prep.gold[d]
            rows.append(int(prep.first[d]) + (g or 0))
        return rows


class DocSampler:
    """Endless, per-rank shuffled passes over a task's fit docs."""

    def __init__(self, n_docs, seed, rank):
        self.n = n_docs
        self.rng = random.Random(f"{seed}-{rank}")
        self.order, self.pos = [], 0

    def take(self, k):
        out = []
        while len(out) < k:
            if self.pos >= len(self.order):
                self.order = list(range(self.n))
                self.rng.shuffle(self.order)
                self.pos = 0
            out.append(self.order[self.pos])
            self.pos += 1
        return out


def make_batch(prep, rows, prefix_ids, pad_id):
    """(input_ids [B, L], score_mask [B, L]) for request rows: MeCo-style
    prefix tokens + the request, right-padded, continuation positions marked."""
    seqs = [prefix_ids + prep.seqs[r] for r in rows]
    L = max(len(s) for s in seqs)
    input_ids = torch.full((len(seqs), L), pad_id, dtype=torch.long)
    mask = torch.zeros((len(seqs), L), dtype=torch.bool)
    for i, (s, r) in enumerate(zip(seqs, rows)):
        input_ids[i, :len(s)] = torch.tensor(s)
        mask[i, len(s) - prep.cont_lens[r]:len(s)] = True
    return input_ids, mask


def doc_batches(prep, doc_ids, max_rows):
    """Split doc_ids into batches of whole docs with <= max_rows requests each
    (a doc with more requests than that gets a batch to itself)."""
    batches, cur, n = [], [], 0
    for d in doc_ids:
        k = len(prep.doc_rows([d]))
        if cur and n + k > max_rows:
            batches.append(cur)
            cur, n = [], 0
        cur.append(d)
        n += k
    if cur:
        batches.append(cur)
    return batches
