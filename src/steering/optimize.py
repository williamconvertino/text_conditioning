"""Search for a task's steering vector (scripts/steering/run_task_vectors.py).
The LM and encoder stay frozen; only the conditioning vector changes.

Three searches, from most to least constrained:

  domain_search  pick the best single domain from a candidate list, by
                 select-split accuracy. The discrete search MeCo can also
                 do (as a domain string prefix), so it is run for both.
  simplex        z = sum_i softmax(w)_i * L_i over a library L of domain
                 vectors (+ the null vector): the best point in the convex
                 hull of real domains, so it stays near conditioning the LM
                 was trained on, and the weights say which domains it mixes.
  free           z = z0 + delta, delta unconstrained in R^H (norm-capped):
                 the most expressive, and the one whose closest domains are
                 the most informative, since nothing forces it toward any
                 particular domain.

simplex and free are optimized with Adam on the fit split against the
choice cross-entropy (src/steering/scoring.py), evaluated on the select split
(with the exact scorer, via select_fn) every `eval_every` steps, and the best select-accuracy iterate is kept
(including step 0, the init) -- so the result is never worse on select than
where it started.
"""

import random
import time

import torch

from src.steering.scoring import doc_loss_and_correct, doc_scores, fast_accuracy


def domain_search(score_fn, candidates, log=print):
    """candidates: [(name, payload)]; score_fn(payload) -> select accuracy.
    Returns (ranked [{name, acc}], best payload)."""
    ranked = []
    best, best_payload = -1, None
    t0 = time.time()
    for i, (name, payload) in enumerate(candidates):
        acc = score_fn(payload)
        ranked.append({"name": name, "acc": acc})
        if acc > best:
            best, best_payload = acc, payload
        if (i + 1) % 10 == 0:
            log(f"    domain search {i + 1}/{len(candidates)}  best {best:.4f}  ({time.time() - t0:.0f}s)")
    ranked.sort(key=lambda r: -r["acc"])
    return ranked, best_payload


class _Param(torch.nn.Module):
    def __init__(self, kind, init, library=None, max_norm=None):
        super().__init__()
        self.kind = kind
        self.max_norm = max_norm
        if kind == "free":
            self.register_buffer("z0", init.clone().float())
            self.delta = torch.nn.Parameter(torch.zeros_like(self.z0))
        elif kind == "simplex":
            self.register_buffer("L", library.float())
            self.logits = torch.nn.Parameter(torch.zeros(len(library)))
        else:
            raise ValueError(kind)

    def weights(self):
        return torch.softmax(self.logits, 0)

    def forward(self):
        if self.kind == "simplex":
            return self.weights() @ self.L
        z = self.z0 + self.delta
        if self.max_norm is not None:
            n = z.norm()
            z = z * torch.clamp(self.max_norm / n, max=1.0)
        return z


def fit_vector(model, preps, kind, pad_id, device, *, init=None, library=None, max_norm=None,
               steps=300, batch_docs=16, lr=0.05, eval_every=25, l2=0.0, seed=0, select_fn=None, log=print):
    """preps: [(fit PreparedDocs, select PreparedDocs)] -- one pair per task;
    each step sums the loss over every task (the joint "all tasks" vector is
    the multi-task case). select_fn(z) -> select accuracy (default: the bf16
    fast scorer); pass the exact scorer so early stopping agrees with the
    reported numbers. Returns (best z [H] on CPU, history, extra)."""
    rng = random.Random(seed)
    param = _Param(kind, init, library, max_norm).to(device)
    opt = torch.optim.Adam(param.parameters(), lr=lr)
    scale = (init.float().norm() ** 2).clamp(min=1.0).to(device) if init is not None else 1.0

    def select_acc():
        with torch.no_grad():
            z = param()
            if select_fn is not None:
                return select_fn(z.detach())
            return sum(fast_accuracy(model, z, sel, pad_id, device) for _, sel in preps) / len(preps)

    snapshot = lambda: param.weights().detach().cpu() if kind == "simplex" else None
    best_acc, best_z, best_w = select_acc(), param().detach().cpu(), snapshot()
    best_step = 0
    history = [{"step": 0, "select_acc": best_acc}]
    log(f"    [{kind}] step 0: select {best_acc:.4f}")
    t0 = time.time()
    for step in range(1, steps + 1):
        opt.zero_grad(set_to_none=True)
        total, n_correct, n_docs = 0.0, 0, 0
        # one backward per task (gradients accumulate), so the joint vector
        # never holds every task's graph at once
        for fit, _ in preps:
            ids = rng.sample(range(fit.n_docs), min(batch_docs, fit.n_docs))
            rows, sums, greedy = doc_scores(model, param(), fit, ids, pad_id, device, max_seqs=64)
            loss, correct = doc_loss_and_correct(fit, ids, rows, sums, greedy)
            (loss / len(preps)).backward()
            total += loss.item() / len(preps)
            n_correct += sum(correct)
            n_docs += len(correct)
        if kind == "free" and l2 > 0:
            reg = l2 * (param.delta ** 2).sum() / scale
            reg.backward()
            total += reg.item()
        opt.step()
        rec = {"step": step, "loss": total, "fit_acc": n_correct / max(1, n_docs)}
        if step % eval_every == 0 or step == steps:
            acc = select_acc()
            rec["select_acc"] = acc
            if acc > best_acc:
                best_acc, best_z, best_w, best_step = acc, param().detach().cpu(), snapshot(), step
            log(f"    [{kind}] step {step}: loss {rec['loss']:.4f}  fit-batch acc {rec['fit_acc']:.3f}  "
                f"select {acc:.4f}  (best {best_acc:.4f} @ {best_step}, {time.time() - t0:.0f}s)")
        history.append(rec)

    extra = {"best_step": best_step, "best_select_acc": best_acc}
    if kind == "simplex":
        extra["weights"] = best_w          # mixture weights of the kept iterate
    return best_z, history, extra
