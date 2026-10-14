#!/usr/bin/env python
"""Steering experiment 2: find a strong conditioning for a task from its
training data, report it on the task's eval split, and see which domains it
resembles (configs/steering/task_vectors.yaml; README "Encoder Analysis &
Steering").

  python scripts/steering/run_task_vectors.py --run url --task arc_easy
  python scripts/steering/run_task_vectors.py --run meco --task all      # joint over every task
  python scripts/steering/run_task_vectors.py --run baseline --task piqa

--run is a key of configs/comparison.yaml (evaluated at the shared
checkpoint). Output: outputs/experiments/<size>/steering/<ckpt label>/task_vectors/<run>/<task>.json
(numbers, rankings, histories, nearest domains) and <task>.pt (vectors and
per-doc eval correctness, for paired comparisons). Figures and tables:
scripts/steering/make_steering_report.py.

Reported select/eval accuracies all come from the exact lm-eval scorer
(src/steering/scoring.py's score_docs), including the optimizers' early
stopping; the bf16 differentiable path only computes the
optimizers' training batches.
"""

import argparse
import json
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

import torch

from src.benchmarks.conditioning import resolve_meco, resolve_none, resolve_url_text
from src.config import load_config
from src.steering.optimize import domain_search, fit_vector
from src.steering.scoring import PreparedDocs, make_lm, score_docs
from src.steering.task_data import build_splits, load_tasks
from src.steering.vectors import load_compared_run, text_spec
from src.url_analysis.analyses import neighbors
from src.url_analysis.encode import load_library, library_vectors
from src.utils import paths


def log(msg):
    print(msg, flush=True)


class Experiment:
    def __init__(self, R, splits, cfg, out_json, out_pt):
        self.R, self.splits, self.cfg = R, splits, cfg
        self.out_json, self.out_pt = out_json, out_pt
        self.result = {"run": R["run_key"], "run_dir": R["run_dir"], "checkpoint": R["ckpt_info"],
                       "ckpt_label": R["ckpt_label"], "tasks": list(splits),
                       "splits": {t: s.info for t, s in splits.items()},
                       "metric": {t: s.metric for t, s in splits.items()}, "methods": {}}
        self.side = {"vectors": {}, "per_doc": {}}

    def lm(self, spec):
        return make_lm(self.R["model"], self.R["tokenizer"], self.R["cfg"], spec, self.cfg["eval_batch_size"])

    def select_acc(self, spec):
        lm = self.lm(spec)
        return sum(score_docs(lm, s, s.select)["acc"] for s in self.splits.values()) / len(self.splits)

    def evaluate(self, name, spec, z=None, **extra):
        t0 = time.time()
        lm = self.lm(spec)
        per_task, per_doc = {}, {}
        for t, s in self.splits.items():
            sel = score_docs(lm, s, s.select)
            ev = score_docs(lm, s, s.eval)
            per_task[t] = {"select": sel["acc"], "eval": ev["acc"], "eval_stderr": ev["stderr"], "n_eval": ev["n"]}
            per_doc[t] = ev["per_doc"]
        mean = lambda k: sum(v[k] for v in per_task.values()) / len(per_task)
        self.result["methods"][name] = {"tasks": per_task, "mean_select": mean("select"), "mean_eval": mean("eval"),
                                        "conditioning": spec.meta, **extra}
        self.side["per_doc"][name] = per_doc
        if z is not None:
            self.side["vectors"][name] = z.detach().float().cpu()
        log(f"  {name:28s} select {mean('select'):.4f}  eval {mean('eval'):.4f}  ({time.time() - t0:.0f}s)")
        self.save()

    def save(self):
        os.makedirs(os.path.dirname(self.out_json), exist_ok=True)
        with open(self.out_json + ".tmp", "w") as f:
            json.dump(self.result, f, indent=1, default=float)
        os.replace(self.out_json + ".tmp", self.out_json)
        torch.save(self.side, self.out_pt)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", required=True, help="configs/comparison.yaml key: baseline | meco | url")
    p.add_argument("--task", required=True, help="An lm-eval task from the suite, or 'all'.")
    p.add_argument("--config", default=os.path.join(ROOT, "configs/steering/task_vectors.yaml"))
    p.add_argument("--comparison-config", default=os.path.join(ROOT, "configs/comparison.yaml"))
    p.add_argument("--tasks-config", default=os.path.join(ROOT, "configs/benchmarks/tasks.yaml"))
    p.add_argument("--sources-config", default=os.path.join(ROOT, "configs/benchmarks/sources.yaml"))
    p.add_argument("--checkpoint", default="shared")
    p.add_argument("--methods", nargs="+", default=None, help="Restrict to these methods (default: all applicable).")
    p.add_argument("--smoke", action="store_true",
                   help="Tiny splits/steps/candidates (results under '<label>-smoke').")
    args = p.parse_args()

    cfg = load_config(args.config)
    sources_cfg = load_config(args.sources_config)
    task_names = cfg.get("tasks") or load_config(args.tasks_config)["tasks"]
    task_names = task_names if args.task == "all" else [args.task]
    sp = cfg["splits"]
    if args.smoke:
        sp = {**sp, "n_fit": 64, "n_select": 48}
        cfg["domain_search"]["n_candidates"] = 4
        cfg["simplex"].update(steps=4, eval_every=2, n_library=16)
        cfg["free"].update(steps=4, eval_every=2)

    R = load_compared_run(ROOT, args.comparison_config, args.run, args.checkpoint)
    variant = R["cfg"]["variant"]
    label = R["ckpt_label"] + ("-smoke" if args.smoke else "")
    out_base = paths.experiments_dir(R["size"], "steering", label, "task_vectors", R["run_key"], args.task)
    tasks = load_tasks(task_names)
    splits = {t: build_splits(t, tasks[t], sp["n_fit"], sp["n_select"], sp["seed"], sp.get("holdout_frac", 0.3),
                              n_eval=100 if args.smoke else None) for t in task_names}
    for t, s in splits.items():
        log(f"{t}: metric {s.metric}  " + "  ".join(f"{k}={v}" for k, v in s.info.items()))

    ex = Experiment(R, splits, cfg, out_base + ".json", out_base + ".pt")
    want = lambda m: args.methods is None or m in args.methods or m.split(":")[0] in args.methods
    lib = None
    if variant in ("meco", "text"):
        lib = load_library(paths.experiments_dir(R["size"], "url_analysis", R["ckpt_label"], "library.pt"))

    if want("unconditioned"):
        ex.evaluate("unconditioned", resolve_none(), z=R["model"].conditioner.null_vector if variant == "text" else None)

    def url_spec(domain=None, source=None):
        return resolve_url_text(R["model"].conditioner, R["tokenizer"], R["encoder"].source, R["device"],
                                source=source, domain=domain, sources_cfg=sources_cfg)

    if variant in ("meco", "text"):
        for s in sources_cfg["sources"]:
            if want(f"source:{s}"):
                spec = resolve_meco(source=s, sources_cfg=sources_cfg) if variant == "meco" else url_spec(source=s)
                ex.evaluate(f"source:{s}", spec, z=spec.z)

    best_domain_z = None
    if variant in ("meco", "text") and want("domain_search"):
        n = cfg["domain_search"]["n_candidates"]
        names = [sources_cfg["sources"][s]["meco_domain"] for s in sources_cfg["sources"]]
        names += [d for d in lib["domains"][:n] if d not in names]
        pos = {d: i for i, d in enumerate(lib["domains"])}
        if variant == "meco":
            cands = [(d, resolve_meco(domain=d)) for d in names]
        else:
            # library root vectors = E(root URL) for library domains (same
            # model/checkpoint as the analysis); named sources encode like --source.
            cands = [(d, text_spec(f"text_domain-{d}", lib["root"][pos[d]], {"domain": d, "input": lib["root_inputs"][pos[d]]})
                      if d in pos else url_spec(domain=f"https://{d}/")) for d in names]
        log(f"domain search over {len(cands)} candidates")
        ranking, best = domain_search(ex.select_acc, cands, log)
        best_domain_z = best.z
        ex.evaluate("domain_search", best, z=best.z, chosen=ranking[0]["name"], ranking=ranking)

    if variant == "text":
        model, pad_id, device = R["model"], R["tokenizer"].pad_token_id, R["device"]
        for prm in model.parameters():
            prm.requires_grad_(False)
        builder = ex.lm(resolve_none())
        preps = [(PreparedDocs(builder, s, s.fit), PreparedDocs(builder, s, s.select)) for s in splits.values()]
        k = cfg["nearest_k"]
        select_fn = lambda z: ex.select_acc(text_spec("candidate", z, {}))

        if want("opt_simplex"):
            sc = cfg["simplex"]
            L = library_vectors(lib, sc["vector"])[:sc["n_library"]]
            L = torch.cat([L, lib["null"][None]])
            lib_names = lib["domains"][:sc["n_library"]] + ["(null)"]
            log(f"simplex over {len(L)} library vectors ({sc['vector']})")
            z, hist, extra = fit_vector(model, preps, "simplex", pad_id, device, library=L, steps=sc["steps"],
                                        batch_docs=sc["batch_docs"], lr=sc["lr"], eval_every=sc["eval_every"],
                                        select_fn=select_fn, log=log)
            w = extra["weights"]
            top = [{"domain": lib_names[i], "weight": float(w[i])} for i in torch.argsort(w, descending=True)[:k]]
            ex.evaluate("opt_simplex", text_spec("opt_simplex", z, {"method": "opt_simplex"}), z=z,
                        history=hist, top_weights=top, weight_entropy=float(-(w * (w + 1e-12).log()).sum()),
                        nearest=neighbors(z[None], ["z"], lib, k)["z"], **{kk: v for kk, v in extra.items() if kk != "weights"})

        if want("opt_free"):
            fc = cfg["free"]
            init = {"domain_search": best_domain_z, "mean": lib["mean"], "null": lib["null"]}[fc["init"]]
            if init is None:
                raise SystemExit("free.init: domain_search needs the domain_search method in this invocation")
            init = init.detach().float().cpu()
            max_norm = fc["max_norm_mult"] * float(lib["centroid"].norm(dim=-1).max())
            log(f"free vector from {fc['init']} init (||z0|| {float(init.norm()):.1f}, cap {max_norm:.1f})")
            z, hist, extra = fit_vector(model, preps, "free", pad_id, device, init=init, max_norm=max_norm,
                                        steps=fc["steps"], batch_docs=fc["batch_docs"], lr=fc["lr"],
                                        eval_every=fc["eval_every"], l2=fc.get("l2", 0.0), select_fn=select_fn,
                                        log=log)
            ex.evaluate("opt_free", text_spec("opt_free", z, {"method": "opt_free", "init": fc["init"]}), z=z,
                        history=hist, init=fc["init"], norm=float(z.norm()),
                        cos_to_init_centered=float(torch.nn.functional.cosine_similarity(
                            z - lib["mean"], init - lib["mean"], dim=0)),
                        nearest=neighbors(z[None], ["z"], lib, k)["z"],
                        nearest_root=neighbors(z[None], ["z"], lib, k, "root")["z"], **extra)

    log(f"wrote {ex.out_json}")


if __name__ == "__main__":
    main()
