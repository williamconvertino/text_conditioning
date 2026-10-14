#!/usr/bin/env python
"""Latent-space analysis of the URL-encoded run's encoder
(configs/url_analysis/encoder_analysis.yaml; README "Encoder Analysis & Steering").

Steps (default: all):
  embed    scan training-shard URLs (src/url_analysis/corpus.py), encode the
           domain library + in-domain page groups (src/url_analysis/encode.py)
           -> library.pt, groups.pt, corpus.pt
  analyze  every analysis in src/url_analysis/analyses.py -> results/*.json:
             stats          norms, anisotropy, null vector, root vs. centroid
             spectrum       PCA spectra / effective dimensionality
             neighbors      nearest domains for queries, top domains, extra inputs
             similarity     centered-cosine matrix among the query domains
             domain_map     t-SNE / PCA / KMeans of the top domains
             retrieval      does a page vector retrieve its own domain?
             variance       share of page-vector variance explained by domain
             in_domain      structure within wikipedia / reddit / nytimes / ...
             year_probe     linear probe for the year in the URL path
             perturbations  sensitivity to surface edits of a URL

Output: outputs/experiments/<size>/url_analysis/<ckpt label>/ (the label of the shared
checkpoint, e.g. step20000, or of --checkpoint). Figures/tables: scripts/url_analysis/plot_encoder_analysis.py.

  python scripts/url_analysis/run_encoder_analysis.py
  python scripts/url_analysis/run_encoder_analysis.py --step analyze
  python scripts/url_analysis/run_encoder_analysis.py --checkpoint ckpt_last.pt   # a specific checkpoint instead
"""

import argparse
import json
import os
import random
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

import torch

from src.comparison.checkpoints import shared_checkpoint
from src.config import load_config
from src.data.shards import split_shards
from src.url_analysis import analyses as A
from src.url_analysis.corpus import lookup_count, pick_stems, scan_corpus
from src.url_analysis.encode import build_library, load_library, load_url_encoder, save_library
from src.utils import paths


def resolve_target(args, cmp_cfg, ana_cfg):
    """(run_dir, checkpoint relative to run_dir, label, provenance)."""
    rel_run_dir = cmp_cfg["runs"][ana_cfg["run"]]
    if args.checkpoint == "shared":
        pin = shared_checkpoint(cmp_cfg, ROOT)
        info = pin["runs"][ana_cfg["run"]]
        checkpoint, label, extra = info["path"], pin["label"], {"step": info["step"], "tokens": info["tokens"]}
    else:
        checkpoint, extra = args.checkpoint, {}
        label = os.path.splitext(os.path.basename(checkpoint))[0]
    provenance = {"run_dir": rel_run_dir, "path": checkpoint, "label": label, **extra}
    return os.path.join(ROOT, rel_run_dir), checkpoint, label, provenance


def save_json(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=1, default=float)


def step_embed(encoder, run_cfg, cfg, out_dir, provenance):
    t0 = time.time()
    tok_dir = os.path.join(ROOT, run_cfg["tok_dir"])   # no-op for an absolute tok_dir
    train_stems, _ = split_shards(tok_dir, run_cfg["n_val_shards"])
    stems = pick_stems(train_stems, cfg["corpus"]["n_shards"])
    groups = {g: spec["domains"] for g, spec in cfg["in_domain"]["groups"].items()}
    corpus = scan_corpus(tok_dir, stems, cfg["library"]["n_domains"], cfg["library"]["pages_per_domain"],
                         groups, cfg["in_domain"]["pages"], cfg["corpus"]["seed"])
    print(f"scanned {corpus['n_urls']:,} URLs over {len(stems)} shards, {corpus['n_domains_seen']:,} domains "
          f"({time.time() - t0:.0f}s)", flush=True)
    torch.save(corpus, os.path.join(out_dir, "corpus.pt"))

    lib = build_library(encoder, corpus)
    save_library(lib, os.path.join(out_dir, "library.pt"), provenance)
    print(f"library: {len(lib['domains'])} domains, {len(lib['page_urls']):,} pages "
          f"({time.time() - t0:.0f}s)", flush=True)

    group_vecs = {g: {"urls": urls, "vecs": encoder.encode_urls(urls)} for g, urls in corpus["groups"].items()}
    torch.save(group_vecs, os.path.join(out_dir, "groups.pt"))
    print("groups: " + ", ".join(f"{g}={len(v['urls'])}" for g, v in group_vecs.items()), flush=True)


def step_analyze(encoder, cfg, out_dir):
    lib = load_library(os.path.join(out_dir, "library.pt"))
    corpus = torch.load(os.path.join(out_dir, "corpus.pt"), weights_only=False)
    groups = torch.load(os.path.join(out_dir, "groups.pt"), weights_only=False)
    res_dir = os.path.join(out_dir, "results")
    seed = cfg.get("seed", 0)

    def done(name, obj, t0):
        save_json(obj, os.path.join(res_dir, f"{name}.json"))
        print(f"  {name:14s} {time.time() - t0:6.1f}s", flush=True)

    t = time.time(); done("stats", A.vector_stats(lib), t)
    t = time.time()
    done("spectrum", {"domain centroids": A.pca_spectrum(lib["centroid"]),
                      "domain roots": A.pca_spectrum(lib["root"]),
                      "pages": A.pca_spectrum(lib["page_vecs"][torch.randperm(len(lib["page_vecs"]))[:20000]])}, t)

    t = time.time()
    ncfg = cfg["neighbors"]
    pos = {d: i for i, d in enumerate(lib["domains"])}
    queries = [q for q in ncfg["queries"] if q in pos]
    missing = [q for q in ncfg["queries"] if q not in pos]
    top = [d for d in lib["domains"][:ncfg["n_top_domains"]] if d not in queries]
    names = queries + top
    nbr = {kind: A.neighbors(lib[kind][[pos[n] for n in names]], names, lib, ncfg["k"], kind)
           for kind in ("centroid", "root")}
    extra = ncfg.get("extra_inputs") or {}
    extra_vecs = encoder.encode([encoder.metadata(u) if "://" in u else u for u in extra.values()]) \
        if extra else torch.zeros(0)
    extra_nbr = A.neighbors(extra_vecs, list(extra), lib, ncfg["k"], "centroid") if extra else {}
    done("neighbors", {"queries": queries, "missing_queries": missing, "top": top, "by_kind": nbr,
                       "extra": extra_nbr, "extra_inputs": extra,
                       "extra_counts": {k: lookup_count(corpus, u) for k, u in extra.items()},
                       "counts": {d: int(lib["counts"][pos[d]]) for d in names}}, t)

    t = time.time()
    labels, sims = A.similarity_matrix(queries, lib, "centroid",
                                       extra={f"[{k}]": v for k, v in zip(extra, extra_vecs)})
    done("similarity", {"labels": labels, "sims": sims}, t)

    t = time.time()
    mcfg = cfg["map"]
    done("domain_map", A.domain_map(lib, mcfg["n_domains"], mcfg["n_clusters"], mcfg["categories"], seed,
                                    mcfg["tsne_perplexity"]), t)

    t = time.time(); done("retrieval", A.domain_retrieval(lib, seed=seed), t)
    t = time.time(); done("variance", A.variance_by_domain(lib["page_vecs"], lib["page_domain"]), t)

    t = time.time()
    icfg = cfg["in_domain"]
    ind = {}
    for g, spec in icfg["groups"].items():
        if len(groups[g]["urls"]) < 50:
            print(f"    skipping in-domain group {g}: only {len(groups[g]['urls'])} pages", flush=True)
            continue
        ind[g] = A.in_domain(groups[g]["urls"], groups[g]["vecs"], spec.get("rules", []),
                             icfg["n_clusters"], top_labels=icfg.get("top_labels", 10), seed=seed)
    done("in_domain", ind, t)

    t = time.time(); done("year_probe", A.year_probe(lib["page_urls"], lib["page_vecs"], seed), t)

    t = time.time()
    rng = random.Random(seed)
    urls = rng.sample(lib["page_urls"], min(cfg["perturbations"]["n_urls"], len(lib["page_urls"])))
    done("perturbations", A.perturbation_study(encoder, urls, lib), t)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default=os.path.join(ROOT, "configs/url_analysis/encoder_analysis.yaml"))
    p.add_argument("--comparison-config", default=os.path.join(ROOT, "configs/comparison.yaml"))
    p.add_argument("--checkpoint", default="shared",
                   help="'shared' (default) or a checkpoint path relative to the run dir.")
    p.add_argument("--step", choices=["embed", "analyze", "all"], default="all")
    p.add_argument("--out-dir", default=None,
                   help="Default: outputs/experiments/<size>/url_analysis/<ckpt label>")
    args = p.parse_args()

    cfg = load_config(args.config)
    cmp_cfg = load_config(args.comparison_config)
    run_dir, checkpoint, label, provenance = resolve_target(args, cmp_cfg, cfg)
    out_dir = args.out_dir or paths.experiments_dir(paths.model_size(run_dir), "url_analysis", label)
    os.makedirs(out_dir, exist_ok=True)
    print(f"encoder: {run_dir} @ {checkpoint} -> {out_dir}", flush=True)

    encoder, _, run_cfg = load_url_encoder(run_dir, checkpoint)
    save_json({**provenance, "field": encoder.field}, os.path.join(out_dir, "provenance.json"))
    if args.step in ("embed", "all"):
        step_embed(encoder, run_cfg, cfg, out_dir, provenance)
    if args.step in ("analyze", "all"):
        step_analyze(encoder, cfg, out_dir)


if __name__ == "__main__":
    main()
