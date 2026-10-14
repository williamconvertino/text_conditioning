#!/usr/bin/env python
"""Figures + tables for the URL-encoder analysis
(scripts/url_analysis/run_encoder_analysis.py's results/*.json).

Reads outputs/experiments/<size>/url_analysis/<ckpt label>/results/ and writes
to outputs/results/<size>/url_analysis/<ckpt label>/{figures,tables} (or
--out-dir):
  figures/domain_map_{tsne,pca}_{category,cluster}   global maps of the top domains
  figures/query_similarity                           centered-cosine heatmap of the query domains
  figures/pca_spectrum, vector_stats                 dimensionality, norms, root vs. centroid
  figures/domain_retrieval, url_perturbations, year_probe
  figures/in_domain_<group>                          t-SNE of one domain's pages
  tables/neighbors_{centroid,root}, neighbors_extra  nearest domains (md + csv)
  tables/clusters, in_domain_summary, in_domain_cluster_exemplars, summary

  python scripts/url_analysis/plot_encoder_analysis.py                      # shared-checkpoint label
  python scripts/url_analysis/plot_encoder_analysis.py --comparison-config configs/comparison_3b.yaml
"""

import argparse
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

from src.comparison.checkpoints import shared_checkpoint
from src.config import load_config
from src.url_analysis import plots as P
from src.utils import paths


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--analysis-dir", default=None,
                   help="Default: outputs/experiments/<size>/url_analysis/<shared ckpt label>")
    p.add_argument("--out-dir", default=None,
                   help="Default: outputs/results/<size>/url_analysis/<shared ckpt label>")
    p.add_argument("--config", default=os.path.join(ROOT, "configs/url_analysis/encoder_analysis.yaml"))
    p.add_argument("--comparison-config", default=os.path.join(ROOT, "configs/comparison.yaml"))
    args = p.parse_args()

    cfg = load_config(args.config)
    if not (args.analysis_dir and args.out_dir):
        cmp_cfg = load_config(args.comparison_config)
        size, label = paths.comparison_size(cmp_cfg), shared_checkpoint(cmp_cfg, ROOT)["label"]
    d = args.analysis_dir or paths.experiments_dir(size, "url_analysis", label)
    out = args.out_dir or paths.results_dir(size, "url_analysis", label)
    res = lambda name: json.load(open(os.path.join(d, "results", f"{name}.json")))
    fig_dir, tab_dir = os.path.join(out, "figures"), os.path.join(out, "tables")

    nb = res("neighbors")
    dm = res("domain_map")
    highlight = nb["queries"]
    P.plot_domain_map(dm, fig_dir, cfg["map"]["n_label"], highlight, "tsne")
    P.plot_domain_map(dm, fig_dir, cfg["map"]["n_label"], highlight, "pca")
    P.write_cluster_table(dm, tab_dir)

    for kind, nbrs in nb["by_kind"].items():
        P.write_neighbor_tables(nbrs, tab_dir, f"neighbors_{kind}", counts=nb["counts"])
    if nb["extra"]:
        P.write_neighbor_tables({f"{k} ({nb['extra_inputs'][k]})": v for k, v in nb["extra"].items()},
                                tab_dir, "neighbors_extra",
                                counts={f"{k} ({nb['extra_inputs'][k]})": c for k, c in nb["extra_counts"].items()})
    sim = res("similarity")
    P.plot_similarity_heatmap(sim["labels"], sim["sims"], fig_dir)

    stats = res("stats")
    P.plot_stats(stats, fig_dir)
    P.plot_spectrum(res("spectrum"), fig_dir)
    ret, var = res("retrieval"), res("variance")
    P.plot_retrieval(ret, var, fig_dir)
    pert = res("perturbations")
    P.plot_perturbations(pert, fig_dir)
    probe = res("year_probe")
    P.plot_year_probe(probe, fig_dir)

    ind = res("in_domain")
    for name, r in ind.items():
        P.plot_in_domain(name, r, fig_dir)
    if ind:
        P.write_in_domain_tables(ind, tab_dir)

    spec = res("spectrum")
    summary = [
        ["Library domains / pages", f"{stats['n_domains']} / {stats['n_pages']:,}"],
        ["Median page-vector norm (null vector norm)",
         f"{stats['norm']['page']['p50']:.2f} ({stats['norm']['null']:.2f})"],
        ["Mean pairwise cos between pages, raw / centered",
         f"{stats['pairwise_cos_pages']['raw_mean']:.3f} / {stats['pairwise_cos_pages']['centered_mean']:.3f}"],
        ["Median centered cos(root, centroid)", f"{stats['root_vs_centroid_cos']['centered']['p50']:.3f}"],
        ["Participation ratio: centroids / pages",
         f"{spec['domain centroids']['participation_ratio']:.1f} / {spec['pages']['participation_ratio']:.1f}"],
        ["Page variance explained by domain (R²)", f"{var['r2_domain']:.3f}"],
        ["Page → own domain retrieval top-1 / top-10 (chance)",
         f"{ret['top1']:.3f} / {ret['top10']:.3f} ({ret['chance_top1']:.4f})"],
        ["Year probe R² (MAE vs mean-baseline MAE)",
         "—" if probe.get("skipped") else f"{probe['r2']:.3f} ({probe['mae']:.1f} vs {probe['mae_predict_mean']:.1f})"],
        ["Null vector's nearest domains", ", ".join(n["domain"] for n in stats["null_nearest_domains"][:6])],
    ]
    P.write_summary(summary, tab_dir)
    print(f"figures -> {fig_dir}\ntables  -> {tab_dir}")


if __name__ == "__main__":
    main()
