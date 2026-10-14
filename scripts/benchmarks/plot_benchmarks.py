#!/usr/bin/env python
"""Compare benchmark results written by scripts/benchmarks/run_benchmarks.py
(src/benchmarks/results.py, src/benchmarks/plots.py).

Three subcommands:

  list              Show every discovered outputs/experiments/<size>/benchmarks/<run>/<tag>/
                     result (run/tag/tasks), so results already produced are
                     easy to find.

  types             Compare conditioning *types* against each other, e.g.
                     Baseline vs. MeCo vs. Text Conditioning, each pinned to
                     one (run, tag) result.

  sources           Compare *sources* for a fixed conditioning type, e.g.
                     Wikipedia vs. Reddit vs. News, each a (run, tag) result
                     for the same run.

`types` and `sources` are the same comparison mechanically (a labeled list of
results); they're kept as separate subcommands only because that's the two
comparisons actually asked for, and separate output stems keep filenames
self-describing. Each produces a per-task breakdown figure and a
macro-averaged summary figure.

Every --run is a "<run_dir>:<conditioning_tag>[:<label>]" triplet, e.g.
"outputs/checkpoints/meco_p0.9_medium:meco_wikipedia:MeCo (Wikipedia)". label defaults to the
tag if omitted.

Examples:
  python scripts/benchmarks/plot_benchmarks.py list

  python scripts/benchmarks/plot_benchmarks.py types \\
      --run outputs/checkpoints/baseline_medium:unconditioned:Baseline \\
      --run outputs/checkpoints/meco_medium:meco_wikipedia:MeCo \\
      --run outputs/checkpoints/text_conditioned_medium:text_wikipedia:"Text Conditioning" \\
      --stem conditioning_types_wikipedia

  python scripts/benchmarks/plot_benchmarks.py sources \\
      --run outputs/checkpoints/text_conditioned_medium:text_wikipedia:Wikipedia \\
      --run outputs/checkpoints/text_conditioned_medium:text_reddit:Reddit \\
      --run outputs/checkpoints/text_conditioned_medium:text_news:News \\
      --stem text_conditioned_sources
"""

import argparse
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

from src.benchmarks.results import discover_results, find_result
from src.benchmarks.plots import plot_aggregate, plot_task_breakdown
from src.utils import paths


def _parse_run_spec(spec):
    parts = spec.split(":", 2)
    if len(parts) < 2:
        raise SystemExit(f"--run {spec!r} must be '<run_dir>:<conditioning_tag>[:<label>]'")
    run_dir, tag = parts[0], parts[1]
    label = parts[2] if len(parts) == 3 else tag
    return run_dir, tag, label


def _labeled_records(run_specs):
    labeled = []
    for spec in run_specs:
        run_dir, tag, label = _parse_run_spec(spec)
        run_name = os.path.basename(run_dir.rstrip("/"))
        rec = find_result(run_name, tag)
        labeled.append((label, rec))
    return labeled


def _add_common_args(p):
    p.add_argument("--run", action="append", required=True, dest="runs",
                   help="'<run_dir>:<conditioning_tag>[:<label>]', repeatable.")
    p.add_argument("--out-dir", default=None,
                   help="Default: outputs/results/<size of the first --run>/benchmarks/figures/.")
    p.add_argument("--stem", required=True, help="Output filename stem (no extension).")
    p.add_argument("--title", default=None)
    p.add_argument("--tasks", nargs="+", default=None, help="Restrict to these tasks (default: all present).")
    p.add_argument("--metric", default=None,
                   help="Exact lm-eval metric key (e.g. acc_norm,none). Default: "
                        "acc_norm,none if present, else acc,none, else first available.")


def cmd_list(args):
    records = discover_results()
    if not records:
        print(f"no benchmark results found under {paths.EXPERIMENTS_DIR} "
             f"(run scripts/benchmarks/run_benchmarks.py first)")
        return
    for rec in records:
        tasks = ", ".join(sorted(rec["results"].keys()))
        print(f"{rec['run_name']:28s} {rec['conditioning_tag']:24s} "
             f"[{rec['variant']}/{rec['conditioning_kind']}]  tasks: {tasks}")


def _run(args, default_title):
    labeled = _labeled_records(args.runs)
    prefer = (args.metric,) if args.metric else ("acc_norm,none", "acc,none")
    title = args.title or default_title
    args.out_dir = args.out_dir or paths.results_dir(paths.model_size(_parse_run_spec(args.runs[0])[0]),
                                                     "benchmarks", "figures")
    plot_task_breakdown(labeled, args.out_dir, f"{args.stem}_by_task", tasks=args.tasks,
                        title=title, prefer=prefer)
    plot_aggregate(labeled, args.out_dir, f"{args.stem}_aggregate", tasks=args.tasks,
                  title=title, prefer=prefer)
    print(f"wrote {os.path.join(args.out_dir, args.stem)}_by_task.png / _aggregate.png (+ .pdf)")


def cmd_types(args):
    _run(args, "Conditioning Type Comparison")


def cmd_sources(args):
    _run(args, "Source Comparison")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="command", required=True)

    sub.add_parser("list", help="List every discovered benchmark result.")

    p_types = sub.add_parser("types", help="Compare conditioning types against each other.")
    _add_common_args(p_types)

    p_sources = sub.add_parser("sources", help="Compare sources for one conditioning type.")
    _add_common_args(p_sources)

    return p.parse_args()


def main():
    args = parse_args()
    {"list": cmd_list, "types": cmd_types, "sources": cmd_sources}[args.command](args)


if __name__ == "__main__":
    main()
