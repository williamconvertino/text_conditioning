"""Benchmark comparison tables (scripts/make_table.py): the lm-eval-harness
counterpart to src/analysis/tables.py's final_conditioning table, for saved
outputs/experiments/<size>/benchmarks/<run>/<tag>/results.json files (src/benchmarks/results.py)
rather than final_conditioning.json. One row per (run, conditioning_tag)
result actually produced by scripts/benchmarks/run_benchmarks.py -- unlike the
final_conditioning table, there's no fixed set of modes to expect per variant,
since a run can have any number of benchmarked conditioning settings (several
sources, ad hoc domains/URLs, ...).

Reuses src/analysis/tables.py's csv/markdown/latex writers (same presentation,
different header/rows) rather than duplicating them.
"""

import os

from src.analysis.tables import write_csv, write_markdown, write_latex
from src.benchmarks.results import best_metric_with_stderr
from src.utils.paths import resolve_run_dir

MISSING = "—"


def _all_tasks(records):
    tasks = []
    for rec in records:
        for t in rec["results"]:
            if t not in tasks:
                tasks.append(t)
    return tasks


def _fmt_cell(cell, pm):
    """cell: (value, stderr) in [0, 1] (stderr may be None), or None for a
    missing result. Renders as percentage accuracy, e.g. 0.4050 -> "40.5",
    plus the stderr as an error bar when present, e.g. (0.4050, 0.0021) ->
    "40.5 ± 0.2". `pm` is the literal "plus/minus" text to join with -- plain
    "±" for csv/markdown, LaTeX math-mode "$\\pm$" for the .tex table."""
    if cell is None:
        return MISSING
    value, stderr = cell
    s = f"{value * 100:.1f}"
    if stderr is not None:
        s += f" {pm} {stderr * 100:.1f}"
    return s


def build_benchmark_table_rows(records, tasks=None, prefer=("acc_norm,none", "acc,none")):
    """records: src/benchmarks/results.py's discover_results() output ->
    (tasks, rows). rows: one per (run, conditioning_tag) result, as
    [run_label, conditioning_tag, *task_cells, mean_cell] where each cell is
    (value, stderr) in [0, 1] (stderr None if not reported) or None for a
    missing task result -- formatting to percentage-with-error-bar text
    happens in _fmt_cell, since the two rendered tables (csv/markdown vs.
    latex) need different plus/minus notation. The Mean cell's stderr is
    always None: combining per-task stderrs into one macro-average stderr
    would assume independence that isn't actually known, so it's left
    unreported rather than invented (README "use -- rather than inventing an
    equivalent measurement") -- only its accuracy is averaged. A missing task
    result is excluded from that row's mean. run_label comes from
    src/analysis/figures.py's display_name when the run's resolved_config.yaml
    is available, else the run dir's basename."""
    from src.analysis.curves import load_run_config  # deferred: see
    from src.analysis.figures import display_name     # write_all_tables's note

    if tasks is None:
        tasks = _all_tasks(records)

    rows = []
    for rec in records:
        cfg = load_run_config(resolve_run_dir(rec["run_dir"]))
        label = display_name(rec["run_name"], cfg)
        cells = []
        for t in tasks:
            task_results = rec["results"].get(t)
            if task_results is None:
                cells.append(None)
                continue
            _, value, stderr = best_metric_with_stderr(task_results, prefer=prefer)
            cells.append(None if value is None else (value, stderr))
        present = [c[0] for c in cells if c is not None]
        mean_cell = (sum(present) / len(present), None) if present else None
        rows.append([label, rec["conditioning_tag"], *cells, mean_cell])
    return tasks, rows


def write_all_benchmark_tables(records, out_dir, tasks=None, prefer=("acc_norm,none", "acc,none")):
    """Build & write the benchmark_results csv/md/tex tables (one row per
    (run, conditioning_tag) result) for the given records. Shared by
    scripts/make_table.py; records normally come from
    src/benchmarks/results.py's discover_results(). Returns (tasks, rows) with
    rows in build_benchmark_table_rows's raw (value, stderr) cell form (not
    the formatted strings actually written), or (None, []) if `records` is
    empty -- callers should skip writing rather than emit an empty table."""
    if not records:
        return None, []

    tasks, rows = build_benchmark_table_rows(records, tasks=tasks, prefer=prefer)
    plain_rows = [[r[0], r[1]] + [_fmt_cell(c, "±") for c in r[2:]] for r in rows]
    latex_rows = [[r[0], r[1]] + [_fmt_cell(c, r"$\pm$") for c in r[2:]] for r in rows]

    header = ["Run", "Conditioning"] + tasks + ["Mean"]
    align_right = [False, False] + [True] * (len(tasks) + 1)

    write_csv(header, plain_rows, os.path.join(out_dir, "benchmark_results.csv"))
    write_markdown(["Run", "Conditioning"] + [f"{t} (%) ↑" for t in tasks] + ["Mean (%) ↑"], plain_rows,
                   os.path.join(out_dir, "benchmark_results.md"), align_right=align_right)
    write_latex(header, latex_rows, os.path.join(out_dir, "benchmark_results.tex"))
    return tasks, rows
