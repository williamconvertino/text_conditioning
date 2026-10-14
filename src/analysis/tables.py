"""Final conditioning tables (README "Final Evaluation Table"): builds the
Model x {No Cond., Self, Same-Domain, Domain-Mean} loss comparison from saved
per-model results (src/eval/conditioning.py output, one JSON file per model --
see scripts/evaluate.py), in csv / markdown / latex, plus a long-format
per-domain breakdown CSV (src/eval/conditioning.py's "top_domains.per_domain").

Every mode is reported at two scopes where that distinction is meaningful
(src/eval/conditioning.py): "No Cond." and "Self" each get a "(Full)" column
(the fixed, domain-independent validation sample) and a "(Top-N)" column (the
same restricted set of documents same_domain/domain_mean are scored on, so
every mode in a row is comparable). same_domain/domain_mean only ever have a
"(Top-N)" value -- they're inherently domain-conditional.
"""

import csv
import json
import os

COLUMNS = [
    ("none", "overall", "No Cond. Loss (Full)"),
    ("none", "top_domains", "No Cond. Loss (Top-N)"),
    ("self", "overall", "Self Loss (Full)"),
    ("self", "top_domains", "Self Loss (Top-N)"),
    ("same_domain", "top_domains", "Same-Domain Loss (Top-N)"),
    ("domain_mean", "top_domains", "Domain-Mean Loss (Top-N)"),
]

DOMAIN_MODE_LABELS = [("none", "No Cond. PPL"), ("self", "Self PPL"),
                      ("same_domain", "Same-Domain PPL"), ("domain_mean", "Domain-Mean PPL"),
                      ("cross_domain", "Cross-Domain PPL")]


def _tupleize_modes(modes_dict):
    return {k: (tuple(v) if v is not None else None) for k, v in modes_dict.items()}


def load_final_results(path):
    """Load one run's src/eval/conditioning.py evaluate_final() output,
    restoring the (loss, ppl) tuples json.dump flattened to lists."""
    raw = json.load(open(path))
    out = {"overall": _tupleize_modes(raw["overall"])}
    td = raw.get("top_domains")
    if td is None:
        out["top_domains"] = None
    else:
        out["top_domains"] = {
            "n_domains": td["n_domains"], "k": td["k"], "min_docs_per_domain": td["min_docs_per_domain"],
            "domains": td["domains"],
            "aggregate": _tupleize_modes(td["aggregate"]),
            "per_domain": {domain: _tupleize_modes(modes) for domain, modes in td["per_domain"].items()},
        }
    return out


def _cell_value(results, mode, scope):
    if scope == "overall":
        return results["overall"].get(mode)
    top_domains = results.get("top_domains")
    return None if top_domains is None else top_domains["aggregate"].get(mode)


def build_table_rows(model_results):
    """model_results: {display_name: results} (results = load_final_results()
    output) -> list of rows [display_name, *loss_or_dash]. A missing/None
    mode/scope becomes "--" rather than an invented measurement (README)."""
    rows = []
    for name, results in model_results.items():
        row = [name]
        for mode, scope, _ in COLUMNS:
            val = _cell_value(results, mode, scope)
            row.append(f"{val[0]:.4f}" if val else "—")
        rows.append(row)
    return rows


def build_domain_rows(model_results):
    """model_results (as above) -> long-format rows, one per (model, domain,
    mode) with a result: {model, domain, count, n_support, n_eval, mode,
    metric, loss, ppl}. Easy to filter/pivot for per-domain analysis (README
    "output results for each domain considered"). Models with no top_domains
    data (e.g. no domain had enough documents) contribute no rows."""
    rows = []
    for name, results in model_results.items():
        top_domains = results.get("top_domains")
        if top_domains is None:
            continue
        domain_meta = {d["domain"]: d for d in top_domains["domains"]}
        for domain, modes in top_domains["per_domain"].items():
            meta = domain_meta.get(domain, {})
            for mode, label in DOMAIN_MODE_LABELS:
                val = modes.get(mode)
                if val is None:
                    continue
                rows.append({
                    "model": name, "domain": domain, "count": meta.get("count"),
                    "n_support": meta.get("n_support"), "n_eval": meta.get("n_eval"),
                    "mode": mode, "metric": label, "loss": val[0], "ppl": val[1],
                })
    return rows


def write_csv(header, rows, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


def write_markdown(header, rows, path, align_right=None):
    """Column-padded markdown table (README "Final Evaluation Table" -- first
    column left-aligned, the rest right-aligned with a ---: separator, all
    cells padded to line up when viewed as raw text). align_right defaults to
    every column but the first (e.g. a leading "Model"/"Run" column)."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if align_right is None:
        align_right = [False] + [True] * (len(header) - 1)
    all_rows = [header] + rows
    widths = [max(len(r[i]) for r in all_rows) for i in range(len(header))]

    def fmt_row(row):
        cells = [cell.rjust(w) if right else cell.ljust(w)
                 for cell, w, right in zip(row, widths, align_right)]
        return "| " + " | ".join(cells) + " |"

    sep = "| " + " | ".join(
        ("-" * (w - 1) + ":") if right else "-" * w
        for w, right in zip(widths, align_right)
    ) + " |"

    lines = [fmt_row(header), sep] + [fmt_row(row) for row in rows]
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


def write_latex(header, rows, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    lines = [r"\begin{tabular}{l" + "r" * (len(header) - 1) + "}", r"\toprule",
             " & ".join(header) + r" \\", r"\midrule"]
    for row in rows:
        lines.append(" & ".join(row) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


def write_all_tables(run_dirs, out_dir):
    """Build & write every table (final_conditioning csv/md/tex, plus the
    per-domain breakdown csv when any run has top_domains data) for the given
    run dirs' final_conditioning.json files. Shared by scripts/evaluate.py
    --table and scripts/make_table.py, which differ only in how they arrive
    at `run_dirs` (an explicit list vs. auto-discovery under outputs/)."""
    from src.utils.paths import final_conditioning_path
    from src.analysis.figures import display_name  # local: tables.py is the
    # lower-level module (figures.py already imports from it); deferred here
    # only to avoid a module-load-order cycle, not a real layering violation.

    model_results = {}
    for run_dir in run_dirs:
        path = final_conditioning_path(run_dir)
        if not os.path.exists(path):
            raise FileNotFoundError(f"{path} missing -- run `scripts/evaluate.py --run-dir {run_dir} "
                                     f"--mode final` first")
        model_results[display_name(os.path.basename(run_dir.rstrip("/")))] = load_final_results(path)

    rows = build_table_rows(model_results)
    header = ["Model"] + [c for _, _, c in COLUMNS]
    write_csv(header, rows, os.path.join(out_dir, "final_conditioning.csv"))
    write_markdown(["Model"] + [c + " ↓" for _, _, c in COLUMNS], rows,
                   os.path.join(out_dir, "final_conditioning.md"))
    write_latex(header, rows, os.path.join(out_dir, "final_conditioning.tex"))

    domain_rows = build_domain_rows(model_results)
    if domain_rows:
        write_domain_csv(domain_rows, os.path.join(out_dir, "domain_breakdown.csv"))
    return rows, domain_rows


def write_domain_csv(rows, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    header = ["model", "domain", "count", "n_support", "n_eval", "mode", "metric", "loss", "ppl"]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=header)
        w.writeheader()
        w.writerows(rows)
