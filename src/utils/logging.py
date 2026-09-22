"""JSONL/CSV metric logging shared across training and evaluation.

Every experiment writes a flat metrics.jsonl (one JSON record per logged event)
and a derived metrics.csv, matching the README's "Training Curves" /
"Presentation-Ready Outputs" requirements. Downstream analysis (src/analysis)
reads these files rather than checkpoints, so figures can be regenerated without
rerunning training or evaluation.
"""

import csv
import json
import os


class MetricLogger:
    def __init__(self, run_dir):
        self.run_dir = run_dir
        os.makedirs(run_dir, exist_ok=True)
        self.jsonl_path = os.path.join(run_dir, "metrics.jsonl")
        self.csv_path = os.path.join(run_dir, "metrics.csv")

    def log(self, record):
        with open(self.jsonl_path, "a") as f:
            f.write(json.dumps(record) + "\n")

    def export_csv(self):
        """Rewrite metrics.csv from metrics.jsonl (union of all keys seen)."""
        if not os.path.exists(self.jsonl_path):
            return
        rows = load_jsonl(self.jsonl_path)
        if not rows:
            return
        fields = []
        for row in rows:
            for k in row:
                if k not in fields:
                    fields.append(k)
        tmp = self.csv_path + ".tmp"
        with open(tmp, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            for row in rows:
                w.writerow(row)
        os.replace(tmp, self.csv_path)


def load_jsonl(path):
    return [json.loads(line) for line in open(path) if line.strip()]
