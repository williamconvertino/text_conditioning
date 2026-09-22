"""Downstream lm-evaluation-harness benchmarking (scripts/benchmarks/), kept
separate from src/eval/ (perplexity-based evaluation, used throughout
training) and src/training/ -- benchmarking is an occasional, expensive,
final-checkpoint-only evaluation axis, not something the training loop or the
perplexity eval path needs to know about. See src/benchmarks/lm_wrapper.py for
the lm-eval-harness adapter and src/benchmarks/conditioning.py for how a run's
conditioning is set for a benchmark pass.
"""
