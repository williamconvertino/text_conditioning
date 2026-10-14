"""lm-eval-harness task documents split for steering-vector search
(scripts/steering/run_task_vectors.py):

  fit     documents the steering vector is optimized on
  select  documents used for model selection (early stopping, choosing the
          best domain) -- never the reported split
  eval    the reported split: lm-eval's own evaluation docs (task.eval_docs,
          i.e. exactly what the benchmark table scores), unless the task has
          nothing else to fit on

Per task:
  * has a training split: fit = a sample of train. select = the validation
    split if lm-eval evaluates on test (arc, openbookqa), else a held-out
    sample of train disjoint from fit (hellaswag, piqa, winogrande, boolq
    evaluate on validation).
  * no training split (test only): fit/select come from a
    fixed `holdout_frac` of the eval docs and eval is the remainder
    (eval_is_subset: True). Every method is reported on that same remainder,
    so comparisons stay fair, but the number is not the full-test benchmark.
"""

import random

from lm_eval.tasks import get_task_dict

from src.benchmarks.tasks import task_manager


class Req:
    """Minimal stand-in for lm_eval.api.instance.Instance: CtrlTokensLM only
    reads .args."""
    __slots__ = ("args",)

    def __init__(self, args):
        self.args = args


def load_tasks(names):
    return get_task_dict(list(names), task_manager())


def headline_metric(task):
    """acc_norm when the task reports it (what the benchmark table shows), else acc."""
    names = [m["metric"] for m in (task.config.metric_list or [])]
    return "acc_norm" if "acc_norm" in names else "acc"


def doc_requests(task, doc, doc_id=0):
    """[(context, continuation), ...] exactly as lm-eval builds them (0-shot)."""
    ctx = task.fewshot_context(doc, 0)
    reqs = task.construct_requests(doc=doc, ctx=ctx, metadata=(task.config.task, doc_id, 1))
    reqs = reqs if isinstance(reqs, list) else [reqs]
    return [r.args for r in reqs]


def gold_index(task, doc):
    """Index of the correct request for a multiple-choice doc (lm-eval's own
    logic in ConfigurableTask.process_results), or None for loglikelihood
    tasks, whose single request is the target."""
    if task.OUTPUT_TYPE != "multiple_choice":
        return None
    if task.multiple_input:
        return int(task.doc_to_text(doc))
    gold = task.doc_to_target(doc)
    if isinstance(gold, str):
        gold = task.doc_to_choice(doc).index(gold)
    return int(gold)


class TaskSplits:
    def __init__(self, name, task, fit, select, eval_docs, info):
        self.name = name
        self.task = task
        self.metric = headline_metric(task)
        self.fit = fit
        self.select = select
        self.eval = eval_docs
        self.info = info


def build_splits(name, task, n_fit, n_select, seed, holdout_frac=0.3, n_eval=None):
    rng = random.Random(f"{seed}-{name}")
    eval_docs = list(task.eval_docs)
    eval_split = task.config.test_split if task.has_test_docs() else task.config.validation_split
    info = {"eval_split": eval_split, "eval_is_subset": False}

    if task.has_training_docs():
        train_split = getattr(task.config, "training_split", None) or "train"
        train = list(task.training_docs())
        rng.shuffle(train)
        if task.has_validation_docs() and task.config.validation_split != eval_split:
            fit = train[:n_fit]
            val = list(task.validation_docs())
            rng.shuffle(val)
            select = val[:n_select]
            info.update(fit_split=train_split, select_split=task.config.validation_split)
        else:
            # select is held out from the *end* of train, so n_fit None (all
            # of train, used by fine-tuning) never overlaps it
            n_fit = len(train) - n_select if n_fit is None else n_fit
            fit = train[:n_fit]
            select = train[n_fit:n_fit + n_select]
            info.update(fit_split=train_split, select_split=f"{train_split} (held out from fit)")
    else:
        order = list(range(len(eval_docs)))
        rng.shuffle(order)
        n_hold = int(round(holdout_frac * len(order)))
        hold, rest = order[:n_hold], sorted(order[n_hold:])
        n_sel = min(n_select, n_hold // 3)
        select = [eval_docs[i] for i in hold[:n_sel]]
        fit = [eval_docs[i] for i in hold[n_sel:None if n_fit is None else n_sel + n_fit]]
        eval_docs = [eval_docs[i] for i in rest]
        info.update(fit_split=f"{eval_split} (held out, {holdout_frac:.0%})",
                    select_split=f"{eval_split} (held out)", eval_is_subset=True)

    if n_eval is not None:                      # smoke tests only
        eval_docs = eval_docs[:n_eval]
        info["eval_is_subset"] = True
    info.update(n_fit=len(fit), n_select=len(select), n_eval=len(eval_docs))
    return TaskSplits(name, task, fit, select, eval_docs, info)
