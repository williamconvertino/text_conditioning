"""Fine-tuning (src/finetune/, README "Fine-Tuning"): FinetuneModel's scores
match the lm-eval adapter's for every variant and mode, peft starts exactly at
the conditioned input, each mode trains the right parameters, and the loop
runs end to end."""

import json
import os
import types

import pytest
import torch

from src.benchmarks.conditioning import ConditioningSpec, resolve_meco, resolve_none
from src.benchmarks.lm_wrapper import CtrlTokensLM
from src.finetune.data import TaskData, doc_batches, make_batch
from src.finetune.loop import Dist, run_finetune
from src.finetune.model import build_finetune_model, encoder_tokens, load_base_model
from src.models.conditioned_lm import ConditionedLM
from src.steering.task_data import build_splits
from src.training.checkpointing import save_eval_checkpoint
from tests.test_eval_conditioning import BLOCK_SIZE, HIDDEN, make_conditioner, make_lm
from tests.test_steering import FakeTask, _fake_splits

URL = "https://en.wikipedia.org/wiki/Cat"


def _cfg(variant):
    cfg = {"variant": variant, "block_size": BLOCK_SIZE, "meco_max_tokens": 8, "encoder_max_tokens": 16}
    if variant == "text":
        cfg.update(text_conditioning_source="url", url_conditioning={"field": "url", "max_tokens": 16})
    return cfg


def _base(variant, tokenizer):
    torch.manual_seed(0)
    cond = make_conditioner(len(tokenizer)) if variant == "text" else None
    return ConditionedLM(make_lm(len(tokenizer)), variant, cond).eval()


def _splits():
    s = _fake_splits()
    s.info = {"eval_is_subset": False}
    return s


def _scores(ft, td, tokenizer):
    prep = td.prep["eval"]
    rows = list(range(len(prep.seqs)))
    ids, mask = make_batch(prep, rows, ft.prefix_ids, tokenizer.pad_token_id)
    with torch.no_grad():
        sums, greedy = ft(ids, mask)
    return sums, greedy


def _reference(base, tokenizer, variant, spec, splits):
    lm = CtrlTokensLM(base, tokenizer, variant, spec, block_size=BLOCK_SIZE, meco_max_tokens=8, batch_size=4)
    reqs = [types.SimpleNamespace(args=a) for d in splits.eval for a in [(d["q"], " yes"), (d["q"], " no")]]
    out = lm.loglikelihood(reqs)
    return torch.tensor([ll for ll, _ in out]), [g for _, g in out]


@pytest.mark.parametrize("variant,mode", [("none", "standard"), ("none", "conditioned"), ("meco", "standard"),
                                          ("meco", "conditioned"), ("text", "standard"), ("text", "conditioned")])
def test_scores_match_lm_eval_adapter(tokenizer, variant, mode):
    base, cfg, splits = _base(variant, tokenizer), _cfg(variant), _splits()
    ft, info = build_finetune_model(base, cfg, tokenizer, mode, condition_url=URL)
    td = TaskData(splits, tokenizer, BLOCK_SIZE, ft.prefix_ids, None, 0)
    sums, greedy = _scores(ft.eval(), td, tokenizer)

    if mode == "standard":
        spec = resolve_none()
    elif variant == "text":
        t = torch.tensor([encoder_tokens(cfg, tokenizer, URL)])
        spec = ConditioningSpec("text", "t", z=base.conditioner.encoder(input_ids=t, attention_mask=torch.ones_like(t))[0])
    else:
        spec = resolve_meco(domain="en.wikipedia.org")
    # the baseline has no conditioning in the adapter; its "conditioned" input is MeCo's prefix
    ref_variant = "meco" if (variant == "none" and mode == "conditioned") else variant
    ref, ref_greedy = _reference(base, tokenizer, ref_variant, spec, splits)
    assert torch.allclose(sums, ref, atol=1e-4)
    assert greedy.tolist() == ref_greedy
    if mode == "conditioned":
        assert info["condition_url"] == URL
        assert (variant == "text") == ("encoder_input" in info)


@pytest.mark.parametrize("variant", ["none", "meco", "text"])
def test_peft_starts_at_conditioned_input_and_trains_only_the_prefix(tokenizer, variant):
    base, cfg, splits = _base(variant, tokenizer), _cfg(variant), _splits()
    cond, _ = build_finetune_model(base, cfg, tokenizer, "conditioned", condition_url=URL)
    peft, info = build_finetune_model(base, cfg, tokenizer, "peft", condition_url=URL, peft_init="condition")
    a = _scores(cond, TaskData(splits, tokenizer, BLOCK_SIZE, cond.prefix_ids, None, 0), tokenizer)[0]
    b = _scores(peft, TaskData(splits, tokenizer, BLOCK_SIZE, peft.prefix_ids, None, 0), tokenizer)[0]
    assert torch.allclose(a, b, atol=1e-4)
    assert [n for n, p in peft.named_parameters() if p.requires_grad] == ["soft_prompt"]
    assert info["n_trainable"] == peft.soft_prompt.numel()
    assert peft.soft_prompt.shape[0] == (1 if variant == "text" else len(cond.prefix_ids))


def test_peft_null_tunes_the_pretrained_null_vector_slot(tokenizer):
    """A text run's peft_null learns the pretrained null vector itself, in the
    one conditioning slot: step 0 is exactly the unconditioned ("standard")
    input, no extra vector is added, and it is the only trainable tensor."""
    base, cfg, splits = _base("text", tokenizer), _cfg("text"), _splits()
    std, _ = build_finetune_model(base, cfg, tokenizer, "standard")
    peft, info = build_finetune_model(base, cfg, tokenizer, "peft", peft_init="null_vector")
    assert torch.equal(peft.soft_prompt.detach(), base.conditioner.null_vector.detach().unsqueeze(0))
    assert peft.n_prefix_vectors == std.n_prefix_vectors == 1 and info["n_prefix_tokens"] == 0
    a = _scores(std, TaskData(splits, tokenizer, BLOCK_SIZE, std.prefix_ids, None, 0), tokenizer)[0]
    b = _scores(peft, TaskData(splits, tokenizer, BLOCK_SIZE, peft.prefix_ids, None, 0), tokenizer)[0]
    assert torch.equal(a, b)
    assert [n for n, p in peft.named_parameters() if p.requires_grad] == ["soft_prompt"]
    assert info["n_trainable"] == HIDDEN


def test_mode_parameter_sets(tokenizer):
    base, cfg = _base("text", tokenizer), _cfg("text")
    ft, _ = build_finetune_model(base, cfg, tokenizer, "standard")
    names = {n for n, p in ft.named_parameters() if p.requires_grad}
    assert names and all(n.startswith("base.lm.") for n in names)
    ft, info = build_finetune_model(base, cfg, tokenizer, "conditioned", condition_url=URL, train_encoder=True)
    assert info["train_encoder"] and any(p.requires_grad for p in base.conditioner.encoder.parameters())
    assert not base.conditioner.null_vector.requires_grad
    ft, _ = build_finetune_model(_base("meco", tokenizer), _cfg("meco"), tokenizer, "peft",
                                 peft_init="vocab_mean", n_vectors=3)
    assert ft.soft_prompt.shape == (3, HIDDEN)
    with pytest.raises(ValueError):
        build_finetune_model(base, cfg, tokenizer, "peft", peft_init="vocab_mean")
    with pytest.raises(ValueError):
        build_finetune_model(base, cfg, tokenizer, "conditioned")


def test_doc_batches_keep_docs_whole(tokenizer):
    splits = _splits()
    td = TaskData(splits, tokenizer, BLOCK_SIZE, [], None, 0)
    prep = td.prep["eval"]
    batches = doc_batches(prep, list(range(prep.n_docs)), max_rows=3)
    assert [d for b in batches for d in b] == list(range(prep.n_docs))
    assert all(len(prep.doc_rows(b)) <= 3 for b in batches)
    assert td.gold_rows(prep, [0, 1]) == [0, 3]      # gold alternates 0, 1; two choices per doc


def test_build_splits_uses_all_training_docs(tokenizer):
    class Task(FakeTask):
        config = types.SimpleNamespace(task="fake", metric_list=[{"metric": "acc"}], test_split=None,
                                       validation_split="validation")
        eval_docs = [{"i": i} for i in range(5)]
        has_training_docs = lambda self: True
        has_validation_docs = lambda self: True
        has_test_docs = lambda self: False
        training_docs = lambda self: [{"i": i} for i in range(10)]
        validation_docs = lambda self: self.eval_docs

    s = build_splits("fake", Task(), None, 3, 0)
    assert len(s.fit) == 7 and len(s.select) == 3
    assert not {d["i"] for d in s.fit} & {d["i"] for d in s.select}
    assert len(build_splits("fake", Task(), 4, 3, 0).fit) == 4


@pytest.mark.parametrize("variant,mode,objective", [("text", "peft", "choice_ce"), ("meco", "standard", "lm"),
                                                    ("none", "conditioned", "choice_ce")])
def test_run_finetune_end_to_end(tmp_path, tokenizer, variant, mode, objective):
    base, cfg = _base(variant, tokenizer), _cfg(variant)
    ft, info = build_finetune_model(base, cfg, tokenizer, mode, condition_url=URL)
    td = TaskData(_splits(), tokenizer, BLOCK_SIZE, ft.prefix_ids, 4, 0)
    ft_cfg = dict(objective=objective, steps=6, batch_docs=4, micro_docs=2, warmup_steps=1, min_lr_frac=0.1,
                  grad_clip=1.0, lr=1e-2, weight_decay=0.0, peft_lr_relative=0.1, eval_every=2, eval_batch_seqs=5,
                  eval_dtype="float32", train_dtype="float32", log_every=1, seed=0)
    D = Dist(0, 1, torch.device("cpu"), False, False)
    out = str(tmp_path / "ft")
    res = run_finetune(ft, ft, [td], mode, ft_cfg, D, out, tokenizer.pad_token_id, info, log=lambda *_: None)
    assert res["best_select"] >= res["base"]["fake"]["select"]
    assert [h["step"] for h in res["history"]] == [0, 2, 4, 6]
    assert res["best"]["fake"]["n_eval"] == len(td.splits.eval)
    recs = [json.loads(l) for l in open(os.path.join(out, "metrics.jsonl"))]
    assert sum(r["split"] == "train" for r in recs) == 6
    assert [r["step"] for r in recs if r["split"] == "eval"] == [0, 2, 4, 6]
    saved = torch.load(os.path.join(out, "prefix.pt"), weights_only=False)
    assert (saved["prefix"] is not None) == (mode == "peft")


def test_load_base_model_reties_embeddings(tmp_path, tokenizer):
    base = _base("none", tokenizer)
    cfg = {**_cfg("none"), "tie_word_embeddings": True, "hidden_size": HIDDEN, "intermediate_size": 64,
           "num_hidden_layers": 2, "num_attention_heads": 4, "num_key_value_heads": 2,
           "max_position_embeddings": BLOCK_SIZE + 1, "rms_norm_eps": 1e-5, "rope_theta": 500000.0,
           "initializer_range": 0.02}
    torch.manual_seed(0)
    from src.training.trainer import build_model
    base, _ = build_model(cfg, tokenizer)
    save_eval_checkpoint(str(tmp_path / "ckpt.pt"), base, 1, 0.0, 0, cfg)
    loaded = load_base_model(str(tmp_path), "ckpt.pt", tokenizer, cfg)
    assert loaded.lm.lm_head.weight is loaded.lm.model.embed_tokens.weight
    assert torch.equal(loaded.lm.lm_head.weight, base.lm.lm_head.weight)


def test_report_aggregates_seeds(tmp_path):
    from src.finetune import report as F
    pin = {"runs": {"meco": {"run_dir": "outputs/meco_p0.9_medium"}}}
    for method, accs in (("standard", [0.50, 0.54]), ("conditioned", [0.53, 0.55])):
        for seed, acc in zip((1234, 1), accs):
            d = tmp_path / "meco_p0.9_medium" / "step1" / method / "piqa" / f"seed{seed}"
            d.mkdir(parents=True)
            (d / "metrics.jsonl").write_text("".join(
                json.dumps({"split": "eval", "task": "piqa", "step": s, "select_acc": acc, "eval_acc": acc + s})
                + "\n" for s in (0, 25)))
            r = lambda a: {"piqa": {"eval": a, "eval_stderr": 0.01}}
            (d / "result.json").write_text(json.dumps({"base": r(0.4), "best": r(acc), "best_step": 25}))
    jobs = F.load_jobs(str(tmp_path), pin, "step1")
    assert F.n_seeds(jobs) == 2
    m, sd = F._cell(jobs[("meco", "standard", "piqa")], "best", "piqa")
    assert m == pytest.approx(0.52) and sd == pytest.approx(0.0283, abs=1e-3)
    assert F._cell(jobs[("meco", "standard", "piqa")], "base", "piqa") == (0.4, 0.01)
    d, dsd = F._paired(jobs[("meco", "conditioned", "piqa")], jobs[("meco", "standard", "piqa")], "best", "best", "piqa")
    assert d == pytest.approx(0.02) and dsd == pytest.approx(0.0141, abs=1e-3)
    steps, mean, std = F._curve(jobs, "meco", "standard", "piqa", 1)
    assert steps == [0, 25] and mean[1] == pytest.approx(25.52)
    assert F._run_color(pin, "meco") == "#0072B2"
    F.result_tables(jobs, pin, ["piqa"], ["standard", "conditioned"], str(tmp_path / "tables"))
    assert (tmp_path / "tables" / "finetune_vs_standard.md").exists()
    F.plot_curves(jobs, pin, ["piqa"], ["standard", "conditioned"], ["meco"], str(tmp_path / "fig"), "c", "t")
