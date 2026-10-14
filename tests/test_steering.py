"""Shared-checkpoint resolution, the URL-encoder analyses, and the steering
experiments' scoring/optimization (README "Encoder Analysis & Steering")."""

import os
import types

import pytest
import torch

from src.benchmarks.conditioning import ConditioningSpec, resolve_none
from src.benchmarks.lm_wrapper import CtrlTokensLM
from src.benchmarks.results import discover_results, result_path, save_result
from src.comparison.checkpoints import checkpoint_for, list_checkpoints, resolve_shared_checkpoint
from src.models.conditioned_lm import ConditionedLM
from src.steering.optimize import fit_vector
from src.steering.scoring import PreparedDocs, doc_scores, fast_accuracy, score_docs
from src.steering.vectors import interpolate, scale
from src.url_analysis import analyses as A
from tests.test_eval_conditioning import BLOCK_SIZE, HIDDEN, make_conditioner, make_lm


# -- src/comparison/checkpoints.py ----------------------------------------------

def _run(tmp_path, name, snapshots=(), last=None):
    d = tmp_path / name
    (d / "checkpoints").mkdir(parents=True)
    for step, tok_m in snapshots:
        (d / "checkpoints" / f"ckpt_step{step:07d}_tok{tok_m}M.pt").write_bytes(b"")
    if last is not None:
        torch.save({"step": last[0], "tokens_seen": last[1]}, d / "ckpt_last.pt")
    return str(d)


def test_shared_step_prefers_exact_final_step(tmp_path):
    a = _run(tmp_path, "a", [(860, 250), (1720, 500)], last=(2000, 580_000_000))
    b = _run(tmp_path, "b", [(862, 250), (1724, 500)], last=(2000, 581_000_000))
    pin = resolve_shared_checkpoint({"a": a, "b": b})
    assert pin["label"] == "step2000" and pin["rule"] == "shared_step"
    assert {i["path"] for i in pin["runs"].values()} == {"ckpt_last.pt"}


def test_token_milestone_fallback_and_error(tmp_path):
    a = _run(tmp_path, "a", [(860, 250), (1720, 500)], last=(1900, 550_000_000))
    b = _run(tmp_path, "b", [(862, 250)], last=(1000, 290_000_000))
    pin = resolve_shared_checkpoint({"a": a, "b": b})
    assert pin["label"] == "tok250M"
    assert pin["runs"]["a"]["step"] == 860 and pin["runs"]["b"]["step"] == 862
    c = _run(tmp_path, "c", [], last=(500, 140_000_000))
    with pytest.raises(ValueError, match="no checkpoint step"):
        resolve_shared_checkpoint({"a": a, "c": c})
    assert [c["kind"] for c in list_checkpoints(a)] == ["snapshot", "snapshot", "last"]
    path, info = checkpoint_for(pin, os.path.join(a, ""))
    assert path.endswith("tok250M.pt") and info["name"] == "a"


def test_labelled_results_are_kept_apart(tmp_path):
    run = tmp_path / "x_medium"
    kw = dict(run_name="x_medium", variant="none", conditioning_kind="none", conditioning_meta={},
              tasks_cfg={}, lm_eval_results={"results": {"piqa": {"acc,none": 0.6}}})
    save_result(str(run), "unconditioned", **kw)
    save_result(str(run), "unconditioned", ckpt_label="step20000", **kw)
    assert os.path.exists(result_path(str(run), "unconditioned", "step20000"))
    assert len(discover_results()) == 1
    assert len(discover_results(ckpt_label="step20000")) == 1


# -- src/steering/vectors.py ------------------------------------------------------

def test_interpolate_and_scale_endpoints():
    a, b = torch.randn(16) * 5, torch.randn(16) * 5
    for mode in ("linear", "slerp"):
        assert torch.allclose(interpolate(a, b, 0.0, mode), a, atol=1e-5)
        assert torch.allclose(interpolate(a, b, 1.0, mode), b, atol=1e-4)
    mid = interpolate(a, b, 0.5, "slerp")
    assert abs(float(mid.norm()) - float((a.norm() + b.norm()) / 2)) < 1e-3
    ref = torch.randn(16)
    assert torch.allclose(scale(a, ref, 1.0), a) and torch.allclose(scale(a, ref, 0.0), ref)


# -- src/url_analysis/analyses.py ---------------------------------------------------

def test_label_url_rules():
    assert A.label_url("https://www.reddit.com/r/AskScience/comments/1/x", "path_after:r") == "askscience"
    assert A.label_url("https://math.stackexchange.com/questions/1", "subdomain") == "math"
    assert A.label_url("https://www.nytimes.com/2012/05/03/sports/x.html", "section") == "sports"
    assert A.label_url("https://www.nytimes.com/2012/05/03/sports/x.html", "year") == "2012"
    assert A.label_url("https://github.com/org/repo", "path:0") == "org"
    assert A.label_url("https://example.com/", "path:0") is None
    v = A.url_variants("https://www.example.com/a/b")
    assert v["path removed (root)"] == "https://www.example.com/" and v["domain only"] == "example.com"


def _synthetic_library(n_dom=6, per=10, H=12, seed=0):
    g = torch.Generator().manual_seed(seed)
    centers = torch.randn(n_dom, H, generator=g) * 5 + 20
    pages = torch.cat([c + 0.3 * torch.randn(per, H, generator=g) for c in centers])
    dom = torch.arange(n_dom).repeat_interleave(per)
    cent = torch.stack([pages[dom == d].mean(0) for d in range(n_dom)])
    return {"domains": [f"d{i}.com" for i in range(n_dom)], "counts": torch.arange(n_dom, 0, -1) * 10,
            "root": cent + 0.1, "centroid": cent, "page_vecs": pages, "page_domain": dom,
            "page_urls": [f"https://d{d}.com/{i}" for i, d in enumerate(dom.tolist())],
            "null": torch.zeros(H), "mean": cent.mean(0)}


def test_domain_statistics_on_separable_library():
    lib = _synthetic_library()
    assert A.variance_by_domain(lib["page_vecs"], lib["page_domain"])["r2_domain"] > 0.9
    ret = A.domain_retrieval(lib, n_pages=60)
    assert ret["top1"] > 0.9
    nb = A.neighbors(lib["centroid"][:2], ["d0.com", "d1.com"], lib, k=3)
    assert all(n["domain"] != "d0.com" for n in nb["d0.com"])
    stats = A.vector_stats(lib)
    assert stats["pairwise_cos_pages"]["raw_mean"] > stats["pairwise_cos_pages"]["centered_mean"]


# -- src/steering/scoring.py + optimize.py ----------------------------------------------

class FakeTask:
    """A 2-choice multiple-choice task with lm-eval's ConfigurableTask surface."""
    OUTPUT_TYPE = "multiple_choice"
    multiple_input = 0
    config = types.SimpleNamespace(task="fake", metric_list=[{"metric": "acc"}])

    def fewshot_context(self, doc, n):
        return doc["q"]

    def construct_requests(self, doc, ctx, **kw):
        return [types.SimpleNamespace(args=(ctx, " " + c)) for c in doc["choices"]]

    def doc_to_target(self, doc):
        return doc["gold"]

    def doc_to_choice(self, doc):
        return doc["choices"]

    def process_results(self, doc, results):
        lls = [ll for ll, _ in results]
        return {"acc": float(max(range(len(lls)), key=lls.__getitem__) == doc["gold"])}


def _fake_splits():
    docs = [{"q": q, "choices": ["yes", "no"], "gold": i % 2}
            for i, q in enumerate(["The sky is", "Water is", "Fire is", "Ice is", "Grass is", "Snow is"])]
    return types.SimpleNamespace(task=FakeTask(), metric="acc", fit=docs, select=docs, eval=docs, name="fake")


def test_fast_path_matches_lm_eval_scorer(tokenizer):
    torch.manual_seed(0)
    model = ConditionedLM(make_lm(len(tokenizer)), "text", make_conditioner(len(tokenizer))).eval()
    z = torch.randn(HIDDEN)
    spec = ConditioningSpec("text", "t", z=z)
    lm = CtrlTokensLM(model, tokenizer, "text", spec, block_size=BLOCK_SIZE, batch_size=4)
    splits = _fake_splits()
    exact = score_docs(lm, splits, splits.eval)

    prep = PreparedDocs(lm, splits, splits.eval)
    device = torch.device("cpu")
    rows, sums, _ = doc_scores(model, z, prep, list(range(prep.n_docs)), tokenizer.pad_token_id, device)
    reqs = [types.SimpleNamespace(args=a) for d in splits.eval for a in [(d["q"], " yes"), (d["q"], " no")]]
    ref = torch.tensor([ll for ll, _ in lm.loglikelihood(reqs)])
    assert torch.allclose(sums, ref, atol=1e-4)
    assert fast_accuracy(model, z, prep, tokenizer.pad_token_id, device) == pytest.approx(exact["acc"])


def test_fit_vector_never_worse_on_select(tokenizer):
    torch.manual_seed(0)
    model = ConditionedLM(make_lm(len(tokenizer)), "text", make_conditioner(len(tokenizer))).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    builder = CtrlTokensLM(model, tokenizer, "text", resolve_none(), block_size=BLOCK_SIZE)
    splits = _fake_splits()
    prep = PreparedDocs(builder, splits, splits.fit)
    device = torch.device("cpu")
    init = torch.randn(HIDDEN)
    start = fast_accuracy(model, init, prep, tokenizer.pad_token_id, device)
    for kind, kw in (("free", {"init": init, "max_norm": 100.0}), ("simplex", {"library": torch.randn(4, HIDDEN)})):
        z, hist, extra = fit_vector(model, [(prep, prep)], kind, tokenizer.pad_token_id, device, steps=3,
                                    batch_docs=4, lr=0.1, eval_every=1, log=lambda *_: None, **kw)
        assert z.shape == (HIDDEN,) and len(hist) == 4
        if kind == "free":
            assert extra["best_select_acc"] >= start
        else:
            assert extra["weights"].sum() == pytest.approx(1.0, abs=1e-5)
