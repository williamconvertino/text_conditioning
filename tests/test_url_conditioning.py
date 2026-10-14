"""URL-encoded conditioning (README "URL-Encoded Conditioning"): the URL
condition source, its collation/model path, the final eval's URL branch,
benchmark steering vectors, and the plotting hooks for the
text_conditioned_url* experiment families."""

import json

import pytest
import torch

from src.analysis.figures import CONDITIONED_PREFIXES, display_name, plot_conditioning_curve
from src.benchmarks.conditioning import resolve_url_text
from src.data.collator import _meco_prefix_ids, collate_encoder_batch, load_val_batches, train_batches
from src.data.shards import TokenizedDocStream, split_shards
from src.data.url_conditioning import UrlConditionSource, build_url_source
from src.eval.conditioning import evaluate_final
from src.models.conditioned_lm import ConditionedLM
from tests.test_domain_conditioning import partner_tok_dir  # noqa: F401 (fixture)
from tests.test_eval_conditioning import EVAL_CFG, domain_tok_dir, make_conditioner, make_lm  # noqa: F401

SOURCES_CFG = {"sources": {"wikipedia": {"meco_domain": "en.wikipedia.org", "match": ["wikipedia.org"]}}}


# -- src/data/url_conditioning.py ---------------------------------------------

def test_domain_field_matches_meco_metadata(tokenizer):
    source = UrlConditionSource(tokenizer, field="domain", max_tokens=48)
    [doc] = source.for_url("https://www.en.wikipedia.org/wiki/Foo")
    sep = tokenizer.encode("\n\n", add_special_tokens=False)
    meco = _meco_prefix_ids(tokenizer, "https://www.en.wikipedia.org/wiki/Foo", 48)
    assert doc["tokens"] == meco[: len(meco) - len(sep)]   # MeCo's prefix minus its separator
    assert source.for_url("") == []                          # no domain -> null vector


def test_url_field_and_max_tokens(tokenizer):
    url = "https://example.com/a/very/long/path/about/something"
    full = UrlConditionSource(tokenizer, field="url", max_tokens=96)
    assert full.metadata_text(url) == url
    capped = UrlConditionSource(tokenizer, field="url", max_tokens=3)
    assert len(capped.for_url(url)[0]["tokens"]) == 3
    with pytest.raises(ValueError):
        UrlConditionSource(tokenizer, field="path")


def test_build_url_source_only_for_url_runs(tokenizer):
    assert build_url_source({"variant": "text"}, tokenizer) is None
    assert build_url_source({"variant": "meco", "text_conditioning_source": "url"}, tokenizer) is None
    src = build_url_source({"variant": "text", "text_conditioning_source": "url",
                            "url_conditioning": {"field": "url", "max_tokens": 7}}, tokenizer)
    assert src.field == "url" and src.encoder_max_tokens == 7


# -- collation / model --------------------------------------------------------

def test_train_batches_with_url_source(partner_tok_dir, tokenizer):
    source = UrlConditionSource(tokenizer, max_tokens=12)
    stream = TokenizedDocStream(partner_tok_dir, ["shard_a", "shard_b"], seed=0)
    batch = next(train_batches(stream, tokenizer, "text", 32, 8, 512, local_batch=9,
                               condition_source=source))
    enc = batch["encoder"]
    assert enc["input_ids"].shape == (8, 12)          # source's max_tokens, not the run's 512
    assert enc["has_condition"].sum().item() == 8     # every doc but the no-URL one
    assert enc["group_index"].tolist() == sorted(enc["group_index"].tolist())

    dropped = next(train_batches(stream, tokenizer, "text", 32, 8, 512, local_batch=9,
                                 keep_mask_fn=lambda: [False] * 9, condition_source=source))
    assert not dropped["encoder"]["has_condition"].any()


def test_url_conditioned_z_is_encoded_domain(partner_tok_dir, tokenizer):
    torch.manual_seed(0)
    conditioner = make_conditioner(len(tokenizer)).eval()
    model = ConditionedLM(make_lm(len(tokenizer)), "text", conditioner).eval()
    source = UrlConditionSource(tokenizer, max_tokens=12)
    [batch] = load_val_batches(partner_tok_dir, ["shard_a"], tokenizer, "text", 32, 8, 512,
                               val_docs=None, val_batch_size=8, condition_source=source)
    with torch.no_grad():
        loss, _, z = model(batch)
        enc = collate_encoder_batch(source.for_text("a.com"), tokenizer, 12)
        expected = conditioner.encoder(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"])[0]
    assert torch.isfinite(loss)
    for row in range(3):                               # a.com/1, a.com/2, www.a.com/3 -> same vector
        assert torch.allclose(z[row], expected, atol=1e-5)
    assert torch.allclose(z[4], conditioner.null_vector)  # no URL


# -- src/eval/conditioning.py -------------------------------------------------

def _url_eval(domain_tok_dir, tokenizer, field):
    torch.manual_seed(0)
    model = ConditionedLM(make_lm(len(tokenizer)), "text", make_conditioner(len(tokenizer))).eval()
    _, val_stems = split_shards(domain_tok_dir, n_val_shards=1)
    cfg = dict(EVAL_CFG, text_conditioning_source="url", url_conditioning={"field": field, "max_tokens": 16})
    return evaluate_final(model, tokenizer, domain_tok_dir, val_stems, cfg)


def test_evaluate_final_url_domain_field(domain_tok_dir, tokenizer):
    results = _url_eval(domain_tok_dir, tokenizer, "domain")
    assert results["overall"]["none"] is not None and results["overall"]["self"] is not None
    agg = results["top_domains"]["aggregate"]
    assert agg["self"] is not None and agg["cross_domain"] is not None
    # identical domain string for every same-domain doc -> would just repeat "self"
    assert agg["same_domain"] is None and agg["domain_mean"] is None


def test_evaluate_final_url_full_field(domain_tok_dir, tokenizer):
    agg = _url_eval(domain_tok_dir, tokenizer, "url")["top_domains"]["aggregate"]
    assert all(agg[m] is not None for m in ("none", "self", "same_domain", "domain_mean", "cross_domain"))


# -- src/benchmarks/conditioning.py -------------------------------------------

def test_resolve_url_text(tokenizer):
    torch.manual_seed(0)
    conditioner = make_conditioner(len(tokenizer))
    source = UrlConditionSource(tokenizer, max_tokens=12)
    by_source = resolve_url_text(conditioner, tokenizer, source, "cpu", source="wikipedia",
                                 sources_cfg=SOURCES_CFG)
    by_domain = resolve_url_text(conditioner, tokenizer, source, "cpu", domain="en.wikipedia.org")
    by_url = resolve_url_text(conditioner, tokenizer, source, "cpu",
                              url="https://en.wikipedia.org/wiki/Anything")
    assert by_source.kind == "text" and by_source.tag == "text_wikipedia"
    assert by_domain.tag == "text_domain-en-wikipedia-org"
    assert by_source.meta["encoded"] == "en.wikipedia.org"
    assert torch.allclose(by_source.z, by_domain.z) and torch.allclose(by_source.z, by_url.z)
    with pytest.raises(AssertionError):
        resolve_url_text(conditioner, tokenizer, source, "cpu", source="wikipedia", domain="x",
                         sources_cfg=SOURCES_CFG)


# -- src/analysis/figures.py --------------------------------------------------

def test_url_families_have_plot_metadata():
    for name in ("text_conditioned_url", "text_conditioned_url_cooldown", "text_conditioned_url_full"):
        assert CONDITIONED_PREFIXES[name][0] == "url_conditioned"
        assert display_name(f"{name}_medium").startswith("URL-Encoded Conditioning")


def test_plot_conditioning_curve_url_run(tmp_path):
    run_dir = tmp_path / "text_conditioned_url_medium"
    run_dir.mkdir()
    with open(run_dir / "metrics.jsonl", "w") as f:
        for i, step in enumerate((500, 1000)):
            f.write(json.dumps({
                "step": step, "split": "val", "tokens": step * 1e6, "loss": 3.0 - 0.1 * i,
                "ppl": 20.0 - i, "url_conditioned_loss": 3.0 - 0.1 * i, "url_conditioned_ppl": 20.0 - i,
                "unconditioned_loss": 3.2 - 0.1 * i, "unconditioned_ppl": 24.0 - i}) + "\n")
    plot_conditioning_curve(str(run_dir), str(tmp_path / "figs"), metric="ppl")
    assert (tmp_path / "figs" / "text_url_conditioning_ppl.png").exists()
