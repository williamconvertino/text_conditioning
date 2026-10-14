import os

import torch

from src.benchmarks.conditioning import ConditioningSpec, resolve_meco, resolve_none, resolve_text
from src.benchmarks.lm_wrapper import CtrlTokensLM
from src.benchmarks.results import best_metric, discover_results, load_result, save_result, task_metric_rows
from src.benchmarks.sources import find_doc_by_url, find_source_docs
from src.data.collator import collate_encoder_batch, doc_dict
from src.data.shards import split_shards
from src.models.conditioned_lm import ConditionedLM
from src.models.conditioning import TextConditioner
from src.models.configuration_llama import LlamaConfig
from src.models.modeling_llama import LlamaForCausalLM
from src.models.text_encoder import TextEncoder, TextEncoderConfig

HIDDEN = 32
BLOCK_SIZE = 16

SOURCES_CFG = {
    "sources": {
        "wikipedia": {"meco_domain": "en.wikipedia.org", "match": ["wikipedia.org"]},
        "news": {"meco_domain": "news.example.com", "match": ["news.example.com"]},
    },
    "n_docs_per_source": 20,
    "seed": 1234,
}


def make_lm(vocab_size):
    cfg = LlamaConfig(vocab_size=vocab_size, hidden_size=HIDDEN, intermediate_size=64,
                      num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
                      max_position_embeddings=BLOCK_SIZE + 1, pad_token_id=0, bos_token_id=1,
                      eos_token_id=2)
    return LlamaForCausalLM(cfg)


def make_conditioner(vocab_size):
    enc_cfg = TextEncoderConfig(vocab_size=vocab_size, hidden_size=16, intermediate_size=32,
                                num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=2,
                                max_position_embeddings=32, output_dim=HIDDEN, pad_token_id=0)
    return TextConditioner(TextEncoder(enc_cfg), HIDDEN)


class FakeReq:
    """Minimal stand-in for lm_eval.api.instance.Instance -- CtrlTokensLM only
    ever reads .args, so tests don't need the real (heavier) dataclass."""

    def __init__(self, context, continuation):
        self.args = (context, continuation)


# -- src/benchmarks/sources.py -----------------------------------------------

def test_find_source_docs_matches_domain(tok_dir):
    train_stems, _ = split_shards(tok_dir, n_val_shards=1)
    docs = find_source_docs(tok_dir, "wikipedia", SOURCES_CFG, train_stems, n_docs=20, seed=1)
    assert len(docs) == 2
    assert all("wikipedia" in (shard.url[row] or "") for shard, row in docs)


def test_find_source_docs_unknown_source_raises(tok_dir):
    train_stems, _ = split_shards(tok_dir, n_val_shards=1)
    try:
        find_source_docs(tok_dir, "not_a_source", SOURCES_CFG, train_stems, n_docs=20, seed=1)
        raise AssertionError("expected KeyError")
    except KeyError:
        pass


def test_find_doc_by_url_exact_match(tok_dir):
    train_stems, _ = split_shards(tok_dir, n_val_shards=1)
    [(shard, row)] = find_source_docs(tok_dir, "wikipedia", SOURCES_CFG, train_stems, n_docs=1, seed=1)
    url = shard.url[row]
    found_shard, found_row = find_doc_by_url(tok_dir, url, train_stems)
    assert found_shard.url[found_row] == url


# -- src/benchmarks/conditioning.py ------------------------------------------

def test_resolve_none():
    spec = resolve_none()
    assert spec.kind == "none"
    assert spec.tag == "unconditioned"


def test_resolve_meco_source_and_domain():
    spec = resolve_meco(source="wikipedia", sources_cfg=SOURCES_CFG)
    assert spec.kind == "meco"
    assert spec.domain == "en.wikipedia.org"
    assert spec.tag == "meco_wikipedia"

    spec2 = resolve_meco(domain="custom.example.org")
    assert spec2.domain == "custom.example.org"
    assert spec2.tag.startswith("meco_domain-")


def test_resolve_text_averages_source_docs(tok_dir, tokenizer):
    conditioner = make_conditioner(len(tokenizer))
    train_stems, _ = split_shards(tok_dir, n_val_shards=1)
    device = torch.device("cpu")

    spec = resolve_text(conditioner, tokenizer, tok_dir, train_stems, encoder_max_tokens=16,
                        device=device, source="wikipedia", sources_cfg=SOURCES_CFG, n_docs=20, seed=1)
    assert spec.kind == "text"
    assert spec.tag == "text_wikipedia"
    assert spec.z.shape == (HIDDEN,)
    assert spec.meta["n_docs"] == 2

    docs = find_source_docs(tok_dir, "wikipedia", SOURCES_CFG, train_stems, n_docs=20, seed=1)
    doc_dicts = [doc_dict(shard, row) for shard, row in docs]
    enc = collate_encoder_batch(doc_dicts, tokenizer, 16)
    with torch.no_grad():
        zs = conditioner.encoder(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"])
    assert torch.allclose(spec.z, zs.mean(dim=0), atol=1e-6)


def test_resolve_text_single_url(tok_dir, tokenizer):
    conditioner = make_conditioner(len(tokenizer))
    train_stems, _ = split_shards(tok_dir, n_val_shards=1)
    [(shard, row)] = find_source_docs(tok_dir, "wikipedia", SOURCES_CFG, train_stems, n_docs=1, seed=1)
    url = shard.url[row]

    spec = resolve_text(conditioner, tokenizer, tok_dir, train_stems, encoder_max_tokens=16,
                        device=torch.device("cpu"), url=url)
    assert spec.kind == "text"
    assert spec.meta["n_docs"] == 1
    assert spec.tag.startswith("text_url-")


# -- src/benchmarks/lm_wrapper.py --------------------------------------------

def test_ctrl_tokens_lm_none_variant(tokenizer):
    model = ConditionedLM(make_lm(len(tokenizer)), "none")
    model.eval()
    lm = CtrlTokensLM(model, tokenizer, "none", resolve_none(), block_size=BLOCK_SIZE)

    reqs = [FakeReq("The quick brown fox", " jumps"), FakeReq("Hello there", " world")]
    results = lm.loglikelihood(reqs)
    assert len(results) == 2
    for logprob, is_greedy in results:
        assert torch.isfinite(torch.tensor(logprob))
        assert isinstance(is_greedy, bool)


def test_ctrl_tokens_lm_meco_prefix_changes_score(tokenizer):
    model = ConditionedLM(make_lm(len(tokenizer)), "meco")
    model.eval()
    context, continuation = "The quick brown fox", " jumps"

    lm_none = CtrlTokensLM(model, tokenizer, "meco", resolve_none(), block_size=BLOCK_SIZE, meco_max_tokens=8)
    lm_meco = CtrlTokensLM(model, tokenizer, "meco", resolve_meco(domain="en.wikipedia.org"),
                           block_size=BLOCK_SIZE, meco_max_tokens=8)

    [(logprob_none, _)] = lm_none.loglikelihood([FakeReq(context, continuation)])
    [(logprob_meco, _)] = lm_meco.loglikelihood([FakeReq(context, continuation)])
    assert logprob_none != logprob_meco


def test_ctrl_tokens_lm_text_variant_null_vs_z(tokenizer):
    conditioner = make_conditioner(len(tokenizer))
    model = ConditionedLM(make_lm(len(tokenizer)), "text", conditioner)
    model.eval()
    context, continuation = "The quick brown fox", " jumps"

    lm_null = CtrlTokensLM(model, tokenizer, "text", resolve_none(), block_size=BLOCK_SIZE)
    z_spec = ConditioningSpec("text", "text_custom", z=torch.randn(HIDDEN))
    lm_z = CtrlTokensLM(model, tokenizer, "text", z_spec, block_size=BLOCK_SIZE)

    [(logprob_null, _)] = lm_null.loglikelihood([FakeReq(context, continuation)])
    [(logprob_z, _)] = lm_z.loglikelihood([FakeReq(context, continuation)])
    assert logprob_null != logprob_z


def test_ctrl_tokens_lm_truncates_long_context(tokenizer):
    model = ConditionedLM(make_lm(len(tokenizer)), "none")
    model.eval()
    lm = CtrlTokensLM(model, tokenizer, "none", resolve_none(), block_size=BLOCK_SIZE)

    seq, clen = lm._build_example(" word" * 100, " end")
    assert len(seq) <= BLOCK_SIZE
    assert 1 <= clen <= BLOCK_SIZE


# -- src/benchmarks/results.py -----------------------------------------------

def test_best_metric_prefers_acc_norm():
    assert best_metric({"acc,none": 0.5, "acc_norm,none": 0.6}) == ("acc_norm,none", 0.6)
    assert best_metric({"acc,none": 0.5}) == ("acc,none", 0.5)
    assert best_metric({"alias": "task", "acc_stderr,none": 0.1}) == (None, None)


def test_save_and_discover_results(tmp_path):
    run_dir = str(tmp_path / "baseline_medium")
    fake_results = {"results": {"hellaswag": {"acc,none": 0.42, "alias": "hellaswag"}},
                    "n-shot": {"hellaswag": 0}, "n-samples": {}, "versions": {}}
    path = save_result(run_dir, "unconditioned", run_name="baseline_medium", variant="none",
                       conditioning_kind="none", conditioning_meta={}, tasks_cfg={"tasks": ["hellaswag"]},
                       lm_eval_results=fake_results)
    assert os.path.exists(path)

    loaded = load_result(path)
    assert loaded["results"]["hellaswag"]["acc,none"] == 0.42

    records = discover_results()
    assert len(records) == 1
    assert records[0]["run_name"] == "baseline_medium"

    rows = task_metric_rows(records)
    assert rows == [{"run_name": "baseline_medium", "conditioning_tag": "unconditioned",
                     "variant": "none", "task": "hellaswag", "metric": "acc,none", "value": 0.42}]
