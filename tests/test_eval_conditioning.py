import json
import os

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from src.analysis.tables import build_domain_rows, build_table_rows, load_final_results, write_domain_csv
from src.data.collator import doc_dict
from src.data.shards import split_shards
from src.eval.conditioning import _combine, _score, evaluate_final
from src.eval.domain_support import build_domain_index, split_support_and_eval, top_domains
from src.models.conditioned_lm import ConditionedLM
from src.models.conditioning import TextConditioner
from src.models.configuration_llama import LlamaConfig
from src.models.modeling_llama import LlamaForCausalLM
from src.models.text_encoder import TextEncoder, TextEncoderConfig

HIDDEN = 32
BLOCK_SIZE = 32

# k=2, min_docs_per_domain=2 -> a domain needs >= 4 documents to qualify.
# big1/big2 (15 docs each) qualify; small (3 docs) does not.
EVAL_CFG = dict(block_size=BLOCK_SIZE, meco_max_tokens=8, encoder_max_tokens=16, val_batch_size=4,
                val_docs=None, n_domains=2, domain_scan_docs=None, domain_support_k=2,
                min_docs_per_domain=2, cross_domain=True, seed=0)


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


def _write_shard(out_dir, stem, texts, urls, tokenizer, max_doc_tokens=24):
    eos = tokenizer.eos_token_id
    starts, lengths, tok_chunks = [], [], []
    cursor = 0
    for t in texts:
        ids = tokenizer.encode(t, add_special_tokens=False)[: max_doc_tokens - 1] + [eos]
        tok_chunks.append(np.asarray(ids, dtype=np.uint32))
        starts.append(cursor)
        lengths.append(len(ids))
        cursor += len(ids)
    tokens = np.concatenate(tok_chunks)
    tokens.tofile(os.path.join(out_dir, f"{stem}.bin"))
    index = pa.table({
        "row": np.arange(len(texts), dtype=np.int32),
        "start": np.asarray(starts, dtype=np.int64),
        "length": np.asarray(lengths, dtype=np.int32),
        "url": urls,
        "doc_id": [f"{stem}-{i}" for i in range(len(texts))],
    })
    pq.write_table(index, os.path.join(out_dir, f"{stem}_index.parquet"))
    open(os.path.join(out_dir, f"{stem}.done"), "w").close()


@pytest.fixture()
def domain_tok_dir(tmp_path, tokenizer):
    out_dir = str(tmp_path / "fineweb_tokenized")
    os.makedirs(out_dir, exist_ok=True)
    manifest = {
        "tokenizer": "gpt2", "vocab_size": len(tokenizer), "eos_id": tokenizer.eos_token_id,
        "pad_id": tokenizer.pad_token_id, "max_doc_tokens": 24,
        "text_column": "text", "url_column": "url", "id_column": "id",
    }
    with open(os.path.join(out_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f)

    _write_shard(out_dir, "shard_00", ["training filler document."],
                ["https://train.example.com/a"], tokenizer)

    texts, urls = [], []
    for i in range(15):
        texts.append(f"Document number {i} about topic big one repeated content for domain one.")
        urls.append(f"https://big1.example.com/article-{i}")
    for i in range(15):
        texts.append(f"Document number {i} about topic big two repeated content for domain two.")
        urls.append(f"https://big2.example.com/article-{i}")
    for i in range(3):
        texts.append(f"Short domain document {i}.")
        urls.append(f"https://small.example.com/article-{i}")
    _write_shard(out_dir, "shard_01_val", texts, urls, tokenizer)
    return out_dir


# -- src/eval/domain_support.py ----------------------------------------------

def test_top_domains_selection(domain_tok_dir):
    _, val_stems = split_shards(domain_tok_dir, n_val_shards=1)
    idx = build_domain_index(domain_tok_dir, val_stems)

    chosen = top_domains(idx, n_domains=2, k=2, min_docs_per_domain=2)
    assert [d for d, _ in chosen] == ["big1.example.com", "big2.example.com"]
    assert all(c == 15 for _, c in chosen)

    # "small.example.com" has only 3 docs (< 2 + 2), so raising n_domains
    # still shouldn't select it.
    chosen_all = top_domains(idx, n_domains=10, k=2, min_docs_per_domain=2)
    assert "small.example.com" not in [d for d, _ in chosen_all]

    assert len(top_domains(idx, n_domains=1, k=2, min_docs_per_domain=2)) == 1


def test_split_support_and_eval_only_domains(domain_tok_dir):
    _, val_stems = split_shards(domain_tok_dir, n_val_shards=1)
    idx = build_domain_index(domain_tok_dir, val_stems)

    support, eval_docs = split_support_and_eval(idx, k=2, min_docs=2, seed=0,
                                                only_domains=["big1.example.com"])
    assert set(support) == {"big1.example.com"}
    assert len(support["big1.example.com"]) == 2
    assert len(eval_docs["big1.example.com"]) == 13
    # "big2.example.com" was never asked for
    assert "big2.example.com" not in support


# -- src/eval/conditioning.py -------------------------------------------------

def test_evaluate_final_none_variant(domain_tok_dir, tokenizer):
    model = ConditionedLM(make_lm(len(tokenizer)), "none")
    model.eval()
    _, val_stems = split_shards(domain_tok_dir, n_val_shards=1)
    results = evaluate_final(model, tokenizer, domain_tok_dir, val_stems, EVAL_CFG)

    assert results["overall"]["none"] is not None
    assert all(results["overall"][m] is None for m in ("self", "same_domain", "domain_mean", "cross_domain"))

    td = results["top_domains"]
    assert td is not None
    assert td["n_domains"] == 2
    assert td["aggregate"]["none"] is not None
    assert all(td["aggregate"][m] is None for m in ("self", "same_domain", "domain_mean", "cross_domain"))
    assert set(td["per_domain"]) == {"big1.example.com", "big2.example.com"}
    for modes in td["per_domain"].values():
        assert modes["none"] is not None
        assert all(modes[m] is None for m in ("self", "same_domain", "domain_mean", "cross_domain"))


def test_evaluate_final_meco_variant(domain_tok_dir, tokenizer):
    model = ConditionedLM(make_lm(len(tokenizer)), "meco")
    model.eval()
    _, val_stems = split_shards(domain_tok_dir, n_val_shards=1)
    results = evaluate_final(model, tokenizer, domain_tok_dir, val_stems, EVAL_CFG)

    assert results["overall"]["none"] is not None
    assert results["overall"]["self"] is not None
    assert results["overall"]["same_domain"] is None

    td = results["top_domains"]
    assert td["aggregate"]["none"] is not None
    assert td["aggregate"]["self"] is not None
    assert td["aggregate"]["same_domain"] is None
    assert td["aggregate"]["domain_mean"] is None
    for modes in td["per_domain"].values():
        assert modes["none"] is not None
        assert modes["self"] is not None
        assert modes["same_domain"] is None


def test_evaluate_final_text_variant(domain_tok_dir, tokenizer):
    conditioner = make_conditioner(len(tokenizer))
    model = ConditionedLM(make_lm(len(tokenizer)), "text", conditioner)
    model.eval()
    _, val_stems = split_shards(domain_tok_dir, n_val_shards=1)

    results = evaluate_final(model, tokenizer, domain_tok_dir, val_stems, EVAL_CFG)

    assert results["overall"]["none"] is not None
    assert results["overall"]["self"] is not None
    # same_domain/domain_mean/cross_domain are inherently top-domain-only
    assert results["overall"]["same_domain"] is None
    assert results["overall"]["domain_mean"] is None
    assert results["overall"]["cross_domain"] is None

    td = results["top_domains"]
    assert td["n_domains"] == 2
    assert {d["domain"] for d in td["domains"]} == {"big1.example.com", "big2.example.com"}
    for mode in ("none", "self", "same_domain", "domain_mean", "cross_domain"):
        assert td["aggregate"][mode] is not None

    assert set(td["per_domain"]) == {"big1.example.com", "big2.example.com"}
    for modes in td["per_domain"].values():
        for mode in ("none", "self", "same_domain", "domain_mean", "cross_domain"):
            assert modes[mode] is not None


def test_evaluate_final_top_domains_aggregate_is_token_weighted(domain_tok_dir, tokenizer):
    """The aggregate reported for "none"/"self" must be the token-weighted
    combination of the per-domain results (not e.g. an unweighted mean of
    per-domain PPLs), recomputed here independently via the same building
    blocks (_score/_combine) evaluate_final itself uses, but on freshly
    recovered per-domain document sets."""
    conditioner = make_conditioner(len(tokenizer))
    model = ConditionedLM(make_lm(len(tokenizer)), "text", conditioner)
    model.eval()
    _, val_stems = split_shards(domain_tok_dir, n_val_shards=1)

    results = evaluate_final(model, tokenizer, domain_tok_dir, val_stems, EVAL_CFG)
    td = results["top_domains"]

    idx = build_domain_index(domain_tok_dir, val_stems)
    chosen = top_domains(idx, EVAL_CFG["n_domains"], EVAL_CFG["domain_support_k"],
                         EVAL_CFG["min_docs_per_domain"])
    _, eval_docs = split_support_and_eval(idx, EVAL_CFG["domain_support_k"], EVAL_CFG["min_docs_per_domain"],
                                          EVAL_CFG["seed"], only_domains=[d for d, _ in chosen])

    for mode, keep_mask_value in (("none", False), ("self", True)):
        scored = []
        for domain, _ in chosen:
            docs = [doc_dict(sh, r) for sh, r in eval_docs[domain]]
            scored.append(_score(model, docs, tokenizer, "text", EVAL_CFG["block_size"],
                                 EVAL_CFG["meco_max_tokens"], EVAL_CFG["encoder_max_tokens"],
                                 EVAL_CFG["val_batch_size"], keep_mask_value=keep_mask_value))
        expected = _combine(scored)
        assert td["aggregate"][mode] == pytest.approx(expected)


def test_evaluate_final_no_domain_qualifies(tok_dir, tokenizer):
    """tests/conftest.py's tiny shared fixture only has 1-2 docs per domain --
    no domain can support even the default k/min_docs split."""
    model = ConditionedLM(make_lm(len(tokenizer)), "none")
    model.eval()
    _, val_stems = split_shards(tok_dir, n_val_shards=1)
    cfg = dict(EVAL_CFG)
    cfg["min_docs_per_domain"] = 5
    results = evaluate_final(model, tokenizer, tok_dir, val_stems, cfg)
    assert results["overall"]["none"] is not None
    assert results["top_domains"] is None


# -- src/analysis/tables.py ---------------------------------------------------

def _fake_baseline_results():
    overall = {"none": (1.0, 2.718), "self": None, "same_domain": None, "domain_mean": None,
              "cross_domain": None}
    top_domains_data = {
        "n_domains": 1, "k": 2, "min_docs_per_domain": 2,
        "domains": [{"domain": "big1.example.com", "count": 15, "n_support": 2, "n_eval": 13}],
        "aggregate": {"none": (0.9, 2.46), "self": None, "same_domain": None, "domain_mean": None,
                     "cross_domain": None},
        "per_domain": {"big1.example.com": {"none": (0.9, 2.46), "self": None, "same_domain": None,
                                            "domain_mean": None, "cross_domain": None}},
    }
    return {"overall": overall, "top_domains": top_domains_data}


def _fake_text_results():
    overall = {"none": (1.0, 2.718), "self": (0.5, 1.6487), "same_domain": None, "domain_mean": None,
              "cross_domain": None}
    top_domains_data = {
        "n_domains": 1, "k": 2, "min_docs_per_domain": 2,
        "domains": [{"domain": "big1.example.com", "count": 15, "n_support": 2, "n_eval": 13}],
        "aggregate": {"none": (0.9, 2.46), "self": (0.4, 1.49), "same_domain": (0.6, 1.82),
                     "domain_mean": (0.7, 2.01), "cross_domain": (1.2, 3.32)},
        "per_domain": {"big1.example.com": {"none": (0.9, 2.46), "self": (0.4, 1.49),
                                            "same_domain": (0.6, 1.82), "domain_mean": (0.7, 2.01),
                                            "cross_domain": (1.2, 3.32)}},
    }
    return {"overall": overall, "top_domains": top_domains_data}


def test_load_final_results_round_trip(tmp_path):
    path = tmp_path / "final_conditioning.json"
    with open(path, "w") as f:
        json.dump(_fake_text_results(), f)
    loaded = load_final_results(str(path))
    assert loaded["overall"]["none"] == (1.0, 2.718)
    assert loaded["top_domains"]["aggregate"]["same_domain"] == (0.6, 1.82)
    assert loaded["top_domains"]["per_domain"]["big1.example.com"]["domain_mean"] == (0.7, 2.01)


def test_build_table_rows_no_top_domains():
    results = {"overall": {"none": (1.0, 2.718), "self": None, "same_domain": None, "domain_mean": None,
                           "cross_domain": None},
              "top_domains": None}
    rows = build_table_rows({"Baseline": results})
    assert rows == [["Baseline", "2.72", "—", "—", "—", "—", "—"]]


def test_build_table_rows_baseline_missing_modes_are_dash():
    rows = build_table_rows({"Baseline": _fake_baseline_results()})
    assert rows == [["Baseline", "2.72", "2.46", "—", "—", "—", "—"]]


def test_build_table_rows_text_variant():
    rows = build_table_rows({"TextCond": _fake_text_results()})
    assert rows == [["TextCond", "2.72", "2.46", "1.65", "1.49", "1.82", "2.01"]]


def test_build_domain_rows_and_csv(tmp_path):
    rows = build_domain_rows({"TextCond": _fake_text_results()})
    assert len(rows) == 5
    assert {r["mode"] for r in rows} == {"none", "self", "same_domain", "domain_mean", "cross_domain"}
    assert all(r["domain"] == "big1.example.com" for r in rows)

    path = tmp_path / "domain_breakdown.csv"
    write_domain_csv(rows, str(path))
    assert path.exists()


def test_build_domain_rows_skips_models_without_top_domains():
    results = {"overall": {m: None for m in ("none", "self", "same_domain", "domain_mean", "cross_domain")},
              "top_domains": None}
    assert build_domain_rows({"NoDomains": results}) == []
