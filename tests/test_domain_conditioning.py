"""Domain-conditioned training (README "Domain-Conditioned Training"): the
URL-domain index, in-domain partner sampling, partner-encoder collation, the
conditioner's per-row partner averaging, and the plotting hooks for the
text_conditioned_domain_* experiment families."""

import json
import os

import pytest
import torch

from src.analysis.figures import CONDITIONED_PREFIXES, display_name, plot_conditioning_curve
from src.data.collator import (
    collate_batch, collate_partner_encoder_batch, doc_dict, load_val_batches, train_batches,
)
from src.data.domain_index import DomainIndex, DomainPartnerSampler
from src.data.shards import Shard, TokenizedDocStream
from src.models.conditioned_lm import ConditionedLM
from src.models.conditioning import TextConditioner
from tests.conftest import _write_shard
from tests.test_conditioning import HIDDEN, make_conditioner, make_lm

# shard_a: 3 x a.com, 1 x solo.com (singleton), 1 x no URL
# shard_b: 2 x a.com, 2 x b.com   -> a.com spans both shards (5 docs)
SHARD_A = [("https://a.com/1", "Alpha one."), ("https://a.com/2", "Alpha two."),
           ("https://www.a.com/3", "Alpha three."), ("https://solo.com/x", "Only one here."),
           ("", "No url at all.")]
SHARD_B = [("https://a.com/4", "Alpha four."), ("https://b.com/1", "Beta one."),
           ("https://a.com/5", "Alpha five."), ("https://b.com/2", "Beta two.")]


@pytest.fixture()
def partner_tok_dir(tmp_path, tokenizer):
    out_dir = str(tmp_path / "tok")
    os.makedirs(out_dir)
    for stem, rows in (("shard_a", SHARD_A), ("shard_b", SHARD_B)):
        _write_shard(out_dir, stem, [t for _, t in rows], [u for u, _ in rows], tokenizer)
    return out_dir


def _row_domain(index, stem, row):
    d = int(index.doc_domain[index.doc_id(stem, row)])
    return None if d < 0 else index.domain_names[d]


# -- src/data/domain_index.py ------------------------------------------------

def test_domain_index_groups_across_shards(partner_tok_dir):
    index = DomainIndex.build(partner_tok_dir, ["shard_a", "shard_b"])
    assert index.n_docs == 9
    assert index.meta["n_unknown"] == 1
    assert sorted(index.domain_names) == ["a.com", "b.com", "solo.com"]
    a_members = index.domain_members(index.doc_id("shard_a", 0))
    assert sorted(a_members.tolist()) == [0, 1, 2, 5, 7]
    assert len(index.domain_members(index.doc_id("shard_a", 4))) == 0   # no URL
    assert _row_domain(index, "shard_a", 2) == "a.com"                  # www. stripped
    assert index.coverage(2) == pytest.approx(7 / 9)


def test_domain_index_doc_offsets_match_shards(partner_tok_dir):
    index = DomainIndex.build(partner_tok_dir, ["shard_a", "shard_b"])
    shard = Shard(partner_tok_dir, "shard_b")
    for row in range(len(shard)):
        gid = index.doc_id("shard_b", row)
        assert index.doc_start[gid] == shard.start[row]
        assert index.doc_length[gid] == shard.length[row]


def test_domain_index_save_load_round_trip(partner_tok_dir, tmp_path):
    built = DomainIndex.build(partner_tok_dir, ["shard_a", "shard_b"], num_workers=2)
    out = str(tmp_path / "index")
    built.save(out)
    loaded = DomainIndex.load(out, expected_stems=["shard_a", "shard_b"])
    for name in ("shard_offsets", "doc_start", "doc_length", "doc_domain", "domain_order", "domain_offsets"):
        assert (getattr(loaded, name) == getattr(built, name)).all()
    with pytest.raises(ValueError):
        DomainIndex.load(out, expected_stems=["shard_a"])
    with pytest.raises(FileNotFoundError):
        DomainIndex.load(str(tmp_path / "missing"))


def test_partner_sampler_domain_random(partner_tok_dir):
    index = DomainIndex.build(partner_tok_dir, ["shard_a", "shard_b"])
    sampler = DomainPartnerSampler(index, "domain_random", seed=0)
    for stem, n in (("shard_a", len(SHARD_A)), ("shard_b", len(SHARD_B))):
        for row in range(n):
            ids = sampler.partner_ids(stem, row)
            gid = index.doc_id(stem, row)
            if _row_domain(index, stem, row) in (None, "solo.com"):
                assert ids == []
                continue
            assert len(ids) == 1 and ids[0] != gid
            assert index.doc_domain[ids[0]] == index.doc_domain[gid]
    # deterministic in (seed, doc id), so resumed runs replay the same partners
    assert sampler.partner_ids("shard_a", 0) == DomainPartnerSampler(index, "domain_random", seed=0) \
        .partner_ids("shard_a", 0)


def test_partner_sampler_domain_mean_caps_at_domain_size(partner_tok_dir, tokenizer):
    index = DomainIndex.build(partner_tok_dir, ["shard_a", "shard_b"])
    sampler = DomainPartnerSampler(index, "domain_mean", n_support=8, seed=0)
    ids = sampler.partner_ids("shard_a", 0)
    assert sorted(ids) == [1, 2, 5, 7]            # every other a.com doc, never itself
    assert len(sampler.partner_ids("shard_b", 1)) == 1   # b.com has one other doc
    small = DomainPartnerSampler(index, "domain_mean", n_support=2, seed=0)
    assert len(small.partner_ids("shard_a", 0)) == 2

    # partner docs carry the right tokens
    shard_b = Shard(partner_tok_dir, "shard_b")
    [partner] = sampler.partners("shard_b", 1)
    assert list(partner["tokens"]) == list(shard_b.doc_tokens(3))


def test_partner_sampler_min_domain_docs(partner_tok_dir):
    index = DomainIndex.build(partner_tok_dir, ["shard_a", "shard_b"])
    sampler = DomainPartnerSampler(index, "domain_mean", min_domain_docs=3, seed=0)
    assert sampler.partner_ids("shard_b", 1) == []      # b.com: only 2 docs
    assert len(sampler.partner_ids("shard_b", 0)) == 4  # a.com: 5 docs


# -- src/data/collator.py ----------------------------------------------------

def test_collate_partner_encoder_batch(partner_tok_dir, tokenizer):
    index = DomainIndex.build(partner_tok_dir, ["shard_a", "shard_b"])
    sampler = DomainPartnerSampler(index, "domain_mean", seed=0)
    partners = [sampler.partners("shard_a", r) for r in range(len(SHARD_A))]
    enc = collate_partner_encoder_batch(partners, tokenizer, encoder_max_tokens=16)
    assert enc["has_condition"].tolist() == [True, True, True, False, False]
    assert enc["input_ids"].shape[0] == sum(len(p) for p in partners) == 12
    assert enc["group_index"].tolist() == [0] * 4 + [1] * 4 + [2] * 4


def test_collate_partner_encoder_batch_placeholder_when_empty(tokenizer):
    enc = collate_partner_encoder_batch([[], []], tokenizer, encoder_max_tokens=16)
    assert enc["input_ids"].shape[0] == 1
    assert enc["group_index"].tolist() == [0]
    assert enc["has_condition"].tolist() == [False, False]


def test_train_batches_with_partner_sampler(partner_tok_dir, tokenizer):
    index = DomainIndex.build(partner_tok_dir, ["shard_a", "shard_b"])
    sampler = DomainPartnerSampler(index, "domain_random", seed=0)
    stream = TokenizedDocStream(partner_tok_dir, ["shard_a", "shard_b"], seed=0)
    batch = next(train_batches(stream, tokenizer, "text", 32, 8, 16, local_batch=9,
                               condition_source=sampler))
    assert batch["encoder"]["has_condition"].sum().item() == 7   # all but solo.com + no-URL
    assert batch["encoder"]["input_ids"].shape[0] == 7

    # a schedule-dropped row gets no partners at all (=> null vector)
    dropped = next(train_batches(stream, tokenizer, "text", 32, 8, 16, local_batch=9,
                                 keep_mask_fn=lambda: [False] * 9, condition_source=sampler))
    assert not dropped["encoder"]["has_condition"].any()


def test_load_val_batches_with_partner_sampler(partner_tok_dir, tokenizer):
    index = DomainIndex.build(partner_tok_dir, ["shard_a"])
    sampler = DomainPartnerSampler(index, "domain_mean", seed=0)
    batches = load_val_batches(partner_tok_dir, ["shard_a"], tokenizer, "text", 32, 8, 16,
                               val_docs=None, val_batch_size=2, condition_source=sampler)
    assert [b["encoder"]["has_condition"].tolist() for b in batches] == \
        [[True, True], [True, False], [False]]
    self_batches = load_val_batches(partner_tok_dir, ["shard_a"], tokenizer, "text", 32, 8, 16,
                                    val_docs=None, val_batch_size=2)
    assert "group_index" not in self_batches[0]["encoder"]


# -- src/models/conditioning.py / conditioned_lm.py ---------------------------

def test_group_mean():
    z = torch.tensor([[1.0, 1.0], [3.0, 3.0], [10.0, 0.0]])
    out = TextConditioner.group_mean(z, torch.tensor([0, 0, 2]), batch_size=3)
    assert torch.allclose(out, torch.tensor([[2.0, 2.0], [0.0, 0.0], [10.0, 0.0]]))


def _partner_batch(partner_tok_dir, tokenizer, source="domain_mean"):
    index = DomainIndex.build(partner_tok_dir, ["shard_a", "shard_b"])
    sampler = DomainPartnerSampler(index, source, seed=0)
    shard = Shard(partner_tok_dir, "shard_a")
    rows = list(range(len(shard)))
    docs = [doc_dict(shard, r) for r in rows]
    partners = [sampler.partners("shard_a", r) for r in rows]
    return docs, partners, collate_batch(docs, tokenizer, "text", 32, 8, 16, partners=partners)


def test_domain_conditioned_forward_and_backward(partner_tok_dir, tokenizer):
    torch.manual_seed(0)
    conditioner = make_conditioner(len(tokenizer))
    model = ConditionedLM(make_lm(len(tokenizer)), "text", conditioner)
    _, partners, batch = _partner_batch(partner_tok_dir, tokenizer)

    loss, n_tok, z = model(batch)
    assert z.shape == (len(partners), HIDDEN)
    assert torch.isfinite(loss) and n_tok.item() > 0
    # rows without partners (solo.com, no URL) got the null vector
    null = conditioner.null_vector.detach()
    assert torch.allclose(z[3], null) and torch.allclose(z[4], null)
    assert not torch.allclose(z[0], null)
    loss.backward()
    assert conditioner.encoder.embed_tokens.weight.grad is not None
    assert conditioner.null_vector.grad is not None


def test_domain_conditioned_z_is_mean_of_partner_encodings(partner_tok_dir, tokenizer):
    torch.manual_seed(0)
    conditioner = make_conditioner(len(tokenizer)).eval()
    model = ConditionedLM(make_lm(len(tokenizer)), "text", conditioner).eval()
    _, partners, batch = _partner_batch(partner_tok_dir, tokenizer)
    with torch.no_grad():
        _, _, z = model(batch)
        own = collate_batch(partners[0], tokenizer, "text", 32, 8, 16)["encoder"]
        expected = conditioner.encoder(input_ids=own["input_ids"],
                                       attention_mask=own["attention_mask"]).mean(dim=0)
    assert torch.allclose(z[0], expected, atol=1e-5)


def test_domain_conditioned_keep_mask_false_matches_null(partner_tok_dir, tokenizer):
    """keep_mask all-False (the unconditioned val metric) must null out every
    row, partners or not -- same loss as a batch nobody has partners for."""
    torch.manual_seed(0)
    model = ConditionedLM(make_lm(len(tokenizer)), "text", make_conditioner(len(tokenizer))).eval()
    docs, partners, batch = _partner_batch(partner_tok_dir, tokenizer)
    empty = collate_batch(docs, tokenizer, "text", 32, 8, 16, partners=[[] for _ in docs])
    off = torch.zeros(len(docs), dtype=torch.bool)
    with torch.no_grad():
        a, _, _ = model(batch, keep_mask=off)
        b, _, _ = model(empty)
    assert torch.allclose(a, b)


# -- src/analysis/figures.py -------------------------------------------------

def test_domain_families_have_plot_metadata():
    for name in ("text_conditioned_domain_mean", "text_conditioned_domain_random"):
        assert CONDITIONED_PREFIXES[name][0] == "domain_conditioned"
        assert display_name(f"{name}_medium") != f"{name}_medium"


def test_plot_conditioning_curve_domain_run(tmp_path):
    run_dir = tmp_path / "text_conditioned_domain_mean_medium"
    run_dir.mkdir()
    with open(run_dir / "metrics.jsonl", "w") as f:
        for i, step in enumerate((500, 1000)):
            f.write(json.dumps({
                "step": step, "split": "val", "tokens": step * 1e6, "loss": 3.0 - 0.1 * i,
                "ppl": 20.0 - i, "domain_conditioned_loss": 3.0 - 0.1 * i,
                "domain_conditioned_ppl": 20.0 - i, "self_conditioned_loss": 2.8 - 0.1 * i,
                "self_conditioned_ppl": 16.0 - i, "unconditioned_loss": 3.2 - 0.1 * i,
                "unconditioned_ppl": 24.0 - i}) + "\n")
    out = tmp_path / "figs"
    plot_conditioning_curve(str(run_dir), str(out), metric="ppl")
    assert (out / "text_domain_mean_conditioning_ppl.png").exists()
