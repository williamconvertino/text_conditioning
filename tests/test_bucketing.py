"""Length-bucketed training steps (src/data/collator.py's
bucketed_step_batches, `length_bucketing: true`): same documents as the
padded path, micro-batches within the token budget, and a step loss equal to
the token-weighted mean over all of the step's documents."""

import pytest
import torch

from src.data.collator import (_length_groups, bucketed_step_batches, collate_batch, count_loss_tokens,
                               doc_dict, train_batches)
from src.data.shards import TokenizedDocStream, split_shards
from src.models.conditioned_lm import ConditionedLM
from tests.test_eval_conditioning import make_conditioner, make_lm


def test_length_groups_budget_and_split():
    lengths = [5, 30, 12, 7, 30, 2, 18, 9]
    groups = _length_groups(lengths, tokens_per_micro=40)
    assert sorted(i for g in groups for i in g) == list(range(8))
    assert all(len(g) * max(lengths[i] for i in g) <= 40 for g in groups)
    more = _length_groups(lengths, 40, n_groups=len(groups) + 2)
    assert len(more) == len(groups) + 2 and sorted(i for g in more for i in g) == list(range(8))
    with pytest.raises(ValueError):
        _length_groups([1, 1], 40, n_groups=3)


@pytest.mark.parametrize("variant", ["none", "meco", "text"])
def test_bucketed_steps_match_padded_steps(tok_dir, tokenizer, variant):
    train, _ = split_shards(tok_dir, 1)
    kw = dict(block_size=64, meco_max_tokens=8, encoder_max_tokens=16)
    padded = train_batches(TokenizedDocStream(tok_dir, train, 0), tokenizer, variant, kw["block_size"],
                           kw["meco_max_tokens"], kw["encoder_max_tokens"], local_batch=2)
    bucketed = bucketed_step_batches(TokenizedDocStream(tok_dir, train, 0), tokenizer, variant, docs_per_step=4,
                                     tokens_per_micro=2 * 64, **kw)
    torch.manual_seed(0)
    cond = make_conditioner(len(tokenizer)) if variant == "text" else None
    model = ConditionedLM(make_lm(len(tokenizer)), variant, cond).eval()
    model.lm.config.max_position_embeddings = 65
    for _ in range(2):
        ref = [next(padded), next(padded)]
        micro = next(bucketed)
        # the same documents (as multisets of token sequences)
        seqs = lambda bs: sorted(tuple(b["input_ids"][i, :b["lengths"][i]].tolist())
                                 for b in bs for i in range(len(b["lengths"])))
        assert seqs(micro) == seqs(ref)
        assert all(b["input_ids"].shape[1] == int(b["lengths"].max()) for b in micro)
        assert all(b["input_ids"].numel() <= 2 * 64 for b in micro)
        # token-weighted step loss == one loss over all the step's tokens
        with torch.no_grad():
            outs = [model(b) for b in micro]
            n = [count_loss_tokens(b, variant) for b in micro]
            assert [int(o[1]) for o in outs] == n
            step_loss = sum(float(o[0]) * k for o, k in zip(outs, n)) / sum(n)
            ref_outs = [model(b) for b in ref]
            ref_loss = sum(float(o[0]) * float(o[1]) for o in ref_outs) / sum(float(o[1]) for o in ref_outs)
        assert step_loss == pytest.approx(ref_loss, rel=1e-4)
