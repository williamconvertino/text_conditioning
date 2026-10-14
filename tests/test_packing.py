"""Packed training batches (src/data/packing.py, `packing: true`): rows are
tiled by document segments, every segment carries its document's
conditioning, the packed forward (varlen / segment-masked attention) equals
running each segment on its own, pretokenized metadata matches live
tokenization, and a saved cursor resumes the exact stream."""

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from src.data.metadata_tokens import tokenize_shard_metadata
from src.data.packing import IGNORE, PackedBatcher
from src.data.shards import discover_shard_stems, split_shards
from src.models.conditioned_lm import ConditionedLM
from tests.test_eval_conditioning import make_conditioner, make_lm

T = 16


def batcher(tok_dir, tokenizer, mode, cursor=None, p=None, rows=4, micro=2):
    train, _ = split_shards(tok_dir, 1)
    return PackedBatcher(tok_dir, train, tokenizer, mode, T, rows_per_step=rows, micro_rows=micro, seed=0,
                         meco_max_tokens=8, url_max_tokens=12, p_condition_fn=(lambda: p) if p else None,
                         cursor=cursor)


def segments(b, with_pad=False):
    """[(row, start, length)] of a micro-batch from its cu_seqlens (padding
    tails, which have no loss tokens, only if with_pad)."""
    cu = b["cu_seqlens"].tolist()
    segs = [(s // T, s % T, e - s) for s, e in zip(cu[:-1], cu[1:])]
    return [x for x in segs if with_pad or (b["labels"][x[0], x[1]:x[1] + x[2]] != IGNORE).any()]


@pytest.mark.parametrize("mode", ["none", "meco", "url"])
def test_rows_tiled_by_conditioned_segments(tok_dir, tokenizer, mode):
    micro, _ = next(iter(batcher(tok_dir, tokenizer, mode)))
    sep = tokenizer.encode("\n\n", add_special_tokens=False)
    for b in micro:
        assert b["input_ids"].shape == (2, T)
        segs = segments(b)
        assert all(0 <= s and s + n <= T for _, s, n in segs)
        for r, s, n in segments(b, with_pad=True):
            assert b["position_ids"][r, s:s + n].tolist() == list(range(n))
            assert b["labels"][r, s + n - 1] == IGNORE            # never predicts across segments
        if mode == "none":
            # every position but each segment's last is trained, on the next token
            for r, s, n in segs:
                assert b["labels"][r, s:s + n - 1].tolist() == b["input_ids"][r, s + 1:s + n].tolist()
        if mode == "meco":
            for r, s, n in segs:
                ids = b["input_ids"][r, s:s + n].tolist()
                k = next(i for i in range(len(ids) - 1) if ids[i:i + len(sep)] == sep) + len(sep)
                assert (b["labels"][r, s:s + k - 1] == IGNORE).all()   # prefix tokens never targets
                assert b["labels"][r, s + k - 1] == ids[k]             # "\n\n" predicts the document
        if mode == "url":
            assert b["slot_index"].tolist() == [r * T + s for r, s, n in segs]
            assert b["encoder"]["has_condition"].all()
            assert b["n_loss_tokens"] == int((b["labels"] != IGNORE).sum())


def test_split_documents_repeat_their_prefix(tok_dir, tokenizer):
    micro, _ = next(iter(batcher(tok_dir, tokenizer, "meco")))
    starts = [b["input_ids"][r, s:s + 3].tolist() for b in micro for r, s, n in segments(b) if n > 3]
    # 2 train shards x 2 docs: every segment starts with one of those 4 domains' prefixes
    assert len(starts) > 4 and len({tuple(x) for x in starts}) <= 4


def test_condition_dropout_is_deterministic(tok_dir, tokenizer):
    a = [m for m, _ in [next(iter(batcher(tok_dir, tokenizer, "url", p=0.5, rows=8)))]][0]
    b = [m for m, _ in [next(iter(batcher(tok_dir, tokenizer, "url", p=0.5, rows=8)))]][0]
    has = torch.cat([x["encoder"]["has_condition"] for x in a])
    assert torch.equal(has, torch.cat([x["encoder"]["has_condition"] for x in b]))
    assert 0 < has.float().mean() < 1


@pytest.mark.parametrize("mode", ["none", "meco", "url"])
def test_resume_from_cursor(tok_dir, tokenizer, mode):
    it = iter(batcher(tok_dir, tokenizer, mode, p=0.7))
    steps = [next(it) for _ in range(5)]
    resumed = iter(batcher(tok_dir, tokenizer, mode, p=0.7, cursor=steps[2][1]))
    for (micro, _), (ref, _) in zip([next(resumed), next(resumed)], steps[3:]):
        for a, b in zip(micro, ref):
            assert torch.equal(a["input_ids"], b["input_ids"]) and torch.equal(a["labels"], b["labels"])
            assert torch.equal(a["cu_seqlens"], b["cu_seqlens"])


@pytest.mark.parametrize("mode", ["meco", "url"])
def test_pretokenized_metadata_matches_live(tok_dir, tokenizer, mode):
    live = next(iter(batcher(tok_dir, tokenizer, mode)))[0]
    for stem in discover_shard_stems(tok_dir):
        tokenize_shard_metadata(tok_dir, stem, tokenizer)
    side = next(iter(batcher(tok_dir, tokenizer, mode)))[0]
    for a, b in zip(live, side):
        assert torch.equal(a["input_ids"], b["input_ids"])
        if mode == "url":
            assert torch.equal(a["encoder"]["input_ids"], b["encoder"]["input_ids"])


@pytest.mark.parametrize("variant,mode,strong", [("none", "none", False), ("meco", "meco", False),
                                                 ("text", "url", False), ("text", "url", True)])
def test_packed_forward_equals_per_segment(tok_dir, tokenizer, variant, mode, strong):
    torch.manual_seed(0)
    cond = make_conditioner(len(tokenizer)) if variant == "text" else None
    lm = make_lm(len(tokenizer))
    lm.config.pinned_condition = strong     # per segment: the unpacked path pins position 0 itself
    model = ConditionedLM(lm, variant, cond).eval()
    batch = next(iter(batcher(tok_dir, tokenizer, mode)))[0][0]
    with torch.no_grad():
        loss, n_tok, z = model(batch)
        emb = model.lm.get_input_embeddings()(batch["input_ids"])
        if variant == "text":
            flat = emb.reshape(-1, emb.shape[-1]).clone()
            flat[batch["slot_index"]] = z
            emb = flat.view_as(emb)
        nll, n = 0.0, 0
        for r, s, k in segments(batch, with_pad=True):            # each segment alone, through the unpacked path
            logits = model.lm(inputs_embeds=emb[r:r + 1, s:s + k]).logits[0]
            lab = batch["labels"][r, s:s + k]
            nll += F.cross_entropy(logits.float(), lab, ignore_index=IGNORE, reduction="sum").item()
            n += int((lab != IGNORE).sum())
    assert int(n_tok) == n == batch["n_loss_tokens"]
    assert float(loss) == pytest.approx(nll / n, rel=1e-5)


def _train_cfg(tok_dir, tmp_path, variant, max_steps):
    import yaml
    enc = tmp_path / "enc.yaml"
    enc.write_text(yaml.safe_dump({"hidden_size": 16, "intermediate_size": 32, "num_hidden_layers": 1,
                                   "num_attention_heads": 2, "num_key_value_heads": 2,
                                   "max_position_embeddings": 32, "rms_norm_eps": 1e-5, "rope_theta": 1e4,
                                   "initializer_range": 0.02}))
    cfg = {"name": f"pack_{variant}", "variant": variant, "tok_dir": tok_dir, "tokenizer": "gpt2",
           "n_val_shards": 1, "val_docs": 2, "val_batch_size": 2, "block_size": T, "meco_max_tokens": 8,
           "encoder_max_tokens": 16, "local_batch": 2, "global_batch_seqs": 4, "grad_accum": 1,
           "packing": True, "lr": 1e-3, "min_lr": 1e-4, "warmup_steps": 1, "max_steps": max_steps,
           "weight_decay": 0.1, "grad_clip": 1.0, "seed": 0, "dtype": "float32", "log_interval": 1,
           "val_interval": 2, "ckpt_save_interval_tokens": 10 ** 9, "hidden_size": 32,
           "intermediate_size": 64, "num_hidden_layers": 2, "num_attention_heads": 4,
           "num_key_value_heads": 2, "max_position_embeddings": T + 1, "rms_norm_eps": 1e-5,
           "rope_theta": 1e4, "initializer_range": 0.02, "save_best_checkpoint": False,
           "conditioning_schedule": {"kind": "constant", "p_condition": 0.7}}
    if variant.startswith("text"):
        cfg.update(variant="text", text_encoder_config=str(enc), text_conditioning_source="url",
                   url_conditioning={"field": "url", "max_tokens": 12},
                   strong_conditioning=variant == "text_strong")
    return cfg


@pytest.mark.parametrize("variant", ["none", "meco", "text", "text_strong"])
def test_packed_training_resumes_exactly(tok_dir, tmp_path, monkeypatch, variant):
    import json
    from src.training.trainer import run

    def losses(run_dir):
        recs = [json.loads(l) for l in open(run_dir / "metrics.jsonl")]
        return {r["step"]: r["loss"] for r in recs if r["split"] == "train"}

    full, split = tmp_path / "full", tmp_path / "split"
    run(_train_cfg(tok_dir, tmp_path, variant, 6), str(full))
    monkeypatch.setenv("TRAIN_STOP_AT", "1")                       # deadline passed: stop after step 1
    run(_train_cfg(tok_dir, tmp_path, variant, 6), str(split))
    monkeypatch.delenv("TRAIN_STOP_AT")
    run(_train_cfg(tok_dir, tmp_path, variant, 6), str(split))     # resumes from step 1
    a, b = losses(full), losses(split)
    assert sorted(a) == sorted(b) == list(range(1, 7))
    for k in range(2, 7):
        assert b[k] == pytest.approx(a[k], rel=1e-5)
    ckpt = torch.load(split / "ckpt_last.pt", weights_only=False)
    assert ckpt["data_cursor"][0][0] > 0
    assert ckpt["run_cfg"].get("strong_conditioning", False) == (variant == "text_strong")


def _layer_inputs(lm, run):
    """Hidden states entering each decoder layer's attention (its input_layernorm)."""
    seen = []
    hooks = [layer.input_layernorm.register_forward_hook(lambda m, i, o: seen.append(i[0].detach().clone()))
             for layer in lm.model.layers]
    try:
        run()
    finally:
        for h in hooks:
            h.remove()
    return seen


def test_strong_conditioning_pins_the_slot_at_every_layer(tokenizer):
    torch.manual_seed(0)
    lm = make_lm(len(tokenizer)).eval()
    x = torch.randn(2, 6, lm.config.hidden_size)
    with torch.no_grad():
        weak = lm(inputs_embeds=x).logits
        lm.config.pinned_condition = True
        ins = _layer_inputs(lm, lambda: lm(inputs_embeds=x))
        strong = lm(inputs_embeds=x).logits
        tokens = _layer_inputs(lm, lambda: lm(input_ids=torch.ones(2, 6, dtype=torch.long)))
    assert all(torch.allclose(h[:, 0], x[:, 0]) for h in ins)          # z itself at every layer
    assert not torch.allclose(ins[1][:, 1:], x[:, 1:])                 # other positions evolve
    assert not torch.allclose(strong[:, 1:], weak[:, 1:])              # the documents see z deeper
    assert not torch.allclose(tokens[1][:, 0], tokens[0][:, 0])        # input_ids calls pin nothing
