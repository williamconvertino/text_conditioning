import os

import torch

from src.data.collator import collate_batch, doc_dict, load_val_batches, train_batches
from src.data.metadata import url_to_domain
from src.data.shards import TokenizedDocStream, iter_shard_docs, load_manifest, rank_shards, split_shards


def test_url_to_domain():
    assert url_to_domain("https://en.wikipedia.org/wiki/X") == "en.wikipedia.org"
    assert url_to_domain("https://www.example.com/a/b") == "example.com"
    assert url_to_domain("") == "unknown"


def test_manifest_and_split(tok_dir):
    manifest = load_manifest(tok_dir)
    assert manifest["tokenizer"] == "gpt2"
    train_stems, val_stems = split_shards(tok_dir, n_val_shards=1)
    assert val_stems == ["shard_02_val"]
    assert train_stems == ["shard_00", "shard_01"]
    my_stems = rank_shards(train_stems, rank=0, world_size=2)
    assert my_stems == ["shard_00"]


def test_doc_stream_yields_all_docs_per_epoch(tok_dir):
    stream = TokenizedDocStream(tok_dir, ["shard_00", "shard_01"], seed=0)
    it = iter(stream)
    seen = set()
    for _ in range(4):  # exactly one epoch (2 docs/shard * 2 shards)
        shard, row = next(it)
        seen.add((shard.stem, row))
    assert len(seen) == 4


def test_resume_skip_docs_lands_on_same_doc(tok_dir):
    stems = ["shard_00", "shard_01"]
    full = TokenizedDocStream(tok_dir, stems, seed=0)
    it = iter(full)
    docs = [next(it) for _ in range(3)]
    resumed = TokenizedDocStream(tok_dir, stems, seed=0, skip_docs=2)
    shard, row = next(iter(resumed))
    assert (shard.stem, row) == (docs[2][0].stem, docs[2][1])


def test_collate_none_variant(tok_dir, tokenizer):
    docs = [doc_dict(shard, row) for shard, row in iter_shard_docs(tok_dir, ["shard_00"])]
    batch = collate_batch(docs, tokenizer, "none", block_size=32, meco_max_tokens=8,
                          encoder_max_tokens=16)
    assert batch["input_ids"].shape == (2, 32)
    assert (batch["text_start"] == 0).all()
    assert "encoder" not in batch


def test_collate_meco_variant_has_nonzero_text_start(tok_dir, tokenizer):
    docs = [doc_dict(shard, row) for shard, row in iter_shard_docs(tok_dir, ["shard_00"])]
    batch = collate_batch(docs, tokenizer, "meco", block_size=32, meco_max_tokens=8,
                          encoder_max_tokens=16)
    assert (batch["text_start"] > 0).all()
    assert (batch["text_start"] < batch["lengths"]).all()


def test_collate_meco_variant_keep_mask_drops_prefix(tok_dir, tokenizer):
    """keep_mask=False rows must come out identical to variant "none" (no
    domain prefix, text_start=0) -- see src/training/schedules.py's
    conditioning_schedule for MeCo."""
    docs = [doc_dict(shard, row) for shard, row in iter_shard_docs(tok_dir, ["shard_00"])]
    conditioned = collate_batch(docs, tokenizer, "meco", block_size=32, meco_max_tokens=8,
                                encoder_max_tokens=16, keep_mask=[True, True])
    dropped = collate_batch(docs, tokenizer, "meco", block_size=32, meco_max_tokens=8,
                            encoder_max_tokens=16, keep_mask=[False, False])
    none_variant = collate_batch(docs, tokenizer, "none", block_size=32, meco_max_tokens=8,
                                 encoder_max_tokens=16)
    assert (conditioned["text_start"] > 0).all()
    assert (dropped["text_start"] == 0).all()
    assert torch.equal(dropped["input_ids"], none_variant["input_ids"])
    assert torch.equal(dropped["lengths"], none_variant["lengths"])


def test_collate_meco_variant_keep_mask_is_per_example(tok_dir, tokenizer):
    docs = [doc_dict(shard, row) for shard, row in iter_shard_docs(tok_dir, ["shard_00"])]
    batch = collate_batch(docs, tokenizer, "meco", block_size=32, meco_max_tokens=8,
                          encoder_max_tokens=16, keep_mask=[True, False])
    assert batch["text_start"][0] > 0
    assert batch["text_start"][1] == 0


def test_collate_text_variant_has_encoder_batch(tok_dir, tokenizer):
    docs = [doc_dict(shard, row) for shard, row in iter_shard_docs(tok_dir, ["shard_00"])]
    batch = collate_batch(docs, tokenizer, "text", block_size=32, meco_max_tokens=8,
                          encoder_max_tokens=16)
    assert (batch["text_start"] == 0).all()
    assert batch["encoder"]["input_ids"].shape == (2, 16)
    assert batch["encoder"]["attention_mask"].sum() > 0


def test_train_batches_generator(tok_dir, tokenizer):
    stream = TokenizedDocStream(tok_dir, ["shard_00", "shard_01"], seed=0)
    gen = train_batches(stream, tokenizer, "none", block_size=32, meco_max_tokens=8,
                        encoder_max_tokens=16, local_batch=3)
    batch = next(gen)
    assert batch["input_ids"].shape[0] == 3
    batch2 = next(gen)
    assert batch2["input_ids"].shape[0] == 3


def test_train_batches_meco_keep_mask_fn(tok_dir, tokenizer):
    """train_batches calls keep_mask_fn() fresh per yielded batch, and only
    for variant "meco" -- src/training/trainer.py's meco_keep_mask_fn closes
    over the current step/schedule this way to implement conditioning_schedule
    for MeCo."""
    stream = TokenizedDocStream(tok_dir, ["shard_00", "shard_01"], seed=0)
    calls = []

    def keep_mask_fn():
        calls.append(1)
        return [False, False, False]

    gen = train_batches(stream, tokenizer, "meco", block_size=32, meco_max_tokens=8,
                        encoder_max_tokens=16, local_batch=3, keep_mask_fn=keep_mask_fn)
    batch = next(gen)
    assert len(calls) == 1
    assert (batch["text_start"] == 0).all()  # keep_mask all-False -> no prefix
    next(gen)
    assert len(calls) == 2  # called again for the second yielded batch


def test_load_val_batches(tok_dir, tokenizer):
    batches = load_val_batches(tok_dir, ["shard_02_val"], tokenizer, "text", block_size=32,
                               meco_max_tokens=8, encoder_max_tokens=16, val_docs=10,
                               val_batch_size=1)
    assert len(batches) == 2
