"""Shared test fixtures: a tiny, self-contained pretokenized shard directory
(same .bin/_index.parquet/manifest.json/.done format tokenize_fineweb.py
writes) and a tiny GPT-2 tokenizer, so tests never need network access or the
real (gated) Llama tokenizer.
"""

import json
import os

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from transformers import AutoTokenizer

DOMAINS = ["en.wikipedia.org", "en.wikipedia.org", "news.example.com",
          "news.example.com", "blog.example.org", "blog.example.org"]
TEXTS = [
    "The quick brown fox jumps over the lazy dog.",
    "Transformers are a type of neural network architecture.",
    "Local news: the city council met on Tuesday to discuss the budget.",
    "In other news, the weather is expected to be sunny this weekend.",
    "Today I want to talk about my favorite hiking trail in the mountains.",
    "This blog post covers three easy recipes for a quick weeknight dinner.",
]


@pytest.fixture(scope="session")
def tokenizer():
    tok = AutoTokenizer.from_pretrained("gpt2")
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    return tok


def _write_shard(output_dir, stem, texts, urls, tokenizer, max_doc_tokens=64):
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
    tokens.tofile(os.path.join(output_dir, f"{stem}.bin"))
    index = pa.table({
        "row": np.arange(len(texts), dtype=np.int32),
        "start": np.asarray(starts, dtype=np.int64),
        "length": np.asarray(lengths, dtype=np.int32),
        "url": urls,
        "doc_id": [f"{stem}-{i}" for i in range(len(texts))],
    })
    pq.write_table(index, os.path.join(output_dir, f"{stem}_index.parquet"))
    open(os.path.join(output_dir, f"{stem}.done"), "w").close()


@pytest.fixture()
def tok_dir(tmp_path, tokenizer):
    out_dir = str(tmp_path / "fineweb_tokenized")
    os.makedirs(out_dir, exist_ok=True)
    manifest = {
        "tokenizer": "gpt2", "vocab_size": len(tokenizer), "eos_id": tokenizer.eos_token_id,
        "pad_id": tokenizer.pad_token_id, "max_doc_tokens": 64,
        "text_column": "text", "url_column": "url", "id_column": "id",
    }
    with open(os.path.join(out_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f)

    urls = [f"https://{d}/article-{i}" for i, d in enumerate(DOMAINS)]
    # two train shards, one val shard -- enough to exercise split_shards/rank_shards
    _write_shard(out_dir, "shard_00", TEXTS[0:2], urls[0:2], tokenizer)
    _write_shard(out_dir, "shard_01", TEXTS[2:4], urls[2:4], tokenizer)
    _write_shard(out_dir, "shard_02_val", TEXTS[4:6], urls[4:6], tokenizer)
    return out_dir
