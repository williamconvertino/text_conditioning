"""Named URL/domain "sources" (configs/benchmarks/sources.yaml), shared by
both conditioning axes exercised during benchmarking
(src/benchmarks/conditioning.py):

  * MeCo: a source's `meco_domain` is prepended as an ordinary text prefix,
    exactly like training (src/data/collator.py's `_meco_prefix_ids`).
  * Text conditioning: a source's `match` patterns select training documents
    whose domain (src/data/metadata.py's `url_to_domain`) contains one of
    them; those documents are encoded and averaged into a single domain-mean
    conditioning vector (README "Domain-Mean Conditioning", applied here to
    benchmark tasks instead of held-out FineWeb documents).

Matching documents are always drawn from the *training* shards (never the
validation shards `src/eval/domain_support.py` reserves for the perplexity
final-conditioning eval), since benchmark tasks are an entirely separate
dataset and there is no leakage concern -- the point is just "a handful of
representative documents from this source".
"""

import random

from src.data.metadata import url_to_domain
from src.data.shards import iter_shard_docs


def source_names(sources_cfg):
    return list(sources_cfg["sources"].keys())


def meco_domain_for_source(source_name, sources_cfg):
    try:
        return sources_cfg["sources"][source_name]["meco_domain"]
    except KeyError:
        raise KeyError(f"unknown source {source_name!r}; known sources: "
                       f"{source_names(sources_cfg)}") from None


def _matches(url, patterns):
    domain = url_to_domain(url)
    return any(p in domain for p in patterns)


def find_source_docs(tok_dir, source_name, sources_cfg, train_stems, n_docs, seed):
    """Deterministically sample up to n_docs training documents whose domain
    matches source_name's `match` patterns."""
    if source_name not in sources_cfg["sources"]:
        raise KeyError(f"unknown source {source_name!r}; known sources: "
                       f"{source_names(sources_cfg)}")
    patterns = sources_cfg["sources"][source_name]["match"]
    matches = [(shard, row) for shard, row in iter_shard_docs(tok_dir, train_stems)
              if _matches(shard.url[row] or "", patterns)]
    if not matches:
        raise ValueError(f"no training documents matched source {source_name!r} "
                         f"(match={patterns}) in {tok_dir}")
    rng = random.Random(seed)
    rng.shuffle(matches)
    return matches[:n_docs]


def find_doc_by_url(tok_dir, url, train_stems):
    """Exact-match lookup of a single training document by its URL, for ad hoc
    single-document text conditioning (--url)."""
    for shard, row in iter_shard_docs(tok_dir, train_stems):
        if shard.url[row] == url:
            return shard, row
    raise ValueError(f"no training document with url == {url!r} found in {tok_dir}")
