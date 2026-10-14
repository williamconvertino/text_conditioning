"""Build and cache per-domain support representations for the final conditioning
evaluation (README "Same-Domain Conditioning" / "Domain-Mean Conditioning").

Support documents are drawn from the validation shards (never training shards),
and for domain-mean conditioning specifically, the K support documents used to
build a domain's mean vector are held out from the set of documents being scored
under that vector -- README: "Support documents must not overlap with the
validation examples being scored".

FineWeb's domain distribution is extremely long-tailed: even scanning an entire
~47k-document validation shard, most domains have only 1-2 documents, and only
a few dozen have enough to support a same-domain/domain-mean split at all (see
`top_domains` below). Capping the scan to a small `val_docs`-sized sample (as
the "overall" perplexity metric does) makes this worse -- almost no domain
clears the K-support-docs bar, which is why `same_domain`/`domain_mean` used to
come back empty for most runs. `src/eval/conditioning.py` therefore scans the
*entire* validation shard when selecting domains (this is cheap: it's a
Python-side histogram over an already-loaded parquet index, not a model forward
pass), then only ever runs the model over the handful of documents belonging to
the domains actually selected.
"""

import os
import random

import torch

from src.data.collator import collate_encoder_batch, collate_partner_encoder_batch, doc_dict
from src.data.metadata import url_to_domain
from src.data.shards import iter_shard_docs


def build_domain_index(tok_dir, val_stems, limit=None):
    """domain -> list of (shard, row), in shard/row order."""
    index = {}
    for shard, row in iter_shard_docs(tok_dir, val_stems, limit=limit):
        domain = url_to_domain(shard.url[row] or "")
        index.setdefault(domain, []).append((shard, row))
    return index


def top_domains(domain_index, n_domains, k, min_docs_per_domain):
    """Domains from `domain_index` with enough documents to support a
    same-domain/domain-mean split (>= min_docs_per_domain + k documents),
    sorted by document count descending and truncated to the top `n_domains`.
    Returns a list of (domain, count) tuples, most-documents-first -- this
    fixed, count-sorted order is what guarantees a same-domain/domain-mean
    split is actually possible for every domain `evaluate_final` picks (README
    "Final Conditioning Evaluation" originally required this implicitly; on
    FineWeb's long tail it has to be selected for explicitly -- see this
    module's docstring)."""
    eligible = [(domain, len(docs)) for domain, docs in domain_index.items()
               if len(docs) >= min_docs_per_domain + k]
    eligible.sort(key=lambda item: (-item[1], item[0]))
    return eligible[:n_domains]


def split_support_and_eval(domain_index, k, min_docs, seed, only_domains=None):
    """For each domain (or each of `only_domains`, if given) with >= min_docs +
    k documents, split into a support set of size k (for same-domain /
    domain-mean conditioning) and the remaining documents to be scored.
    Domains with too few documents are skipped entirely -- if `only_domains`
    was chosen via `top_domains` above with matching `k`/`min_docs`, every one
    of them clears this bar by construction."""
    support, eval_docs = {}, {}
    domains = domain_index.keys() if only_domains is None else only_domains
    for domain in domains:
        docs = domain_index.get(domain, [])
        if len(docs) < min_docs + k:
            continue
        rng = random.Random(seed + (hash(domain) % (2 ** 31)))
        shuffled = docs[:]
        rng.shuffle(shuffled)
        support[domain] = shuffled[:k]
        eval_docs[domain] = shuffled[k:]
    return support, eval_docs


@torch.no_grad()
def encode_support_vectors(conditioner, tokenizer, support, encoder_max_tokens, device, batch_size=16):
    """domain -> mean z over its support set (README "Domain-Mean Conditioning")."""
    conditioner.eval()
    out = {}
    for domain, items in support.items():
        docs = [doc_dict(shard, row) for shard, row in items]
        zs = []
        for s in range(0, len(docs), batch_size):
            chunk = docs[s:s + batch_size]
            enc = collate_encoder_batch(chunk, tokenizer, encoder_max_tokens)
            z = conditioner.encoder(input_ids=enc["input_ids"].to(device),
                                    attention_mask=enc["attention_mask"].to(device))
            zs.append(z)
        out[domain] = torch.cat(zs, dim=0).mean(dim=0)
    return out


@torch.no_grad()
def encode_single_doc(conditioner, tokenizer, shard, row, encoder_max_tokens, device):
    """z for one document (README "Same-Domain Conditioning": encode a single
    other document from the same domain)."""
    doc = doc_dict(shard, row)
    enc = collate_encoder_batch([doc], tokenizer, encoder_max_tokens)
    z = conditioner.encoder(input_ids=enc["input_ids"].to(device),
                            attention_mask=enc["attention_mask"].to(device))
    return z[0]


@torch.no_grad()
def encode_condition_lists(conditioner, tokenizer, input_lists, encoder_max_tokens, device):
    """input_lists: one list of encoder inputs per row (e.g.
    src/data/url_conditioning.py's UrlConditionSource.for_url) -> [B, H]: the
    mean encoding of each row's inputs, or the null vector for an empty list
    -- the same computation training uses for condition-source batches."""
    enc = collate_partner_encoder_batch(input_lists, tokenizer, encoder_max_tokens)
    return conditioner(enc["input_ids"].to(device), enc["attention_mask"].to(device),
                       group_index=enc["group_index"], has_condition=enc["has_condition"])


def save_support_cache(path, support_vectors):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({k: v.cpu() for k, v in support_vectors.items()}, path)


def load_support_cache(path):
    return torch.load(path, map_location="cpu")
