"""Resolve a benchmarking conditioning choice
(scripts/benchmarks/run_benchmarks.py's --mode/--source/--domain/--url) into a
`ConditioningSpec` that src/benchmarks/lm_wrapper.py's CtrlTokensLM knows how
to apply, plus a stable `tag` used to organize results
(src/benchmarks/results.py: outputs/<run>/benchmarks/<tag>/results.json).

Three conditioning kinds, matching the three ConditionedLM variants
(src/models/conditioned_lm.py):

  * "none": no conditioning -- valid for every variant (for "meco"/"text" runs
    this is their own unconditioned mode, i.e. z_null / no domain prefix).
  * "meco": a domain string prepended as text, exactly like training. Either a
    named --source (configs/benchmarks/sources.yaml) or an ad hoc --domain.
  * "text": a conditioning vector z. Either a named --source (averaged over
    n_docs_per_source matching training documents) or an ad hoc --url (a
    single specific training document).
"""

import re

import torch

from src.benchmarks.sources import find_doc_by_url, find_source_docs, meco_domain_for_source
from src.data.collator import collate_encoder_batch, doc_dict


def _slug(s, max_len=40):
    return re.sub(r"[^a-z0-9]+", "-", s.lower()).strip("-")[:max_len]


class ConditioningSpec:
    """kind: "none" | "meco" | "text".
    tag: stable name identifying this conditioning setting for result storage.
    domain: MeCo prefix domain string (kind == "meco").
    z: FloatTensor [hidden_size] conditioning vector (kind == "text").
    meta: JSON-serializable provenance (source name, docs used, ...).
    """

    def __init__(self, kind, tag, domain=None, z=None, meta=None):
        assert kind in ("none", "meco", "text")
        self.kind = kind
        self.tag = tag
        self.domain = domain
        self.z = z
        self.meta = meta or {}


def resolve_none():
    return ConditioningSpec("none", "unconditioned")


def resolve_meco(source=None, domain=None, sources_cfg=None):
    assert (source is None) != (domain is None), "pass exactly one of --source/--domain"
    if source is not None:
        domain = meco_domain_for_source(source, sources_cfg)
        tag = f"meco_{source}"
    else:
        tag = f"meco_domain-{_slug(domain)}"
    return ConditioningSpec("meco", tag, domain=domain, meta={"source": source, "domain": domain})


@torch.no_grad()
def resolve_text(conditioner, tokenizer, tok_dir, train_stems, encoder_max_tokens, device,
                 source=None, url=None, sources_cfg=None, n_docs=20, seed=1234):
    assert (source is None) != (url is None), "pass exactly one of --source/--url"
    was_training = conditioner.training
    conditioner.eval()
    try:
        if source is not None:
            docs = find_source_docs(tok_dir, source, sources_cfg, train_stems, n_docs, seed)
            tag = f"text_{source}"
        else:
            docs = [find_doc_by_url(tok_dir, url, train_stems)]
            tag = f"text_url-{_slug(url)}"

        doc_dicts = [doc_dict(shard, row) for shard, row in docs]
        enc = collate_encoder_batch(doc_dicts, tokenizer, encoder_max_tokens)
        zs = conditioner.encoder(input_ids=enc["input_ids"].to(device),
                                 attention_mask=enc["attention_mask"].to(device))
        z = zs.mean(dim=0)
    finally:
        conditioner.train(was_training)

    meta = {"source": source, "url": url, "n_docs": len(docs),
            "doc_urls": [shard.url[row] for shard, row in docs]}
    return ConditioningSpec("text", tag, z=z, meta=meta)
