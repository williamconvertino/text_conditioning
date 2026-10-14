"""Final conditioning evaluation (README "Final Conditioning Evaluation" /
"Final Evaluation Table"): for held-out validation documents, score each
variant's model under the conditioning modes that apply to it. Expensive modes
(same-domain / domain-mean, which require encoding many extra documents) are
meant for a single run on the final checkpoint -- see src/training/trainer.py
for the cheap self/unconditioned metrics already tracked throughout training.

Mode applicability by variant (src/analysis/tables.py renders anything left as
None as "--", per README "use -- rather than inventing an equivalent
measurement"):
  none (baseline):  only "none" -- there is nothing to remove or vary.
  meco:              "none" (URL/domain prefix stripped) and "self" (its normal
                     domain-conditioned eval, reused as the "self" column since
                     the domain string is a deterministic function of the
                     document's own URL). "same_domain" is not reported
                     separately: a same-domain document has the exact same
                     domain string as the target, so it would just repeat
                     "self". "domain_mean" has no meaning for text metadata
                     (there's nothing to average).
  text:              all of none / self / same_domain / domain_mean, plus the
                     optional cheap cross_domain sanity check.
  text, URL-encoded  (text_conditioning_source: url, src/data/url_conditioning.py)
                     the encoder only ever sees URL metadata, so every mode
                     encodes URLs rather than document text: "self" = the
                     document's own URL (MeCo's "self", in vector space),
                     same_domain = another same-domain document's URL,
                     domain_mean = the mean over the support documents' URLs,
                     cross_domain = another domain's mean. With field
                     "domain" (the default), every same-domain document's
                     metadata is the identical domain string, so same_domain
                     and domain_mean would just repeat "self" and are left
                     as None (same reasoning as MeCo); cross_domain is kept
                     -- it's the "steer to the wrong domain" test.

Two evaluation scopes, per mode (README "output results for each domain
considered as well as some aggregation"):
  "overall":      the fixed val_docs-sized validation sample (same one
                   training-time validation uses), independent of domain.
                   Only "none"/"self" apply here -- same_domain/domain_mean/
                   cross_domain are inherently domain-conditional (see below).
  "top_domains":  restricted to a fixed set of `n_domains` domains chosen for
                   having enough documents to support a same-domain/
                   domain-mean split in the first place (src/eval/
                   domain_support.py's `top_domains` -- FineWeb's domain
                   distribution is extremely long-tailed, so this has to be
                   selected for explicitly rather than assumed). "none" and
                   "self" are *also* computed on this restricted, per-domain
                   subset (not just "overall") so that every mode is being
                   compared on the exact same documents -- results include
                   both a combined aggregate and a per-domain breakdown.
"""

import math
import random

import torch

from src.data.collator import collate_batch, doc_dict
from src.data.shards import iter_shard_docs
from src.data.url_conditioning import build_url_source
from src.eval.domain_support import (
    build_domain_index, encode_condition_lists, encode_single_doc, encode_support_vectors,
    split_support_and_eval, top_domains,
)

MODES = ("none", "self", "same_domain", "domain_mean", "cross_domain")
# same_domain/domain_mean/cross_domain need a per-document conditioning vector
# drawn from other documents in (or outside) the same domain, so they only
# ever make sense restricted to the chosen top domains, never "overall".
TOP_DOMAIN_ONLY_MODES = ("same_domain", "domain_mean", "cross_domain")


def _empty_modes():
    return {m: None for m in MODES}


@torch.no_grad()
def _score(model, docs, tokenizer, variant_for_batch, block_size, meco_max_tokens,
          encoder_max_tokens, batch_size, keep_mask_value=None, z_provider=None):
    """Score `docs` under one conditioning setting, batching internally.

    variant_for_batch: how to collate the batch ("none"/"meco"/"text").
    keep_mask_value: True/False/None -- self-conditioned / null-conditioned /
      the model's default, only meaningful for variant_for_batch == "text"
      using the model's own conditioner (i.e. no z_provider).
    z_provider(chunk) -> FloatTensor [len(chunk), H]: supplies an externally
      computed conditioning vector per example instead of running the batch's
      own documents through the encoder (same-domain / domain-mean /
      cross-domain conditioning).

    Returns (loss, ppl, n_tok), or None if `docs` is empty -- n_tok lets
    callers combine several _score() calls into a correctly token-weighted
    aggregate (see `_combine` below) rather than just averaging PPLs.
    """
    if not docs:
        return None
    device = next(model.parameters()).device
    was_training = model.training
    model.eval()
    total_loss, total_tok = 0.0, 0.0
    for s in range(0, len(docs), batch_size):
        chunk = docs[s:s + batch_size]
        batch = collate_batch(chunk, tokenizer, variant_for_batch, block_size, meco_max_tokens,
                              encoder_max_tokens)
        if z_provider is not None:
            z = z_provider(chunk)
            loss, n_tok = model.forward_with_explicit_condition(batch, z)
        else:
            keep_mask = None
            if model.variant == "text" and keep_mask_value is not None:
                keep_mask = torch.full((len(chunk),), keep_mask_value, dtype=torch.bool, device=device)
            loss, n_tok, _ = model(batch, keep_mask=keep_mask)
        total_loss += loss.item() * n_tok.item()
        total_tok += n_tok.item()
    if was_training:
        model.train()
    if total_tok == 0:
        return None
    mean_loss = total_loss / total_tok
    return (mean_loss, math.exp(min(mean_loss, 20.0)), total_tok)


def _combine(scored):
    """[(loss, ppl, n_tok) or None, ...] -> token-weighted (loss, ppl), or None
    if every entry was None (e.g. a mode with zero eligible domains)."""
    total_loss, total_tok = 0.0, 0.0
    for s in scored:
        if s is None:
            continue
        loss, _, n_tok = s
        total_loss += loss * n_tok
        total_tok += n_tok
    if total_tok == 0:
        return None
    mean_loss = total_loss / total_tok
    return (mean_loss, math.exp(min(mean_loss, 20.0)))


def _drop_n_tok(scored):
    return None if scored is None else (scored[0], scored[1])


def evaluate_final(model, tokenizer, tok_dir, val_stems, cfg):
    """cfg: configs/eval/conditioning.yaml, merged with the run's block_size /
    meco_max_tokens / encoder_max_tokens / variant / text_conditioning_source /
    url_conditioning (see scripts/evaluate.py). Returns:

        {"overall": {mode: (loss, ppl) or None, ...},
         "top_domains": None, or {
             "n_domains": int, "k": int, "min_docs_per_domain": int,
             "domains": [{"domain": str, "count": int, "n_support": int, "n_eval": int}, ...],
             "aggregate": {mode: (loss, ppl) or None, ...},
             "per_domain": {domain: {mode: (loss, ppl) or None, ...}, ...},
         }}
    """
    device = next(model.parameters()).device
    block_size = cfg["block_size"]
    meco_max_tokens = cfg["meco_max_tokens"]
    encoder_max_tokens = cfg.get("encoder_max_tokens", 512)
    batch_size = cfg["val_batch_size"]
    variant = model.variant

    def score(docs, variant_for_batch, **kwargs):
        return _score(model, docs, tokenizer, variant_for_batch, block_size, meco_max_tokens,
                     encoder_max_tokens, batch_size, **kwargs)

    url_source = build_url_source({**cfg, "variant": variant}, tokenizer)

    def url_z(docs):
        """Each doc's own URL encoding (null vector without a parseable URL)."""
        return encode_condition_lists(model.conditioner, tokenizer,
                                      [url_source.for_url(d["url"]) for d in docs],
                                      url_source.encoder_max_tokens, device)

    # -- "overall": the fixed val_docs-sized sample, independent of domain --
    all_docs = [doc_dict(sh, r) for sh, r in iter_shard_docs(tok_dir, val_stems, limit=cfg.get("val_docs"))]
    overall = _empty_modes()
    if url_source is not None:
        overall["none"] = _drop_n_tok(score(all_docs, "text", keep_mask_value=False))
        overall["self"] = _drop_n_tok(score(all_docs, "text", z_provider=url_z))
    elif variant == "none":
        overall["none"] = _drop_n_tok(score(all_docs, "none"))
    elif variant == "meco":
        overall["none"] = _drop_n_tok(score(all_docs, "none"))
        overall["self"] = _drop_n_tok(score(all_docs, "meco"))
    else:  # "text"
        overall["none"] = _drop_n_tok(score(all_docs, "text", keep_mask_value=False))
        overall["self"] = _drop_n_tok(score(all_docs, "text", keep_mask_value=True))

    # -- "top_domains": domains with enough documents to support a
    # same-domain/domain-mean split, found by scanning the *entire* validation
    # shard (cheap -- see src/eval/domain_support.py's module docstring) --
    k = cfg.get("domain_support_k", 8)
    min_docs = cfg.get("min_docs_per_domain", 2)
    n_domains = cfg.get("n_domains", 50)
    seed = cfg.get("seed", 1234)

    full_domain_index = build_domain_index(tok_dir, val_stems, limit=cfg.get("domain_scan_docs"))
    chosen = top_domains(full_domain_index, n_domains, k, min_docs)
    if not chosen:
        return {"overall": overall, "top_domains": None}

    domain_names = [d for d, _ in chosen]
    support, eval_docs = split_support_and_eval(full_domain_index, k, min_docs, seed,
                                                only_domains=domain_names)

    domain_means = None
    if url_source is not None:
        # one input list per domain holding every support doc's URL -> its mean
        domain_names_ = list(support)
        zs = encode_condition_lists(
            model.conditioner, tokenizer,
            [[x for sh, r in support[d] for x in url_source.for_url(sh.url[r])] for d in domain_names_],
            url_source.encoder_max_tokens, device)
        domain_means = dict(zip(domain_names_, zs))
    elif variant == "text":
        domain_means = encode_support_vectors(model.conditioner, tokenizer, support,
                                              encoder_max_tokens, device)
    rng = random.Random(seed)

    per_domain = {}
    pooled = {m: [] for m in MODES}
    for domain, count in chosen:
        docs = [doc_dict(sh, r) for sh, r in eval_docs.get(domain, [])]
        if not docs:
            continue
        result = _empty_modes()

        if url_source is not None:
            result["none"] = score(docs, "text", keep_mask_value=False)
            result["self"] = score(docs, "text", z_provider=url_z)
            z_mean = domain_means[domain].unsqueeze(0).to(device)
            if url_source.field == "url":
                support_docs = [doc_dict(sh, r) for sh, r in support[domain]]
                result["same_domain"] = score(
                    docs, "text", z_provider=lambda chunk: url_z([rng.choice(support_docs) for _ in chunk]))
                result["domain_mean"] = score(
                    docs, "text", z_provider=lambda chunk: z_mean.expand(len(chunk), -1))
            other_domains = [d for d in domain_means if d != domain]
            if cfg.get("cross_domain", False) and other_domains:
                result["cross_domain"] = score(
                    docs, "text",
                    z_provider=lambda chunk: torch.stack(
                        [domain_means[rng.choice(other_domains)] for _ in chunk], dim=0).to(device))
        elif variant == "none":
            result["none"] = score(docs, "none")
        elif variant == "meco":
            result["none"] = score(docs, "none")
            result["self"] = score(docs, "meco")
        else:  # "text"
            result["none"] = score(docs, "text", keep_mask_value=False)
            result["self"] = score(docs, "text", keep_mask_value=True)

            support_items = support[domain]
            result["same_domain"] = score(
                docs, "text",
                z_provider=lambda chunk: torch.stack(
                    [encode_single_doc(model.conditioner, tokenizer, *rng.choice(support_items),
                                       encoder_max_tokens, device) for _ in chunk], dim=0))

            z_mean = domain_means[domain].unsqueeze(0).to(device)
            result["domain_mean"] = score(
                docs, "text", z_provider=lambda chunk: z_mean.expand(len(chunk), -1))

            other_domains = [d for d in domain_means if d != domain]
            if cfg.get("cross_domain", False) and other_domains:
                result["cross_domain"] = score(
                    docs, "text",
                    z_provider=lambda chunk: torch.stack(
                        [domain_means[rng.choice(other_domains)] for _ in chunk], dim=0).to(device))

        for m in MODES:
            pooled[m].append(result[m])
        per_domain[domain] = {m: _drop_n_tok(result[m]) for m in MODES}

    top_domains_out = {
        "n_domains": len(chosen),
        "k": k,
        "min_docs_per_domain": min_docs,
        "domains": [{"domain": d, "count": c, "n_support": len(support.get(d, [])),
                    "n_eval": len(eval_docs.get(d, []))} for d, c in chosen],
        "aggregate": {m: _combine(pooled[m]) for m in MODES},
        "per_domain": per_domain,
    }
    return {"overall": overall, "top_domains": top_domains_out}
