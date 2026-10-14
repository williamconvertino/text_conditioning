"""Encode URL metadata with a trained URL-encoded run's encoder
(README "URL-Encoded Conditioning"), plus the saved "domain library" that
both the encoder analysis and the steering experiments build on.

A URL-encoded run conditions the LM on z = E(metadata(url)), where metadata is
the domain (url_conditioning.field: domain) or the full URL (field: url).
UrlEncoder reproduces exactly that computation for arbitrary strings, batched.

Domain library (build_library -> library.pt), per library domain d:
  root      E(root input): the root URL "<scheme>://<host>/" for a full-URL
            encoder, the domain string for a domain encoder -- the input
            MeCo-style steering by --source uses (src/benchmarks/conditioning.py)
  centroid  mean of E(page URL) over d's sampled pages -- the average
            conditioning vector the LM actually saw for d during training
plus the null vector and the library mean (used to center cosine
similarities; see src/url_analysis/analyses.py).
"""

import os

import torch

from src.data.url_conditioning import build_url_source
from src.eval.domain_support import encode_condition_lists
from src.training.checkpointing import load_run


class UrlEncoder:
    def __init__(self, conditioner, tokenizer, url_source, device):
        self.conditioner = conditioner.eval()
        self.tokenizer = tokenizer
        self.source = url_source
        self.device = device

    @property
    def field(self):
        return self.source.field

    def metadata(self, url):
        """What the encoder sees for a page URL (None: no parseable domain -> null vector)."""
        return self.source.metadata_text(url)

    def root_input(self, entry):
        """Encoder input standing for a whole library domain."""
        return entry["root_url"] if self.field == "url" else entry["domain"]

    @torch.no_grad()
    def encode(self, texts, batch_size=1024):
        """texts: metadata strings (None -> null vector) -> float32 [N, H] on CPU."""
        out = []
        for s in range(0, len(texts), batch_size):
            chunk = [[] if t is None else self.source.for_text(t) for t in texts[s:s + batch_size]]
            z = encode_condition_lists(self.conditioner, self.tokenizer, chunk,
                                       self.source.encoder_max_tokens, self.device)
            out.append(z.float().cpu())
        return torch.cat(out) if out else torch.zeros(0, self.null_vector.numel())

    def encode_urls(self, urls, batch_size=1024):
        return self.encode([self.metadata(u) for u in urls], batch_size)

    @property
    def null_vector(self):
        return self.conditioner.null_vector.detach().float().cpu()


def load_url_encoder(run_dir, checkpoint, device=None, keep_model=False):
    """(UrlEncoder, model or None, cfg). The causal LM is dropped unless
    keep_model (the analysis only needs the encoder + null vector)."""
    model, tokenizer, cfg = load_run(run_dir, checkpoint, device=device)
    url_source = build_url_source(cfg, tokenizer)
    if url_source is None:
        raise SystemExit(f"{run_dir} is not a URL-encoded run (text_conditioning_source != 'url')")
    device = next(model.parameters()).device
    enc = UrlEncoder(model.conditioner, tokenizer, url_source, device)
    if not keep_model:
        model.lm = None
        model = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return enc, model, cfg


def build_library(encoder, corpus, batch_size=1024):
    """corpus: src/url_analysis/corpus.py's scan_corpus output -> library dict
    (see module docstring). Page vectors are kept (page_vecs / page_domain) for
    the in-domain statistics."""
    lib = corpus["library"]
    roots = encoder.encode([encoder.root_input(e) for e in lib], batch_size)

    page_urls, page_domain = [], []
    for d, e in enumerate(lib):
        page_urls.extend(e["pages"])
        page_domain.extend([d] * len(e["pages"]))
    page_vecs = encoder.encode_urls(page_urls, batch_size)
    page_domain = torch.tensor(page_domain, dtype=torch.long)

    H = roots.shape[1]
    sums = torch.zeros(len(lib), H).index_add_(0, page_domain, page_vecs)
    n = torch.bincount(page_domain, minlength=len(lib)).clamp(min=1).unsqueeze(1)
    centroids = sums / n

    return {
        "domains": [e["domain"] for e in lib],
        "counts": torch.tensor([e["count"] for e in lib]),
        "root_inputs": [encoder.root_input(e) for e in lib],
        "root": roots,
        "centroid": centroids,
        "page_urls": page_urls,
        "page_domain": page_domain,
        "page_vecs": page_vecs,
        "null": encoder.null_vector,
        "mean": centroids.mean(0),
        "field": encoder.field,
    }


def save_library(lib, path, provenance):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({**lib, "provenance": provenance}, path)


def load_library(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} does not exist: run scripts/url_analysis/run_encoder_analysis.py "
                                f"embed first (slurm/url_analysis/encoder_analysis.slurm)")
    return torch.load(path, map_location="cpu", weights_only=False)


def library_vectors(lib, kind):
    """kind: "root" | "centroid" -> [N, H]."""
    if kind not in ("root", "centroid"):
        raise ValueError(f"library vector kind must be root|centroid, got {kind!r}")
    return lib[kind]
