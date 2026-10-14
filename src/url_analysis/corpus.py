"""Sample the URLs the URL-encoder analysis encodes
(scripts/url_analysis/run_encoder_analysis.py), straight from the training
shards' index parquets (url column only -- no document tokens are read).

Produces:
  * domain frequencies over the scanned shards (the "library" is the
    n_domains most frequent domains);
  * each library domain's root URL, the most common "<scheme>://<host>/"
    observed for it, i.e. the closest thing to a bare-domain input for a
    full-URL encoder (for a domain-field encoder the root is the domain itself);
  * up to pages_per_domain uniformly sampled page URLs per library domain;
  * up to `pages` page URLs for each in-domain analysis group (a group is a
    set of domains matched by suffix, e.g. every *.stackexchange.com site).

Only training shards are scanned (the same shards the encoder was trained on):
the question is how the encoder represents the metadata it learned from, not
generalization to held-out URLs.
"""

import collections
import random
from urllib.parse import urlparse

import numpy as np
import pyarrow.parquet as pq

from src.data.domain_index import UNKNOWN_DOMAIN
from src.data.metadata import url_to_domain


def pick_stems(train_stems, n_shards):
    """n_shards stems spread evenly over the (sorted) training stems, so the
    scan covers every FineWeb dump rather than the first one only."""
    if n_shards >= len(train_stems):
        return list(train_stems)
    step = len(train_stems) / n_shards
    return [train_stems[int(i * step)] for i in range(n_shards)]


def read_urls(tok_dir, stems):
    urls = []
    for stem in stems:
        table = pq.read_table(f"{tok_dir}/{stem}_index.parquet", columns=["url"])
        urls.extend(u or "" for u in table.column("url").to_pylist())
    return urls


def root_url(url):
    p = urlparse(url)
    return f"{p.scheme or 'http'}://{p.netloc}/" if p.netloc else None


def domain_matches(domain, pattern):
    return domain == pattern or domain.endswith("." + pattern)


def scan_corpus(tok_dir, stems, n_domains, pages_per_domain, groups, group_pages, seed):
    """groups: {group name: [domain suffix, ...]}. Returns a dict (see module
    docstring) of plain python lists, ready to torch.save."""
    urls = read_urls(tok_dir, stems)
    codes = {}
    domain_id = np.fromiter((codes.setdefault(url_to_domain(u), len(codes)) for u in urls),
                            dtype=np.int64, count=len(urls))
    names = list(codes)
    counts = np.bincount(domain_id, minlength=len(names))

    unknown = codes.get(UNKNOWN_DOMAIN)
    order = [d for d in np.argsort(-counts, kind="stable") if d != unknown][:n_domains]
    lib_set = set(int(d) for d in order)

    by_domain = collections.defaultdict(list)
    for i, d in enumerate(domain_id):
        if d in lib_set:
            by_domain[int(d)].append(i)

    rng = random.Random(seed)
    library = []
    for d in order:
        d = int(d)
        idx = by_domain[d]
        sample = rng.sample(idx, min(pages_per_domain, len(idx)))
        roots = collections.Counter(root_url(urls[i]) for i in idx[:500])
        roots.pop(None, None)
        library.append({
            "domain": names[d],
            "count": int(counts[d]),
            "root_url": roots.most_common(1)[0][0] if roots else f"http://{names[d]}/",
            "pages": [urls[i] for i in sample],
        })

    group_urls = {}
    for gname, patterns in groups.items():
        matched_domains = [d for d, name in enumerate(names) if any(domain_matches(name, p) for p in patterns)]
        matched = np.flatnonzero(np.isin(domain_id, matched_domains)) if matched_domains else []
        picked = rng.sample(list(matched), min(group_pages, len(matched)))
        group_urls[gname] = [urls[i] for i in picked]

    return {
        "stems": list(stems),
        "n_urls": len(urls),
        "n_domains_seen": len(names) - (unknown is not None),
        "library": library,
        "groups": group_urls,
        "domain_counts": {names[d]: int(counts[d]) for d in np.argsort(-counts)[:max(n_domains * 4, 20000)]},
    }


def lookup_count(corpus, domain):
    """Frequency of `domain` in the scanned shards (0 if it never appeared in
    the top domains recorded)."""
    return corpus["domain_counts"].get(url_to_domain(domain), 0)
