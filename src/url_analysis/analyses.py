"""Analyses of the URL encoder's latent space
(scripts/url_analysis/run_encoder_analysis.py). Every function takes vectors
already produced by src/url_analysis/encode.py and returns plain,
JSON/torch-serializable results; plotting lives in src/url_analysis/plots.py.

Similarity is *centered* cosine by default: learned conditioning vectors
typically share a large common component (every vector has to look like
"a prepended conditioning position" to the LM), which makes raw cosines
uniformly high and neighbor lists dominated by that shared direction.
Subtracting the library mean (equal weight per domain) removes it; raw
cosines are reported alongside in vector_stats so the anisotropy is visible.
"""

import collections
import re
from urllib.parse import urlparse

import numpy as np
import torch
import torch.nn.functional as F


# -- similarity helpers --------------------------------------------------------

def normed(x, mean=None):
    x = x.float()
    if mean is not None:
        x = x - mean.float()
    return F.normalize(x, dim=-1)


def cosine(a, b, mean=None):
    """[N, H] x [M, H] -> [N, M] (centered by `mean` when given)."""
    return normed(a, mean) @ normed(b, mean).T


def top_k(sims, names, k, exclude=None):
    """sims: [M] similarities to `names`; exclude: indices to skip."""
    order = torch.argsort(sims, descending=True).tolist()
    out = []
    for j in order:
        if exclude is not None and j in exclude:
            continue
        out.append({"domain": names[j], "cos": float(sims[j])})
        if len(out) == k:
            break
    return out


# -- global statistics ---------------------------------------------------------

def vector_stats(lib):
    """Norms, anisotropy and root-vs-centroid agreement."""
    root, cent, pages, null, mean = lib["root"], lib["centroid"], lib["page_vecs"], lib["null"], lib["mean"]
    idx = torch.randperm(len(pages), generator=torch.Generator().manual_seed(0))[:4000]
    sub = pages[idx]
    raw = cosine(sub, sub)
    ctr = cosine(sub, sub, mean)
    off = ~torch.eye(len(sub), dtype=torch.bool)
    root_vs_cent_raw = F.cosine_similarity(root, cent, dim=-1)
    root_vs_cent_ctr = F.cosine_similarity(root - mean, cent - mean, dim=-1)
    null_to_lib = cosine(null[None], cent, mean)[0]
    q = lambda t: {f"p{p}": float(torch.quantile(t.float(), p / 100)) for p in (5, 25, 50, 75, 95)}
    return {
        "hidden_size": int(root.shape[1]),
        "n_domains": int(len(root)),
        "n_pages": int(len(pages)),
        "norm": {"page": q(pages.norm(dim=-1)), "root": q(root.norm(dim=-1)),
                 "centroid": q(cent.norm(dim=-1)), "null": float(null.norm())},
        "pairwise_cos_pages": {"raw_mean": float(raw[off].mean()), "centered_mean": float(ctr[off].mean())},
        "root_vs_centroid_cos": {"raw": q(root_vs_cent_raw), "centered": q(root_vs_cent_ctr)},
        "null_to_centroid_cos_centered": q(null_to_lib),
        "null_nearest_domains": top_k(null_to_lib, lib["domains"], 15),
        "root_vs_centroid_per_domain": [float(x) for x in root_vs_cent_ctr],
        "page_norms_sample": [float(x) for x in sub.norm(dim=-1)],
    }


def pca_spectrum(x, n=100):
    """Explained-variance ratio of the top n principal components and the
    participation ratio (sum(l)^2 / sum(l^2), an effective dimensionality)."""
    x = x.float() - x.float().mean(0)
    s = torch.linalg.svdvals(x)
    var = s ** 2
    ratio = var / var.sum()
    return {"explained_variance_ratio": [float(r) for r in ratio[:n]],
            "participation_ratio": float(var.sum() ** 2 / (var ** 2).sum())}


def pca_2d(x):
    x = x.float() - x.float().mean(0)
    _, _, v = torch.pca_lowrank(x, q=2, center=False)
    return (x @ v[:, :2]).numpy()


def tsne_2d(x, seed=0, perplexity=30, metric="cosine"):
    from sklearn.manifold import TSNE
    perplexity = min(perplexity, max(5, (len(x) - 1) // 3))
    return TSNE(n_components=2, perplexity=perplexity, metric=metric, init="pca",
                random_state=seed).fit_transform(x.float().numpy())


def kmeans(x, k, seed=0):
    from sklearn.cluster import KMeans
    km = KMeans(n_clusters=k, n_init=10, random_state=seed).fit(normed(x).numpy())
    return km.labels_, km.cluster_centers_


# -- neighbors -----------------------------------------------------------------

def neighbors(query_vecs, query_names, lib, k=15, kind="centroid"):
    """Nearest library domains (centered cosine on `kind` vectors) for each
    query. A query that is itself a library domain never lists itself."""
    vecs = lib[kind]
    sims = cosine(query_vecs, vecs, lib["mean"])
    pos = {d: i for i, d in enumerate(lib["domains"])}
    return {name: top_k(sims[i], lib["domains"], k, exclude={pos[name]} if name in pos else None)
            for i, name in enumerate(query_names)}


def similarity_matrix(names, lib, kind="centroid", extra=None):
    """Centered cosine among `names` (library domains) plus optional extra
    {label: vector} points -> (labels, [n, n] list)."""
    pos = {d: i for i, d in enumerate(lib["domains"])}
    labels, vecs = [], []
    for n in names:
        if n in pos:
            labels.append(n)
            vecs.append(lib[kind][pos[n]])
    for label, v in (extra or {}).items():
        labels.append(label)
        vecs.append(v)
    if not vecs:
        return [], []
    sims = cosine(torch.stack(vecs), torch.stack(vecs), lib["mean"])
    return labels, sims.tolist()


# -- global map ----------------------------------------------------------------

def categorize(domain, categories):
    """First matching category from an ordered {category: [substring, ...]} map."""
    for cat, needles in categories.items():
        if any(n in domain for n in needles):
            return cat
    return "other"


def domain_map(lib, n_domains, n_clusters, categories, seed=0, perplexity=30, kind="centroid"):
    """t-SNE + PCA coordinates and KMeans clusters for the n_domains most
    frequent library domains, plus each cluster's most frequent members."""
    n = min(n_domains, len(lib["domains"]))
    x = lib[kind][:n] - lib["mean"]
    labels, _ = kmeans(x, n_clusters, seed)
    names = lib["domains"][:n]
    counts = lib["counts"][:n].tolist()
    clusters = collections.defaultdict(list)
    for i, c in enumerate(labels):
        clusters[int(c)].append((counts[i], names[i]))
    summary = {c: [nm for _, nm in sorted(m, reverse=True)[:15]] for c, m in sorted(clusters.items())}
    return {
        "domains": names,
        "counts": counts,
        "category": [categorize(d, categories) for d in names],
        "cluster": [int(c) for c in labels],
        "tsne": tsne_2d(x, seed, perplexity).tolist(),
        "pca": pca_2d(x).tolist(),
        "cluster_members": summary,
        "cluster_sizes": {c: len(m) for c, m in clusters.items()},
    }


# -- domain identity in page vectors --------------------------------------------

def variance_by_domain(page_vecs, page_domain):
    """Share of page-vector variance explained by the domain (between-domain
    sum of squares / total), i.e. the R^2 of predicting a page's vector by its
    domain centroid."""
    x = page_vecs.float()
    mu = x.mean(0)
    n_dom = int(page_domain.max()) + 1
    sums = torch.zeros(n_dom, x.shape[1]).index_add_(0, page_domain, x)
    n = torch.bincount(page_domain, minlength=n_dom).clamp(min=1).unsqueeze(1).float()
    cent = sums / n
    total = ((x - mu) ** 2).sum()
    within = ((x - cent[page_domain]) ** 2).sum()
    return {"r2_domain": float(1 - within / total)}


def domain_retrieval(lib, n_pages=20000, seed=0, ks=(1, 5, 10)):
    """For sampled pages: is the page's own domain among the k nearest
    domain centroids (leave-one-out: the page is removed from its own
    centroid)? High accuracy = the page vector is dominated by domain
    identity; low = by path content."""
    pages, dom, cent = lib["page_vecs"].float(), lib["page_domain"], lib["centroid"].float()
    n_dom = len(cent)
    count = torch.bincount(dom, minlength=n_dom).float()
    g = torch.Generator().manual_seed(seed)
    idx = torch.randperm(len(pages), generator=g)[:n_pages]
    idx = idx[count[dom[idx]] > 1]
    hits = {k: 0 for k in ks}
    per_domain = collections.defaultdict(lambda: [0, 0])
    for s in range(0, len(idx), 2048):
        b = idx[s:s + 2048]
        v, d = pages[b], dom[b]
        sims = cosine(v, cent, lib["mean"])
        n = count[d].unsqueeze(1)
        loo = (cent[d] * n - v) / (n - 1)
        sims[torch.arange(len(b)), d] = F.cosine_similarity(v - lib["mean"], loo - lib["mean"], dim=-1)
        rank = (sims > sims[torch.arange(len(b)), d].unsqueeze(1)).sum(1)
        for k in ks:
            hits[k] += int((rank < k).sum())
        for di, r in zip(d.tolist(), rank.tolist()):
            per_domain[di][0] += int(r == 0)
            per_domain[di][1] += 1
    top1 = {lib["domains"][d]: h / t for d, (h, t) in per_domain.items() if t >= 5}
    return {"n_pages": int(len(idx)), "n_domains": n_dom,
            **{f"top{k}": hits[k] / max(1, len(idx)) for k in ks},
            "chance_top1": 1 / n_dom, "top1_per_domain": top1}


# -- in-domain structure ---------------------------------------------------------

_YEAR_RE = re.compile(r"(?<!\d)(19[5-9]\d|20[0-2]\d)(?!\d)")


def label_url(url, rule):
    """Coarse label of a page URL used to color/score in-domain structure.
    Rules: subdomain | section (first non-numeric path segment) | year |
    path:<i> | path_after:<segment> (e.g. path_after:r -> subreddit)."""
    p = urlparse(url)
    segs = [s for s in p.path.split("/") if s]
    if rule == "subdomain":
        host = p.netloc.lower().split(":")[0]
        host = host[4:] if host.startswith("www.") else host
        return host.split(".")[0] if host.count(".") >= 2 else "(root)"
    if rule == "section":
        return next((s.lower() for s in segs if not s.isdigit() and not _YEAR_RE.fullmatch(s)), None)
    if rule == "year":
        m = _YEAR_RE.search(p.path)
        return m.group(1) if m else None
    if rule.startswith("path:"):
        i = int(rule.split(":")[1])
        return segs[i].lower() if len(segs) > i else None
    if rule.startswith("path_after:"):
        key = rule.split(":", 1)[1]
        for a, b in zip(segs, segs[1:]):
            if a.lower() == key:
                return b.lower()
        return None
    raise ValueError(f"unknown label rule {rule!r}")


def in_domain(urls, vecs, rules, n_clusters=8, top_labels=10, seed=0, perplexity=30):
    """Structure among pages of one domain group: t-SNE coordinates, KMeans
    clusters with their most central URLs, and per label rule the labels,
    their silhouette score (cosine; > 0 means pages sharing a label sit
    together) and mean within-label vs. overall similarity."""
    from sklearn.metrics import silhouette_score

    mean = vecs.float().mean(0)
    x = vecs.float() - mean
    xn = F.normalize(x, dim=-1)
    k = max(2, min(n_clusters, len(urls) // 20))
    clusters, centers = kmeans(x, k, seed)
    centers = torch.tensor(centers, dtype=torch.float32)
    exemplars = {}
    for c in range(k):
        members = np.flatnonzero(clusters == c)
        sims = xn[members] @ F.normalize(centers[c], dim=0)
        exemplars[c] = [urls[members[j]] for j in torch.argsort(sims, descending=True)[:8].tolist()]

    per_rule = {}
    for rule in rules:
        labels = [label_url(u, rule) for u in urls]
        freq = collections.Counter(l for l in labels if l is not None)
        keep = {l for l, _ in freq.most_common(top_labels)}
        shown = [l if l in keep else ("(none)" if l is None else "(other)") for l in labels]
        mask = np.array([l in keep for l in labels])
        sil = None
        if mask.sum() > 10 and len({l for l, m in zip(labels, mask) if m}) > 1:
            sil = float(silhouette_score(xn[mask].numpy(), np.array(labels)[mask], metric="cosine"))
        within = []
        for l in keep:
            m = torch.tensor([lab == l for lab in labels])
            if m.sum() > 1:
                s = xn[m] @ xn[m].T
                within.append(float((s.sum() - m.sum()) / (m.sum() * (m.sum() - 1))))
        overall = float(((xn @ xn.T).sum() - len(xn)) / (len(xn) * (len(xn) - 1)))
        per_rule[rule] = {"labels": shown, "top_labels": [l for l, _ in freq.most_common(top_labels)],
                          "coverage": float(sum(freq.values()) / max(1, len(urls))),
                          "silhouette": sil,
                          "within_label_cos": float(np.mean(within)) if within else None,
                          "overall_cos": overall}
    return {"n_pages": len(urls), "urls": urls, "tsne": tsne_2d(x, seed, perplexity).tolist(),
            "cluster": [int(c) for c in clusters], "cluster_exemplars": exemplars,
            "mean_cos_to_centroid": float((xn @ F.normalize(x.mean(0), dim=0)).mean()),
            "rules": per_rule}


def year_probe(urls, vecs, seed=0, min_pages=200):
    """Ridge probe: predict the year in a page URL's path from its vector
    (library pages whose path contains a 1950-2029 year). Held-out R^2 / MAE
    against a predict-the-mean baseline -- does the encoder carry time?"""
    from sklearn.linear_model import RidgeCV
    from sklearn.model_selection import train_test_split

    years = [label_url(u, "year") for u in urls]
    idx = [i for i, y in enumerate(years) if y is not None]
    if len(idx) < min_pages:
        return {"n_pages": len(idx), "skipped": True}
    x = vecs[idx].float().numpy()
    y = np.array([int(years[i]) for i in idx], dtype=np.float64)
    xtr, xte, ytr, yte = train_test_split(x, y, test_size=0.25, random_state=seed)
    model = RidgeCV(alphas=np.logspace(-2, 4, 13)).fit(xtr, ytr)
    pred = model.predict(xte)
    mae_mean = float(np.abs(yte - ytr.mean()).mean())
    return {"n_pages": len(idx), "r2": float(model.score(xte, yte)),
            "mae": float(np.abs(pred - yte).mean()), "mae_predict_mean": mae_mean,
            "test_year": yte.tolist(), "test_pred": pred.tolist()}


# -- robustness to surface edits of the URL -----------------------------------------

def url_variants(url):
    """Surface edits of one URL that keep (or deliberately drop) its meaning."""
    p = urlparse(url)
    host = p.netloc
    bare = host[4:] if host.startswith("www.") else host
    other_scheme = "http" if p.scheme == "https" else "https"
    path = p.path or "/"
    v = {
        "original": url,
        "scheme swapped": f"{other_scheme}://{host}{path}",
        "www toggled": f"{p.scheme}://{bare if host.startswith('www.') else 'www.' + host}{path}",
        "trailing slash toggled": f"{p.scheme}://{host}{path.rstrip('/') if path.endswith('/') and path != '/' else path + '/'}",
        "uppercased path": f"{p.scheme}://{host}{path.upper()}",
        "query string added": f"{p.scheme}://{host}{path}?utm_source=feed",
        "path removed (root)": f"{p.scheme}://{host}/",
        "domain only": bare,
    }
    return v


def perturbation_study(encoder, urls, lib):
    """Centered cosine between E(original URL) and E(variant), per variant
    kind, averaged over `urls`; plus a same-domain / other-domain reference
    (the mean cosine of two random pages of the same / different library
    domains), so the numbers have a scale."""
    kinds, sims = None, collections.defaultdict(list)
    for url in urls:
        v = url_variants(url)
        kinds = list(v)
        z = encoder.encode(list(v.values()))
        s = cosine(z[:1], z, lib["mean"])[0]
        for kind, val in zip(kinds, s.tolist()):
            sims[kind].append(val)

    pages, dom = lib["page_vecs"], lib["page_domain"]
    g = torch.Generator().manual_seed(0)
    a = torch.randint(len(pages), (5000,), generator=g)
    b = torch.randint(len(pages), (5000,), generator=g)
    cs = F.cosine_similarity(pages[a] - lib["mean"], pages[b] - lib["mean"], dim=-1)
    same = dom[a] == dom[b]
    # same-domain pairs are rare under uniform sampling; draw them explicitly
    same_vals = []
    starts = torch.searchsorted(dom, torch.arange(int(dom.max()) + 1))
    for d in torch.randint(int(dom.max()) + 1, (5000,), generator=g).tolist():
        lo = int(starts[d]); hi = int(starts[d + 1]) if d + 1 < len(starts) else len(dom)
        if hi - lo < 2:
            continue
        i, j = torch.randperm(hi - lo, generator=g)[:2].tolist()
        same_vals.append(float(F.cosine_similarity(pages[lo + i] - lib["mean"], pages[lo + j] - lib["mean"], dim=0)))
    return {"kinds": kinds, "mean": {k: float(np.mean(v)) for k, v in sims.items()},
            "per_url": {k: v for k, v in sims.items()}, "urls": urls,
            "reference": {"same_domain_pages": float(np.mean(same_vals)) if same_vals else None,
                          "different_domain_pages": float(cs[~same].mean())}}
