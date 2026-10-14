# Learned Text Conditioning for Efficient Language Model Pretraining

This repository investigates whether a language model can learn its own fine-grained, document-level conditioning signal during pretraining.

Recent work such as **Metadata Conditioning Accelerates Language Model Pre-training (MeCo)** and **Beyond URLs: Metadata Diversity and Position for Efficient LLM Pretraining** shows that prepending sufficiently fine-grained metadata can improve language-model pretraining efficiency. URL-derived source information is particularly effective, while coarse metadata such as broad format, topic, or quality categories appears substantially less useful.

These results suggest that useful conditioning needs to capture relatively fine-grained information about the distribution from which a document was generated.

Instead of defining this information manually, this project asks:

> **Can we learn a compact document-level conditioning representation directly from the text itself, and can this representation improve language-model pretraining?**

We encode each training document with a small bidirectional Transformer, reduce the document to a single conditioning vector, and prepend that vector to the causal language model's context.

The resulting model is trained with the normal autoregressive next-token objective.

---

## Motivation

Metadata-conditioned pretraining can be viewed as giving the language model additional information about the current document:

$$
p(x_t \mid x_{<t}, m)
$$

where \(m\) might identify the source URL, domain, quality category, or some other document attribute.

Fine-grained metadata appears particularly useful, but hand-designed metadata has several limitations:

* useful metadata may not exist for every dataset;
* the best metadata representation may be task- or corpus-dependent;
* predefined annotations constrain what information can be communicated;
* generating detailed annotations with another model adds preprocessing cost and external assumptions.

Instead, we propose learning the conditioning representation directly:

$$
z = E(x_1,\ldots,x_T)
$$

where \(E\) is a small bidirectional document encoder.

The causal language model then predicts

$$
p_\theta(x_t \mid x_{<t}, z).
$$

Because the encoder observes the entire document, \(z\) may contain several kinds of useful information simultaneously, including:

* source or domain characteristics;
* topic;
* style and tone;
* document structure;
* vocabulary distribution;
* formatting;
* entities and low-frequency vocabulary;
* other document-specific information discovered automatically during training.

The representation is intentionally compressed into a **single vector**, forcing the model to summarize the document rather than directly exposing a second token sequence to the causal LM.

---

# Research Questions

The initial experiments focus on three questions.

### 1. Does learned text conditioning improve language modeling?

Does providing a learned document vector reduce validation perplexity relative to standard autoregressive pretraining?

### 2. Does the representation contain transferable document-level information?

A self-conditioned vector may contain information specific to the exact validation document.

We therefore also test whether representations extracted from **other documents from the same source/domain** improve perplexity.

If they do, the encoder is learning reusable distributional information rather than only an individual document fingerprint.

### 3. Does conditioning improve the LM backbone itself?

For pretraining efficiency, the most important metric is eventually performance **without access to the encoder**.

We therefore evaluate the text-conditioned model both with and without its conditioning signal and experiment with conditioning schedules that encourage the backbone to retain benefits after conditioning is removed.

---

# Data

Experiments use the **FineWeb 100BT subset**.

FineWeb is especially convenient for this project because each document includes its original URL. This gives us both:

1. the raw document text required by the proposed encoder; and
2. URL-derived metadata for the MeCo baseline.

For metadata conditioning, the default source identifier will be derived from the document URL, e.g. its hostname/domain.

Example:

```text
https://en.wikipedia.org/wiki/Transformer_(deep_learning_architecture)

→ en.wikipedia.org
```

The preprocessing pipeline should preserve at least:

```python
{
    "input_ids": ...,
    "url": ...,
    "domain": ...
}
```

Training examples should otherwise remain identical between experiments.

The training and validation splits, tokenization, sequence packing, and document ordering should be shared across runs whenever possible.

---

# Model

## Language Model

The causal LM will use a **custom Hugging Face-compatible implementation** modeled after **Llama 3.2 3B**.

The implementation should remain close to the standard Llama architecture while keeping individual components easy to modify for research.

The default architecture should therefore include:

* causal self-attention;
* rotary positional embeddings;
* grouped-query attention;
* RMSNorm;
* SwiGLU MLPs;
* tied or standard Llama output embeddings as appropriate;
* Llama-compatible initialization and configuration.

Tokenization uses the **Llama tokenizer**.

The model should subclass the appropriate Hugging Face `PreTrainedModel` / `PretrainedConfig` abstractions so that checkpoints can be loaded, saved, and evaluated using familiar HF interfaces.

The architecture should not depend directly on the official Llama implementation so that conditioning behavior can be modified freely.

---

# Text Conditioning

The proposed model adds a small bidirectional Transformer encoder:

```text
document tokens
      │
      ▼
┌───────────────────────┐
│ Bidirectional Encoder │
│     2–3 layers        │
└───────────────────────┘
      │
      ▼
document representation
      │
      ▼
 projection / normalization
      │
      ▼
conditioning vector z
```

The encoder should be substantially smaller than the causal LM.

Its exact width, number of heads, number of layers, and pooling mechanism should be configurable.

A reasonable initial default is:

* 2–3 Transformer layers;
* much smaller hidden dimension than the LM;
* bidirectional attention;
* one document representation obtained through a learned summary/CLS token or configurable pooling;
* normalization followed by projection into the LM embedding dimension.

The resulting vector is prepended to the LM embeddings:

```text
[z] [x1] [x2] [x3] ... [xT]
 │    │    │    │        │
 └──────── causal LM ─────┘
```

The loss is computed only over document tokens.

Conceptually:

$$
z = E(x)
$$

$$
\mathcal{L}
=
-\sum_t
\log p_\theta(x_t \mid x_{<t},z).
$$

The conditioning position itself does not receive a language-model loss.

---

# Unconditioned Mode

The text-conditioned model must also support evaluation without document information.

When conditioning is unavailable, the document vector should be replaced with a learned or fixed **null conditioning vector**:

```text
[z_null] [x1] [x2] ... [xT]
```

This keeps the LM input structure identical while removing document-specific information.

During the initial experiments, a configurable fraction of training examples will have their conditioning vector dropped and replaced with `z_null`.

Initial default:

```text
conditioning probability = 0.90
conditioning dropout     = 0.10
```

This is intentionally simple for the initial proof-of-concept. More sophisticated cooldown schedules will be evaluated separately.

MeCo has the same notion of unconditioned mode, with the domain-prefix tokens dropped entirely (rather than a null vector) for a dropped example -- see "Experiment E — Conditioning Schedule" below and `src/data/collator.py`'s `collate_lm_batch`.

---

# Initial Experiments

The initial project should remain small enough to run and interpret within roughly two weeks.

## Week 1: Proof of Concept

Train three models.

### Experiment A — Baseline

Standard Llama-style causal LM.

```text
document → causal LM → next-token loss
```

This establishes the normal pretraining curve.

---

### Experiment B — MeCo

Same causal LM backbone as the baseline, with URL/domain-derived metadata prepended during training.

```text
domain metadata + document → causal LM → next-token loss
```

This provides an established fine-grained conditioning baseline.

The LM parameter count should match the baseline.

`configs/experiment/meco.yaml` uses the same constant \(p = 0.9\) `conditioning_schedule` as text conditioning (C) and URL-encoded conditioning (H). A dropped example has its domain prefix removed entirely, so MeCo's unconditioned mode is trained too. `slurm/experiments/train_meco.slurm` passes `--p-condition 0.9`, which writes to `outputs/checkpoints/meco_p0.9_medium`. `outputs/checkpoints/meco_medium` holds the earlier run at MeCo's original \(p = 1.0\).

---

### Experiment C — Learned Text Conditioning

Same causal LM backbone as the baseline plus the lightweight bidirectional encoder.

```text
document ───────────────► encoder ─► z
   │                                │
   └────────────────────────────────┼─► causal LM
                                    │
                              prepended vector
```

The **causal backbone parameter count is identical** to the baseline.

The encoder therefore represents additional parameters and computation.

This experiment answers the simplest question:

> Does adding learned document conditioning to the same language model improve pretraining?

---

# Week 2: Controls and Scheduling

Assuming the initial text-conditioning result is promising, add two targeted experiments.

## Experiment D — Total-Parameter-Matched Text Conditioning

Reduce the causal LM slightly so that:

$$
P_\text{backbone} + P_\text{encoder}
\approx
P_\text{baseline}.
$$

This tests whether allocating part of a fixed parameter budget to a document encoder is still beneficial.

Exact equality is not necessary; the parameter counts should simply be close and clearly reported.

---

## Experiment E — Conditioning Schedule

Run one scheduling ablation chosen based on the Week 1 result.

For example, if the conditioned model performs very well with its encoder but poorly without it, gradually remove conditioning near the end of training:

```text
Early training:

P(conditioned) ≈ 1.0

             ↓

Final portion of training:

P(conditioned) → 0.0
```

The goal is to use document conditioning as **privileged information during learning** while encouraging the final causal backbone to operate independently.

Scheduling should be implemented generically so that later experiments can easily test:

* constant conditioning probability;
* random conditioning dropout;
* linear cooldown;
* cosine cooldown;
* step-based removal.

The first two-week experiment should not attempt a large schedule sweep.

Scheduling is not text-conditioning-specific: MeCo (`variant: meco`) supports the exact same `conditioning_schedule` config (`src/training/schedules.py`, `configs/experiment/meco_cooldown.yaml`) -- with the domain prefix dropped from the token sequence entirely for a given example, rather than substituted with a null vector, since MeCo's conditioning is ordinary tokens rather than something addable in embedding space (`src/data/collator.py`'s `collate_lm_batch`/`train_batches`). Every metric, plot, and table that tracks a text-conditioned run's self-conditioned vs. unconditioned loss (`self_conditioned_*`/`unconditioned_*` in `metrics.jsonl`) has a MeCo equivalent (`meco_conditioned_*`/`unconditioned_*`) already, and both are used generically wherever the analysis code looks up a run's "conditioned" column.

---

# Domain-Conditioned Training

The final evaluation asks whether a vector encoded from *another* document of the same domain ("Same-Domain") or a mean over several ("Domain-Mean") helps, but the Experiment C/D/E models only ever train on self-conditioning. Two further experiments train on those transfer conditions directly, so the encoder is pushed to capture reusable domain-level information rather than a per-document fingerprint:

| Experiment | Config | Conditioning vector for training document \(x_i\) |
| --- | --- | --- |
| F — Domain-Mean | `configs/experiment/text_conditioned_domain_mean.yaml` | \(\frac{1}{K}\sum_j E(x_j)\) over up to \(K\) = `n_support` (default 8) random other documents from \(x_i\)'s domain |
| G — Same-Domain | `configs/experiment/text_conditioned_domain_random.yaml` | \(E(x_j)\) for one random other document from \(x_i\)'s domain |

Details:

* **Domains** are URL hostnames (`src/data/metadata.py`'s `url_to_domain`, the same definition MeCo and the final eval use), grouped over the *entire* training split, not just the current batch or shard.
* **The target is never among its own conditioning documents.** A document whose domain has no other training document (or whose URL has no parseable domain) gets the null vector, i.e. the default "unknown" conditioning. `domain_conditioning.min_domain_docs` raises that bar.
* **`conditioning_schedule` still applies.** With probability \(1 - p_\text{condition}(t)\), a document with in-domain partners still gets the null vector. Both configs use constant \(p = 0.9\), like Experiment C. `--p-condition` / `--conditioning-schedule` give a sweep its own run directory, exactly as for the other runs. The schedule is applied at collation time, so a dropped example's partners are never fetched or encoded.
* **Gradients flow through the partner encodings**, so the encoder is trained only through documents other than the one being predicted.
* **Partners are deterministic** in `(seed, document)`, so a resumed run replays the same conditioning.
* **Selection is a config switch, not a new variant.** Both are `variant: text` runs with `text_conditioning_source: domain_mean | domain_random` (`self` is the default and the original behavior). Checkpoints therefore load, evaluate (`scripts/evaluate.py --mode final`) and benchmark exactly like any other text-conditioned run.

**Domain index.** Finding "another document from this domain" across ~150M documents needs a corpus-wide lookup. `scripts/build_domain_index.py` builds it once, in a few minutes of CPU time, into `data/domain_index/100BT/`: flat `.npy` arrays that every training rank memory-maps (`src/data/domain_index.py`). Training refuses to start if the index was built for a different shard split.

**Training-time metrics.** These runs log `val/domain_conditioned_{loss,ppl}` (the training condition; also the headline `val/loss`, and the metric `ckpt_best.pt` is selected on), plus the usual `val/self_conditioned_*` and `val/unconditioned_*`. Validation partners come from the validation shard only, mirroring the final eval. Because that shard is small, only about 20% of validation documents have an in-domain partner there; the rest are scored with the null vector. The final eval's Top-N columns are the clean comparison. `scripts/make_figures.py` picks these runs up automatically: their conditioning curves (`text_domain_{mean,random}_conditioning_*`) show unconditioned, domain-conditioned and self-conditioned lines.

```bash
jid=$(sbatch --parsable slurm/domain_conditioned/build_domain_index.slurm)
sbatch --dependency=afterok:$jid slurm/domain_conditioned/train_text_conditioned_domain_mean.slurm
sbatch --dependency=afterok:$jid slurm/domain_conditioned/train_text_conditioned_domain_random.slurm
# after training:
sbatch slurm/domain_conditioned/eval_text_conditioned_domain_mean.slurm
sbatch slurm/domain_conditioned/eval_text_conditioned_domain_random.slurm
sbatch slurm/domain_conditioned/bench_text_conditioned_domain_mean.slurm
sbatch slurm/domain_conditioned/bench_text_conditioned_domain_random.slurm
```

---

# URL-Encoded Conditioning

MeCo conditions on URL metadata by prepending the domain string as tokens. These experiments feed the **same metadata through the text encoder** instead, so the LM is conditioned on one learned vector \(z = E(\text{domain})\) in embedding space, exactly like text conditioning:

```text
MeCo:         [en . wikipedia . org \n\n] [x1] [x2] ...
URL-encoded:  [E("en.wikipedia.org")]      [x1] [x2] ...
```

This makes MeCo's conditioning a continuous representation: it costs the LM one position regardless of metadata length, has a trained null vector for "no metadata", and can be steered in vector space (any domain string, averages of domains, interpolations) rather than only by editing a token prefix.

| Experiment | Config | Encoder input | Schedule |
| --- | --- | --- | --- |
| H — URL-Encoded | `configs/experiment/text_conditioned_url.yaml` | domain (byte-for-byte MeCo's metadata) | constant \(p = 0.9\) |
| H′ — URL-Encoded (Cooldown) | `configs/experiment/text_conditioned_url_cooldown.yaml` | domain | `meco_cooldown.yaml`'s schedule (1.0 → 0.0 over the last 30%) |
| H″ — URL-Encoded (Full URL) | `configs/experiment/text_conditioned_url_full.yaml` | full URL, path included | constant \(p = 0.9\) |

Details:

* **Config.** All three are `variant: text` runs with `text_conditioning_source: url` and a `url_conditioning: {field: domain | url, max_tokens}` block (`src/data/url_conditioning.py`). Checkpoints are ordinary text-conditioned checkpoints, so `load_run`, evaluation, benchmarking and plotting all work unchanged.
* **Comparisons.** H vs. MeCo, and H′ vs. MeCo (Cooldown), differ only in how the identical domain string reaches the LM. H uses \(p = 0.9\) rather than MeCo's 1.0 so the null vector is actually trained and "No Cond." is meaningful, as for every other vector-conditioned run.
* **Unparseable URLs** get the null vector, as in the domain-conditioned runs.
* **Cheap encoder input.** It is only `max_tokens` long, versus 512 document tokens for text conditioning.
* **Training-time metrics:** `val/url_conditioned_{loss,ppl}` (the headline `val/loss`; `ckpt_best.pt` is selected on it) and `val/unconditioned_*`. There is no `self_conditioned_*`, because the encoder never sees document text.
* **Final eval** (`src/eval/conditioning.py`). Every mode encodes URLs, not text:
  * "Self" is the document's own URL vector, MeCo's "Self" in vector space.
  * "Cross-Domain" conditions on another domain's vector.
  * For `field: domain`, Same-Domain and Domain-Mean are `—`: every same-domain document has the identical domain string, so they would repeat "Self" (the same reasoning as for MeCo).
  * `field: url` reports all modes.
* **Benchmark steering.** `--mode text` on these runs encodes a string rather than averaging documents:
  * `--source wikipedia` encodes the source's `meco_domain`, the same string MeCo would prepend.
  * `--domain <any string>` encodes an ad hoc steering target, even one never seen in training.
  * `--url <url>` encodes that URL's metadata.
  * Result tags match the other runs (`text_<source>`), so they line up in `plot_benchmarks.py` and `benchmark_results.*`.

```bash
sbatch slurm/url_conditioned/train_text_conditioned_url.slurm
sbatch slurm/url_conditioned/train_text_conditioned_url_cooldown.slurm
sbatch slurm/url_conditioned/train_text_conditioned_url_full.slurm
# after training (same for _cooldown / _full):
sbatch slurm/url_conditioned/eval_text_conditioned_url.slurm
sbatch slurm/url_conditioned/bench_text_conditioned_url.slurm
python scripts/benchmarks/run_benchmarks.py --run-dir outputs/checkpoints/text_conditioned_url_medium --mode text --domain arxiv.org
```

---

# Encoder Analysis & Steering

The URL-encoded run (H″, `text_conditioned_url_full`) matches MeCo, and it also gives MeCo's conditioning a vector form, \(z = E(\text{url})\). These experiments look at that vector space, then use it for steering. They compare exactly three runs, listed in `configs/comparison.yaml`: **Baseline** (`baseline_medium`), **MeCo** (`meco_p0.9_medium`), and **URL-Encoded** (`text_conditioned_url_full_medium`).

## Shared checkpoint

A comparison is fair only if every run gets the same training budget. Every script below therefore works out, at startup, the furthest checkpoint all three runs share (`src/comparison/checkpoints.py`):

1. It first looks for the largest **step** that every run has a checkpoint at. In practice that is the final step, once every run has finished: `ckpt_last.pt` at `max_steps`.
2. Otherwise it takes the largest **token milestone** that every run has a snapshot for (`checkpoints/ckpt_step*_tok<N>M.pt`). Runs reach a milestone a few steps apart; the result records each run's exact step.

`ckpt_best.pt` is never used, since each run selects it by its own val loss. To see which checkpoint will be used, run `python scripts/comparison/show_checkpoint.py`.

Results are stored under that checkpoint's label (e.g. `step20000`): `outputs/experiments/<size>/benchmarks/<run>/<label>/<tag>/results.json`, `outputs/experiments/<size>/url_analysis/<label>/`, and `outputs/experiments/<size>/steering/<label>/`, with figures and tables under `outputs/results/<size>/{comparison,url_analysis,steering}/<label>/` (see [Outputs Layout](#outputs-layout)). They never mix with the default `ckpt_best.pt` results (`outputs/experiments/<size>/benchmarks/<run>/<tag>/`). If a run is still training when you launch, results land under an earlier label rather than being mixed, so launch once every run has finished.

## Running everything

Submit the jobs manually, in this order:

```bash
sbatch slurm/comparison/bench_comparison.slurm      # 13 tasks: Baseline / MeCo / URL, unconditioned + 5 sources
sbatch slurm/url_analysis/encoder_analysis.slurm    # encoder analysis; builds the domain library
# once encoder_analysis has finished (mixtures and task vectors need its library.pt):
sbatch slurm/steering/mixtures.slurm                # 6 shards
sbatch slurm/steering/task_vectors.slurm            # 30 = 3 runs x (9 tasks + all)
# once task_vectors has finished:
sbatch slurm/steering/task_transfer.slurm
# any time (CPU; draws whatever results exist so far):
sbatch slurm/comparison/report.slurm
```

`bash slurm/steering/submit_all.sh` queues the same jobs, with the dependencies above.

Figures and tables can be regenerated at any time from saved results (CPU only):

```bash
python scripts/comparison/make_comparison.py        # outputs/results/<size>/comparison/<label>/
python scripts/url_analysis/plot_encoder_analysis.py # outputs/results/<size>/url_analysis/<label>/{figures,tables}
python scripts/steering/make_steering_report.py      # outputs/results/<size>/steering/<label>/{figures,tables}
```

## Encoder analysis

`scripts/url_analysis/run_encoder_analysis.py` (`configs/url_analysis/encoder_analysis.yaml`, `src/url_analysis/`) scans URLs from 8 training shards (~8M URLs). It builds a **domain library** of the 5000 most frequent domains. Each domain gets two vectors:

* a **root** vector, \(E(\)`<scheme>://<host>/`\()\), which is the input `--source` steering uses;
* a **centroid**: the mean of \(E(\text{page URL})\) over 32 sampled pages, i.e. the average conditioning the LM saw for that domain.

Similarities are **centered cosines** (the library mean is subtracted first). Raw URL vectors share a large common direction: pages have raw pairwise cosine ≈ 0.8, and norm ≈ 45 against the null vector's ≈ 1. Without centering, every neighbor list is dominated by that shared direction.

| Result | Question |
| --- | --- |
| `tables/neighbors_{centroid,root}` | Nearest domains for the configured queries and the 60 most frequent domains |
| `tables/neighbors_extra` | Where the encoder places arbitrary strings, including rare or unseen ones (4chan, arxiv, a made-up news site, a single Wikipedia article URL) |
| `figures/query_similarity` | Similarity matrix among the query domains |
| `figures/domain_map_{tsne,pca}_{category,cluster}`, `tables/clusters` | Global map of the top 2000 domains, colored by keyword category and by KMeans cluster |
| `figures/domain_retrieval` | Does a page's vector retrieve its own domain (leave-one-out centroid)? Also the share of page-vector variance explained by domain |
| `figures/in_domain_<group>`, `tables/in_domain_*` | Structure within a domain: Wikipedia; Reddit by subreddit; StackExchange by site; NYT and Guardian by section and year; GitHub by owner; Blogspot. Includes silhouettes and cluster exemplar URLs |
| `figures/year_probe` | A linear probe for the year in a URL path: do the vectors carry time? |
| `figures/url_perturbations` | Sensitivity to surface edits (scheme, www, trailing slash, case, query string, root, bare domain), against same-domain and different-domain reference levels |
| `figures/pca_spectrum`, `figures/vector_stats`, `tables/summary` | Effective dimensionality, norms, the null vector, root vs. centroid |

To query nearest domains for any strings, or for their mean:

```bash
python scripts/url_analysis/query_neighbors.py https://boards.4chan.org/ https://arxiv.org/abs/1706.03762
python scripts/url_analysis/query_neighbors.py --mix https://en.wikipedia.org/ https://stackoverflow.com/
```

## Steering experiment 1: combining vectors

`scripts/steering/run_mixtures.py` (`configs/steering/mixtures.yaml`) evaluates the full lm-eval suite with the URL-encoded run conditioned on combined vectors:

* **Interpolations** \((1-\alpha) z_a + \alpha z_b\), with \(\alpha \in \{0, .25, .5, .75, 1\}\), for these pairs: wikipedia↔stackexchange, reddit↔4chan, wikipedia↔reddit, nytimes↔foxnews, wikipedia↔fanfiction, stackexchange↔github, and wikipedia↔simple.wikipedia. `interpolation: slerp` keeps the norm on the arc between the two vectors.
* **Steering strength** \(\bar z + s (z_a - \bar z)\) around the library mean \(\bar z\). Here \(s = 0\) is the "average domain", \(s = 1\) is the anchor, and \(s > 1\) exaggerates it. The sweep is done around the mean, not the origin or the null vector: scaling around those mostly changes the norm, which the LM never saw vary.
* **Equal-weight means** of several anchors.

The anchor strings are the ones `--source` encodes, so \(\alpha \in \{0, 1\}\) reproduces the comparison's `text_<source>` rows. 4chan is essentially absent from FineWeb, so its vector is the encoder extrapolating; `tables/neighbors_extra` shows where it lands. Outputs:

* `figures/mixtures_all_pairs_mean`, `figures/mixture_<a>+<b>_by_task`, `figures/mixtures_strength_mean` (with each run's unconditioned accuracy as a reference line);
* `tables/mixtures`;
* `tables/mixture_neighbors`: the nearest library domains of each mixed vector.

## Steering experiment 2: task-optimized conditioning

`scripts/steering/run_task_vectors.py` (`configs/steering/task_vectors.yaml`, `src/steering/`) searches, for each task and jointly over all tasks (`all`), for the conditioning that maximizes accuracy on the task's own training data.

**Splits** (`src/steering/task_data.py`):

* **fit** (2000 docs) comes from the train split.
* **select** (500 docs) is used for early stopping and domain choice. It is the validation split when lm-eval reports test (ARC, OpenBookQA), and otherwise train documents held out from fit.
* **eval** is lm-eval's own evaluation split, so the numbers line up with the benchmark table.

MMLU (`mmlu_cont`) has no train split, so it fits on its validation split and selects on dev; it is still reported on the full test set. A task with only a test split would fit and select on 30% of it and report on the remaining 70%, and `tables/sanity_scorer_vs_lm_eval` flags that.

**Methods:**

| Method | Runs | Conditioning |
| --- | --- | --- |
| none | all | Unconditioned (null vector / no prefix / baseline) |
| best source | MeCo, URL | The named source with the best select accuracy |
| best domain | MeCo, URL | The best of the 100 most frequent library domains plus the named sources, chosen on select. This is the discrete search MeCo can also do, so the two runs are compared like for like |
| simplex-optimized | URL | \(z = \sum_i \text{softmax}(w)_i L_i\): the best convex mix of 256 library centroids plus the null vector. Stays on the "real domain" manifold; the weights name the domains it uses |
| free-optimized | URL | \(z = z_0 + \delta\), with \(\delta\) unconstrained (norm-capped), initialized at the best domain. Its nearest library domains show what "a good vector for task X" resembles |

The optimized vectors are trained with Adam against the choice cross-entropy on fit (length-normalized for `acc_norm` tasks). The LM stays frozen. The iterate kept is the best on select, and step 0 counts, so a method is never worse on select than its initialization.

Every reported number, and every early-stopping decision, comes from `src/steering/scoring.py`'s `score_docs`. That is lm-eval's scoring path (`CtrlTokensLM` + `task.process_results`), so on full eval splits it reproduces `run_benchmarks.py`; `tables/sanity_scorer_vs_lm_eval` checks this. A differentiable bf16 path is used only for the training batches.

`scripts/steering/run_task_transfer.py` then evaluates each task's vector, and MeCo's chosen domain, on every other task.

Outputs:

* `tables/task_vectors_eval`, `tables/task_vectors_joint`;
* `tables/task_vectors_delta_vs_unconditioned`: paired per-document differences ± stderr;
* `tables/task_vectors_domains`: the chosen domains, simplex weights, and nearest domains of the free vector;
* `tables/domain_search_rankings`: the top domains per task, MeCo vs. URL;
* `figures/task_vectors_by_task`, `figures/task_vectors_training`, `figures/transfer_<run>_<method>`.

## Smoke tests

`--limit N` on `run_mixtures.py` writes to `<label>-limitN`, and `--smoke` on `run_task_vectors.py` / `run_task_transfer.py` writes to `<label>-smoke`. Both use tiny settings, and neither ever touches the real results.

---

# Fine-Tuning

The steering experiments keep the model frozen. These experiments fine-tune the compared runs (Baseline, MeCo and URL-Encoded, from `configs/comparison.yaml` or a per-scale `configs/comparison_<scale>.yaml`) on each benchmark task, and record accuracy against step. The question is whether pretraining with conditioning still helps once a task's own training data is available, and whether that conditioning is worth using while fine-tuning.

## Modes

`src/finetune/model.py` defines three modes, each available for all three runs:

| Mode (`--method`) | Trained | Conditioning |
| --- | --- | --- |
| `standard` | the whole LM | the run's unconditioned mode: nothing for Baseline or MeCo, the null vector for URL. The fine-tuning baseline |
| `conditioned` | the whole LM (the encoder stays frozen; `conditioned_encoder` trains it too) | fixed conditioning from a condition URL. MeCo gets its domain prefix tokens, built exactly as in training. URL gets \(z = E(\text{url})\) through its own encoder. Baseline gets MeCo's prefix too, as a control: the same input, on a model never trained to use it |
| `peft` | only the conditioning; the model is frozen | URL learns the vector in the conditioning slot it was pretrained to read, starting from \(E(\text{url})\) (`peft_null` starts from the null vector). Baseline and MeCo learn a soft prompt, initialized at the embeddings of MeCo's prefix tokens, so step 0 is exactly the `conditioned` input (`peft_1vec`: one vector initialized at the mean token embedding, parameter-matched with URL's \(z\)) |

The condition URL is `condition.url` in `configs/finetune/default.yaml`, `https://en.wikipedia.org/` by default, and `condition.per_task` can override it per task. Both runs read the same metadata they read in pretraining: MeCo prepends `en.wikipedia.org`, and the full-URL encoder reads the URL. It is the same string the `wikipedia` benchmark source steers with.

## Protocol

* **Splits.** These are `src/steering/task_data.py`'s. Fit is all of the task's training data. Select (500 docs) is used for model selection. Eval is lm-eval's own evaluation split. MMLU has no training split, so it fits on its validation split and selects on dev, as in the steering experiments.
* **Objective.** The default, `objective: choice_ce`, is the cross-entropy over a document's choices, scored by their (length-normalized, for `acc_norm` tasks) log-likelihoods. It is the loss the steering vectors were optimized with, so `peft` on the URL run is the steering experiment's free vector trained on all of the task data. `objective: lm` is ordinary supervised fine-tuning on the gold answer; its jobs are tagged `<method>_lm`.
* **Optimization.** Every job runs 500 steps of 32 documents, with warmup and cosine decay. Full fine-tuning uses AdamW at 1e-5. PEFT uses Adam at 0.015 × the learned vectors' RMS, which puts URL vectors (RMS ≈ 1.4) and token embeddings (≈ 100× smaller) on the same relative step.
* **Curves.** Step 0 and every 25 steps score the select split and a fixed 1000-document subset of eval (all of it for smaller splits).
* **Reported numbers.** The best iterate on select is kept (step 0 included). It is scored on the full eval split, alongside the final iterate and step 0 (the zero-shot number).
* **Seeds.** A job's `seed` (`--seed`, or `SEEDS="1 2"` for the SLURM scripts) sets only the training-data order. The splits and the curve subset are fixed by `splits.seed`, so every seed scores the same documents. Results go to `<method>/<task>/seed<k>/`. The report averages over seeds: table cells are the mean ± the std across seeds, curves are the mean with a ±1 std band, and `tables/finetune_vs_standard` gives each method minus standard fine-tuning, paired by seed. With a single seed, a cell's ± is only the stderr over eval documents, which says nothing about run-to-run variance.
* **Scoring.** Every accuracy uses lm-eval's `process_results` on fp32 log-likelihoods. `tests/test_finetune.py` checks these against the benchmark adapter's (`CtrlTokensLM`), so the zero-shot rows match the benchmark tables.

## Running it

```bash
python scripts/finetune/run_finetune.py --model url --method peft --task arc_easy
python scripts/finetune/run_finetune.py --model meco --method conditioned --task all   # one model, every task
python scripts/finetune/run_finetune.py --model baseline --method standard --task piqa --set objective=lm

bash slurm/finetune/submit_all.sh          # 3 runs x 3 methods x 9 tasks = 81 jobs + the report
bash slurm/finetune/submit_all.sh 1.5b     # the 1.5B runs (configs/comparison_1.5b.yaml)
python scripts/finetune/make_finetune_report.py [--comparison-config configs/comparison_1.5b.yaml]
```

`--model` is a comparison-config key or a run dir. It is not `--run`, because under torchrun that would be taken as an abbreviation of torchrun's own `--run-path`. `--checkpoint shared` (the default) starts from the furthest checkpoint the compared runs share, i.e. the final one once all have finished. `--set key=value` overrides any config entry, and `--smoke` writes tiny runs to `<label>-smoke`. Full fine-tuning at 8B does not fit on one GPU: `slurm/finetune/submit_all.sh 8b` gives those jobs 4 GPUs with FSDP (`--set fsdp=true` under torchrun), while PEFT stays on one.

Outputs:

* per job, `outputs/experiments/<size>/finetune/<run dir>/<ckpt label>/<method>/<task>/seed<k>/`:
  * `metrics.jsonl`: train loss, and select / eval-curve accuracy per eval step;
  * `result.json`: step-0, final and best accuracies, and the conditioning used;
  * `prefix.pt`: per-doc correctness and the learned vectors;
  * `model.pt` with `--set save_model=true`, loadable with `load_run`;
* the report, in `outputs/results/<size>/finetune/<ckpt label>/`:
  * `figures/curves_eval`: eval accuracy vs. step, per task and mean, colored by run with a line style per method;
  * `figures/curves_select`, `figures/curves_<method>`, `figures/curves_run_<run>`, `figures/curves_joint`;
  * `tables/finetune_results`: zero-shot and best-on-select accuracy;
  * `tables/finetune_final`, `tables/finetune_delta` (best minus step 0), `tables/finetune_vs_standard` (paired by seed), `tables/finetune_best_steps`.

---

# Scaling Up

`configs/scale/{1.5b,3b,8b}.yaml` train any experiment at a Llama scale. `scripts/train.py --scale <s>` merges the scale file over the experiment config, so the variant, conditioning and schedule stay the experiment's own. The run lands in `outputs/checkpoints/<experiment>_<s>`.

| Scale | Architecture (`configs/model/`) | Params | Layers × width, FFN | Heads (Q / KV) | Embeddings |
| --- | --- | --- | --- | --- | --- |
| `1.5b` | Beyond URLs' 1.5B Llama (`llama3_1.5b.yaml`) | 1.50B | 16 × 2048, 8192 | 32 / 8 | untied |
| `3b` | Llama-3.2-3B (`llama3_3b.yaml`) | 3.21B | 28 × 3072, 8192 | 24 / 8 | tied |
| `8b` | Llama-3.1-8B (`llama3_8b.yaml`) | 8.03B | 32 × 4096, 14336 | 32 / 8 | untied |

The 3B and 8B parameter counts match the released models exactly. The 1.5B model is the one Fan et al. pretrain in "URLs Help, Topics Guide" and "Beyond URLs" (arXiv:2511.21613). The papers give 16 layers, d_model 2048 and 1.5B parameters. The other fields are Llama-3.2-1B's, and with untied embeddings that comes to 1.50B parameters. All three use RoPE θ = 500k, RMSNorm ε = 1e-5, SwiGLU and the Llama 3 tokenizer. There are two differences from the released models. `max_position_embeddings` is the block size + 1, the extra position being the conditioning slot. There is no "llama3" RoPE scaling, which only matters beyond the original 8K context. `llama3_3b.yaml` previously had untied embeddings; Llama-3.2-3B ties them, and the config now does too.

## Data budgets

Each scale is scheduled for a Chinchilla-optimal run of 20 tokens per parameter, capped at the data. The training split holds 83.8B usable tokens: 147.6M documents averaging 568 tokens, each truncated at the 2048-token block.

| Scale | Target tokens | Docs / step (≈ tokens) | `max_steps` | Parallelism (8 × H200) | Eval snapshot every |
| --- | --- | --- | --- | --- | --- |
| `1.5b` | 100B (Beyond URLs) | 512 packed 4096-token rows (2.10M) | 47.7k | DDP, compiled, packed | 2B tokens (6GB) |
| `3b` | 64.3B | 2048 (1.16M) | 55.3k | DDP, 4 × 2048-token micro-batches | 1B tokens (13GB) |
| `8b` | 83.8B, one epoch (Chinchilla: 161B) | 4096 (2.33M) | 36k | FSDP + activation checkpointing | 2B tokens (32GB) |

`global_batch_docs` fixes the documents per step whatever the GPU count (`global_batch_seqs`, the packed rows, at 1.5B), and `grad_accum` is derived from it.

The 1.5B scale follows Beyond URLs' recipe: sequence length 4096, about 2.1M tokens per step, AdamW with weight decay 0.1, and a cosine LR from 3e-4 to 3e-5 with 2000 warmup steps, for 100B tokens. Training uses packed sequences (`src/data/packing.py`, see `packing.py` below). With packing, `tokens_seen` counts row positions, so every variant reaches the same token count at the same step. 100B tokens is about 1.1 epochs of the 91.2B-token FineWeb sample (no truncation, since packing splits long documents).

The runs are not expected to finish. Each model trains for as long as the budget allows, and models are compared at matching token milestones. Every run at a scale has the same LR schedule and data order, so the largest snapshot all of them share (`src/comparison/checkpoints.py`) is a fair comparison. The price is that a run stopped early is compared partway through its cosine schedule, at a higher LR than a finished run would have.

## Strong conditioning

`strong_conditioning: true` (`configs/experiment/text_conditioned_url_full_strong.yaml`) pins the conditioning slot at every depth. Before each attention and each MLP sublayer, the slot's hidden state is reset to the conditioning vector z (`DecoderLayer` in `src/models/modeling_llama.py`, `pinned_condition` in the LM config). Every layer therefore attends to z itself, not to a residual-stream state derived from it that can drift or wash out deeper in the model. The parameter count is unchanged, and z gets gradient from every layer.

* Packed training pins every segment's slot.
* Unpacked calls that pass `inputs_embeds` pin position 0, which is where every caller (validation, benchmarks, fine-tuning, steering) puts z. So a strong run evaluates through the same code as the other URL runs.
* Calls with `input_ids` pin nothing.

## FineWeb-Edu

`configs/data/fineweb_edu.yaml` switches the corpus to FineWeb-Edu's `sample/100BT` split, the data "URLs Help, Topics Guide" and "Beyond URLs" pretrain on (286GB of parquet, same columns as FineWeb). `bash slurm/data/prepare_fineweb_edu.sh` chains three CPU jobs:

1. download to `data/fineweb_edu_raw/`;
2. tokenize to `data/fineweb_edu_tokenized/100BT` (35-way array);
3. pretokenize the URL metadata.

Train on it with `scripts/train.py --data fineweb_edu` (or `DATA=fineweb_edu bash slurm/scale/submit.sh 1.5b <experiment> <hours>`). The run dir gets a `_fwedu` tag, for example `outputs/checkpoints/baseline_fwedu_1.5b`, so it never resumes a FineWeb run. The comparison set is `configs/comparison_1.5b_fwedu.yaml`. Its `url_strong` entry is the strong-conditioning run, which the generic comparison reports pick up; the steering reports only special-case `url`. Val loss is measured on FineWeb-Edu's own held-out shard, so it is not comparable with FineWeb runs.

## Throughput

**Packing** (`packing: true`, 1.5B). Packed 4096-token rows with per-document varlen FlashAttention, compiled decoder layers (`compile: true`) and DDP. Measured on one H200 (`slurm/scale/bench_1.5b_throughput.slurm`, `scripts/profile_packed_step.py`):

| 1.5B, micro-batch | Throughput (tokens/s/GPU) | Peak memory |
| --- | --- | --- |
| baseline, 8 × 4096 (the config) | 68k | 81 GiB |
| URL-encoded, 8 × 4096 | 67k | 81 GiB |
| URL-encoded, 4 × 4096 | 49k | 53 GiB |
| URL-encoded, 16 × 4096 | 62k | 136 GiB (allocator OOM warnings at validation) |

That is about 54h per 100B-token run on 8 GPUs. The text encoder's attention never uses the cuDNN backend (`src/models/text_encoder.py`). cuDNN builds a plan per input shape, and the encoder's shape changes every batch, which made URL-encoded training 20% slower than the baseline.

**Length bucketing** (`length_bucketing: true`, the 3B / 8B scale configs). Padding every document to 2048 tokens spent about 72% of compute on padding. With bucketing, each step still takes exactly the same documents, but they are sorted by length and grouped into micro-batches of at most `local_batch × 2048` tokens, each trimmed to its longest document. The step loss becomes the token-weighted mean over the step's documents, and token counts become exact. The medium configs keep the old padded behavior.

**Measured on H200s, 3B** (`slurm/scale/check_3b_parallelism.slurm`), in real tokens/s per GPU:

| Setup | Throughput | Peak memory |
| --- | --- | --- |
| DDP, no activation checkpointing, 4 × 2048 micro-batches | 13.0k | 115 GiB |
| DDP + checkpointing, 8 × 2048 | 9.7k | 92 GiB |
| FSDP + checkpointing, 8 × 2048 | 8.8k | 58 GiB |
| DDP + checkpointing, padded (no bucketing) | 4.1k | 92 GiB |

DDP without checkpointing at 8 × 2048 runs out of memory. The first row, now the 3B config, is about 3.2× faster than the padded setup and processes ~21B tokens per 60 hours on 8 GPUs. FSDP (`fsdp: true`, `src/utils/sharding.py`) is needed only at 8B. It uses PyTorch's `fully_shard` per decoder layer, keeps fp32 parameters with the same autocast as DDP, and writes the same full checkpoints. A run's `ckpt_last.pt` resumes only under the parallelism that wrote it.

## Running

```bash
bash slurm/scale/submit.sh 3b text_conditioned_url_full 60      # one run, up to 60h on 8 GPUs
bash slurm/scale/submit.sh 3b meco 60 0.9                       # MeCo, p_condition 0.9
DEPENDENCY=afterany:<jobid> bash slurm/scale/submit.sh 3b baseline 60
```

Rerunning the same command continues the run. Each job validates, writes `ckpt_last.pt` and exits 45 minutes before its time limit (`TRAIN_STOP_AT`). If a run crashes, for example on an out-of-memory batch, the job resumes it with activation checkpointing for the rest of its time. The scale configs skip `ckpt_best.pt`, since comparisons use the milestone snapshots.

Afterwards, every comparison, steering and fine-tuning script takes `--comparison-config configs/comparison_<scale>.yaml`, as do `slurm/scale/{eval,bench}.slurm`, `slurm/finetune/submit_all.sh <scale>`, `slurm/steering/submit_all.sh <scale>`, and (via `--export=ALL,SCALE=<scale>`) every `slurm/steering/*`, `slurm/url_analysis/*` and `slurm/comparison/report.slurm` job. Each config's `model_size` sends its results to `outputs/{experiments,results}/<scale>/`, so a scale's results never mix with the medium runs'.

A comparison config can pin its checkpoint (`checkpoint: tok7000M` or `step<N>`) instead of resolving the furthest shared one. `configs/comparison_3b.yaml` does this: only the URL run is trained so far, so it is evaluated at its 7B-token snapshot, and Baseline / MeCo results at the same milestone will line up with it once those runs reach 7B tokens and are added back to the config. Runs left out of a config are skipped by the launchers (`slurm/steering/submit_all.sh` sizes its arrays from the listed runs):

```bash
bash slurm/steering/submit_all.sh 3b                      # bench + encoder analysis + steering + report
RUNS=url bash slurm/finetune/submit_all.sh 3b             # fine-tuning + report
sbatch --export=ALL,RUN_DIR=outputs/checkpoints/text_conditioned_url_full_3b,"EXTRA_ARGS=--checkpoint checkpoints/ckpt_step0006016_tok7001M.pt" slurm/scale/eval.slurm
```

---

# Evaluation

The initial project intentionally focuses on **cross-entropy and perplexity**.

Downstream LM-Eval tasks can be added once the basic conditioning effect has been established.

This keeps the first experiment easy to run, debug, and interpret.

(Downstream lm-evaluation-harness benchmarking has since been added -- see "Downstream Benchmarking" below.)

---

# Downstream Benchmarking (lm-evaluation-harness)

Once a run's basic conditioning effect is established via the perplexity metrics above, `scripts/benchmarks/` runs the standard [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) downstream-task suite against it. This is a separate script/module tree from `scripts/evaluate.py` / `src/eval/` on purpose (see "Codebase Design" below): it is an occasional, final-checkpoint evaluation, not something the training loop or the perplexity eval path needs to know about.

## Task selection

`configs/benchmarks/tasks.yaml` centralizes the default task list, few-shot count, batch size, and an optional per-task doc `limit` (for fast smoke tests). The default suite is deliberately loglikelihood-only multiple-choice/binary accuracy tasks -- `hellaswag`, `piqa`, `arc_easy`, `arc_challenge`, `winogrande`, `boolq`, `openbookqa`, plus two knowledge tasks defined in `configs/benchmarks/lm_eval_tasks/`: `sciq_closed` (SciQ without its answer-bearing support passage) and `mmlu_cont` (MMLU, all subjects as one task, scored on the answer text) -- so every task is cheap enough to run repeatedly across every (run, conditioning) combination:

* No free-form generation tasks (e.g. GSM8K): the custom causal LM (`src/models/modeling_llama.py`) has no `generate()`/KV-cache implementation.
* No rolling-window perplexity tasks (e.g. WikiText): this project already tracks val loss/ppl throughout training (`src/eval/perplexity.py`); downstream accuracy is meant to add a complementary signal, not duplicate that.

Override the task list, `--limit`, `--num-fewshot`, or `--batch-size` per invocation via `scripts/benchmarks/run_benchmarks.py`'s flags.

## Conditioning during benchmarking

Benchmark tasks have no document to condition on the way FineWeb training documents do, so `src/benchmarks/conditioning.py` resolves a conditioning *setting* (fixed for an entire benchmark run) instead:

| `--mode` | applies to variant | conditioning applied |
| --- | --- | --- |
| `none` | any | that variant's own unconditioned mode (`z_null` for text, no domain prefix for MeCo) |
| `meco --source <name>` | `meco` | the named source's domain string (`configs/benchmarks/sources.yaml`), prepended exactly like training |
| `meco --domain <string>` | `meco` | an ad hoc domain/URL string, for one-off experimentation |
| `text --source <name>` | `text` | a domain-mean vector: `n_docs_per_source` (default 20) training documents matching the source are encoded and averaged, then prepended in embedding space for every request |
| `text --url <url>` | `text` | a vector from one specific training document |

`configs/benchmarks/sources.yaml` defines the named sources (`wikipedia`, `reddit`, `stackexchange`, `news`, `github` by default) so a source only has to be defined once and reused across every run/experiment.

## Running it

```bash
python scripts/benchmarks/run_benchmarks.py --run-dir outputs/checkpoints/baseline_medium --mode none
python scripts/benchmarks/run_benchmarks.py --run-dir outputs/checkpoints/meco_medium --mode meco --source wikipedia
python scripts/benchmarks/run_benchmarks.py --run-dir outputs/checkpoints/text_conditioned_medium --mode text --source wikipedia
python scripts/benchmarks/run_benchmarks.py --run-dir outputs/checkpoints/text_conditioned_medium --mode text --url https://en.wikipedia.org/wiki/Foo

# SLURM
sbatch --export=RUN_DIR=outputs/checkpoints/meco_medium,MODE=meco,SOURCE=wikipedia slurm/benchmarks/run_benchmarks.slurm
```

## Results and comparison

Every `(run, conditioning)` combination writes to its own predictable path, `outputs/experiments/<size>/benchmarks/<run_name>/<conditioning_tag>/results.json` (`src/benchmarks/results.py`) -- e.g. `outputs/experiments/medium/benchmarks/text_conditioned_medium/text_wikipedia/results.json` -- so results never overwrite each other and are easy to find later. `python scripts/benchmarks/plot_benchmarks.py list` prints every result discovered under `outputs/experiments/`.

`scripts/benchmarks/plot_benchmarks.py` builds the two requested comparisons, each as a per-task breakdown figure plus a macro-averaged summary figure (`outputs/results/<size>/benchmarks/figures/` by default):

```bash
# i. conditioning types against each other, at a matched source
python scripts/benchmarks/plot_benchmarks.py types \
    --run outputs/checkpoints/baseline_medium:unconditioned:Baseline \
    --run outputs/checkpoints/meco_medium:meco_wikipedia:MeCo \
    --run outputs/checkpoints/text_conditioned_medium:text_wikipedia:"Text Conditioning" \
    --stem conditioning_types_wikipedia

# ii. sources against each other, for one conditioning type
python scripts/benchmarks/plot_benchmarks.py sources \
    --run outputs/checkpoints/text_conditioned_medium:text_wikipedia:Wikipedia \
    --run outputs/checkpoints/text_conditioned_medium:text_reddit:Reddit \
    --run outputs/checkpoints/text_conditioned_medium:text_news:News \
    --stem text_conditioned_sources
```

`scripts/make_table.py` (the same script that builds the perplexity "Final Evaluation Table" below) also builds a flat `benchmark_results.{csv,md,tex}` from every discovered `results.json`: one row per `(run, conditioning_tag)` result, one column per task (its headline metric -- `acc_norm,none` if present, else `acc,none`), plus a macro-averaged `Mean` column. Cells are percentage accuracy with the task's reported stderr as an error bar where available, e.g. `40.5 ± 0.2` (`$\pm$` in the `.tex` table); the `Mean` column never gets an error bar of its own, since combining per-task stderrs into one macro-average stderr would assume independence that isn't actually known (README "use -- rather than inventing an equivalent measurement"). Unlike the plots above, it needs no `--run` list -- it just tables everything found under `outputs/experiments/`, one table per model size (`outputs/results/<size>/benchmarks/tables/`):

```bash
python scripts/make_table.py
```

---

# Training Curves

All experiments should periodically evaluate on a fixed validation subset.

Each run should record:

```text
train/loss
val/loss
val/perplexity
```

Conditioned models should additionally track the relevant validation modes.

For the text-conditioned model:

```text
val/self_conditioned_loss
val/self_conditioned_ppl

val/unconditioned_loss
val/unconditioned_ppl
```

These two metrics should be evaluated throughout training because they are inexpensive and directly measure the central tradeoff.

For MeCo:

```text
val/meco_conditioned_loss
val/meco_conditioned_ppl

val/unconditioned_loss
val/unconditioned_ppl
```

This should make it possible to produce figures such as:

### Figure 1 — Unconditioned Validation Perplexity

```text
Perplexity
    │
    │       Baseline
    │      /
    │     /       MeCo
    │    /       /
    │   /       /    TextCond
    │  /       /    /
    └──────────────────────
             Tokens
```

### Figure 2 — Text Conditioning

```text
Perplexity
    │
    │      Unconditioned
    │     /
    │    /
    │   /
    │  /   Self-conditioned
    │ /   /
    └──────────────────────
             Tokens
```

These plots are expected to be the main figures used for early presentations.

---

# Final Conditioning Evaluation

Some conditioning modes require encoding many additional documents and should therefore **not be recomputed at every checkpoint**.

Instead, run one more comprehensive evaluation on the final checkpoint.

For each validation document \(x_i\), evaluate:

## 1. No Conditioning

```text
z = z_null
```

Measures standalone LM performance.

---

## 2. Self Conditioning

```text
z = E(x_i)
```

The encoder sees the exact document being evaluated.

This represents the maximum amount of document-specific information available through the learned bottleneck.

---

## 3. Same-Domain Conditioning

Choose another document \(x_j\) from the same domain:

```text
domain(x_j) = domain(x_i)
x_j != x_i
```

and evaluate:

$$
p(x_i \mid E(x_j)).
$$

Example:

```text
Target:
Wikipedia article A

Conditioning:
Wikipedia article B
```

This tests whether the learned representation contains information that transfers between related documents.

---

## 4. Domain-Mean Conditioning

For each evaluated domain, construct a separate support set:

$$
D_s=\{d_1,\ldots,d_K\}.
$$

Encode the support documents and compute:

$$
z_\text{domain}
=
\frac{1}{K}
\sum_{j=1}^K E(d_j).
$$

Use this single vector to condition all validation documents from the corresponding domain.

Support documents must not overlap with the validation examples being scored.

The number of support documents \(K\) should be fixed across domains whenever possible.

This evaluation is particularly relevant to future steering applications: a representation can be estimated from a collection of reference documents and then reused for generation.

---

## Optional: Cross-Domain Conditioning

As a cheap additional sanity check, condition each document using a representation from an unrelated domain.

For example:

```text
Target:
Wikipedia document

Conditioning:
Reddit / news / blog document
```

This is not required for the initial proof-of-concept but can be included if the embeddings are already cached.

---

# Final Evaluation Table

The evaluation script should automatically produce a clean table such as:

| Model                  | No Cond. PPL ↓ | Self PPL ↓ | Same-Domain PPL ↓ | Domain-Mean PPL ↓ |
| ---------------------- | -------------: | ---------: | ----------------: | ----------------: |
| Baseline               |              — |          — |                 — |                 — |
| MeCo                   |              — |          — |                 — |                 — |
| TextCond               |              — |          — |                 — |                 — |
| TextCond Param-Matched |              — |          — |                 — |                 — |

For models where a conditioning mode does not apply, the table generator should use `—` rather than inventing an equivalent measurement.

The underlying values should also be saved in machine-readable form.

**Top-domain restriction (added after the initial implementation).** FineWeb's domain distribution is extremely long-tailed: even across an entire ~47k-document validation shard, most domains have only 1-2 documents, so requiring every evaluated domain to support a same-domain/domain-mean split (as originally implemented) left `same_domain`/`domain_mean` empty for nearly every run. `src/eval/conditioning.py` now selects the `n_domains` (default 50) validation domains with the most documents -- guaranteeing each one can support the split -- and restricts `same_domain`/`domain_mean`/`cross_domain` to those. `none`/`self` are reported at *two* scopes so every mode stays comparable: "Full" (the original domain-independent validation sample) and "Top-N" (the same restricted, per-domain document set `same_domain`/`domain_mean` use). The full per-domain breakdown (which documents, how many, and every mode's score for each of the `n_domains` domains) is saved alongside the aggregate -- see "Downstream Benchmarking"-style per-domain analysis in `final_conditioning.json`'s `top_domains.per_domain`, and `scripts/evaluate.py --table`'s `tables/domain_breakdown.csv`.

---

# Presentation-Ready Outputs

Analysis should be treated as part of the codebase rather than performed manually after training.

Every experiment should produce:

```text
metrics.jsonl
metrics.csv
```

The analysis scripts generate everything under `outputs/results/<size>/` (next section), one set per model size.

## Outputs Layout

`outputs/` is split into three trees, each organized by model size where that applies. Every script resolves these paths through `src/utils/paths.py`.

```text
outputs/
├── checkpoints/<run>/            (i) training runs: ckpt_last.pt, ckpt_best.pt, checkpoints/ snapshots,
│                                     metrics.jsonl/.csv, resolved_config.yaml (scripts/train.py)
├── experiments/<size>/           (ii) intermediate results, regenerated by the eval/benchmark jobs
│   ├── eval/<run>/final_conditioning.json                     scripts/evaluate.py --mode final
│   ├── benchmarks/<run>/[<ckpt label>/]<tag>/results.json     scripts/benchmarks/run_benchmarks.py
│   ├── finetune/<run>/<ckpt label>/<method>/<task>/seed<k>/   scripts/finetune/run_finetune.py
│   ├── url_analysis/<ckpt label>/                             library.pt, corpus.pt, results/*.json
│   └── steering/<ckpt label>/{task_vectors,mixtures}/
├── results/<size>/               (iii) presentation-ready figures and tables
│   ├── training/figures/                       training and validation curves  (scripts/make_figures.py)
│   ├── final_eval/{figures,tables}/            final conditioning eval   (make_conditioning_plot.py, make_table.py)
│   ├── benchmarks/{figures,tables}/            lm-eval results            (make_table.py, plot_benchmarks.py)
│   ├── comparison/<ckpt label>/{figures,tables}/     scripts/comparison/make_comparison.py
│   ├── finetune/<ckpt label>/{figures,tables}/       scripts/finetune/make_finetune_report.py
│   ├── steering/<ckpt label>/{figures,tables}/       scripts/steering/make_steering_report.py
│   └── url_analysis/<ckpt label>/{figures,tables}/   scripts/url_analysis/plot_encoder_analysis.py
└── archive/                      superseded runs (runs/) and tables (tables/)
```

`<size>` is the run's `model_size` (`medium`, `1.5b`, `3b`, `8b`). Figures and tables are only ever drawn from runs of the same size. `scripts/make_figures.py`, `make_table.py` and `make_conditioning_plot.py` write one set per size (`--sizes 3b` restricts them, and `--out-dir` combines the given runs into one set instead). The shared-checkpoint scripts take their size from the comparison config's `model_size`. `TC_OUTPUTS_DIR` moves the whole tree, for example for a smoke test.

Figures should use:

* consistent dimensions;
* consistent typography;
* readable axis labels;
* clear experiment names;
* minimal visual clutter;
* sufficiently large text for slides;
* high-resolution raster output;
* vector PDF output when possible.

The plotting API should allow optional smoothing for visual presentation, while always retaining the unsmoothed raw measurements.

Plot labels should use human-readable names such as:

```text
Baseline
MeCo
Text Conditioning
Text Conditioning (Param Matched)
```

rather than config or checkpoint identifiers.

All figure generation should be reproducible from saved metrics without needing to rerun evaluation.

---

# Suggested Repository Structure

```text
.
├── README.md
│
├── configs/
│   ├── model/
│   │   ├── llama3_1.5b.yaml / llama3_3b.yaml / llama3_8b.yaml
│   │   └── text_encoder.yaml
│   │
│   ├── scale/                  # 1.5b.yaml, 3b.yaml, 8b.yaml (scripts/train.py --scale)
│   ├── finetune/default.yaml   # scripts/finetune/run_finetune.py
│   │
│   ├── experiment/
│   │   ├── baseline.yaml
│   │   ├── meco.yaml
│   │   ├── meco_cooldown.yaml
│   │   ├── text_conditioned.yaml
│   │   ├── text_conditioned_param_matched.yaml
│   │   └── text_conditioned_cooldown.yaml
│   │
│   ├── train/
│   │   └── default.yaml
│   │
│   ├── eval/
│   │   └── conditioning.yaml
│   │
│   └── benchmarks/
│       ├── tasks.yaml
│       └── sources.yaml
│
├── src/
│   ├── data/
│   │   ├── fineweb.py
│   │   ├── metadata.py
│   │   ├── packing.py
│   │   └── collator.py
│   │
│   ├── models/
│   │   ├── configuration_llama.py
│   │   ├── modeling_llama.py
│   │   ├── text_encoder.py
│   │   └── conditioning.py
│   │
│   ├── training/
│   │   ├── trainer.py
│   │   ├── schedules.py
│   │   └── checkpointing.py
│   │
│   ├── eval/
│   │   ├── perplexity.py
│   │   ├── conditioning.py
│   │   └── domain_support.py
│   │
│   ├── analysis/
│   │   ├── curves.py
│   │   ├── figures.py
│   │   └── tables.py
│   │
│   ├── benchmarks/
│   │   ├── sources.py
│   │   ├── conditioning.py
│   │   ├── lm_wrapper.py
│   │   ├── results.py
│   │   └── plots.py
│   │
│   ├── finetune/               # model.py, data.py, loop.py, report.py
│   │
│   └── utils/
│       ├── distributed.py
│       ├── sharding.py         # FSDP
│       └── logging.py
│
├── scripts/
│   ├── prepare_data.py
│   ├── train.py
│   ├── evaluate.py
│   ├── make_figures.py
│   ├── benchmarks/
│   │   ├── run_benchmarks.py
│   │   └── plot_benchmarks.py
│   └── finetune/
│       ├── run_finetune.py
│       └── make_finetune_report.py
│
├── tests/
│   ├── test_model.py
│   ├── test_conditioning.py
│   ├── test_data.py
│   └── test_benchmarks.py
│
└── outputs/                     see "Outputs Layout"
    ├── checkpoints/<run>/
    ├── experiments/<size>/
    ├── results/<size>/
    └── archive/
```

---

# Codebase Design

## `src/data/`

Responsible only for dataset processing.

### `fineweb.py`

* load/stream FineWeb 100BT;
* create deterministic train/validation splits;
* expose text and URL metadata;
* tokenize documents.

### `metadata.py`

Parse metadata used by MeCo and group-based evaluation.

Example:

```python
url -> hostname -> domain_id
```

This module should be independent of model code.

### `packing.py`

Pack tokenized documents into fixed-context training examples while preserving document/source metadata (`packing: true`, the 1.5B scale; baseline, MeCo and URL-encoded runs only).

Documents are concatenated into `block_size` rows with no padding. Each document piece in a row is its own segment: RoPE positions restart at 0, and attention is causal within the segment only (`torch.nn.attention.varlen.varlen_attn`; a dense block-diagonal mask on CPU). A packed baseline row therefore computes exactly what running each document alone would (`tests/test_packing.py`). Conditioning stays document-level because every segment starts with its own document's conditioning:

```text
baseline   [x1 x2 ... EOS][y1 y2 ... EOS][z1 ...
MeCo       [dom_x \n\n x1 x2 ... EOS][dom_y \n\n y1 ... EOS] ...    prefix tokens loss-masked
URL        [E(url_x) x1 x2 ... EOS][E(url_y) y1 ... EOS] ...        one slot per segment, filled in embedding space
```

A document that doesn't fit in the rest of a row is split, and its continuation starts the next row with the same prefix or slot (Fan et al. 2025 do the same). The conditioning schedule's keep/drop decision is a deterministic hash per document, and the trainer stores each rank's stream position in `ckpt_last.pt`, so a resumed run continues the exact data stream. Domain and URL token ids are pretokenized once into `<tok_dir>/metadata/` (`python -m src.data.metadata_tokens --tok-dir ...`, `slurm/tokenize_metadata.slurm`). Without them the packer tokenizes live. A background thread packs ahead of the GPU.

### `collator.py`

Build batches for:

* baseline training;
* MeCo training;
* text-conditioned training.

---

# `src/models/`

## `modeling_llama.py`

Custom causal Llama implementation.

This should contain only the normal LM architecture and expose a clean way to provide `inputs_embeds`.

The conditioning implementation should not be deeply embedded throughout the Transformer blocks.

---

## `text_encoder.py`

Small bidirectional document encoder.

Responsibilities:

```text
tokens
  ↓
bidirectional Transformer
  ↓
pooling
  ↓
normalization
  ↓
document representation
```

The encoder should expose a simple interface such as:

```python
z = encoder(
    input_ids=input_ids,
    attention_mask=attention_mask,
)
```

---

## `conditioning.py`

Responsible for turning metadata or text representations into LM inputs.

For example:

```python
condition = conditioner(batch)

inputs_embeds = torch.cat(
    [condition[:, None, :], token_embeddings],
    dim=1,
)
```

It should support:

```text
none
text
meco
```

without requiring separate LM implementations.

Conditioning dropout should also live here or in the training scheduler rather than inside the encoder.

---

# `src/training/`

## `trainer.py`

Shared training loop for all experiments.

The experiment config, rather than separate training scripts, should determine whether a run is:

```text
baseline
meco
text-conditioned
```

The training implementation should support distributed multi-GPU execution from the beginning.

---

## `schedules.py`

Defines conditioning probability:

```python
p_condition(t)
```

Initial schedules might include:

```text
constant
linear_cooldown
cosine_cooldown
step_cooldown
```

For Week 1:

```text
constant p = 0.90
```

is sufficient.

---

# `src/eval/`

## `perplexity.py`

Standard validation CE/perplexity.

This is used frequently during training.

---

## `conditioning.py`

Evaluate:

```text
none
self
same_domain
domain_mean
cross_domain
```

The expensive group-level modes should primarily be used for final evaluation.

---

## `domain_support.py`

Build and cache support representations.

For example:

```text
domain
   │
   ├── document embedding 1
   ├── document embedding 2
   ├── document embedding 3
   │
   └── ...
        ↓
     mean vector
```

Caching these vectors avoids repeatedly running the encoder during evaluation.

---

# `src/analysis/`

Analysis code should consume saved metrics and evaluation outputs rather than checkpoints.

This ensures figures can be regenerated quickly.

Example:

```bash
python scripts/make_figures.py \
    outputs/checkpoints/baseline_medium \
    outputs/checkpoints/meco_medium \
    outputs/checkpoints/text_conditioned_medium
```

This should regenerate all comparison plots and final tables.

---

# Initial Success Criteria

The project is intended as a proof-of-concept rather than a complete evaluation of the method.

The strongest initial result would show:

### 1. Strong self-conditioning

$$
L_\text{self}
<
L_\text{none}.
$$

This demonstrates that the encoder can learn a useful document representation.

### 2. Transfer across documents

$$
L_\text{same-domain}
<
L_\text{none}.
$$

This would suggest that the representation captures reusable document-distribution information rather than only encoding the specific sample.

### 3. Useful aggregate representations

$$
L_\text{domain-mean}
<
L_\text{none}.
$$

This would provide early evidence that learned document vectors could eventually be used for reference-based generation steering.

### 4. Improved unconditioned learning

Most importantly for training efficiency:

$$
L_\text{TextCond,None}
<
L_\text{Baseline}.
$$

If this improvement persists after conditioning is removed, it would suggest that learned document context can improve the underlying causal LM rather than simply making prediction easier while the context is available.

---

# Scope of the Initial Project

The first two weeks deliberately do **not** attempt to answer every question about learned conditioning.

Out of scope for the initial proof-of-concept:

* LM-Eval downstream benchmarks;
* extensive conditioning-dimension sweeps;
* discrete/VQ conditioning;
* multiple encoder architectures;
* detailed steering experiments;
* representation probing;
* large scheduling sweeps;
* full compute-normalized scaling studies.

These become worthwhile only if the initial experiments show a clear conditioning effect.

The immediate goal is much narrower:

> **Determine whether a compact representation learned directly from document text provides a useful fine-grained conditioning signal during language-model pretraining, whether that signal transfers across related documents, and whether some of its benefit can be retained when conditioning is removed.**

---

# Implementation Notes

The codebase follows the layout proposed above with a few adjustments made while
implementing it:

* **`data/` vs `src/data/`.** `src/data/` holds the pipeline *code*
  (`download_fineweb.py`, `tokenize_fineweb.py`, `tokenizer_utils.py`,
  `metadata.py`, `shards.py`, `packing.py`, `collator.py`); `data/` at the repo
  root holds the actual *artifacts* (`data/fineweb_raw/`,
  `data/fineweb_tokenized/100BT/`), matching `outputs/` being an artifacts
  directory rather than code. The `.bin` / `_index.parquet` / `manifest.json` /
  `.done` shard format is unchanged from the original preprocessing scripts, so
  any in-progress or completed tokenization is compatible as-is.
* **Variants.** The three experiments share one `ConditionedLM`
  (`src/models/conditioned_lm.py`) parameterized by `variant: none | meco |
  text`, rather than separate model classes, per the "without requiring
  separate LM implementations" guidance for `conditioning.py`.
* **Model scale.** `configs/model/llama_medium.yaml` (~400M params, a more
  standard Llama-style scale) is the default for all experiment configs, not
  the literal Llama-3.2-3B spec — training an actual 3B model within the
  two-week POC window isn't realistic on a shared cluster.
  `configs/model/llama_small.yaml` (~170M) is kept around for faster smoke
  tests. The Llama 3 1B / 3B / 8B architectures (`configs/model/llama3_*.yaml`)
  are trained through `scripts/train.py --scale` (see "Scaling Up"). Every
  `configs/model/*.yaml` sets a `model_size` (`small` / `medium` / `1.5b` / `3b`
  / `8b`) that `scripts/train.py` appends to the run's output directory
  (`outputs/checkpoints/<experiment>_<model_size>`), so switching an experiment's model
  scale starts a fresh run dir rather than resuming or overwriting a
  smaller-model run's checkpoints and logs.
* **Final conditioning eval for MeCo.** MeCo's domain string is a deterministic
  function of a document's URL, not a learned per-document representation, so
  "same-domain" conditioning would just repeat "self" (the same domain string),
  and "domain-mean" has no meaning for text metadata. `src/eval/conditioning.py`
  reports MeCo's normal domain-conditioned eval under the "Self" column and
  leaves "Same-Domain"/"Domain-Mean" as `--` for it; see that module's
  docstring for the reasoning.
* **Top-domain restriction for same-domain/domain-mean conditioning.**
  FineWeb's domain distribution is extremely long-tailed (see "Final
  Evaluation Table" above), so `src/eval/conditioning.py` only ever runs
  same-domain/domain-mean/cross-domain conditioning on the `n_domains` domains
  with the most validation documents (`src/eval/domain_support.py`'s
  `top_domains`, selected by scanning the *entire* validation shard -- cheap,
  since it's just a URL histogram, not a model forward pass). `none`/`self`
  are reported both on the original domain-independent sample ("overall") and
  restricted to that same top-domain document set ("top_domains"), so every
  mode in a result is being compared on the same documents; the per-domain
  breakdown underlying the aggregate is kept too.
* **Benchmarking kept out of `src/eval/`/`scripts/`.** `src/benchmarks/` and
  `scripts/benchmarks/` are a deliberately separate module/script tree from
  perplexity evaluation, per the "Downstream Benchmarking" section above --
  benchmarking is occasional and final-checkpoint-only, so it doesn't need to
  share a namespace with code the training loop imports every step.
* **Why not `lm_eval.models.huggingface.HFLM`.** `src/benchmarks/lm_wrapper.py`
  implements `lm_eval.api.model.LM` directly rather than subclassing HFLM:
  HFLM assumes `generate()`/KV-cache support (this project's causal LM has
  none) and has no hook for splicing a conditioning vector into embedding
  space before the first token without breaking its internal sequence-length
  bookkeeping. The custom adapter only supports loglikelihood-based tasks,
  which is exactly what `configs/benchmarks/tasks.yaml`'s default suite uses.

## Running things

```bash
# 1. data (see slurm/ for the SLURM versions used on the actual 100BT sample)
python scripts/prepare_data.py --limit-shards 2   # smoke test
sbatch slurm/download_fineweb.slurm
sbatch slurm/tokenize_fineweb.slurm

# 2. train (experiment config selects the variant)
python scripts/train.py --experiment configs/experiment/baseline.yaml
sbatch --export=EXPERIMENT=text_conditioned slurm/train.slurm

# 3. evaluate (run dirs are named <experiment>_<model_size>, e.g. _medium by default)
python scripts/evaluate.py --run-dir outputs/checkpoints/text_conditioned_medium --mode final
python scripts/evaluate.py --table outputs/checkpoints/baseline_medium outputs/checkpoints/meco_medium outputs/checkpoints/text_conditioned_medium

# 4. figures
python scripts/make_figures.py                      # every run, one set per model size
python scripts/make_figures.py --sizes 3b           # outputs/results/3b/training/figures/

# 5. downstream benchmarks (see "Downstream Benchmarking" above)
python scripts/benchmarks/run_benchmarks.py --run-dir outputs/checkpoints/text_conditioned_medium --mode text --source wikipedia
python scripts/benchmarks/plot_benchmarks.py list

# 6. domain-conditioned training (see "Domain-Conditioned Training" above)
python scripts/build_domain_index.py
python scripts/train.py --experiment configs/experiment/text_conditioned_domain_mean.yaml 

# 7. URL-encoded conditioning (see "URL-Encoded Conditioning" above; no index needed)
python scripts/train.py --experiment configs/experiment/text_conditioned_url.yaml 

# 8. fine-tuning on the benchmark tasks (see "Fine-Tuning" above)
python scripts/finetune/run_finetune.py --model url --method peft --task arc_easy

# 9. a Llama 3 scale (see "Scaling Up" above)
torchrun --standalone --nproc_per_node=8 scripts/train.py --experiment configs/experiment/meco.yaml \
    --p-condition 0.9 --scale 1.5b

# tests
pip install -r requirements.txt
pytest tests/
```
