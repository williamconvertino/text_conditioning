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

# Evaluation

The initial project intentionally focuses on **cross-entropy and perplexity**.

Downstream LM-Eval tasks can be added once the basic conditioning effect has been established.

This keeps the first experiment easy to run, debug, and interpret.

(Downstream lm-evaluation-harness benchmarking has since been added -- see "Downstream Benchmarking" below.)

---

# Downstream Benchmarking (lm-evaluation-harness)

Once a run's basic conditioning effect is established via the perplexity metrics above, `scripts/benchmarks/` runs the standard [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) downstream-task suite against it. This is a separate script/module tree from `scripts/evaluate.py` / `src/eval/` on purpose (see "Codebase Design" below): it is an occasional, final-checkpoint evaluation, not something the training loop or the perplexity eval path needs to know about.

## Task selection

`configs/benchmarks/tasks.yaml` centralizes the default task list, few-shot count, batch size, and an optional per-task doc `limit` (for fast smoke tests). The default suite is deliberately loglikelihood-only multiple-choice/binary accuracy tasks -- `hellaswag`, `piqa`, `arc_easy`, `arc_challenge`, `winogrande`, `boolq`, `openbookqa`, `lambada_openai` -- so every task is cheap enough to run repeatedly across every (run, conditioning) combination:

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
python scripts/benchmarks/run_benchmarks.py --run-dir outputs/baseline_medium --mode none
python scripts/benchmarks/run_benchmarks.py --run-dir outputs/meco_medium --mode meco --source wikipedia
python scripts/benchmarks/run_benchmarks.py --run-dir outputs/text_conditioned_medium --mode text --source wikipedia
python scripts/benchmarks/run_benchmarks.py --run-dir outputs/text_conditioned_medium --mode text --url https://en.wikipedia.org/wiki/Foo

# SLURM
sbatch --export=RUN_DIR=outputs/meco_medium,MODE=meco,SOURCE=wikipedia slurm/benchmarks/run_benchmarks.slurm
```

## Results and comparison

Every `(run, conditioning)` combination writes to its own predictable path, `outputs/<run_name>/benchmarks/<conditioning_tag>/results.json` (`src/benchmarks/results.py`) -- e.g. `outputs/text_conditioned_medium/benchmarks/text_wikipedia/results.json` -- so results never overwrite each other and are easy to find later. `python scripts/benchmarks/plot_benchmarks.py list` prints every result discovered under `outputs/`.

`scripts/benchmarks/plot_benchmarks.py` builds the two requested comparisons, each as a per-task breakdown figure plus a macro-averaged summary figure (`outputs/benchmark_figures/` by default):

```bash
# i. conditioning types against each other, at a matched source
python scripts/benchmarks/plot_benchmarks.py types \
    --run outputs/baseline_medium:unconditioned:Baseline \
    --run outputs/meco_medium:meco_wikipedia:MeCo \
    --run outputs/text_conditioned_medium:text_wikipedia:"Text Conditioning" \
    --stem conditioning_types_wikipedia

# ii. sources against each other, for one conditioning type
python scripts/benchmarks/plot_benchmarks.py sources \
    --run outputs/text_conditioned_medium:text_wikipedia:Wikipedia \
    --run outputs/text_conditioned_medium:text_reddit:Reddit \
    --run outputs/text_conditioned_medium:text_news:News \
    --stem text_conditioned_sources
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

The analysis scripts should generate:

```text
figures/
├── training_loss.png
├── training_loss.pdf
├── validation_ppl.png
├── validation_ppl.pdf
├── text_conditioning_ppl.png       # any text_conditioned* run
├── text_conditioning_ppl.pdf
├── meco_conditioning_ppl.png       # any meco/meco_cooldown run
└── meco_conditioning_ppl.pdf

tables/
├── final_conditioning.csv
├── final_conditioning.md
└── final_conditioning.tex
```

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
│   │   ├── llama3_3b.yaml
│   │   └── text_encoder.yaml
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
│   └── utils/
│       ├── distributed.py
│       └── logging.py
│
├── scripts/
│   ├── prepare_data.py
│   ├── train.py
│   ├── evaluate.py
│   ├── make_figures.py
│   └── benchmarks/
│       ├── run_benchmarks.py
│       └── plot_benchmarks.py
│
├── tests/
│   ├── test_model.py
│   ├── test_conditioning.py
│   ├── test_data.py
│   └── test_benchmarks.py
│
└── outputs/
    └── <experiment_name>/
        ├── checkpoints/
        ├── metrics.jsonl
        ├── metrics.csv
        ├── figures/
        ├── tables/
        └── benchmarks/
            └── <conditioning_tag>/
                └── results.json
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

Pack tokenized documents into fixed-context training examples while preserving document/source metadata.

Because text conditioning is document-level, packing behavior must not accidentally mix unrelated documents under a single conditioning representation.

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
    outputs/baseline_medium \
    outputs/meco_medium \
    outputs/text_conditioned_medium
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
  tests, and `configs/model/llama3_3b.yaml` matches the README's named target
  and can be swapped in later by pointing an experiment config's `include:` at
  it. Every `configs/model/*.yaml` sets a `model_size` (`small` / `medium` /
  `full`) that `scripts/train.py` appends to the run's output directory
  (`outputs/<experiment>_<model_size>`), so switching an experiment's model
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
python scripts/train.py --experiment configs/experiment/baseline.yaml --out-dir outputs
sbatch --export=EXPERIMENT=text_conditioned slurm/train.slurm

# 3. evaluate (run dirs are named <experiment>_<model_size>, e.g. _medium by default)
python scripts/evaluate.py --run-dir outputs/text_conditioned_medium --mode final
python scripts/evaluate.py --table outputs/baseline_medium outputs/meco_medium outputs/text_conditioned_medium

# 4. figures
python scripts/make_figures.py outputs/baseline_medium outputs/meco_medium outputs/text_conditioned_medium \
    --out-dir outputs/figures

# 5. downstream benchmarks (see "Downstream Benchmarking" above)
python scripts/benchmarks/run_benchmarks.py --run-dir outputs/text_conditioned_medium --mode text --source wikipedia
python scripts/benchmarks/plot_benchmarks.py list

# tests
pip install -r requirements.txt
pytest tests/
```
