"""A pretrained run (src/models/conditioned_lm.py's ConditionedLM) wrapped for
fine-tuning on a benchmark task (README "Fine-Tuning"). Three modes, each
defined for all three variants:

  standard     Full fine-tuning in the run's own unconditioned mode: no
               prefix (baseline, MeCo) or the null vector (text / URL runs).
               The baseline every other mode is compared against.
  conditioned  Full fine-tuning with fixed conditioning from a condition URL:
               MeCo's domain prefix tokens, built exactly as in training
               (src/data/collator.py's _meco_prefix_ids), or, for a
               vector-conditioned run, z = E(metadata(url)) through the
               run's own encoder (src/data/url_conditioning.py). The baseline
               gets MeCo's prefix too, as a control: same input, but a model
               never trained to use it. The encoder stays frozen unless
               train_encoder is set.
  peft         The whole model frozen; only the conditioning is learned.
               Text / URL runs learn the vector z in the conditioning slot the
               model was pretrained to read (initialized at E(url) or the null
               vector). Baseline / MeCo learn a soft prompt of token
               embeddings prepended to the input (initialized at the
               embeddings of MeCo's prefix for the condition URL, so step 0
               is exactly the "conditioned" input, or at n_vectors copies of
               the mean token embedding).

FinetuneModel.forward is the single scoring path for training and
evaluation: per-request log p(continuation) and greedy-match flags, the
quantities lm-eval's loglikelihood requests return. In fp32 it reproduces
src/benchmarks/lm_wrapper.py's CtrlTokensLM scores (tests/test_finetune.py),
so fine-tuning accuracies line up with the benchmark tables. Running
everything through forward() also keeps FSDP (src/utils/sharding.py) correct,
which needs the root module's forward as the entry point.
"""

import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.data.collator import _meco_prefix_ids
from src.data.url_conditioning import build_url_source
from src.training.trainer import build_model

MODES = ("standard", "conditioned", "peft")
PEFT_INITS = ("condition", "null_vector", "vocab_mean")


class FinetuneModel(nn.Module):
    def __init__(self, base, mode, prefix_ids=(), encoder_ids=None, soft_prompt=None):
        super().__init__()
        assert mode in MODES
        self.base = base
        self.mode = mode
        self.variant = base.variant
        self.prefix_ids = list(prefix_ids)
        self.register_buffer("encoder_ids", None if encoder_ids is None else
                             torch.tensor([encoder_ids], dtype=torch.long), persistent=False)
        self.soft_prompt = nn.Parameter(soft_prompt.detach().clone().float()) if soft_prompt is not None else None

    @property
    def n_prefix_vectors(self):
        """Embedding-space positions ahead of the token sequence."""
        if self.soft_prompt is not None:
            return self.soft_prompt.shape[0]
        return 1 if self.variant == "text" else 0

    def prefix_vectors(self):
        """[P, H] vectors prepended in embedding space, or None."""
        if self.soft_prompt is not None:
            return self.soft_prompt
        if self.variant != "text":
            return None
        conditioner = self.base.conditioner
        if self.encoder_ids is not None:
            mask = torch.ones_like(self.encoder_ids)
            return conditioner.encoder(input_ids=self.encoder_ids, attention_mask=mask)
        return conditioner.null_vector.unsqueeze(0)

    def forward(self, input_ids, score_mask):
        """input_ids [B, L]: token sequences (self.prefix_ids included),
        right-padded. score_mask [B, L]: the continuation positions to score.
        Returns (sum log p over scored tokens [B] fp32, all-greedy [B] bool).

        Token j of input_ids sits at position P + j of the LM input (P prefix
        vectors first) and is predicted by the hidden state at P + j - 1, so
        a token at LM position 0 is never scored -- as in CtrlTokensLM. The
        lm_head (and its fp32 log-softmax) is applied only at scored
        positions, not over [B, L, vocab]."""
        lm = self.base.lm
        B = input_ids.shape[0]
        x = lm.get_input_embeddings()(input_ids)
        pv = self.prefix_vectors()
        P = 0
        if pv is not None:
            P = pv.shape[0]
            x = torch.cat([pv.to(x.dtype).unsqueeze(0).expand(B, -1, -1), x], dim=1)
        h = lm.model(inputs_embeds=x).last_hidden_state

        rows, cols = score_mask.nonzero(as_tuple=True)
        keep = cols + P >= 1
        rows, cols = rows[keep], cols[keep]
        logp = F.log_softmax(lm.lm_head(h[rows, cols + P - 1]).float(), dim=-1)
        targets = input_ids[rows, cols]
        tok_lp = logp.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
        sums = torch.zeros(B, device=tok_lp.device, dtype=tok_lp.dtype).index_add(0, rows, tok_lp)
        miss = torch.zeros(B, device=tok_lp.device, dtype=torch.long).index_add(
            0, rows, (logp.argmax(-1) != targets).long())
        return sums, miss == 0

    def trainable_parameters(self):
        return [p for p in self.parameters() if p.requires_grad]


def load_base_model(run_dir, checkpoint, tokenizer, cfg):
    """ConditionedLM with `checkpoint`'s weights, on CPU. Built on the meta
    device and filled straight from the (memory-mapped) checkpoint, so an 8B
    model is never randomly initialized first or held twice in memory -- this
    matters when every FSDP rank loads it."""
    with torch.device("meta"):
        model, _ = build_model(cfg, tokenizer)
    ckpt = torch.load(os.path.join(run_dir, checkpoint), map_location="cpu", mmap=True, weights_only=False)
    model.load_state_dict(ckpt["model"], assign=True)
    if cfg.get("tie_word_embeddings", False):
        # assign=True installs the two checkpoint entries as separate Parameters
        model.lm.lm_head.weight = model.lm.model.embed_tokens.weight
    return model


def encoder_tokens(cfg, tokenizer, url):
    """What a text run's encoder reads for the condition URL: its URL
    metadata (domain or full URL, as in training) for a URL-encoded run, else
    the string itself as document text."""
    url_source = build_url_source(cfg, tokenizer)
    if url_source is not None:
        docs = url_source.for_url(url)
        if not docs:
            raise ValueError(f"condition url {url!r} has no parseable domain")
        return list(docs[0]["tokens"])
    return tokenizer.encode(url, add_special_tokens=False)[: cfg.get("encoder_max_tokens", 512)]


def build_finetune_model(base, cfg, tokenizer, mode, condition_url=None, peft_init="condition",
                         n_vectors=1, train_encoder=False):
    """Wrap `base` for `mode` and set requires_grad accordingly. Returns
    (FinetuneModel, info) -- info records the conditioning actually used."""
    variant = cfg["variant"]
    info = {"mode": mode, "variant": variant, "condition_url": None}
    prefix_ids, enc_ids, soft = [], None, None
    meco_prefix = lambda: _meco_prefix_ids(tokenizer, condition_url, cfg.get("meco_max_tokens", 48))

    needs_url = mode == "conditioned" or (mode == "peft" and peft_init == "condition")
    if needs_url:
        if not condition_url:
            raise ValueError(f"mode {mode!r} (peft init {peft_init!r}) needs a condition url")
        info["condition_url"] = condition_url

    if mode == "conditioned":
        if variant == "text":
            enc_ids = encoder_tokens(cfg, tokenizer, condition_url)
            info["encoder_input"] = tokenizer.decode(enc_ids)
        else:
            prefix_ids = meco_prefix()
            info["prefix"] = tokenizer.decode(prefix_ids)

    elif mode == "peft":
        if peft_init not in PEFT_INITS:
            raise ValueError(f"peft init must be one of {PEFT_INITS}, got {peft_init!r}")
        with torch.no_grad():
            if variant == "text":
                if peft_init == "condition":
                    ids = encoder_tokens(cfg, tokenizer, condition_url)
                    info["encoder_input"] = tokenizer.decode(ids)
                    t = torch.tensor([ids])
                    soft = base.conditioner.encoder(input_ids=t, attention_mask=torch.ones_like(t))
                elif peft_init == "null_vector":
                    soft = base.conditioner.null_vector.unsqueeze(0)
                else:
                    raise ValueError("text runs learn their conditioning vector: peft init must be "
                                     "condition | null_vector")
            else:
                emb = base.lm.get_input_embeddings().weight
                if peft_init == "condition":
                    ids = meco_prefix()
                    info["prefix"] = tokenizer.decode(ids)
                    soft = emb[torch.tensor(ids)]
                elif peft_init == "vocab_mean":
                    soft = emb.mean(0, keepdim=True).repeat(n_vectors, 1)
                else:
                    raise ValueError("baseline / MeCo runs have no null vector: peft init must be "
                                     "condition | vocab_mean")
        info["peft_init"] = peft_init

    model = FinetuneModel(base, mode, prefix_ids=prefix_ids, encoder_ids=enc_ids, soft_prompt=soft)
    for p in model.parameters():
        p.requires_grad_(False)
    if mode == "peft":
        model.soft_prompt.requires_grad_(True)
    else:
        for p in base.lm.parameters():
            p.requires_grad_(True)
        if mode == "conditioned" and variant == "text" and train_encoder:
            for p in base.conditioner.encoder.parameters():
                p.requires_grad_(True)
    info["train_encoder"] = bool(train_encoder and mode == "conditioned" and variant == "text")
    info["n_prefix_vectors"] = model.n_prefix_vectors
    info["n_prefix_tokens"] = len(prefix_ids)
    info["n_trainable"] = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return model, info
