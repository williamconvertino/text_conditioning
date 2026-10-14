"""Custom Llama-3-style causal LM: causal self-attention, rotary positional
embeddings, grouped-query attention, RMSNorm, SwiGLU MLPs. Deliberately
reimplemented (not a subclass of the official Llama modeling code) so the
conditioning behavior added in src/models/conditioning.py and
src/models/conditioned_lm.py can be changed freely without fighting an upstream
implementation.

Conditioning itself does not appear anywhere in this file: the model only knows
how to run over `input_ids` or a caller-supplied `inputs_embeds` (e.g. a
conditioning vector concatenated with token embeddings) and exposes a generic
next-token loss helper. Everything variant-specific lives in conditioning.py /
conditioned_lm.py, per the README's "the conditioning implementation should not
be deeply embedded throughout the Transformer blocks".
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
try:
    from torch.nn.attention.varlen import varlen_attn
except ImportError:          # torch < 2.9: packed batches fall back to a dense mask
    varlen_attn = None
from transformers import PreTrainedModel
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast

from src.models.configuration_llama import LlamaConfig


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        dtype = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x.to(dtype) * self.weight


def build_rope_cache(head_dim, seq_len, theta, device, dtype):
    inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))
    t = torch.arange(seq_len, device=device).float()
    freqs = torch.outer(t, inv_freq)
    emb = torch.cat([freqs, freqs], dim=-1)
    return emb.cos().to(dtype), emb.sin().to(dtype)


def rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([-x2, x1], dim=-1)


def apply_rope(q, k, cos, sin):
    # cos/sin: [T, head_dim] broadcast over [B, H, T, head_dim]
    cos = cos[None, None, :, :]
    sin = sin[None, None, :, :]
    q = (q * cos) + (rotate_half(q) * sin)
    k = (k * cos) + (rotate_half(k) * sin)
    return q, k


def apply_rope_packed(q, k, cos, sin):
    # q/k: [B, T, H, head_dim]; cos/sin: [B, T, head_dim], already gathered at
    # each token's within-segment position
    cos, sin = cos[:, :, None, :], sin[:, :, None, :]
    return (q * cos) + (rotate_half(q) * sin), (k * cos) + (rotate_half(k) * sin)


def segment_causal_mask(cu_seqlens, n):
    """Dense [n, n] bool mask: causal within each segment of cu_seqlens, nothing
    across segments. The CPU / no-varlen fallback for packed attention."""
    idx = torch.arange(n, device=cu_seqlens.device)
    seg = torch.searchsorted(cu_seqlens[1:].to(torch.long), idx, right=True)
    return (seg[:, None] == seg[None, :]) & (idx[None, :] <= idx[:, None])


def packed_attention(q, k, v, cu_seqlens, max_seqlen):
    """q: [N, H, D], k/v: [N, H_kv, D] over all packed tokens -> [N, H, D],
    causal within each segment of cu_seqlens (int32 [S + 1])."""
    if varlen_attn is not None and q.is_cuda:
        return varlen_attn(q, k, v, cu_seqlens, cu_seqlens, max_seqlen, max_seqlen,
                           window_size=(-1, 0), enable_gqa=k.shape[1] != q.shape[1])
    n_rep = q.shape[1] // k.shape[1]
    mask = segment_causal_mask(cu_seqlens, q.shape[0])
    qh, kh, vh = (t.transpose(0, 1).unsqueeze(0) for t in (q, k, v))   # [1, H, N, D]
    out = F.scaled_dot_product_attention(qh, repeat_kv(kh, n_rep), repeat_kv(vh, n_rep), attn_mask=mask)
    return out.squeeze(0).transpose(0, 1)


def repeat_kv(x, n_rep):
    if n_rep == 1:
        return x
    B, H_kv, T, D = x.shape
    x = x[:, :, None, :, :].expand(B, H_kv, n_rep, T, D)
    return x.reshape(B, H_kv * n_rep, T, D)


class SwiGLU(nn.Module):
    def __init__(self, hidden_size, intermediate_size):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.n_rep = self.num_heads // self.num_kv_heads
        self.q_proj = nn.Linear(config.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, config.hidden_size, bias=False)

    def forward(self, x, cos, sin, attn_mask, cu_seqlens=None, max_seqlen=None):
        B, T, _ = x.shape
        if cu_seqlens is not None:
            q = self.q_proj(x).view(B, T, self.num_heads, self.head_dim)
            k = self.k_proj(x).view(B, T, self.num_kv_heads, self.head_dim)
            v = self.v_proj(x).view(B * T, self.num_kv_heads, self.head_dim)
            q, k = apply_rope_packed(q, k, cos, sin)
            out = packed_attention(q.reshape(B * T, self.num_heads, self.head_dim).to(v.dtype),
                                   k.reshape(B * T, self.num_kv_heads, self.head_dim).to(v.dtype),
                                   v, cu_seqlens, max_seqlen)
            return self.o_proj(out.reshape(B, T, -1))
        q = self.q_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.num_kv_heads, self.head_dim).transpose(1, 2)
        q, k = apply_rope(q, k, cos, sin)
        k = repeat_kv(k, self.n_rep)
        v = repeat_kv(v, self.n_rep)
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, is_causal=(attn_mask is None))
        out = out.transpose(1, 2).contiguous().view(B, T, -1)
        return self.o_proj(out)


class DecoderLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.input_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.self_attn = CausalSelfAttention(config)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.mlp = SwiGLU(config.hidden_size, config.intermediate_size)

    def forward(self, x, cos, sin, attn_mask, cu_seqlens=None, max_seqlen=None, pin_mask=None, pin_values=None):
        """pin_mask [B, T, 1] bool / pin_values [B, T, H] ("strong"
        conditioning): pinned positions are reset to pin_values before each
        sublayer, so every attention and MLP sees the conditioning vector
        itself rather than a residual-stream state derived from it."""
        if pin_mask is not None:
            x = torch.where(pin_mask, pin_values.to(x.dtype), x)
        x = x + self.self_attn(self.input_layernorm(x), cos, sin, attn_mask, cu_seqlens, max_seqlen)
        if pin_mask is not None:
            x = torch.where(pin_mask, pin_values.to(x.dtype), x)
        x = x + self.mlp(self.post_attention_layernorm(x))
        return x


class LlamaPreTrainedModel(PreTrainedModel):
    config_class = LlamaConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True

    def _init_weights(self, module):
        std = self.config.initializer_range
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
        elif isinstance(module, RMSNorm):
            module.weight.data.fill_(1.0)


class LlamaModel(LlamaPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size,
                                         padding_idx=config.pad_token_id)
        self.layers = nn.ModuleList([DecoderLayer(config) for _ in range(config.num_hidden_layers)])
        self.norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.head_dim = config.hidden_size // config.num_attention_heads
        # Recompute each layer's activations in backward instead of storing
        # them (training only). Set from the run config's gradient_checkpointing
        # key (src/training/trainer.py's build_model), needed to fit the 3B/8B
        # configs (configs/scale/).
        self.gradient_checkpointing = False
        self.post_init()

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    def forward(self, input_ids=None, inputs_embeds=None, attention_mask=None,
                position_ids=None, cu_seqlens=None, max_seqlen=None, pinned_mask=None):
        """Packed batches (src/data/packing.py) pass position_ids [B, T]
        (restarting at 0 in every segment) and cu_seqlens (int32 [S + 1],
        segment boundaries over the flattened B*T tokens): attention is then
        causal within each segment only.

        config.pinned_condition ("strong" conditioning, see DecoderLayer):
        the positions in pinned_mask [B, T] (the conditioning slots) keep
        their input embedding at every sublayer. Without an explicit
        pinned_mask, a call with inputs_embeds pins position 0 -- where every
        unpacked caller puts the conditioning vector (ConditionedLM,
        benchmarks, fine-tuning, steering); calls with input_ids pin nothing."""
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
        B, T, _ = inputs_embeds.shape
        cos, sin = build_rope_cache(self.head_dim, max_seqlen or T, self.config.rope_theta,
                                    inputs_embeds.device, inputs_embeds.dtype)

        attn_mask = None
        if cu_seqlens is not None:
            cos, sin = cos[position_ids], sin[position_ids]          # [B, T, head_dim]
        elif attention_mask is not None:
            # additive mask combining the causal triangle with padding, [B, 1, T, T]
            keep_col = attention_mask[:, None, None, :].to(torch.bool)   # [B,1,1,T]
            causal = torch.tril(torch.ones(T, T, device=inputs_embeds.device, dtype=torch.bool))
            keep = keep_col & causal[None, None, :, :]
            attn_mask = torch.zeros(B, 1, T, T, device=inputs_embeds.device, dtype=inputs_embeds.dtype)
            attn_mask.masked_fill_(~keep, torch.finfo(inputs_embeds.dtype).min)

        pin_mask = None
        if getattr(self.config, "pinned_condition", False) and (pinned_mask is not None or
                                                                (input_ids is None and cu_seqlens is None)):
            if pinned_mask is None:
                pinned_mask = torch.zeros(B, T, dtype=torch.bool, device=inputs_embeds.device)
                pinned_mask[:, 0] = True
            pin_mask = pinned_mask.unsqueeze(-1)

        x = inputs_embeds
        pin = (pin_mask, inputs_embeds if pin_mask is not None else None)
        for layer in self.layers:
            if self.gradient_checkpointing and self.training and torch.is_grad_enabled():
                x = checkpoint(layer, x, cos, sin, attn_mask, cu_seqlens, max_seqlen, *pin, use_reentrant=False)
            else:
                x = layer(x, cos, sin, attn_mask, cu_seqlens, max_seqlen, *pin)
        x = self.norm(x)
        return BaseModelOutputWithPast(last_hidden_state=x)


class LlamaForCausalLM(LlamaPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.model = LlamaModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        if config.tie_word_embeddings:
            self.lm_head.weight = self.model.embed_tokens.weight
        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def forward(self, input_ids=None, inputs_embeds=None, attention_mask=None, **packed):
        out = self.model(input_ids=input_ids, inputs_embeds=inputs_embeds, attention_mask=attention_mask,
                         **packed)
        logits = self.lm_head(out.last_hidden_state)
        return CausalLMOutputWithPast(logits=logits)


def packed_token_loss(logits, labels):
    """Mean next-token cross-entropy for a packed batch: labels [B, T] holds
    the token each position predicts (-100 = no loss; src/data/packing.py
    already did the shift). Returns (loss, n_tokens)."""
    flat = labels.reshape(-1)
    nll = F.cross_entropy(logits.reshape(flat.shape[0], -1).float(), flat, ignore_index=-100, reduction="sum")
    n_tok = (flat != -100).sum().clamp_min(1)
    return nll / n_tok, n_tok


def shifted_token_loss(logits, target_ids, loss_mask):
    """Standard next-token cross-entropy: logits[:, :-1] predicts
    target_ids[:, 1:], masked by loss_mask[:, 1:].

    `target_ids` and `loss_mask` must have the same sequence length as the
    model's input (input_ids, or the inputs_embeds sequence they stand in for).
    This lets every variant share one loss function:
      * baseline/meco: target_ids == the fused input_ids sequence itself;
        loss_mask marks real document-token positions (see
        src/models/conditioned_lm.py), so metadata-prefix and padding positions
        are excluded automatically.
      * text-conditioned: the model's input is [z, doc_token_embeds], one
        position longer than input_ids. target_ids is `input_ids` with one dummy
        value prepended (never used: loss_mask is 0 there), so logits[0]
        (produced from z) lines up with predicting doc_input_ids[0], which is
        exactly the point of conditioning -- see README "The conditioning
        position itself does not receive a language-model loss."
    """
    logits = logits[:, :-1, :].float()
    targets = target_ids[:, 1:]
    mask = loss_mask[:, 1:].to(logits.dtype)
    logp = F.log_softmax(logits, dim=-1)
    nll = -logp.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    nll = nll * mask
    n_tok = mask.sum().clamp_min(1.0)
    loss = nll.sum() / n_tok
    return loss, n_tok
