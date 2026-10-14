"""Small bidirectional document encoder (README "Text Conditioning"):

    document tokens -> bidirectional Transformer -> pooling -> normalization ->
    projection -> conditioning vector z

Substantially smaller than the causal LM (2-3 layers, small hidden dim by
default); width/heads/layers/pooling are all configurable via
configs/model/text_encoder.yaml.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel
from transformers import PretrainedConfig, PreTrainedModel

from src.models.modeling_llama import RMSNorm, SwiGLU, build_rope_cache, apply_rope, repeat_kv


class TextEncoderConfig(PretrainedConfig):
    model_type = "text_encoder"

    def __init__(
        self,
        vocab_size=128256,
        hidden_size=256,
        intermediate_size=1024,
        num_hidden_layers=3,
        num_attention_heads=4,
        num_key_value_heads=4,
        max_position_embeddings=1024,
        rms_norm_eps=1e-5,
        rope_theta=10000.0,
        initializer_range=0.02,
        output_dim=2048,          # must match the causal LM's hidden_size
        pooling="cls",            # "cls" or "mean"
        pad_token_id=None,
        **kwargs,
    ):
        assert pooling in ("cls", "mean")
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.max_position_embeddings = max_position_embeddings
        self.rms_norm_eps = rms_norm_eps
        self.rope_theta = rope_theta
        self.initializer_range = initializer_range
        self.output_dim = output_dim
        self.pooling = pooling
        super().__init__(pad_token_id=pad_token_id, **kwargs)


# Never cuDNN attention: it builds an execution plan per input shape, and the
# encoder's shape (inputs x length) changes every batch, so on H200 that
# planning cost ~65ms of CPU per layer per batch (packed URL-encoded 1.5B
# training ran ~20% slower than the baseline because of it).
ENCODER_SDPA_BACKENDS = [SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH]


class BidirectionalAttention(nn.Module):
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

    def forward(self, x, cos, sin, attn_mask):
        B, T, _ = x.shape
        q = self.q_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.num_kv_heads, self.head_dim).transpose(1, 2)
        q, k = apply_rope(q, k, cos, sin)
        k = repeat_kv(k, self.n_rep)
        v = repeat_kv(v, self.n_rep)
        with sdpa_kernel(ENCODER_SDPA_BACKENDS):
            out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, is_causal=False)
        out = out.transpose(1, 2).contiguous().view(B, T, -1)
        return self.o_proj(out)


class EncoderLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.input_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.attn = BidirectionalAttention(config)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.mlp = SwiGLU(config.hidden_size, config.intermediate_size)

    def forward(self, x, cos, sin, attn_mask):
        x = x + self.attn(self.input_layernorm(x), cos, sin, attn_mask)
        x = x + self.mlp(self.post_attention_layernorm(x))
        return x


class TextEncoder(PreTrainedModel):
    """z = encoder(input_ids=..., attention_mask=...) -> FloatTensor [B, output_dim]."""

    config_class = TextEncoderConfig
    base_model_prefix = "encoder"

    def __init__(self, config):
        super().__init__(config)
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size,
                                         padding_idx=config.pad_token_id)
        self.cls_embedding = nn.Parameter(torch.zeros(1, 1, config.hidden_size))
        self.layers = nn.ModuleList([EncoderLayer(config) for _ in range(config.num_hidden_layers)])
        self.norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.proj = nn.Linear(config.hidden_size, config.output_dim, bias=False)
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.post_init()
        nn.init.normal_(self.cls_embedding, std=config.initializer_range)

    def _init_weights(self, module):
        std = self.config.initializer_range
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
        elif isinstance(module, RMSNorm):
            module.weight.data.fill_(1.0)

    def forward(self, input_ids, attention_mask=None):
        B, T = input_ids.shape
        tok_emb = self.embed_tokens(input_ids)
        cls = self.cls_embedding.expand(B, -1, -1)
        x = torch.cat([cls, tok_emb], dim=1)                         # [B, T+1, H]

        if attention_mask is None:
            attention_mask = torch.ones(B, T, device=input_ids.device, dtype=torch.long)
        cls_mask = torch.ones(B, 1, device=input_ids.device, dtype=attention_mask.dtype)
        full_mask = torch.cat([cls_mask, attention_mask], dim=1)      # [B, T+1]

        cos, sin = build_rope_cache(self.head_dim, T + 1, self.config.rope_theta, x.device, x.dtype)
        keep = full_mask[:, None, None, :].to(torch.bool)             # bidirectional: only pad masking
        attn_mask = torch.zeros(B, 1, T + 1, T + 1, device=x.device, dtype=x.dtype)
        attn_mask.masked_fill_(~keep, torch.finfo(x.dtype).min)

        for layer in self.layers:
            x = layer(x, cos, sin, attn_mask)
        x = self.norm(x)

        if self.config.pooling == "cls":
            pooled = x[:, 0, :]
        else:
            m = attention_mask.unsqueeze(-1).to(x.dtype)
            pooled = (x[:, 1:, :] * m).sum(1) / m.sum(1).clamp_min(1.0)

        return self.proj(pooled)
