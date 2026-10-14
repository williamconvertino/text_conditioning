"""Config for the custom causal LM (src/models/modeling_llama.py).

Modeled after Llama 3.2's architecture (RoPE, grouped-query attention, RMSNorm,
SwiGLU) but implemented independently so conditioning behavior can be changed
freely -- see README "the architecture should not depend directly on the
official Llama implementation".
"""

from transformers import PretrainedConfig


class LlamaConfig(PretrainedConfig):
    model_type = "custom_llama"

    def __init__(
        self,
        vocab_size=128256,
        hidden_size=2048,
        intermediate_size=5504,
        num_hidden_layers=16,
        num_attention_heads=16,
        num_key_value_heads=4,
        max_position_embeddings=4096,
        rms_norm_eps=1e-5,
        rope_theta=500000.0,
        initializer_range=0.02,
        tie_word_embeddings=False,
        pad_token_id=None,
        bos_token_id=128000,
        eos_token_id=128001,
        pinned_condition=False,
        **kwargs,
    ):
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
        # "strong" conditioning (src/models/modeling_llama.py's DecoderLayer):
        # the conditioning position's hidden state is reset to the
        # conditioning vector before every attention and MLP sublayer
        self.pinned_condition = pinned_condition
        super().__init__(
            pad_token_id=pad_token_id, bos_token_id=bos_token_id, eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings, **kwargs)
