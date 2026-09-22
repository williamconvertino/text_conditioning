import torch

from src.models.configuration_llama import LlamaConfig
from src.models.modeling_llama import LlamaForCausalLM, shifted_token_loss
from src.models.text_encoder import TextEncoder, TextEncoderConfig

VOCAB = 200


def tiny_lm_config(**overrides):
    cfg = dict(vocab_size=VOCAB, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
              num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=64,
              pad_token_id=0, bos_token_id=1, eos_token_id=2)
    cfg.update(overrides)
    return LlamaConfig(**cfg)


def test_lm_forward_shapes():
    model = LlamaForCausalLM(tiny_lm_config())
    input_ids = torch.randint(0, VOCAB, (2, 10))
    out = model(input_ids=input_ids)
    assert out.logits.shape == (2, 10, VOCAB)


def test_lm_forward_inputs_embeds_one_longer():
    model = LlamaForCausalLM(tiny_lm_config())
    input_ids = torch.randint(0, VOCAB, (2, 10))
    tok_embeds = model.get_input_embeddings()(input_ids)
    z = torch.randn(2, 1, tok_embeds.shape[-1])
    inputs_embeds = torch.cat([z, tok_embeds], dim=1)
    out = model(inputs_embeds=inputs_embeds)
    assert out.logits.shape == (2, 11, VOCAB)


def test_shifted_token_loss_matches_manual_standard_lm():
    torch.manual_seed(0)
    B, T, V = 2, 6, 10
    logits = torch.randn(B, T, V)
    target_ids = torch.randint(0, V, (B, T))
    loss_mask = torch.ones(B, T, dtype=torch.bool)
    loss, n_tok = shifted_token_loss(logits, target_ids, loss_mask)
    expected = torch.nn.functional.cross_entropy(
        logits[:, :-1, :].reshape(-1, V), target_ids[:, 1:].reshape(-1))
    assert torch.allclose(loss, expected, atol=1e-5)
    assert n_tok.item() == B * (T - 1)


def test_shifted_token_loss_respects_mask():
    B, T, V = 1, 4, 5
    logits = torch.zeros(B, T, V)
    target_ids = torch.zeros(B, T, dtype=torch.long)
    loss_mask = torch.tensor([[0, 0, 1, 1]], dtype=torch.bool)
    _, n_tok = shifted_token_loss(logits, target_ids, loss_mask)
    assert n_tok.item() == 2  # only positions 2,3 of loss_mask are counted (shifted)


def test_text_encoder_output_shape_and_pad_invariance():
    enc_cfg = TextEncoderConfig(vocab_size=VOCAB, hidden_size=16, intermediate_size=32,
                                num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=2,
                                max_position_embeddings=32, output_dim=32, pad_token_id=0)
    encoder = TextEncoder(enc_cfg)
    input_ids = torch.randint(1, VOCAB, (2, 8))
    input_ids[:, 6:] = 0
    attention_mask = torch.ones(2, 8, dtype=torch.long)
    attention_mask[:, 6:] = 0

    z = encoder(input_ids=input_ids, attention_mask=attention_mask)
    assert z.shape == (2, 32)

    # padding beyond the real content should not change the pooled output
    input_ids_more_pad = input_ids.clone()
    input_ids_more_pad[:, 6:] = torch.randint(1, VOCAB, (2, 2))  # garbage in padded region
    z2 = encoder(input_ids=input_ids_more_pad, attention_mask=attention_mask)
    assert torch.allclose(z, z2, atol=1e-5)
