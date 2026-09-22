import torch

from src.data.collator import collate_batch, doc_dict
from src.data.shards import iter_shard_docs
from src.models.configuration_llama import LlamaConfig
from src.models.conditioned_lm import ConditionedLM
from src.models.conditioning import TextConditioner
from src.models.modeling_llama import LlamaForCausalLM
from src.models.text_encoder import TextEncoder, TextEncoderConfig

HIDDEN = 32


def make_lm(vocab_size):
    cfg = LlamaConfig(vocab_size=vocab_size, hidden_size=HIDDEN, intermediate_size=64,
                      num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
                      max_position_embeddings=64, pad_token_id=0, bos_token_id=1, eos_token_id=2)
    return LlamaForCausalLM(cfg)


def make_conditioner(vocab_size):
    enc_cfg = TextEncoderConfig(vocab_size=vocab_size, hidden_size=16, intermediate_size=32,
                                num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=2,
                                max_position_embeddings=32, output_dim=HIDDEN, pad_token_id=0)
    return TextConditioner(TextEncoder(enc_cfg), HIDDEN)


def get_batch(tok_dir, tokenizer, variant):
    docs = [doc_dict(shard, row) for shard, row in iter_shard_docs(tok_dir, ["shard_00"])]
    return collate_batch(docs, tokenizer, variant, block_size=32, meco_max_tokens=8,
                         encoder_max_tokens=16)


def test_none_variant_forward_and_backward(tok_dir, tokenizer):
    model = ConditionedLM(make_lm(len(tokenizer)), "none")
    batch = get_batch(tok_dir, tokenizer, "none")
    loss, n_tok, z = model(batch)
    assert z is None
    assert torch.isfinite(loss)
    assert n_tok.item() > 0
    loss.backward()
    assert model.lm.get_input_embeddings().weight.grad is not None


def test_meco_variant_forward_and_backward(tok_dir, tokenizer):
    model = ConditionedLM(make_lm(len(tokenizer)), "meco")
    batch = get_batch(tok_dir, tokenizer, "meco")
    loss, n_tok, z = model(batch)
    assert z is None
    assert torch.isfinite(loss)
    loss.backward()


def test_text_variant_forward_and_backward(tok_dir, tokenizer):
    conditioner = make_conditioner(len(tokenizer))
    model = ConditionedLM(make_lm(len(tokenizer)), "text", conditioner)
    batch = get_batch(tok_dir, tokenizer, "text")

    keep_mask = torch.tensor([True, False])
    loss, n_tok, z = model(batch, keep_mask=keep_mask)
    assert z.shape == (2, HIDDEN)
    assert torch.isfinite(loss)
    loss.backward()
    assert conditioner.null_vector.grad is not None
    assert conditioner.encoder.embed_tokens.weight.grad is not None


def test_null_vector_ignores_document_content(tok_dir, tokenizer):
    """With keep_mask all-False, the conditioning vector is always the null
    vector regardless of what the encoder was shown -- so the loss must not
    depend on the encoder's input document at all."""
    conditioner = make_conditioner(len(tokenizer))
    model = ConditionedLM(make_lm(len(tokenizer)), "text", conditioner)
    model.eval()

    batch_a = get_batch(tok_dir, tokenizer, "text")
    docs_b = [doc_dict(shard, row) for shard, row in iter_shard_docs(tok_dir, ["shard_01"])]
    from src.data.collator import collate_batch as _cb
    batch_b = _cb(docs_b, tokenizer, "text", block_size=32, meco_max_tokens=8, encoder_max_tokens=16)
    # keep the causal-LM document tokens identical; only the encoder's input differs
    batch_b["input_ids"] = batch_a["input_ids"]
    batch_b["text_start"] = batch_a["text_start"]
    batch_b["lengths"] = batch_a["lengths"]

    keep_mask = torch.zeros(2, dtype=torch.bool)
    with torch.no_grad():
        loss_a, _, _ = model(batch_a, keep_mask=keep_mask)
        loss_b, _, _ = model(batch_b, keep_mask=keep_mask)
    assert torch.allclose(loss_a, loss_b, atol=1e-5)


def test_forward_with_explicit_condition(tok_dir, tokenizer):
    conditioner = make_conditioner(len(tokenizer))
    model = ConditionedLM(make_lm(len(tokenizer)), "text", conditioner)
    batch = get_batch(tok_dir, tokenizer, "text")
    z = torch.randn(2, HIDDEN)
    loss, n_tok = model.forward_with_explicit_condition(batch, z)
    assert torch.isfinite(loss)
    assert n_tok.item() > 0
