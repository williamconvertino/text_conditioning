"""Top-level model: wires the causal LM (modeling_llama.py) together with the
conditioning source (none / meco / text; src/models/conditioning.py) into the
single forward() call used by both training (src/training/trainer.py) and
evaluation (src/eval/*.py).

Not a PreTrainedModel subclass itself, since it composes two already
HF-compatible models (LlamaForCausalLM and, for the "text" variant,
TextEncoder) rather than defining new weights of its own beyond the null
conditioning vector (which lives on the conditioner, see conditioning.py).
"""

import torch
import torch.nn as nn

from src.models.modeling_llama import shifted_token_loss


class ConditionedLM(nn.Module):
    def __init__(self, lm, variant, conditioner=None):
        super().__init__()
        assert variant in ("none", "meco", "text")
        assert (variant == "text") == (conditioner is not None), \
            "conditioner must be given iff variant == 'text'"
        self.lm = lm
        self.variant = variant
        self.conditioner = conditioner

    @staticmethod
    def _mask_from_lengths(text_start, lengths, block_size, device):
        pos = torch.arange(block_size, device=device).unsqueeze(0)
        return (pos >= text_start.unsqueeze(1)) & (pos < lengths.unsqueeze(1))

    def forward(self, batch, keep_mask=None):
        """batch: dict from src/data/collator.py. keep_mask (variant == "text"
        only): bool [B], True = condition on the real encoded document, False =
        null vector -- see src/training/schedules.py for how it's sampled during
        training.

        Returns (loss, n_tokens, z_or_None).
        """
        device = next(self.lm.parameters()).device
        input_ids = batch["input_ids"].to(device)
        text_start = batch["text_start"].to(device)
        lengths = batch["lengths"].to(device)
        B, block_size = input_ids.shape

        if self.variant in ("none", "meco"):
            out = self.lm(input_ids=input_ids)
            loss_mask = self._mask_from_lengths(text_start, lengths, block_size, device)
            loss, n_tok = shifted_token_loss(out.logits, input_ids, loss_mask)
            return loss, n_tok, None

        # variant == "text"
        enc = batch["encoder"]
        z = self.conditioner(enc["input_ids"].to(device), enc["attention_mask"].to(device),
                             keep_mask=keep_mask)                     # [B, H]
        loss, n_tok = self.forward_with_explicit_condition(batch, z)
        return loss, n_tok, z

    def forward_with_explicit_condition(self, batch, z):
        """Score `batch` documents using a caller-provided conditioning vector z
        (e.g. same-domain / domain-mean eval, src/eval/conditioning.py), instead
        of running the batch's own encoder input through the conditioner.
        variant must be "text". Returns (loss, n_tokens)."""
        device = next(self.lm.parameters()).device
        input_ids = batch["input_ids"].to(device)
        lengths = batch["lengths"].to(device)
        B, block_size = input_ids.shape

        tok_embeds = self.lm.get_input_embeddings()(input_ids)
        inputs_embeds = torch.cat([z.unsqueeze(1).to(tok_embeds.dtype), tok_embeds], dim=1)
        out = self.lm(inputs_embeds=inputs_embeds)

        real_mask = torch.arange(block_size, device=device).unsqueeze(0) < lengths.unsqueeze(1)
        dummy_first = torch.zeros(B, 1, dtype=input_ids.dtype, device=device)
        target_ids = torch.cat([dummy_first, input_ids], dim=1)
        loss_mask = torch.cat(
            [torch.zeros(B, 1, dtype=torch.bool, device=device), real_mask], dim=1)
        return shifted_token_loss(out.logits, target_ids, loss_mask)
