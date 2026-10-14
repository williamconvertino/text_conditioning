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

from src.models.modeling_llama import packed_token_loss, shifted_token_loss


class ConditionedLM(nn.Module):
    def __init__(self, lm, variant, conditioner=None):
        super().__init__()
        assert variant in ("none", "meco", "text")
        assert (variant == "text") == (conditioner is not None), \
            "conditioner must be given iff variant == 'text'"
        self.lm = lm
        self.variant = variant
        self.conditioner = conditioner
        self._packed_loss = packed_token_loss

    def compile_for_training(self):
        """torch.compile each decoder layer and the packed loss in place
        (state_dict keys unchanged). Packed rows have one fixed shape, so this
        compiles once; the text encoder (variable slot count) stays eager."""
        for layer in self.lm.model.layers:
            layer.compile()
        self._packed_loss = torch.compile(packed_token_loss)

    @staticmethod
    def _mask_from_lengths(text_start, lengths, block_size, device):
        pos = torch.arange(block_size, device=device).unsqueeze(0)
        return (pos >= text_start.unsqueeze(1)) & (pos < lengths.unsqueeze(1))

    def forward(self, batch, keep_mask=None):
        """batch: dict from src/data/collator.py. keep_mask (variant == "text"
        only): bool [B], True = condition on the real encoded document, False =
        null vector -- see src/training/schedules.py for how it's sampled during
        training. For domain-conditioned batches (batch["encoder"] has
        group_index/has_condition, src/data/collator.py) the encoded documents
        are each row's in-domain partners, averaged per row.

        Packed batches (src/data/packing.py, any batch with "cu_seqlens") go
        through forward_packed.

        Returns (loss, n_tokens, z_or_None).
        """
        if "cu_seqlens" in batch:
            return self.forward_packed(batch)
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
                             keep_mask=keep_mask, group_index=enc.get("group_index"),
                             has_condition=enc.get("has_condition"))  # [B, H]
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

    def forward_packed(self, batch):
        """Packed training batch (src/data/packing.py): fixed [B, T] rows of
        document segments, attention causal within each segment.

        "none" / "meco": the LM runs on input_ids directly (MeCo's prefixes
        are already tokens in the row). "text" (URL-encoded): every segment
        starts with a slot position; the encoder's vector for each slot
        (batch["encoder"], one row per slot, the null vector where
        has_condition is False -- no URL, or dropped by the conditioning
        schedule) replaces that position's embedding.

        Returns (loss, n_tokens, z_or_None) like forward()."""
        device = self.lm.lm_head.weight.device
        nb = dict(non_blocking=True)
        input_ids = batch["input_ids"].to(device, **nb)
        packed = {"position_ids": batch["position_ids"].to(device, **nb),
                  "cu_seqlens": batch["cu_seqlens"].to(device, **nb),
                  "max_seqlen": batch["max_seqlen"]}
        labels = batch["labels"].to(device, **nb)
        z = None
        if self.variant == "text":
            enc = batch["encoder"]
            has = enc["has_condition"].to(device, **nb)
            z = self.conditioner(enc["input_ids"].to(device, **nb), enc["attention_mask"].to(device, **nb),
                                 keep_mask=has)                               # [n_slots, H]
            emb = self.lm.get_input_embeddings()(input_ids)
            B, T, H = emb.shape
            slots = batch["slot_index"].to(device, **nb)
            flat = emb.reshape(B * T, H).index_copy(0, slots, z.to(emb.dtype))
            if self.lm.config.pinned_condition:     # "strong" conditioning: pin every slot
                packed["pinned_mask"] = torch.zeros(B * T, dtype=torch.bool, device=device).index_fill(
                    0, slots, True).view(B, T)
            out = self.lm(inputs_embeds=flat.view(B, T, H), **packed)
        else:
            out = self.lm(input_ids=input_ids, **packed)
        loss, n_tok = self._packed_loss(out.logits, labels)
        return loss, n_tok, z
