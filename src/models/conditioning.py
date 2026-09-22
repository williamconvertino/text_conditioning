"""Turns document text (or nothing) into an LM conditioning vector.

- variant "none": no conditioner at all; src/models/conditioned_lm.py runs the
  causal LM directly on input_ids.
- variant "meco": no conditioner either -- the domain/URL text is already part
  of input_ids (src/data/collator.py prepends it as ordinary tokens before
  training ever sees the batch), so meco needs no embedding-space conditioning.
- variant "text": this module owns the encoder call, the null vector, and
  applying conditioning dropout -- see README "Unconditioned Mode" and
  "Experiment E: Conditioning Schedule".

The dropout *probability* p_condition(t) is computed by
src/training/schedules.py; this module only applies a given per-example
keep/drop mask.
"""

import torch
import torch.nn as nn


class TextConditioner(nn.Module):
    def __init__(self, encoder, hidden_size):
        super().__init__()
        self.encoder = encoder
        self.null_vector = nn.Parameter(torch.zeros(hidden_size))
        nn.init.normal_(self.null_vector, std=0.02)

    def null_like(self, batch_size, device, dtype):
        return self.null_vector.to(device=device, dtype=dtype).unsqueeze(0).expand(batch_size, -1)

    def forward(self, encoder_input_ids, encoder_attention_mask, keep_mask=None):
        """keep_mask: bool tensor [B] (True = use the real encoded vector, False
        = use the null vector). None => always use the real vector (e.g. eval
        "self" conditioning). The encoder always runs over the whole batch
        (cheap, small model) so the same call path works regardless of whether
        any example in the batch ends up using z_null."""
        z = self.encoder(input_ids=encoder_input_ids, attention_mask=encoder_attention_mask)
        if keep_mask is None:
            return z
        z_null = self.null_like(z.shape[0], z.device, z.dtype)
        keep_mask = keep_mask.to(z.device).view(-1, 1)
        return torch.where(keep_mask, z, z_null)
