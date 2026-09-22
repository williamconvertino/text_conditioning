"""lm-evaluation-harness LM adapter for this repo's ConditionedLM
(src/models/conditioned_lm.py).

Deliberately its own implementation rather than a subclass of
lm_eval.models.huggingface.HFLM: HFLM assumes a model whose forward
signature/generation interface matches transformers' AutoModelForCausalLM,
including KV-cached `generate()`, which src/models/modeling_llama.py does not
implement (see that module's docstring -- the causal LM only knows how to run
a full forward pass over `input_ids`/`inputs_embeds`). It also has no way to
splice a conditioning vector into embedding space before the first token
without breaking HFLM's internal sequence-length bookkeeping. Only
loglikelihood-based tasks (multiple-choice / binary accuracy, no free-form
generation, no rolling-window perplexity) are supported -- see
configs/benchmarks/tasks.yaml for why that's the right tradeoff here.

Conditioning is fixed for the lifetime of one CtrlTokensLM instance (one
ConditioningSpec from src/benchmarks/conditioning.py, applied to every
request): a MeCo domain-string prefix, a fixed text-conditioning vector z
(possibly the model's own null vector, for the "none"/unconditioned setting),
or nothing at all for variant "none". This matches how this project's
conditioning is meant to be used for downstream generation/scoring -- see
README "Domain-Mean Conditioning": "particularly relevant to future steering
applications: a representation can be estimated from a collection of
reference documents and then reused for generation."
"""

import torch
import torch.nn.functional as F
from tqdm import tqdm

from lm_eval.api.model import LM

from src.data.collator import _meco_prefix_ids


class CtrlTokensLM(LM):
    def __init__(self, model, tokenizer, variant, conditioning_spec, block_size,
                meco_max_tokens=48, batch_size=16, device=None):
        super().__init__()
        assert variant in ("none", "meco", "text")
        self.model = model
        self.tokenizer = tokenizer
        self.variant = variant
        self.spec = conditioning_spec
        self.block_size = block_size
        self.max_length = block_size
        self.batch_size = batch_size
        self._device = device or next(model.parameters()).device

        self._prefix_ids = []
        if variant == "meco" and conditioning_spec.kind == "meco":
            # reuse the exact same construction training used, so the prefix
            # this run sees at eval time is identical to what it was trained
            # on for that domain string.
            self._prefix_ids = _meco_prefix_ids(tokenizer, conditioning_spec.domain, meco_max_tokens)

        self._z = None
        if variant == "text":
            self._z = (conditioning_spec.z if conditioning_spec.kind == "text"
                      else model.conditioner.null_vector.detach())
            self._z = self._z.to(self._device)

    @property
    def device(self):
        return self._device

    # -- tokenization ------------------------------------------------------
    def _encode(self, text):
        return self.tokenizer.encode(text, add_special_tokens=False) if text else []

    def _encode_pair(self, context, continuation):
        """Encode (context, continuation) as one joined string then split, so
        BPE tokenization at the boundary matches how the model would see the
        same text in one piece (lm-eval-harness convention) -- and move a
        trailing context space into the continuation for the same reason.
        Mirrors the training pipeline's own convention of never adding
        special tokens (src/data/tokenizer_utils.py, src/data/collator.py)."""
        if not context:
            return [self.tokenizer.eos_token_id], self._encode(continuation) or [self.tokenizer.eos_token_id]
        n_spaces = len(context) - len(context.rstrip())
        if n_spaces > 0:
            continuation = context[-n_spaces:] + continuation
            context = context[:-n_spaces]
        context_enc = self._encode(context)
        whole_enc = self._encode(context + continuation)
        continuation_enc = whole_enc[len(context_enc):]
        if not continuation_enc:
            continuation_enc = [self.tokenizer.eos_token_id]
        return context_enc, continuation_enc

    def _build_example(self, context, continuation):
        """-> (token_ids, continuation_len), truncated to fit self.block_size
        (prefix included), truncating the context from the left and keeping
        the continuation intact whenever possible -- only a continuation
        alone longer than the budget gets truncated (from the left, keeping
        its tail, since that's the part being scored)."""
        context_enc, continuation_enc = self._encode_pair(context, continuation)
        room = self.block_size - len(self._prefix_ids)
        if len(continuation_enc) >= room:
            continuation_enc = continuation_enc[-max(1, room):]
            context_enc = []
        else:
            max_ctx = room - len(continuation_enc)
            if len(context_enc) > max_ctx:
                context_enc = context_enc[-max_ctx:]
        seq = self._prefix_ids + context_enc + continuation_enc
        return seq, len(continuation_enc)

    # -- scoring -------------------------------------------------------------
    def _forward_batch(self, seqs, cont_lens):
        """seqs: token id lists (already truncated to <= self.block_size).
        Returns a list of (sum_logprob, is_greedy), one per example.

        Batched with right-padding only (no attention_mask): causal masking
        already prevents any position from attending to later, padded
        positions, so padding doesn't affect the (unpadded) logits actually
        scored below -- the same assumption src/models/conditioned_lm.py's
        forward_with_explicit_condition makes."""
        pad_id = self.tokenizer.pad_token_id
        B = len(seqs)
        T = max(len(s) for s in seqs)
        input_ids = torch.full((B, T), pad_id, dtype=torch.long)
        loss_mask = torch.zeros((B, T), dtype=torch.bool)
        for i, (seq, clen) in enumerate(zip(seqs, cont_lens)):
            L = len(seq)
            input_ids[i, :L] = torch.tensor(seq, dtype=torch.long)
            loss_mask[i, L - clen:L] = True
        input_ids = input_ids.to(self._device)
        loss_mask = loss_mask.to(self._device)

        lm = self.model.lm
        with torch.no_grad():
            if self.variant == "text":
                tok_embeds = lm.get_input_embeddings()(input_ids)
                z = self._z.to(dtype=tok_embeds.dtype).unsqueeze(0).expand(B, -1)
                inputs_embeds = torch.cat([z.unsqueeze(1), tok_embeds], dim=1)
                out = lm(inputs_embeds=inputs_embeds)
                dummy_first = torch.zeros(B, 1, dtype=input_ids.dtype, device=self._device)
                target_ids = torch.cat([dummy_first, input_ids], dim=1)
                full_mask = torch.cat(
                    [torch.zeros(B, 1, dtype=torch.bool, device=self._device), loss_mask], dim=1)
            else:
                out = lm(input_ids=input_ids)
                target_ids = input_ids
                full_mask = loss_mask

            logits = out.logits[:, :-1, :].float()
            targets = target_ids[:, 1:]
            mask = full_mask[:, 1:]
            logp = F.log_softmax(logits, dim=-1)
            tok_logp = logp.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
            greedy_ok = logp.argmax(-1) == targets

        results = []
        for i in range(B):
            m = mask[i]
            sum_logp = (tok_logp[i] * m).sum().item()
            is_greedy = bool((greedy_ok[i] | ~m).all().item())
            results.append((sum_logp, is_greedy))
        return results

    # -- lm_eval.api.model.LM interface --------------------------------------
    def loglikelihood(self, requests):
        seqs, cont_lens = [], []
        for req in requests:
            context, continuation = req.args
            seq, clen = self._build_example(context, continuation)
            seqs.append(seq)
            cont_lens.append(clen)

        # longest-first batching (stable OOM behavior, matches HFLM convention)
        order = sorted(range(len(seqs)), key=lambda i: -len(seqs[i]))
        results = [None] * len(seqs)
        for start in tqdm(range(0, len(order), self.batch_size), desc="loglikelihood"):
            idx_batch = order[start:start + self.batch_size]
            batch_results = self._forward_batch([seqs[i] for i in idx_batch],
                                                [cont_lens[i] for i in idx_batch])
            for i, res in zip(idx_batch, batch_results):
                results[i] = res
        return results

    def loglikelihood_rolling(self, requests):
        raise NotImplementedError(
            "loglikelihood_rolling (rolling-window perplexity tasks, e.g. wikitext) is not "
            "supported -- this project already tracks val loss/ppl during training "
            "(src/eval/perplexity.py); pick loglikelihood-only tasks for benchmarking "
            "(see configs/benchmarks/tasks.yaml).")

    def generate_until(self, requests):
        raise NotImplementedError(
            "generate_until (free-form generation tasks) is not supported -- "
            "src/models/modeling_llama.py has no generate()/KV-cache implementation. "
            "Pick loglikelihood-only tasks for benchmarking (see configs/benchmarks/tasks.yaml).")
