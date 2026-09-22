#!/usr/bin/env python
"""Print backbone/encoder parameter counts for one or more experiment configs,
without loading any data or tokenizer. Used to tune Experiment D
(configs/experiment/text_conditioned_param_matched.yaml) so that
backbone + encoder params stay close to the baseline's -- see README
"Total-Parameter-Matched Text Conditioning".

Usage:
  python scripts/count_params.py configs/experiment/baseline.yaml \\
      configs/experiment/text_conditioned.yaml \\
      configs/experiment/text_conditioned_param_matched.yaml
"""

import argparse
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.config import load_config
from src.models.configuration_llama import LlamaConfig
from src.models.modeling_llama import LlamaForCausalLM
from src.models.text_encoder import TextEncoderConfig, TextEncoder

VOCAB_SIZE = 128256   # Llama-3 tokenizer vocab; only affects the embedding/lm_head count


def count(module):
    return sum(p.numel() for p in module.parameters())


def main():
    p = argparse.ArgumentParser()
    p.add_argument("experiments", nargs="+")
    args = p.parse_args()

    for path in args.experiments:
        cfg = load_config(path)
        lm_cfg = LlamaConfig(
            vocab_size=VOCAB_SIZE, hidden_size=cfg["hidden_size"],
            intermediate_size=cfg["intermediate_size"], num_hidden_layers=cfg["num_hidden_layers"],
            num_attention_heads=cfg["num_attention_heads"],
            num_key_value_heads=cfg["num_key_value_heads"],
            max_position_embeddings=cfg["max_position_embeddings"],
            tie_word_embeddings=cfg.get("tie_word_embeddings", False))
        lm = LlamaForCausalLM(lm_cfg)
        n_lm = count(lm)

        n_enc = 0
        if cfg.get("variant") == "text":
            enc_cfg_dict = load_config(cfg["text_encoder_config"])
            enc_cfg = TextEncoderConfig(
                vocab_size=VOCAB_SIZE, hidden_size=enc_cfg_dict["hidden_size"],
                intermediate_size=enc_cfg_dict["intermediate_size"],
                num_hidden_layers=enc_cfg_dict["num_hidden_layers"],
                num_attention_heads=enc_cfg_dict["num_attention_heads"],
                num_key_value_heads=enc_cfg_dict["num_key_value_heads"],
                max_position_embeddings=enc_cfg_dict["max_position_embeddings"],
                output_dim=cfg["hidden_size"])
            encoder = TextEncoder(enc_cfg)
            n_enc = count(encoder)

        print(f"{os.path.basename(path):45s} backbone={n_lm/1e6:8.1f}M  "
              f"encoder={n_enc/1e6:7.1f}M  total={(n_lm+n_enc)/1e6:8.1f}M")


if __name__ == "__main__":
    main()
