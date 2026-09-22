"""Minimal, dependency-light YAML config loading.

An experiment config (configs/experiment/*.yaml) can `include:` any number of
other yaml files -- typically one from configs/model/ and one from
configs/train/. Included files are merged first, in order, and then the
including file's own keys are applied on top, so an experiment yaml reads as
exactly "the model + train defaults, plus whatever this experiment overrides".

This intentionally does not pull in a full config framework (hydra/omegaconf):
every config here is small and flat enough that a recursive dict-merge is
sufficient, and it keeps the codebase's dependency surface small.

A `text_encoder_config` key (used by the "text" variant) is treated specially:
it points at its own yaml file (configs/model/text_encoder.yaml) rather than
being merged in, since its field names (hidden_size, num_hidden_layers, ...)
would otherwise collide with the causal LM's. It is resolved to an absolute path
here so callers can load it later without needing to know which yaml file it was
declared in.
"""

import copy
import os

import yaml


def _deep_merge(base, override):
    out = copy.deepcopy(base)
    for k, v in override.items():
        if k == "include":
            continue
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def load_config(path):
    path = os.path.abspath(path)
    base_dir = os.path.dirname(path)
    raw = yaml.safe_load(open(path)) or {}

    merged = {}
    for inc in raw.get("include", []):
        inc_path = inc if os.path.isabs(inc) else os.path.join(base_dir, inc)
        merged = _deep_merge(merged, load_config(inc_path))
    merged = _deep_merge(merged, raw)
    merged.pop("include", None)

    tec = merged.get("text_encoder_config")
    if tec and not os.path.isabs(tec):
        merged["text_encoder_config"] = os.path.normpath(os.path.join(base_dir, tec))

    return merged


def save_config(cfg, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
