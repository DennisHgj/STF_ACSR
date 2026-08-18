"""Checkpoint normalization shared by training and evaluation entry points."""

from collections.abc import Mapping

import torch


def load_checkpoint(path, map_location="cpu"):
    checkpoint = torch.load(path, map_location=map_location)
    if not isinstance(checkpoint, Mapping):
        raise ValueError(f"Checkpoint {path} does not contain a state mapping.")
    return checkpoint


def extract_state_dict(checkpoint):
    for key in ("state_dict", "model_state_dict"):
        value = checkpoint.get(key)
        if isinstance(value, Mapping):
            return dict(value)
    if all(isinstance(key, str) for key in checkpoint):
        return dict(checkpoint)
    raise ValueError("Unable to find a state_dict in the checkpoint.")


def extract_model_state_dict(checkpoint):
    """Return state keys suitable for the inner E2E model."""
    state = extract_state_dict(checkpoint)
    if any(key.startswith("model.") for key in state):
        state = {
            key[len("model.") :]: value
            for key, value in state.items()
            if key.startswith("model.")
        }
    return state


def load_lightning_weights(module, path, map_location="cpu", strict=True):
    checkpoint = load_checkpoint(path, map_location=map_location)
    state = extract_state_dict(checkpoint)
    result = module.load_state_dict(state, strict=strict)
    return list(result.missing_keys), list(result.unexpected_keys)
