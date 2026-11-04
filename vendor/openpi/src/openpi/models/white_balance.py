# Copyright 2026 AI Robot Association (AIRoA)
# SPDX-License-Identifier: Apache-2.0
"""Sample one mild warm/cool RGB gain per training example, shared across views."""

import dataclasses

import jax
import jax.numpy as jnp


def augment(observation, rng, *, strength, probability):
    """Operate after preprocessing; preserve masks, padding, labels and all targets.

    This is a log RGB gain perturbation, not an estimate of physical Kelvin.
    Call only from the training objective. Stateless RNG follows the training RNG
    checkpoint, independently of DataLoader worker count and prefetch.
    """
    if strength == 0 or probability == 0:
        return observation, {
            "white_balance_applied_fraction": jnp.asarray(0.0),
            "white_balance_abs_temperature": jnp.asarray(0.0),
        }
    temperature_rng, apply_rng = jax.random.split(rng)
    shape = observation.state.shape[:-1]
    temperature = jax.random.uniform(
        temperature_rng, shape, minval=-strength, maxval=strength
    )
    apply = jax.random.bernoulli(apply_rng, probability, shape)
    temperature = jnp.where(apply, temperature, 0.0)
    gains = jnp.stack(
        (jnp.exp(temperature), jnp.ones_like(temperature), jnp.exp(-temperature)),
        axis=-1,
    )
    gains = gains / jnp.mean(gains, axis=-1, keepdims=True)

    def recolor(image, valid_mask):
        if valid_mask is None:
            raise ValueError("White balance requires preprocessed image valid masks")
        view_gain = gains.reshape((*shape, *((1,) * (image.ndim - len(shape) - 1)), 3))
        changed = jnp.clip((image * 0.5 + 0.5) * view_gain, 0.0, 1.0) * 2.0 - 1.0
        selection = apply.reshape((*shape, *((1,) * (image.ndim - len(shape)))))
        # Identity samples and invalid pixels remain bitwise unchanged.
        return jnp.where(selection & (valid_mask[..., None] >= 0.5), changed, image)

    images = {
        key: recolor(value, observation.image_valid_masks[key])
        for key, value in observation.images.items()
    }
    history = observation.image_history
    if history is not None:
        history = {
            key: recolor(value, observation.image_history_valid_masks[key])
            for key, value in history.items()
        }
    return dataclasses.replace(observation, images=images, image_history=history), {
        "white_balance_applied_fraction": jnp.mean(apply.astype(jnp.float32)),
        "white_balance_abs_temperature": jnp.mean(jnp.abs(temperature)),
    }
