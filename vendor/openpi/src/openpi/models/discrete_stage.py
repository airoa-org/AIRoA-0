# Copyright 2026 AI Robot Association (AIRoA)
# SPDX-License-Identifier: Apache-2.0
"""Observation-predicted task stage; independent of gripper/latch transitions."""

import dataclasses

import jax
import jax.numpy as jnp


def supervised_loss(logits, labels, weight):
    if labels.shape != logits.shape[:-1] or not jnp.issubdtype(
        labels.dtype, jnp.integer
    ):
        raise ValueError("Stage labels must be integer class IDs with shape [batch]")
    valid = (labels >= 0) & (labels < logits.shape[-1])
    safe_labels = jnp.clip(labels, 0, logits.shape[-1] - 1)
    ce = -jnp.take_along_axis(
        jax.nn.log_softmax(logits), safe_labels[..., None], axis=-1
    )[..., 0]
    # Dataset transforms reject unknown labels. Do not silently mask corruption
    # that bypasses them: produce a non-finite objective for the trainer to reject.
    loss = jnp.where(jnp.all(valid), ce.mean(), jnp.nan)
    predicted = jnp.argmax(logits, -1)
    return weight * loss, {
        "stage_ce": loss,
        "stage_loss": weight * loss,
        "stage_accuracy": jnp.mean((predicted == labels).astype(jnp.float32)),
        "stage_label_valid_fraction": valid.astype(jnp.float32).mean(),
    }


def guard_previous_plan(observation, selected_stage, previous_stage):
    """Prevent another stage's TCR residual from conditioning this action chunk."""
    if previous_stage is not None and previous_stage.shape != selected_stage.shape:
        raise ValueError("previous_plan_stage must have shape [batch]")
    same = (
        jnp.zeros_like(selected_stage, dtype=jnp.bool_)
        if previous_stage is None
        else previous_stage == selected_stage
    )
    had_context = jnp.any(observation.previous_plan_mask, axis=-1)
    mask = observation.previous_plan_mask & same[:, None]
    return dataclasses.replace(
        observation, previous_plan_mask=mask
    ), had_context & ~same
