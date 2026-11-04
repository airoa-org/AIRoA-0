# Copyright 2026 AI Robot Association (AIRoA)
# SPDX-License-Identifier: Apache-2.0
"""PA teacher forcing and stage classification, independent of a model."""

import jax
import jax.numpy as jnp


def pa_attention_mask(prefix_mask, target_mask):
    """Isolate answer tokens from the observation prefix; decoder inputs are shifted."""
    prefix_len, target_len = prefix_mask.shape[1], target_mask.shape[1]
    block = jnp.concatenate(
        [
            jnp.concatenate([jnp.ones((prefix_len, prefix_len), bool), jnp.zeros((prefix_len, target_len), bool)], 1),
            jnp.concatenate(
                [jnp.ones((target_len, prefix_len), bool), jnp.tril(jnp.ones((target_len, target_len), bool))], 1
            ),
        ],
        0,
    )
    valid = jnp.concatenate([prefix_mask, target_mask], 1).astype(bool)
    return block[None] & valid[:, :, None] & valid[:, None, :]


def pa_token_cross_entropy(decode, hidden, targets, loss_mask, *, chunk_size=8):
    """Full-vocabulary CE, normalized per example and evaluated in sequence chunks.

    ``decode`` maps hidden states to vocabulary logits and must be differentiable.
    EOS is included in ``loss_mask``; padding is excluded. Chunk rematerialization
    avoids retaining all vocabulary logits during the backward pass.
    """
    if hidden.shape[:2] != targets.shape or targets.shape != loss_mask.shape:
        raise ValueError("PA hidden/target/loss-mask shapes disagree")
    batch, length = targets.shape
    if chunk_size < 1:
        raise ValueError("PA token chunk size must be positive")
    padding = (-length) % chunk_size
    chunks = (length + padding) // chunk_size
    chunk_hidden = jnp.pad(hidden, ((0, 0), (0, padding), (0, 0))).reshape(
        batch, chunks, chunk_size, hidden.shape[-1]
    ).swapaxes(0, 1)
    chunk_targets = jnp.pad(targets, ((0, 0), (0, padding))).reshape(
        batch, chunks, chunk_size
    ).swapaxes(0, 1)
    chunk_mask = jnp.pad(loss_mask, ((0, 0), (0, padding))).reshape(
        batch, chunks, chunk_size
    ).swapaxes(0, 1)

    @jax.checkpoint
    def chunk(args):
        values, labels, mask = args
        logits = decode(values).astype(jnp.float32)
        labels = jnp.where(mask, labels, 0)
        selected = jnp.take_along_axis(logits, labels[..., None], axis=-1)[..., 0]
        ce = jax.nn.logsumexp(logits, axis=-1) - selected
        return jnp.where(mask, ce, 0.0)

    losses = (
        jax.lax.map(chunk, (chunk_hidden, chunk_targets, chunk_mask))
        .swapaxes(0, 1)
        .reshape(batch, length + padding)[:, :length]
    )
    return losses.sum(-1) / jnp.maximum(loss_mask.sum(-1), 1)


def pa_loss(decode, hidden, targets, loss_mask, *, weight=0.01, chunk_size=8):
    """Average per-example CE over examples with at least one supervised token."""
    per_sample = pa_token_cross_entropy(decode, hidden, targets, loss_mask, chunk_size=chunk_size)
    active = jnp.any(loss_mask, axis=-1)
    ce = per_sample.sum() / jnp.maximum(active.sum(), 1)
    weighted = weight * ce
    return weighted, {
        "pa_aux_ce": ce,
        "pa_aux_loss": weighted,
        "pa_aux_sht_fraction": jnp.mean(active.astype(jnp.float32)),
    }


def stage_loss(logits, labels, *, weight=0.1):
    """Stage CE; invalid class IDs produce NaN so the caller can reject the update."""
    if labels.shape != logits.shape[:-1] or not jnp.issubdtype(labels.dtype, jnp.integer):
        raise ValueError("Stage labels must be integer class IDs with shape [batch]")
    valid = (labels >= 0) & (labels < logits.shape[-1])
    safe_labels = jnp.clip(labels, 0, logits.shape[-1] - 1)
    ce = -jnp.take_along_axis(jax.nn.log_softmax(logits), safe_labels[..., None], axis=-1)[..., 0]
    loss = jnp.where(jnp.all(valid), ce.mean(), jnp.nan)
    predicted = jnp.argmax(logits, -1)
    return weight * loss, {
        "stage_ce": loss,
        "stage_loss": weight * loss,
        "stage_accuracy": jnp.mean((predicted == labels).astype(jnp.float32)),
        "stage_label_valid_fraction": valid.astype(jnp.float32).mean(),
    }
