# Copyright 2026 AI Robot Association (AIRoA)
# SPDX-License-Identifier: Apache-2.0
"""Assemble the action objective with PA, stage, and visual budget terms."""

import jax.numpy as jnp


def post_training_loss(
    action_loss,
    *,
    visual_budget_terms=None,
    pa_loss=None,
    stage_loss=None,
    action_weight=1.0,
    budget_base_weight=0.1,
    budget_wrist_weight=0.1,
    anchor_weight=0.0175,
):
    """Return total and scalar metrics.

    PA and stage inputs are weighted scalars returned by ``pa_loss`` and
    ``stage_loss``. Visual-budget inputs are raw terms from
    ``visual_budget_statistics``. The action prediction uses one gated forward.
    """
    action_loss = jnp.asarray(action_loss)
    if action_loss.ndim != 0:
        raise ValueError("action_loss must be a scalar")
    zero = jnp.asarray(0.0, dtype=action_loss.dtype)
    budget = anchor = zero
    if visual_budget_terms is not None:
        required = {"budget_base", "budget_wrist", "anchor_base", "anchor_wrist"}
        if set(visual_budget_terms) != required or any(jnp.shape(x) != () for x in visual_budget_terms.values()):
            raise ValueError("visual_budget_terms must contain four scalar budget/anchor terms")
        budget = budget_base_weight * visual_budget_terms["budget_base"] + budget_wrist_weight * visual_budget_terms["budget_wrist"]
        anchor = jnp.asarray(anchor_weight, dtype=action_loss.dtype) * (
            visual_budget_terms["anchor_base"] + visual_budget_terms["anchor_wrist"]
        )
    pa = zero if pa_loss is None else jnp.asarray(pa_loss)
    stage = zero if stage_loss is None else jnp.asarray(stage_loss)
    if pa.ndim != 0 or stage.ndim != 0:
        raise ValueError("PA and stage losses must be weighted scalars")
    action = jnp.asarray(action_weight, dtype=action_loss.dtype) * action_loss
    total = action + budget + anchor
    total = total + stage
    total = total + pa
    return total, {"loss_action": action_loss, "loss_action_weighted": action,
                   "loss_budget": budget, "loss_anchor": anchor,
                   "loss_stage": stage, "loss_pa": pa, "loss_total": total}
