# Copyright 2026 AI Robot Association (AIRoA)
# SPDX-License-Identifier: Apache-2.0
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from post_training_losses import post_training_loss


def test_weights_are_applied_once_and_no_full_action_branch_is_added():
    terms = {"budget_base": jnp.asarray(2.), "budget_wrist": jnp.asarray(3.),
             "anchor_base": jnp.asarray(4.), "anchor_wrist": jnp.asarray(5.)}
    total, metrics = post_training_loss(jnp.asarray(7.), visual_budget_terms=terms,
        pa_loss=jnp.asarray(.03), stage_loss=jnp.asarray(.2), anchor_weight=.02)
    np.testing.assert_allclose(total, 7. + .2 + .3 + .18 + .2 + .03, rtol=1e-6)
    np.testing.assert_allclose(metrics["loss_pa"], .03)
    np.testing.assert_allclose(jax.grad(lambda x: post_training_loss(x, visual_budget_terms=terms)[0])(7.), 1.)


def test_optional_terms_disable_cleanly_and_invalid_stage_still_invalidates_total():
    np.testing.assert_array_equal(post_training_loss(jnp.asarray(2.))[0], 2.)
    assert jnp.isnan(post_training_loss(jnp.asarray(2.), stage_loss=jnp.asarray(jnp.nan))[0])
    with pytest.raises(ValueError):
        post_training_loss(jnp.ones(2))
    with pytest.raises(ValueError):
        post_training_loss(jnp.asarray(2.), visual_budget_terms={"budget_base": jnp.asarray(0.)})
