# Copyright 2026 AI Robot Association (AIRoA)
# SPDX-License-Identifier: Apache-2.0
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from post_training_losses.flow import (
    action_time_loss_mean,
    action_time_loss_per_sample,
    exact_action_dim_reduced_mse,
    flow_matching_inputs,
    flow_matching_loss,
)


def test_dimension_masks_normalize_active_dimensions_and_exclude_model_padding():
    prediction = jnp.array([[[1., 3., 100.], [2., 4., 999.]]])
    dynamic = jnp.array([[[True, False], [False, True]]])
    actual = exact_action_dim_reduced_mse(
        prediction, jnp.zeros_like(prediction), jnp.array([1., 1., 0.]), dynamic)
    np.testing.assert_array_equal(actual, [[1., 16.]])
    np.testing.assert_array_equal(
        exact_action_dim_reduced_mse(prediction, prediction), [[0., 0.]])


def test_sample_weights_scale_numerator_without_weighted_denominator():
    errors = jnp.array([[1., 9.], [4., 16.]])
    valid = jnp.array([[True, False], [True, True]])
    actual = action_time_loss_mean(errors, valid, sample_weights=jnp.array([3., .25]))
    np.testing.assert_allclose(actual, 8. / 3.)
    np.testing.assert_array_equal(action_time_loss_per_sample(errors, valid), [1., 10.])
    np.testing.assert_allclose(
        action_time_loss_mean(errors, sample_weights=jnp.array([3., .25])), 8.75)


def test_flow_gradient_ignores_padding_and_invalid_time_but_keeps_active_scale():
    target = jnp.zeros((1, 3, 3))
    valid = jnp.array([[True, False, True]])
    dimensions = jnp.array([[[True, False], [True, True], [True, True]]])
    def loss(prediction):
        return flow_matching_loss(
            prediction, target, action_time_dim_valid_mask=dimensions,
            action_time_valid_mask=valid)
    prediction = jnp.ones_like(target)
    gradient = jax.grad(loss)(prediction)
    np.testing.assert_allclose(gradient, [[[1., 0., 0.], [0., 0., 0.], [.5, .5, 0.]]])
    np.testing.assert_allclose(jax.jit(loss)(prediction), 1.)


def test_finite_all_invalid_rows_reduce_to_zero_with_zero_gradient():
    valid = jnp.zeros((2, 3), dtype=bool)
    loss = lambda p: flow_matching_loss(p, jnp.zeros_like(p), action_time_valid_mask=valid)
    prediction = jnp.ones((2, 3, 4))
    np.testing.assert_array_equal(loss(prediction), 0.)
    np.testing.assert_array_equal(jax.grad(loss)(prediction), jnp.zeros_like(prediction))
    np.testing.assert_array_equal(action_time_loss_per_sample(jnp.ones((2, 3)), valid), [0., 0.])


def test_mask_multiplication_preserves_source_nan_behavior():
    # The reduction requires finite operands even at zero-validity positions.
    assert np.isnan(action_time_loss_mean(jnp.array([[jnp.nan]]), jnp.array([[False]])))
    assert np.isnan(exact_action_dim_reduced_mse(
        jnp.array([[[jnp.nan]]]), jnp.zeros((1, 1, 1)), jnp.array([0.]))[0, 0])


def test_flow_interpolation_uses_clean_zero_noise_one_and_noise_minus_clean_target():
    actions = jnp.array([[[1., -2.]], [[3., 4.]]])
    noise = jnp.array([[[5., 6.]], [[7., -8.]]])
    state, target = flow_matching_inputs(actions, noise, jnp.array([0., 1.]))
    np.testing.assert_array_equal(state[0], actions[0])
    np.testing.assert_array_equal(state[1], noise[1])
    np.testing.assert_array_equal(target, noise - actions)


@pytest.mark.parametrize('call', [
    lambda: action_time_loss_mean(jnp.array(1.)),
    lambda: action_time_loss_mean(jnp.ones((2, 3)), jnp.ones((2,), bool)),
    lambda: action_time_loss_mean(jnp.ones((2, 3)), sample_weights=jnp.ones((2, 1))),
    lambda: action_time_loss_per_sample(jnp.ones((2, 3)), jnp.ones((3,), bool)),
    lambda: exact_action_dim_reduced_mse(jnp.ones((1, 2, 3)), jnp.zeros((1, 2, 3)),
                                        action_dim_mask=jnp.ones((2,))),
    lambda: exact_action_dim_reduced_mse(jnp.ones((1, 2, 3)), jnp.zeros((1, 2, 3)),
                                        action_time_dim_valid_mask=jnp.ones((1, 2, 4))),
])
def test_mask_shape_contracts(call):
    with pytest.raises(ValueError):
        call()
