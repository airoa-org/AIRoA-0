# Copyright 2026 AI Robot Association (AIRoA)
# SPDX-License-Identifier: Apache-2.0
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from post_training_losses.visual_budget import (
    attention_log_prior, attention_strength, visual_budget_statistics,
    embedding_gates, linear_schedule, project_attention_probabilities,
)


@pytest.mark.parametrize('dtype', [jnp.float32, jnp.bfloat16])
def test_embedding_gate_has_raw_sigmoid_and_independent_floor_with_live_gradient(dtype):
    logits = jnp.array([[-1., .25, 1.5]], dtype)
    valid = jnp.array([[True, False, True]])
    raw, effective = embedding_gates(logits, valid, tau=.805, gate_floor=.1)
    sigmoid = jax.nn.sigmoid(logits / jnp.asarray(.805, dtype))
    expected = jnp.where(valid, jnp.asarray(.1, dtype) + (1 - jnp.asarray(.1, dtype)) * sigmoid, 0)
    np.testing.assert_array_equal(effective, expected)
    np.testing.assert_array_equal(raw[:, 1], 0.)
    actual = jax.grad(lambda z: embedding_gates(z, valid, tau=.805, gate_floor=.1)[1].sum())(logits)
    analytic = jnp.where(valid, (1 - jnp.asarray(.1, dtype)) * sigmoid * (1 - sigmoid) / jnp.asarray(.805, dtype), 0)
    assert bool(jnp.all(jnp.isfinite(actual))) and bool(jnp.all(actual[:, jnp.array([0, 2])] > 0))
    np.testing.assert_allclose(np.asarray(actual, dtype=np.float32), np.asarray(analytic, dtype=np.float32), rtol=.025, atol=.002)


def test_projection_is_one_sided_per_sample_and_all_invalid_nans_are_inert():
    logits = jnp.array([[-3., -2., 0.], [8., 4., -1.], [jnp.nan] * 3])
    valid = jnp.array([[True, True, False], [True, True, True], [False] * 3])
    p, nu, raw_mean, projected_mean, binding = project_attention_probabilities(
        logits, valid, tau=.805, keep_ratio=.4)
    np.testing.assert_array_equal(binding, [False, True, False])
    np.testing.assert_array_equal(nu[jnp.array([0, 2])], 0.)
    np.testing.assert_array_equal(p[0, :2], jax.nn.sigmoid(logits[0, :2] / .805))
    np.testing.assert_array_equal(p[2], 0.)
    assert float(nu[1]) > 0 and float(raw_mean[1]) > .4
    assert float(projected_mean[1]) <= .4 + 1e-7
    np.testing.assert_allclose(projected_mean[1], .4, atol=2e-7)
    assert bool(jnp.all(jnp.isfinite(p)))


def test_projection_returns_feasible_upper_endpoint_after_requested_iterations():
    logits = jnp.array([[5., 4.]])
    valid = jnp.ones((1, 2), bool)
    _, nu, _, mean, binding = project_attention_probabilities(
        logits, valid, tau=1., keep_ratio=.2, iterations=1)
    upper = 5. - (np.log(.2) - np.log1p(-.2))
    midpoint = upper / 2
    midpoint_mean = np.mean(1 / (1 + np.exp(-(np.array([5., 4.]) - midpoint))))
    expected_upper = upper if midpoint_mean > .2 else midpoint
    np.testing.assert_allclose(nu, [expected_upper], rtol=1e-6)
    assert bool(binding[0]) and float(mean[0]) <= .2


@pytest.mark.parametrize('dtype', [jnp.float32, jnp.bfloat16])
def test_attention_prior_is_noncentered_projected_floored_and_both_outputs_detached(dtype):
    logits = jnp.array([[5., 3., 0.]], dtype)
    valid = jnp.array([[True, True, False]])
    prior, probability = attention_log_prior(logits, valid, tau=.805, keep_ratio=.4)
    projected = project_attention_probabilities(logits, valid, tau=.805, keep_ratio=.4)[0]
    floor = jnp.exp(-jnp.float32(4.))
    expected = jnp.where(valid, jnp.clip(jnp.log(floor + (1 - floor) * projected), -4., 0.), 0.)
    np.testing.assert_array_equal(prior, expected)
    np.testing.assert_array_equal(probability, projected)
    assert prior.dtype == jnp.float32 and probability.dtype == jnp.float32
    assert bool(jnp.all(prior[0, :2] < 0))  # No valid-token centering.
    for index in (0, 1):
        gradient = jax.grad(lambda z: attention_log_prior(z, valid, tau=.805, keep_ratio=.4)[index].sum())(logits)
        np.testing.assert_array_equal(gradient, jnp.zeros_like(logits))


def test_kind_budget_averages_sample_view_errors_not_global_token_or_view_means():
    probabilities = {'base_a': np.array([[.2, .8], [.4, .4]]),
                     'base_b': np.array([[.8, .8], [.7, .7]]),
                     'wrist': np.array([[.1, .1], [.6, .6]])}
    logits = {k: jnp.asarray(np.log(p / (1 - p)), jnp.float32) for k, p in probabilities.items()}
    valid = {'base_a': jnp.array([[True, True], [True, False]]),
             'base_b': jnp.ones((2, 2), bool), 'wrist': jnp.ones((2, 2), bool)}
    present = {'base_a': jnp.array([True, True]), 'base_b': jnp.array([True, False]),
               'wrist': jnp.array([False, True])}
    prior = {k: jnp.zeros_like(v) for k, v in logits.items()}
    result = visual_budget_statistics(logits, valid, prior, view_kinds={'base_a': 'base', 'base_b': 'base', 'wrist': 'wrist'},
                                tau=1., keep_ratio_base=.9, keep_ratio_wrist=.9, view_valid_by_view=present)
    np.testing.assert_allclose(result['budget_base'], ((.5 - .9)**2 + (.4 - .9)**2 + (.8 - .9)**2) / 3, rtol=1e-6)
    np.testing.assert_allclose(result['budget_wrist'], (.6 - .9)**2, rtol=1e-6)
    np.testing.assert_array_equal(result['anchor_base'], 0.)
    np.testing.assert_array_equal(result['anchor_wrist'], 0.)


def test_anchor_averages_only_positive_prior_mass_and_matches_weighted_token_hinge():
    p = jnp.array([[.05, .3], [.05, .05], [.01, .01]])
    logits = {'w': jnp.log(p / (1 - p))}
    valid = {'w': jnp.ones((3, 2), bool)}
    prior = {'w': jnp.array([[2., 1.], [0., 0.], [1., 1.]])}
    result = visual_budget_statistics(logits, valid, prior, view_kinds={'w': 'wrist'}, tau=1.,
                                view_valid_by_view={'w': jnp.array([True, True, False])})
    np.testing.assert_allclose(result['anchor_wrist'], 2 * (.2 - .05) / 3, rtol=1e-6)
    np.testing.assert_array_equal(result['anchor_base'], 0.)
    np.testing.assert_array_equal(result['budget_base'], 0.)


@pytest.mark.parametrize('dtype', [jnp.float32, jnp.bfloat16])
def test_budget_statistic_preserves_live_finite_scalar_gradient(dtype):
    logits = jnp.array([[0., 0.]], dtype)
    def value(z):
        return visual_budget_statistics({'a': z}, {'a': jnp.ones((1, 2), bool)},
                                  {'a': jnp.zeros((1, 2), dtype)}, view_kinds={'a': 'base'}, tau=1.)['budget_base']
    gradient = jax.grad(value)(logits)
    # d(mean(sigmoid(z)) - .9)^2/dz_i at z=0 equals -.1.
    np.testing.assert_allclose(np.asarray(gradient, dtype=np.float32), -.1, rtol=.025, atol=.002)
    assert bool(jnp.all(jnp.isfinite(gradient)))


def test_schedule_boundaries_terminal_single_step_and_independent_attention_warmup():
    np.testing.assert_allclose([linear_schedule(.1, .02, step=s, total_steps=5) for s in (-1, 0, 2, 4, 9)], [.1, .1, .06, .02, .02], atol=1e-7)
    np.testing.assert_allclose(linear_schedule(.1, .02, step=0, total_steps=1), .02, atol=1e-7)
    np.testing.assert_allclose([attention_strength(s) for s in (-1, 0, 250, 500, 999)], [0, 0, .5, 1, 1])
    with pytest.raises(ValueError):
        attention_strength(0, warmup_steps=0)


@pytest.mark.parametrize('ratio', [0., 1., -.1])
def test_projection_rejects_invalid_soft_budget(ratio):
    with pytest.raises(ValueError):
        project_attention_probabilities(jnp.zeros((1, 2)), jnp.ones((1, 2), bool), tau=1., keep_ratio=ratio)
