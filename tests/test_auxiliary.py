# Copyright 2026 AI Robot Association (AIRoA)
# SPDX-License-Identifier: Apache-2.0
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from post_training_losses.auxiliary import (
    pa_attention_mask, pa_loss, pa_token_cross_entropy, stage_loss,
)


def dense_pa(decode, hidden, targets, mask):
    logits = decode(hidden).astype(jnp.float32)
    safe = jnp.where(mask, targets, 0)
    ce = -jnp.take_along_axis(jax.nn.log_softmax(logits), safe[..., None], -1)[..., 0]
    return jnp.where(mask, ce, 0).sum(-1) / jnp.maximum(mask.sum(-1), 1)


@pytest.mark.parametrize('dtype', [jnp.float32, jnp.bfloat16])
@pytest.mark.parametrize('chunk_size', [1, 3, 8])
def test_chunked_full_vocabulary_values_and_parameter_gradients_match_dense(dtype, chunk_size):
    hidden = (jnp.arange(60).reshape(3, 5, 4) / 30 - 1).astype(dtype)
    weights = (jnp.arange(28).reshape(4, 7) / 25 - .5).astype(dtype)
    targets = jnp.array([[0, 2, 6, 1, 4], [3, 5, 99, 99, 99], [99] * 5])
    mask = jnp.array([[1] * 5, [1, 1, 0, 0, 0], [0] * 5], bool)
    def actual(h, w):
        return pa_token_cross_entropy(lambda x: x @ w, h, targets, mask,
                                      chunk_size=chunk_size).sum()
    def expected(h, w):
        return dense_pa(lambda x: x @ w, h, targets, mask).sum()
    tolerance = .02 if dtype == jnp.bfloat16 else 2e-6
    np.testing.assert_allclose(actual(hidden, weights), expected(hidden, weights), rtol=2e-6)
    for actual_gradient, expected_gradient in zip(
            jax.grad(actual, (0, 1))(hidden, weights),
            jax.grad(expected, (0, 1))(hidden, weights)):
        assert bool(jnp.all(jnp.isfinite(actual_gradient)))
        np.testing.assert_allclose(np.asarray(actual_gradient, dtype=np.float32),
                                   np.asarray(expected_gradient, dtype=np.float32),
                                   rtol=tolerance, atol=tolerance)


def test_pa_reduces_tokens_per_example_then_only_active_examples_and_keeps_eos():
    logits = jnp.array([[[5., 0.], [0., 4.], [2., 0.]],
                        [[0., 0.], [0., 0.], [0., 0.]],
                        [[8., -8.], [8., -8.], [8., -8.]]])
    target = jnp.zeros((3, 3), jnp.int32)
    mask = jnp.array([[True, True, True], [True, False, False], [False] * 3])
    per_example = dense_pa(lambda x: x, logits, target, mask)
    weighted, metrics = pa_loss(lambda x: x, logits, target, mask)
    np.testing.assert_allclose(weighted, .01 * (per_example[0] + per_example[1]) / 2)
    np.testing.assert_allclose(metrics['pa_aux_sht_fraction'], 2 / 3)
    changed_eos = logits.at[0, 2].set(jnp.array([-5., 5.]))
    assert float(pa_loss(lambda x: x, changed_eos, target, mask)[0]) > float(weighted)
    changed_padding = logits.at[2].set(-logits[2])
    np.testing.assert_array_equal(pa_loss(lambda x: x, changed_padding, target, mask)[0], weighted)


def test_all_inactive_pa_is_zero_with_zero_hidden_gradient():
    hidden = jnp.ones((2, 3, 5))
    labels = jnp.full((2, 3), 99)
    mask = jnp.zeros((2, 3), bool)
    loss = lambda h: pa_loss(lambda x: x, h, labels, mask)[0]
    np.testing.assert_array_equal(loss(hidden), 0.)
    np.testing.assert_array_equal(jax.grad(loss)(hidden), jnp.zeros_like(hidden))


def test_teacher_forcing_prefix_never_reads_answers_and_decoder_cannot_read_future():
    mask = pa_attention_mask(jnp.array([[True, True]]), jnp.array([[True] * 3]))[0]
    expected = np.array([[1, 1, 0, 0, 0], [1, 1, 0, 0, 0],
                         [1, 1, 1, 0, 0], [1, 1, 1, 1, 0], [1, 1, 1, 1, 1]], bool)
    np.testing.assert_array_equal(mask, expected)
    def attend(values):
        return (mask.astype(jnp.float32) @ values) / mask.sum(-1)
    jacobian = jax.jacrev(attend)(jnp.arange(5, dtype=jnp.float32))
    np.testing.assert_array_equal(jacobian[:2, 2:], 0.)
    np.testing.assert_array_equal(jacobian[2, 3:], 0.)
    padded = pa_attention_mask(jnp.array([[True, False]]), jnp.array([[True, True, False]]))[0]
    np.testing.assert_array_equal(padded[jnp.array([1, 4])], False)
    np.testing.assert_array_equal(padded[:, jnp.array([1, 4])], False)


@pytest.mark.parametrize('labels', [jnp.array([-1, 1]), jnp.array([0, 3])])
def test_invalid_stage_ids_produce_nonfinite_whole_objective(labels):
    loss, metrics = stage_loss(jnp.zeros((2, 3)), labels)
    assert bool(jnp.isnan(loss)) and bool(jnp.isnan(metrics['stage_ce']))
    np.testing.assert_allclose(metrics['stage_label_valid_fraction'], .5)


@pytest.mark.parametrize('labels', [jnp.zeros((2, 1), jnp.int32), jnp.zeros((2,), jnp.float32)])
def test_stage_rejects_shape_or_noninteger_labels(labels):
    with pytest.raises(ValueError):
        stage_loss(jnp.zeros((2, 3)), labels)


@pytest.mark.parametrize('dtype', [jnp.float32, jnp.bfloat16])
def test_stage_loss_and_gradient_are_weighted_batch_mean(dtype):
    logits = jnp.array([[.5, -.25, 1.], [-1., 2., .25]], dtype)
    labels = jnp.array([2, 0])
    expected = lambda x: -.1 * jax.nn.log_softmax(x)[jnp.arange(2), labels].mean()
    actual = lambda x: stage_loss(x, labels)[0]
    np.testing.assert_array_equal(actual(logits), expected(logits))
    np.testing.assert_allclose(np.asarray(jax.grad(actual)(logits), dtype=np.float32),
                               np.asarray(jax.grad(expected)(logits), dtype=np.float32), rtol=.02, atol=.002)
    assert bool(jnp.isnan(stage_loss(logits.at[0, 0].set(jnp.nan), labels)[0]))


@pytest.mark.parametrize('chunk_size', [0, -1])
def test_pa_rejects_nonpositive_chunk_size(chunk_size):
    with pytest.raises(ValueError):
        pa_token_cross_entropy(lambda x: x, jnp.zeros((1, 2, 3)),
                               jnp.zeros((1, 2), jnp.int32), jnp.ones((1, 2), bool),
                               chunk_size=chunk_size)
