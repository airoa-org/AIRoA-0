# Copyright 2026 AI Robot Association (AIRoA)
# SPDX-License-Identifier: Apache-2.0
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from post_training_losses.nuf import (
    flow_levels,
    flow_substeps,
    nuf_loss,
    pack,
    packed_causal_attention,
    post_levels,
    training_inputs,
)


def test_pack_right_aligns_prefixes_with_chronological_chunk_order():
    values = jnp.array([[[1.], [2.], [3.], [4.], [5.], [6.]]])
    packed = pack(values, chunk_size=2, buffer_size=3)
    np.testing.assert_array_equal(packed[0, :, 0],
        [0., 0., 0., 0., 1., 2., 0., 0., 1., 2., 3., 4., 1., 2., 3., 4., 5., 6.])


def test_prefix_exposure_is_n_to_one_and_loss_uses_valid_token_denominator():
    values = jnp.array([[[1.], [2.], [3.]]])
    prediction = pack(values, chunk_size=1, buffer_size=3)
    valid = pack(jnp.ones((1, 3), bool), chunk_size=1, buffer_size=3, padding=False)
    # Three exposures of chunk1, two of chunk2, one of chunk3: 20/6.
    np.testing.assert_allclose(nuf_loss(prediction, jnp.zeros_like(prediction), valid), 20. / 6.)
    grad = jax.grad(lambda p: nuf_loss(p, jnp.zeros_like(p), valid))(prediction)
    np.testing.assert_array_equal(np.asarray(grad)[~np.asarray(valid)], 0.)


def test_attention_is_causal_by_chunk_and_isolates_packed_states():
    valid = pack(jnp.ones((1, 6), bool), chunk_size=2, buffer_size=3, padding=False)
    mask, positions = packed_causal_attention(valid, chunk_size=2, buffer_size=3)
    expected = np.zeros((1, 18, 18), dtype=bool)
    v = np.asarray(valid)[0]
    for query in range(18):
        for key in range(18):
            expected[0, query, key] = (v[query] and v[key] and query // 6 == key // 6
                                       and (key % 6) // 2 <= (query % 6) // 2)
    np.testing.assert_array_equal(mask, expected)
    np.testing.assert_array_equal(positions, [[-1, -1, -1, -1, 0, 1, -1, -1, 0, 1, 2, 3,
                                             0, 1, 2, 3, 4, 5]])


def test_invalid_source_actions_are_sanitized_before_noise_interpolation():
    actions = jnp.array([[[2., jnp.nan], [jnp.nan, jnp.nan], [3., 4.], [5., 6.]]])
    temporal = jnp.array([[True, False, True, True]])
    dimensions = jnp.array([[[True, False], [False, False], [True, True], [True, True]]])
    rng = jax.random.key(7)
    actual = training_inputs(rng, actions, chunk_size=2, buffer_size=2,
                             temporal=temporal, dimensions=dimensions)
    sanitized = jnp.array([[[2., 0.], [0., 0.], [3., 4.], [5., 6.]]])
    reference = training_inputs(rng, sanitized, chunk_size=2, buffer_size=2,
                                temporal=temporal, dimensions=dimensions)
    for a, b in zip(actual, reference, strict=True):
        np.testing.assert_array_equal(a, b)
    assert np.isfinite(np.asarray(actual[0])).all()
    assert np.isfinite(np.asarray(actual[1])).all()
    invalid = ~np.asarray(actual[3])
    np.testing.assert_array_equal(np.asarray(actual[0])[invalid], 0.)
    np.testing.assert_array_equal(np.asarray(actual[1])[invalid], 0.)


def test_uniform_training_rng_and_interpolation_arithmetic_are_preserved():
    rng = jax.random.key(12)
    actions = jnp.arange(12, dtype=jnp.float32).reshape(2, 3, 2)
    x, target, times, valid, dimensions = training_inputs(rng, actions, chunk_size=1, buffer_size=3)
    time_rng, noise_rng = jax.random.split(rng)
    u = jax.random.beta(time_rng, 1.5, 1.0, (2, 3, 3))
    expected_time = (0.001 + 0.999 * (jnp.arange(3, dtype=jnp.float32)[None, None, :] + u) / 3).reshape(2, 9)
    clean = pack(actions, chunk_size=1, buffer_size=3)
    noise = jax.random.normal(noise_rng, clean.shape)
    np.testing.assert_array_equal(times, expected_time)
    np.testing.assert_array_equal(x, jnp.where(valid[..., None],
        (1 - times[..., None]) * clean + times[..., None] * noise, 0))
    np.testing.assert_array_equal(target, jnp.where(valid[..., None], noise - clean, 0))
    assert dimensions is None


def test_valid_nan_is_not_silently_repaired_and_narrow_masks_keep_source_width_behavior():
    rng = jax.random.key(9)
    actions = jnp.array([[[jnp.nan, 7.]]])
    output = training_inputs(rng, actions, chunk_size=1, buffer_size=1)
    assert np.isnan(np.asarray(output[0])[0, 0, 0])
    assert np.isnan(np.asarray(output[1])[0, 0, 0])
    # A physical mask narrower than model width preserves the original input
    # construction. The loss mask, rather than input construction, excludes
    # trailing model dimensions; no extra numerical repair is inserted here.
    narrow = jnp.array([[[True]]])
    a = training_inputs(rng, jnp.array([[[2., 7.]]]), chunk_size=1, buffer_size=1,
                        dimensions=narrow)
    b = training_inputs(rng, jnp.array([[[2., 0.]]]), chunk_size=1, buffer_size=1,
                        dimensions=narrow)
    assert not np.array_equal(np.asarray(a[0]), np.asarray(b[0]))
    np.testing.assert_allclose(nuf_loss(a[1], a[1], a[3], dimensions=a[4]), 0.)


def test_nonuniform_times_and_substeps_stay_in_their_original_slot_intervals():
    rng = jax.random.key(44)
    actions = jnp.zeros((64, 3, 2))
    levels = (.1, .4, 1.)
    paths = ((.1, 0.), (.4, .2, .1), (1., .7, .4))
    assert flow_levels(levels, buffer_size=3) == levels
    assert post_levels(levels) == (0., .1, .4)
    assert flow_substeps(paths, levels=levels, buffer_size=3) == paths
    for kwargs in ({'levels': levels}, {'levels': levels, 'substeps': paths}):
        times = np.asarray(training_inputs(rng, actions, chunk_size=1, buffer_size=3, **kwargs)[2]).reshape(64, 3, 3)
        for slot, (low, high) in enumerate(zip((.0001, .1, .4), levels, strict=True)):
            assert np.all(times[:, :, slot] >= np.float32(low))
            assert np.all(times[:, :, slot] <= np.float32(high))
    # None keeps the legacy RNG arithmetic; an explicit uniform clock is a
    # distinct supported sampling branch rather than a numerical alias.


@pytest.mark.parametrize('levels', [(0., 1.), (.5, .5), (.2, .9), (True, 1.), (np.nan, 1.)])
def test_invalid_nonuniform_clocks_fail(levels):
    with pytest.raises(ValueError):
        flow_levels(levels, buffer_size=2)


def test_substeps_require_exact_decreasing_pre_to_post_paths():
    with pytest.raises(ValueError):
        flow_substeps(((.2, 0.), (1., .1)), levels=(.2, 1.), buffer_size=2)
    with pytest.raises(ValueError):
        flow_substeps(((.2, .3, 0.), (1., .2)), levels=(.2, 1.), buffer_size=2)
