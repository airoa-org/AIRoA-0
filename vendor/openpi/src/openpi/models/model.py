# Modified by AI Robot Association (AIRoA) in 2026 for AIRoA-0 (A-0) post-training.
# Derived from Physical Intelligence openpi (Apache-2.0), revision 215abfb.
import abc
from collections.abc import Sequence
import dataclasses
import enum
import functools
import logging
import pathlib
from typing import Generic, TypeVar
import augmax
from flax import nnx
from flax import struct
from flax import traverse_util
import jax
import jax.numpy as jnp
import numpy as np
import orbax.checkpoint as ocp
import torch
from openpi.shared import image_tools
import openpi.shared.array_typing as at

logger = logging.getLogger("openpi")
ArrayT = TypeVar("ArrayT", bound=jax.Array | torch.Tensor | np.ndarray)


class ModelType(enum.Enum):
    """Supported model types."""

    PI0 = "pi0"
    PI0_FAST = "pi0_fast"
    PI05 = "pi05"


IMAGE_KEYS = ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb")
IMAGE_RESOLUTION = (224, 224)
_IMAGE_PRIOR_LEGACY_STRENGTH_KEY = ("image_prior_strength",)
_IMAGE_PRIOR_RAW_STRENGTH_KEY = ("image_prior_strength_raw",)
_IMAGE_PRIOR_INIT_EPS = 0.0001


def _cast_like(reference, value):
    if isinstance(reference, jax.Array):
        return jnp.asarray(value, dtype=reference.dtype)
    if isinstance(reference, np.ndarray):
        return np.asarray(value, dtype=reference.dtype)
    return np.asarray(value, dtype=getattr(reference, "dtype", None))


def _effective_to_raw_strength(value) -> np.ndarray:
    effective = np.asarray(value, dtype=np.float32)
    effective = np.clip(effective, 0.0, 1.0)
    effective = np.clip(effective, _IMAGE_PRIOR_INIT_EPS, 1.0 - _IMAGE_PRIOR_INIT_EPS)
    return np.log(effective) - np.log1p(-effective)


def _align_optional_image_prior_params(
    expected: at.Params, params: at.Params
) -> at.Params:
    flat_expected = traverse_util.flatten_dict(expected)
    flat_params = traverse_util.flatten_dict(params)
    expected_raw_keys = [
        kp for kp in flat_expected if kp[:1] == _IMAGE_PRIOR_RAW_STRENGTH_KEY
    ]
    loaded_raw_keys = [
        kp for kp in flat_params if kp[:1] == _IMAGE_PRIOR_RAW_STRENGTH_KEY
    ]
    if expected_raw_keys and _IMAGE_PRIOR_LEGACY_STRENGTH_KEY in flat_params:
        legacy_raw = _effective_to_raw_strength(
            flat_params.pop(_IMAGE_PRIOR_LEGACY_STRENGTH_KEY)
        )
        if (
            len(expected_raw_keys) == 1
            and expected_raw_keys[0] == _IMAGE_PRIOR_RAW_STRENGTH_KEY
        ):
            flat_params[_IMAGE_PRIOR_RAW_STRENGTH_KEY] = _cast_like(
                flat_expected[_IMAGE_PRIOR_RAW_STRENGTH_KEY], legacy_raw
            )
        else:
            for kp in expected_raw_keys:
                flat_params.setdefault(kp, _cast_like(flat_expected[kp], legacy_raw))
    loaded_raw_keys = [
        kp for kp in flat_params if kp[:1] == _IMAGE_PRIOR_RAW_STRENGTH_KEY
    ]
    if (
        expected_raw_keys
        and _IMAGE_PRIOR_RAW_STRENGTH_KEY in flat_params
        and any((len(kp) > 1 for kp in expected_raw_keys))
    ):
        raw_strength = flat_params.pop(_IMAGE_PRIOR_RAW_STRENGTH_KEY)
        for kp in expected_raw_keys:
            flat_params.setdefault(kp, _cast_like(flat_expected[kp], raw_strength))
    loaded_raw_keys = [
        kp for kp in flat_params if kp[:1] == _IMAGE_PRIOR_RAW_STRENGTH_KEY
    ]
    if (
        _IMAGE_PRIOR_RAW_STRENGTH_KEY in flat_expected
        and loaded_raw_keys
        and (_IMAGE_PRIOR_RAW_STRENGTH_KEY not in flat_params)
    ):
        raw_strength = np.mean(
            np.stack(
                [
                    np.asarray(flat_params[kp], dtype=np.float32)
                    for kp in loaded_raw_keys
                ]
            )
        )
        flat_params[_IMAGE_PRIOR_RAW_STRENGTH_KEY] = _cast_like(
            flat_expected[_IMAGE_PRIOR_RAW_STRENGTH_KEY], raw_strength
        )
    for kp, ref in flat_expected.items():
        if kp[:1] == _IMAGE_PRIOR_RAW_STRENGTH_KEY and kp not in flat_params:
            flat_params[kp] = ref
    return traverse_util.unflatten_dict(flat_params)


def _find_shape_dtype_struct_leaves(
    tree: at.PyTree[ArrayT],
) -> list[tuple[tuple[str, ...], jax.ShapeDtypeStruct]]:
    flat_tree = traverse_util.flatten_dict(tree)
    return [
        (kp, value)
        for (kp, value) in flat_tree.items()
        if isinstance(value, jax.ShapeDtypeStruct)
    ]


def normalize_image_baked_padding_by_view(
    image_baked_padding_by_view: dict[str, tuple[int, int, int, int]] | None,
) -> dict[str, tuple[int, int, int, int]] | None:
    """Normalize optional per-view baked-padding settings.

    The mapping values are interpreted as `(top, bottom, left, right)` pixel
    counts already present in the incoming image before runtime preprocessing.
    """
    if image_baked_padding_by_view is None:
        return None
    normalized: dict[str, tuple[int, int, int, int]] = {}
    for key, padding in image_baked_padding_by_view.items():
        if len(padding) != 4:
            raise ValueError(
                f"image_baked_padding_by_view[{key!r}] must be a 4-tuple `(top, bottom, left, right)`, got {padding!r}."
            )
        (top, bottom, left, right) = (int(v) for v in padding)
        if min(top, bottom, left, right) < 0:
            raise ValueError(
                f"image_baked_padding_by_view[{key!r}] must be non-negative, got {(top, bottom, left, right)!r}."
            )
        normalized[str(key)] = (top, bottom, left, right)
    return normalized


def resolve_image_baked_padding(
    image_baked_padding_by_view: dict[str, tuple[int, int, int, int]] | None,
    image_key: str,
) -> tuple[int, int, int, int] | None:
    if image_baked_padding_by_view is None:
        return None
    if image_key in image_baked_padding_by_view:
        return image_baked_padding_by_view[image_key]
    return image_baked_padding_by_view.get("*")


@at.typecheck
@struct.dataclass
class Observation(Generic[ArrayT]):
    """Holds observations, i.e., inputs to the model.

    See `Observation.from_dict` to see the expected dictionary form. This is the format
    that should be produced by the data transforms.
    """

    images: dict[str, at.Float[ArrayT, "*b h w c"]]
    image_masks: dict[str, at.Bool[ArrayT, "*b"]]
    state: at.Float[ArrayT, "*b s"]
    image_priors: dict[str, at.Float[ArrayT, "*b hp wp"]] | None = None
    image_valid_masks: dict[str, at.Bool[ArrayT, "*b h w"]] | None = None
    image_history: dict[str, at.Float[ArrayT, "*b th h w c"]] | None = None
    image_history_masks: dict[str, at.Bool[ArrayT, "*b th"]] | None = None
    image_history_ages: dict[str, at.Float[ArrayT, "*b th"]] | None = None
    image_history_valid_masks: dict[str, at.Bool[ArrayT, "*b th h w"]] | None = None
    sparse_history_tokens: dict[str, ArrayT] | None = None
    sparse_history_token_masks: dict[str, ArrayT] | None = None
    sparse_current_tokens: dict[str, ArrayT] | None = None
    state_history: at.Float[ArrayT, "*b ts s"] | None = None
    state_history_mask: at.Bool[ArrayT, "*b ts"] | None = None
    state_history_age: at.Float[ArrayT, "*b ts"] | None = None
    tokenized_prompt: at.Int[ArrayT, "*b l"] | None = None
    tokenized_prompt_mask: at.Bool[ArrayT, "*b l"] | None = None
    pa_input_tokens: at.Int[ArrayT, "*b pa_l"] | None = None
    pa_target_tokens: at.Int[ArrayT, "*b pa_l"] | None = None
    pa_loss_mask: at.Bool[ArrayT, "*b pa_l"] | None = None
    progress_target: at.Float[ArrayT, "*b"] | None = None
    stage_label: at.Int[ArrayT, "*b"] | None = None
    token_ar_mask: at.Int[ArrayT, "*b l"] | None = None
    token_loss_mask: at.Bool[ArrayT, "*b l"] | None = None
    bot_token_mask: at.Bool[ArrayT, "*b l"] | None = None
    dataset_root_index: at.Int[ArrayT, "*b"] | None = None
    action_time_valid_mask: at.Bool[ArrayT, "*b ah"] | None = None
    action_time_dim_valid_mask: at.Bool[ArrayT, "*b ah ad"] | None = None
    previous_plan_actions: at.Float[ArrayT, "*b ah pad"] | None = None
    previous_plan_mask: at.Bool[ArrayT, "*b ah"] | None = None
    previous_plan_confidence: at.Float[ArrayT, "*b ah"] | None = None

    @classmethod
    def from_dict(cls, data: at.PyTree[ArrayT]) -> "Observation[ArrayT]":
        """This method defines the mapping between unstructured data (i.e., nested dict) to the structured Observation format."""
        if ("tokenized_prompt" in data) != ("tokenized_prompt_mask" in data):
            raise ValueError(
                "tokenized_prompt and tokenized_prompt_mask must be provided together."
            )
        for key in data["image"]:
            if data["image"][key].dtype == np.uint8:
                data["image"][key] = (
                    data["image"][key].astype(np.float32) / 255.0 * 2.0 - 1.0
                )
            elif (
                hasattr(data["image"][key], "dtype")
                and data["image"][key].dtype == torch.uint8
            ):
                data["image"][key] = (
                    data["image"][key].to(torch.float32).permute(0, 3, 1, 2)
                    / 255.0
                    * 2.0
                    - 1.0
                )
        image_history = data.get("image_history")
        if image_history is not None:
            for key, value in image_history.items():
                if value.dtype == np.uint8:
                    image_history[key] = value.astype(np.float32) / 255.0 * 2.0 - 1.0
                elif hasattr(value, "dtype") and value.dtype == torch.uint8:
                    image_history[key] = value.to(torch.float32) / 255.0 * 2.0 - 1.0
        return cls(
            images=data["image"],
            image_masks=data["image_mask"],
            image_valid_masks=data.get("image_valid_mask"),
            image_priors=data.get("image_prior"),
            image_history=image_history,
            image_history_masks=data.get("image_history_mask"),
            image_history_ages=data.get("image_history_age"),
            image_history_valid_masks=data.get("image_history_valid_mask"),
            sparse_history_tokens=data.get("sparse_history_tokens"),
            sparse_history_token_masks=data.get("sparse_history_token_masks"),
            sparse_current_tokens=data.get("sparse_current_tokens"),
            state=data["state"],
            state_history=data.get("state_history"),
            state_history_mask=data.get("state_history_mask"),
            state_history_age=data.get("state_history_age"),
            tokenized_prompt=data.get("tokenized_prompt"),
            tokenized_prompt_mask=data.get("tokenized_prompt_mask"),
            pa_input_tokens=data.get("pa_input_tokens"),
            pa_target_tokens=data.get("pa_target_tokens"),
            pa_loss_mask=data.get("pa_loss_mask"),
            progress_target=data.get("progress_target"),
            stage_label=data.get("stage_label"),
            token_ar_mask=data.get("token_ar_mask"),
            token_loss_mask=data.get("token_loss_mask"),
            bot_token_mask=data.get("bot_token_mask"),
            dataset_root_index=data.get("dataset_root_index"),
            action_time_valid_mask=data.get("action_time_valid_mask"),
            action_time_dim_valid_mask=data.get("action_time_dim_valid_mask"),
            previous_plan_actions=data.get("previous_plan_actions"),
            previous_plan_mask=data.get("previous_plan_mask"),
            previous_plan_confidence=data.get("previous_plan_confidence"),
        )

    def to_dict(self) -> at.PyTree[ArrayT]:
        """Convert the Observation to a nested dict."""
        result = dataclasses.asdict(self)
        result["image"] = result.pop("images")
        result["image_mask"] = result.pop("image_masks")
        result["image_valid_mask"] = result.pop("image_valid_masks")
        result["image_prior"] = result.pop("image_priors")
        result["image_history"] = result.pop("image_history")
        result["image_history_mask"] = result.pop("image_history_masks")
        result["image_history_age"] = result.pop("image_history_ages")
        result["image_history_valid_mask"] = result.pop("image_history_valid_masks")
        return result


Actions = at.Float[ArrayT, "*b ah ad"]


def action_time_loss_mean(
    loss: at.ArrayLike,
    valid_mask: at.ArrayLike | None = None,
    *,
    sample_weights: at.ArrayLike | None = None,
) -> at.Float[at.Array, ""]:
    """Reduce ``[*batch, horizon]`` loss while excluding invalid target rows.

    ``sample_weights`` scales the numerator but deliberately not the validity
    denominator. This preserves the stochastic S2 estimator's existing
    ``2 * E[branch_weight * loss]`` semantics.
    """
    loss = jnp.asarray(loss)
    if loss.ndim < 1:
        raise ValueError(
            f"Action-time loss must have at least one dimension, got {loss.shape}."
        )
    weights = None
    if sample_weights is not None:
        weights = jnp.asarray(sample_weights, dtype=loss.dtype)
        if weights.shape != loss.shape[:-1]:
            raise ValueError(
                f"sample_weights must have shape {loss.shape[:-1]}, got {weights.shape}."
            )
        weights = weights[..., None]
    if valid_mask is None:
        return jnp.mean(loss if weights is None else loss * weights)
    valid = jnp.asarray(valid_mask, dtype=jnp.bool_)
    if valid.shape != loss.shape:
        raise ValueError(
            f"action_time_valid_mask must have shape {loss.shape}, got {valid.shape}."
        )
    valid_weight = valid.astype(loss.dtype)
    numerator = loss * valid_weight
    if weights is not None:
        numerator = numerator * weights
    denominator = jnp.maximum(jnp.sum(valid_weight), jnp.asarray(1.0, dtype=loss.dtype))
    return jnp.sum(numerator) / denominator


def action_time_loss_per_sample(
    loss: at.ArrayLike, valid_mask: at.ArrayLike | None = None
) -> at.Float[at.Array, "*b"]:
    """Reduce only the action-time axis, returning zero for all-invalid rows."""
    loss = jnp.asarray(loss)
    if loss.ndim < 1:
        raise ValueError(
            f"Action-time loss must have at least one dimension, got {loss.shape}."
        )
    if valid_mask is None:
        return jnp.mean(loss, axis=-1)
    valid = jnp.asarray(valid_mask, dtype=jnp.bool_)
    if valid.shape != loss.shape:
        raise ValueError(
            f"action_time_valid_mask must have shape {loss.shape}, got {valid.shape}."
        )
    valid_weight = valid.astype(loss.dtype)
    denominator = jnp.maximum(
        jnp.sum(valid_weight, axis=-1), jnp.asarray(1.0, dtype=loss.dtype)
    )
    return jnp.sum(loss * valid_weight, axis=-1) / denominator


@functools.partial(jax.jit, static_argnums=(1, 2))
def _resize_prior_with_pad(
    priors: at.Float[at.Array, "*b h w"],
    height: int,
    width: int,
    method: jax.image.ResizeMethod = jax.image.ResizeMethod.LINEAR,
) -> at.Float[at.Array, "*b {height} {width}"]:
    """Resize a soft spatial prior without distortion and pad missing regions with zeros."""
    has_batch_dim = priors.ndim == 3
    if not has_batch_dim:
        priors = priors[None]
    (cur_height, cur_width) = priors.shape[1:3]
    ratio = max(cur_width / width, cur_height / height)
    resized_height = int(cur_height / ratio)
    resized_width = int(cur_width / ratio)
    resized_priors = jax.image.resize(
        priors[..., None],
        (priors.shape[0], resized_height, resized_width, 1),
        method=method,
    )
    resized_priors = resized_priors.clip(0.0, 1.0)
    (pad_h0, remainder_h) = divmod(height - resized_height, 2)
    pad_h1 = pad_h0 + remainder_h
    (pad_w0, remainder_w) = divmod(width - resized_width, 2)
    pad_w1 = pad_w0 + remainder_w
    padded_priors = jnp.pad(
        resized_priors,
        ((0, 0), (pad_h0, pad_h1), (pad_w0, pad_w1), (0, 0)),
        constant_values=0.0,
    )[..., 0]
    if not has_batch_dim:
        padded_priors = padded_priors[0]
    return padded_priors


def preprocess_observation(
    rng: at.KeyArrayLike | None,
    observation: Observation,
    *,
    train: bool = False,
    image_keys: Sequence[str] = IMAGE_KEYS,
    image_resolution: tuple[int, int] = IMAGE_RESOLUTION,
    image_baked_padding_by_view: dict[str, tuple[int, int, int, int]] | None = None,
    skip_color_jitter: bool = False,
) -> Observation:
    """Preprocess the observations by performing image augmentations (if train=True), resizing (if necessary), and
    filling in a default image mask (if necessary).
    """
    if not set(image_keys).issubset(observation.images):
        raise ValueError(
            f"images dict missing keys: expected {image_keys}, got {list(observation.images)}"
        )
    if observation.image_history is not None and (
        not set(observation.image_history).issubset(observation.images)
    ):
        raise ValueError(
            f"image_history keys must be a subset of images keys, got history={list(observation.image_history)} images={list(observation.images)}"
        )
    batch_shape = observation.state.shape[:-1]

    def _resize_history_stack(
        history: jax.Array, valid_mask: jax.Array | None, *, height: int, width: int
    ) -> tuple[jax.Array, jax.Array]:
        if history.ndim < 5:
            raise ValueError(
                f"image_history must have shape [*b, t, h, w, c], got {history.shape}"
            )
        (batch_size, time_steps) = (history.shape[0], history.shape[1])
        flat_history = history.reshape((batch_size * time_steps, *history.shape[-3:]))
        if flat_history.shape[1:3] != (height, width):
            flat_history = image_tools.resize_with_pad(flat_history, height, width)
        if valid_mask is None:
            flat_valid_mask = image_tools.resize_with_pad_valid_mask(
                flat_history, height, width
            ).astype(jnp.float32)
        else:
            if valid_mask.ndim == history.ndim and valid_mask.shape[-1] == 1:
                valid_mask = valid_mask[..., 0]
            if valid_mask.ndim != history.ndim - 1:
                raise ValueError(
                    f"image_history_valid_mask must have shape [*b, t, h, w] or [*b, t, h, w, 1], got {valid_mask.shape}"
                )
            flat_valid_mask = valid_mask.reshape(
                (batch_size * time_steps, *valid_mask.shape[-2:])
            )
            if flat_valid_mask.shape[1:3] != (height, width):
                flat_valid_mask = _resize_prior_with_pad(flat_valid_mask, height, width)
        return (
            flat_history.reshape(
                (batch_size, time_steps, height, width, flat_history.shape[-1])
            ),
            flat_valid_mask.reshape((batch_size, time_steps, height, width)),
        )

    def _augment_history_stack(
        history: jax.Array,
        valid_mask: jax.Array,
        *,
        color_rngs: jax.Array,
        geom_rngs: jax.Array,
        geom_chain: augmax.Chain | None,
    ) -> tuple[jax.Array, jax.Array]:
        if geom_chain is not None:
            joint = jnp.concatenate([history, valid_mask[..., None]], axis=-1)
            joint = jax.vmap(
                lambda sample_rng, sample_joint: jax.vmap(
                    lambda frame_joint: geom_chain(sample_rng, frame_joint)
                )(sample_joint)
            )(geom_rngs, joint)
            history = joint[..., : history.shape[-1]]
            valid_mask = joint[..., history.shape[-1] : history.shape[-1] + 1][..., 0]
        color_chain = augmax.Chain(
            augmax.ColorJitter(brightness=0.3, contrast=0.4, saturation=0.5)
        )
        history = jax.vmap(
            lambda sample_rng, sample_history: jax.vmap(
                lambda frame: color_chain(sample_rng, frame)
            )(sample_history)
        )(color_rngs, history)
        return (history, valid_mask)

    out_images = {}
    out_valid_masks = {}
    out_priors = {} if observation.image_priors is not None else None
    out_image_history = {} if observation.image_history is not None else None
    out_image_history_valid_masks = (
        {} if observation.image_history is not None else None
    )
    out_image_history_masks = {} if observation.image_history is not None else None
    out_image_history_ages = {} if observation.image_history is not None else None
    for key in image_keys:
        image = observation.images[key]
        prior = None
        valid_mask = None
        history = (
            None
            if observation.image_history is None
            else observation.image_history.get(key)
        )
        history_valid_mask = None
        if (
            observation.image_valid_masks is not None
            and key in observation.image_valid_masks
        ):
            valid_mask = jnp.asarray(
                observation.image_valid_masks[key], dtype=jnp.float32
            )
            if valid_mask.ndim == image.ndim and valid_mask.shape[-1] == 1:
                valid_mask = valid_mask[..., 0]
            if valid_mask.ndim != image.ndim - 1:
                raise ValueError(
                    f"image_valid_mask[{key}] must have shape [*b, h, w] or [*b, h, w, 1], got {valid_mask.shape}"
                )
        if (
            observation.image_history_valid_masks is not None
            and key in observation.image_history_valid_masks
        ):
            history_valid_mask = jnp.asarray(
                observation.image_history_valid_masks[key], dtype=jnp.float32
            )
        if observation.image_priors is not None and key in observation.image_priors:
            prior = jnp.asarray(observation.image_priors[key], dtype=jnp.float32)
            if prior.ndim == image.ndim and prior.shape[-1] == 1:
                prior = prior[..., 0]
            if prior.ndim != image.ndim - 1:
                raise ValueError(
                    f"image_prior[{key}] must have shape [*b, h, w] or [*b, h, w, 1], got {prior.shape}"
                )
        baked_padding = resolve_image_baked_padding(image_baked_padding_by_view, key)
        if baked_padding is not None:
            baked_valid_mask = image_tools.fixed_padding_valid_mask(
                image, *baked_padding
            ).astype(jnp.float32)
            valid_mask = (
                baked_valid_mask
                if valid_mask is None
                else valid_mask * baked_valid_mask
            )
        if image.shape[1:3] != image_resolution:
            logger.info(
                f"Resizing image {key} from {image.shape[1:3]} to {image_resolution}"
            )
            if valid_mask is None:
                valid_mask = image_tools.resize_with_pad_valid_mask(
                    image, *image_resolution
                ).astype(jnp.float32)
            image = image_tools.resize_with_pad(image, *image_resolution)
        elif valid_mask is None:
            valid_mask = jnp.ones(image.shape[:-1], dtype=jnp.float32)
        if valid_mask is not None and valid_mask.shape[1:3] != image_resolution:
            logger.info(
                f"Resizing image valid mask {key} from {valid_mask.shape[1:3]} to {image_resolution}"
            )
            valid_mask = _resize_prior_with_pad(valid_mask, *image_resolution)
        if prior is not None and prior.shape[1:3] != image_resolution:
            logger.info(
                f"Resizing image prior {key} from {prior.shape[1:3]} to {image_resolution}"
            )
            prior = _resize_prior_with_pad(prior, *image_resolution)
        if history is not None:
            history = jnp.asarray(history)
            (history, history_valid_mask) = _resize_history_stack(
                history,
                history_valid_mask,
                height=image_resolution[0],
                width=image_resolution[1],
            )
            if baked_padding is not None:
                flat_history = history.reshape(
                    (history.shape[0] * history.shape[1], *history.shape[-3:])
                )
                baked_history_valid_mask = image_tools.fixed_padding_valid_mask(
                    flat_history, *baked_padding
                ).astype(jnp.float32)
                baked_history_valid_mask = baked_history_valid_mask.reshape(
                    history_valid_mask.shape
                )
                history_valid_mask = history_valid_mask * baked_history_valid_mask
        if prior is not None:
            prior = jnp.clip(prior, 0.0, 1.0)
        if train:
            image = image / 2.0 + 0.5
            if history is not None:
                history = history / 2.0 + 0.5
            geom_transforms = []
            if "wrist" not in key:
                (height, width) = image.shape[1:3]
                geom_transforms += [
                    augmax.RandomCrop(int(width * 0.95), int(height * 0.95)),
                    augmax.Resize(width, height),
                    augmax.Rotate((-5, 5)),
                ]
            geom_rngs = jax.random.split(rng, image.shape[0])
            geom_chain = augmax.Chain(*geom_transforms) if geom_transforms else None
            if geom_transforms:
                if prior is not None and valid_mask is not None:
                    joint = jnp.concatenate(
                        [image, prior[..., None], valid_mask[..., None]], axis=-1
                    )
                    joint = jax.vmap(geom_chain)(geom_rngs, joint)
                    image = joint[..., : image.shape[-1]]
                    prior = jnp.clip(
                        joint[..., image.shape[-1] : image.shape[-1] + 1][..., 0],
                        0.0,
                        1.0,
                    )
                    valid_mask = joint[..., image.shape[-1] + 1 : image.shape[-1] + 2][
                        ..., 0
                    ]
                elif valid_mask is not None:
                    joint = jnp.concatenate([image, valid_mask[..., None]], axis=-1)
                    joint = jax.vmap(geom_chain)(geom_rngs, joint)
                    image = joint[..., : image.shape[-1]]
                    valid_mask = joint[..., image.shape[-1] : image.shape[-1] + 1][
                        ..., 0
                    ]
                elif prior is not None:
                    joint = jnp.concatenate([image, prior[..., None]], axis=-1)
                    joint = jax.vmap(geom_chain)(geom_rngs, joint)
                    image = joint[..., : image.shape[-1]]
                    prior = jnp.clip(joint[..., image.shape[-1] :][..., 0], 0.0, 1.0)
                else:
                    image = jax.vmap(geom_chain)(geom_rngs, image)
            color_rngs = jax.random.split(jax.random.fold_in(rng, 1), image.shape[0])
            if not skip_color_jitter:
                image = jax.vmap(
                    augmax.Chain(
                        augmax.ColorJitter(brightness=0.3, contrast=0.4, saturation=0.5)
                    )
                )(color_rngs, image)
            elif history is not None:
                raise ValueError(
                    "Eval photometric v1 supports current images and proprio history, not RGB history"
                )
            if history is not None and history_valid_mask is not None:
                (history, history_valid_mask) = _augment_history_stack(
                    history,
                    history_valid_mask,
                    color_rngs=color_rngs,
                    geom_rngs=geom_rngs,
                    geom_chain=geom_chain,
                )
            image = image * 2.0 - 1.0
            if history is not None:
                history = history * 2.0 - 1.0
        out_images[key] = image
        out_valid_masks[key] = jnp.asarray(valid_mask >= 0.5, dtype=jnp.bool_)
        if out_priors is not None and prior is not None:
            out_priors[key] = prior
        if (
            out_image_history is not None
            and history is not None
            and (history_valid_mask is not None)
        ):
            out_image_history[key] = history
            out_image_history_valid_masks[key] = jnp.asarray(
                history_valid_mask >= 0.5, dtype=jnp.bool_
            )
            if (
                observation.image_history_masks is not None
                and key in observation.image_history_masks
            ):
                out_image_history_masks[key] = jnp.asarray(
                    observation.image_history_masks[key]
                )
            else:
                out_image_history_masks[key] = jnp.ones(
                    history.shape[:2], dtype=jnp.bool_
                )
            if (
                out_image_history_ages is not None
                and observation.image_history_ages is not None
                and (key in observation.image_history_ages)
            ):
                out_image_history_ages[key] = jnp.asarray(
                    observation.image_history_ages[key], dtype=jnp.float32
                )
    out_masks = {}
    for key in out_images:
        if key not in observation.image_masks:
            out_masks[key] = jnp.ones(batch_shape, dtype=jnp.bool)
        else:
            out_masks[key] = jnp.asarray(observation.image_masks[key])
    if train and any(
        (
            x is not None
            for x in (
                observation.sparse_history_tokens,
                observation.sparse_history_token_masks,
                observation.sparse_current_tokens,
            )
        )
    ):
        raise ValueError("Inference visual caches cannot be used during training")
    return Observation(
        images=out_images,
        image_masks=out_masks,
        image_valid_masks=out_valid_masks,
        image_priors=out_priors,
        image_history=out_image_history,
        image_history_masks=out_image_history_masks,
        image_history_ages=observation.image_history_ages
        if observation.sparse_history_tokens is not None
        else out_image_history_ages,
        image_history_valid_masks=out_image_history_valid_masks,
        sparse_history_tokens=observation.sparse_history_tokens,
        sparse_history_token_masks=observation.sparse_history_token_masks,
        sparse_current_tokens=observation.sparse_current_tokens,
        state=observation.state,
        state_history=observation.state_history,
        state_history_mask=observation.state_history_mask,
        state_history_age=observation.state_history_age,
        tokenized_prompt=observation.tokenized_prompt,
        tokenized_prompt_mask=observation.tokenized_prompt_mask,
        pa_input_tokens=observation.pa_input_tokens,
        pa_target_tokens=observation.pa_target_tokens,
        pa_loss_mask=observation.pa_loss_mask,
        progress_target=observation.progress_target,
        stage_label=observation.stage_label,
        previous_plan_actions=observation.previous_plan_actions,
        previous_plan_mask=observation.previous_plan_mask,
        previous_plan_confidence=observation.previous_plan_confidence,
        token_ar_mask=observation.token_ar_mask,
        token_loss_mask=observation.token_loss_mask,
        bot_token_mask=observation.bot_token_mask,
        dataset_root_index=observation.dataset_root_index,
        action_time_valid_mask=observation.action_time_valid_mask,
        action_time_dim_valid_mask=observation.action_time_dim_valid_mask,
    )


@dataclasses.dataclass(frozen=True)
class BaseModelConfig(abc.ABC):
    """Configuration shared by all models. Specific models should inherit from this class, and implement the `create`
    method to create the corresponding model.
    """

    action_dim: int
    action_horizon: int
    max_token_len: int
    image_baked_padding_by_view: dict[str, tuple[int, int, int, int]] | None = None

    @property
    @abc.abstractmethod
    def model_type(self) -> ModelType:
        """The model type."""

    @abc.abstractmethod
    def create(self, rng: at.KeyArrayLike) -> "BaseModel":
        """Create a new model, initializing parameters."""

    @abc.abstractmethod
    def inputs_spec(self, *, batch_size: int = 1) -> tuple[Observation, Actions]:
        """Returns the input specification for the model. Values are jax.ShapeDtypeStruct."""

    def fake_obs(self, batch_size: int = 1) -> Observation:
        (observation_spec, _) = self.inputs_spec(batch_size=batch_size)
        return jax.tree.map(lambda x: jnp.ones(x.shape, x.dtype), observation_spec)

    def fake_act(self, batch_size: int = 1) -> Actions:
        (_, action_spec) = self.inputs_spec(batch_size=batch_size)
        return jax.tree.map(lambda x: jnp.ones(x.shape, x.dtype), action_spec)


@dataclasses.dataclass
class BaseModel(nnx.Module, abc.ABC):
    """Base class for all model implementations. Specific models should inherit from this class. They should call
    super().__init__() to initialize the shared attributes (action_dim, action_horizon, and max_token_len).
    """

    action_dim: int
    action_horizon: int
    max_token_len: int

    @abc.abstractmethod
    def compute_loss(
        self,
        rng: at.KeyArrayLike,
        observation: Observation,
        actions: Actions,
        *,
        train: bool = False,
        action_dim_mask: at.ArrayLike | None = None,
    ) -> at.Float[at.Array, "*b ah"]: ...

    @abc.abstractmethod
    def sample_actions(
        self, rng: at.KeyArrayLike, observation: Observation, **kwargs
    ) -> Actions: ...


def restore_params(
    params_path: pathlib.Path | str,
    *,
    restore_type: type[np.ndarray] | type[jax.Array] = jax.Array,
    dtype: jnp.dtype | None = None,
    sharding: jax.sharding.Sharding | None = None,
) -> at.Params:
    """Restores unstructured params PyTree from a checkpoint.

    This works with checkpoints saved with `save_state` during openpi training (see `training/checkpoints.py`) as
    well as pre-trained checkpoints released for openpi.

    Args:
        params_path: The local path to the checkpoint directory.
        restore_type: The type to restore the params as. Can be set to `np.ndarray` to load the params as a numpy array.
        dtype: The dtype to restore all params as. If not provided, will use the original dtype from the checkpoint.
        sharding: The sharding to use for the params. If not provided, the params will be replicated across all devices.

    Returns:
        The restored params.
    """
    params_path = (
        pathlib.Path(params_path).resolve()
        if not str(params_path).startswith("gs://")
        else params_path
    )
    if restore_type is jax.Array and sharding is None:
        mesh = jax.sharding.Mesh(jax.devices(), ("x",))
        sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())
    with ocp.PyTreeCheckpointer() as ckptr:
        metadata = ckptr.metadata(params_path)
        item = {"params": metadata["params"]}
        params = ckptr.restore(
            params_path,
            ocp.args.PyTreeRestore(
                item=item,
                restore_args=jax.tree.map(
                    lambda _: ocp.ArrayRestoreArgs(
                        sharding=sharding, restore_type=restore_type, dtype=dtype
                    ),
                    item,
                ),
            ),
        )["params"]
    flat_params = traverse_util.flatten_dict(params)
    if all((kp[-1] == "value" for kp in flat_params)):
        flat_params = {kp[:-1]: v for (kp, v) in flat_params.items()}
    return traverse_util.unflatten_dict(flat_params)
