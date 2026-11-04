# Copyright 2026 AI Robot Association (AIRoA)
# SPDX-License-Identifier: Apache-2.0
"""Prepared LeRobot data adapters, feature mapping and validity masks."""

import dataclasses
import numpy as np
from openpi import transforms
from openpi.training import config as upstream


@dataclasses.dataclass(frozen=True)
class ResizeWithMask:
    def __call__(self, data):
        from openpi.shared import image_tools as geometry
        from openpi_client import image_tools as images

        result = dict(data)
        padding = {
            k: np.asarray(geometry.resize_with_pad_valid_mask(v, 224, 224), np.bool_)
            for k, v in data["image"].items()
        }
        if "image_valid_mask" in data:
            from PIL import Image

            for key, valid in data["image_valid_mask"].items():
                rgb = np.repeat(np.asarray(valid, np.uint8)[..., None], 3, axis=-1)
                resized = images.resize_with_pad(
                    rgb, 224, 224, method=Image.Resampling.NEAREST
                )
                padding[key] = padding[key] & resized[..., 0].astype(bool)
        result["image_valid_mask"] = padding
        result["image"] = {
            k: images.resize_with_pad(v, 224, 224) for k, v in data["image"].items()
        }
        return result


VIEWS = ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb")


def get_field(record, name):
    if name in record:
        return record[name]
    value = record
    for part in name.split("."):
        value = value[part]
    return value


def rgb_uint8(value):
    image = np.asarray(value)
    if image.ndim != 3:
        raise ValueError(f"Expected a single RGB frame, got {image.shape}")
    if image.shape[-1] != 3 and image.shape[0] == 3:
        image = np.moveaxis(image, 0, -1)
    if image.shape[-1] != 3:
        raise ValueError(f"Expected RGB HWC or CHW, got {image.shape}")
    if image.dtype == np.uint8:
        return image
    if np.issubdtype(image.dtype, np.floating) and np.isfinite(image).all():
        if image.min() >= 0 and image.max() <= 1:
            return np.rint(image * 255).astype(np.uint8)
    raise ValueError(
        "Images must be RGB uint8 [0,255] or floating point [0,1]; no guessed rescaling"
    )


def valid_mask(value, shape, name):
    value = np.asarray(value)
    if value.shape != shape or value.dtype != np.bool_:
        raise ValueError(f"{name} must be boolean with shape {shape}")
    return value


@dataclasses.dataclass(frozen=True)
class Inputs(transforms.DataTransformFn):
    state_key: str
    action_key: str
    state_dim: int
    action_dim: int
    cameras: tuple[tuple[str, str], ...]
    prompt_source: str
    instruction: str
    history_slots: int = 0
    history_valid_key: str = ""
    history_time_key: str = "timestamp"

    def __call__(self, record):
        state = np.asarray(get_field(record, self.state_key), dtype=np.float32)
        if state.shape != (self.state_dim,) or not np.isfinite(state).all():
            raise ValueError(
                f"Invalid state: expected finite ({self.state_dim},), got {state.shape}"
            )
        images = {name: rgb_uint8(get_field(record, key)) for name, key in self.cameras}
        present = set(images)
        reference = next(iter(images.values()))
        for name in VIEWS:
            if name not in images:
                images[name] = np.zeros_like(reference)
        prompt = (
            record.get("prompt") if self.prompt_source == "task" else self.instruction
        )
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError(
                "Missing task instruction; check task_index/tasks metadata or use a fixed prompt"
            )
        result = {
            "image": images,
            "image_mask": {v: np.bool_(v in present) for v in VIEWS},
            "state": state,
            "prompt": prompt,
        }
        if "image_mask" in record:
            for name in present:
                if name in record["image_mask"]:
                    result["image_mask"][name] = valid_mask(
                        record["image_mask"][name], (), "image_mask"
                    )
        if "image_valid_mask" in record:
            result["image_valid_mask"] = {
                name: valid_mask(
                    record["image_valid_mask"][name],
                    images[name].shape[:2],
                    "image_valid_mask",
                )
                if name in record["image_valid_mask"]
                else np.ones(images[name].shape[:2], bool)
                for name in VIEWS
            }
        if self.history_slots:
            shape = (self.history_slots, self.state_dim)
            history = np.asarray(record["state_history"], np.float32)
            mask = valid_mask(
                record["state_history_mask"],
                (self.history_slots,),
                "state_history_mask",
            )
            ages = np.asarray(record["state_history_age"], np.float32)
            if history.shape != shape or ages.shape != (self.history_slots,):
                raise ValueError(
                    "History shape differs from configured slots/state dimensions"
                )
            if (
                not np.isfinite(history[mask]).all()
                or not np.isfinite(ages).all()
                or np.any(ages < 0)
                or np.any(ages[mask] <= 0)
            ):
                raise ValueError(
                    "History must contain finite strictly past measured states and ages"
                )
            if np.any(np.diff(ages[mask]) <= 0):
                raise ValueError(
                    "History must be ordered newest first without repeated measurements"
                )
            result.update(
                state_history=np.where(mask[:, None], history, 0.0),
                state_history_mask=mask,
                state_history_age=np.where(mask, ages, 0.0),
            )
        actions = get_field(record, self.action_key)
        actions = np.asarray(actions, dtype=np.float32)
        if actions.ndim != 2 or actions.shape[-1] != self.action_dim:
            raise ValueError(
                f"Invalid actions: expected [H,{self.action_dim}], got {actions.shape}"
            )
        horizon = actions.shape[0]
        temporal = (
            valid_mask(
                record["action_time_valid_mask"], (horizon,), "action_time_valid_mask"
            )
            if "action_time_valid_mask" in record
            else np.ones(horizon, bool)
        )
        dimensions = (
            valid_mask(
                record["action_time_dim_valid_mask"],
                actions.shape,
                "action_time_dim_valid_mask",
            )
            if "action_time_dim_valid_mask" in record
            else np.ones(actions.shape, bool)
        )
        for name in {self.action_key + "_is_pad", "action_is_pad"}:
            try:
                padded = get_field(record, name)
            except KeyError:
                continue
            temporal = temporal & ~valid_mask(padded, (horizon,), name)
        temporal = temporal & np.any(dimensions, axis=-1)
        dimensions = dimensions & temporal[..., None]
        if not np.isfinite(actions[dimensions]).all():
            raise ValueError("Eligible action targets must be finite")
        result["actions"] = np.where(dimensions, actions, 0.0)
        result["action_time_valid_mask"] = temporal
        result["action_time_dim_valid_mask"] = dimensions
        return result


def fit_normalization(batches, *, state_dim, action_dim):
    """Fit prepared targets per physical dimension, excluding invalid entries."""
    from openpi.shared import normalize

    state = normalize.RunningStats()
    actions = [normalize.RunningStats() for _ in range(action_dim)]
    counts = np.zeros(action_dim, np.int64)
    seen = eligible_anchors = 0
    for batch in batches:
        values = np.asarray(batch["actions"])
        states = np.asarray(batch["state"])
        if (
            values.ndim != 3
            or values.shape[-1] != action_dim
            or states.shape != (values.shape[0], state_dim)
        ):
            raise ValueError(
                "Normalization batch differs from declared physical dimensions"
            )
        temporal = valid_mask(
            batch["action_time_valid_mask"], values.shape[:2], "action_time_valid_mask"
        )
        dimensions = valid_mask(
            batch["action_time_dim_valid_mask"],
            values.shape,
            "action_time_dim_valid_mask",
        )
        eligible = dimensions & temporal[..., None]
        anchors = np.any(eligible, axis=(1, 2))
        if (
            not np.isfinite(states[anchors]).all()
            or not np.isfinite(values[eligible]).all()
        ):
            raise ValueError("Eligible normalization values must be finite")
        if np.any(anchors):
            state.update(states[anchors])
        for dim, stats in enumerate(actions):
            selected = values[..., dim][eligible[..., dim]]
            if selected.size:
                stats.update(selected.reshape(-1, 1))
                counts[dim] += selected.size
        seen += len(states)
        eligible_anchors += int(anchors.sum())
    if eligible_anchors < 2 or np.any(counts < 2):
        raise ValueError(
            "Normalization needs at least two eligible states and targets per physical dimension"
        )
    per_dimension = [stats.get_statistics() for stats in actions]
    action_stats = normalize.NormStats(
        **{
            name: np.concatenate([getattr(stats, name) for stats in per_dimension])
            for name in ("mean", "std", "q01", "q99")
        }
    )
    return {"state": state.get_statistics(), "actions": action_stats}, {
        "sampled_anchors": seen,
        "eligible_state_anchors": eligible_anchors,
        "eligible_action_values_per_dimension": counts.tolist(),
    }


@dataclasses.dataclass(frozen=True)
class RuntimeDataConfig(upstream.DataConfig):
    runtime_contract: dict | None = None


@dataclasses.dataclass(frozen=True)
class LeRobotData(upstream.DataConfigFactory):
    adapter: Inputs | None = None
    label_transform: object | None = None
    runtime_contract: dict | None = None

    def create(self, assets_dirs, model_config):
        if self.adapter is None:
            raise ValueError("A dataset adapter is required")
        model_transforms = upstream.ModelTransformFactory()(model_config)
        if model_config.stage_pa_conditioning:
            from post_training.language import StagePrefix, tokenizer_for

            model_transforms = transforms.Group(
                inputs=[
                    ResizeWithMask(),
                    StagePrefix(tokenizer_for(model_config.max_token_len)),
                    transforms.PadStatesAndActions(model_config.action_dim),
                ],
                outputs=[],
            )
        if (
            model_config.use_visual_budget_gate
            and not model_config.stage_pa_conditioning
        ):
            model_transforms = transforms.Group(
                inputs=[
                    ResizeWithMask() if isinstance(t, transforms.ResizeImages) else t
                    for t in model_transforms.inputs
                ],
                outputs=[],
            )
        if self.label_transform is not None:
            model_transforms = transforms.Group(
                inputs=[self.label_transform, *model_transforms.inputs],
                outputs=[],
            )
        base = self.create_base_config(assets_dirs, model_config)
        base = RuntimeDataConfig(
            **{f.name: getattr(base, f.name) for f in dataclasses.fields(base)},
            runtime_contract=self.runtime_contract,
        )
        return dataclasses.replace(
            base,
            action_sequence_keys=(self.adapter.action_key,),
            prompt_from_task=self.adapter.prompt_source == "task",
            data_transforms=transforms.Group(inputs=[self.adapter], outputs=[]),
            model_transforms=model_transforms,
        )
