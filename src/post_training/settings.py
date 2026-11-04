# Copyright 2026 AI Robot Association (AIRoA)
# SPDX-License-Identifier: Apache-2.0
"""User-facing configuration, independent of the accelerator runtime."""

from dataclasses import dataclass
from pathlib import Path
import hashlib
import json
import re
import tomllib


@dataclass(frozen=True)
class Settings:
    path: Path
    raw: dict

    @classmethod
    def read(cls, filename):
        path = Path(filename).expanduser().resolve()
        with path.open("rb") as stream:
            raw = tomllib.load(stream)
        defaults = {
            "model": {
                "action_horizon": 20,
                "max_token_len": 200,
                "nuf_chunk_size": 0,
                "nuf_buffer_size": 0,
                "nuf_flow_times": None,
            },
            "training": {
                "batch_size": 64,
                "steps": 20000,
                "learning_rate": 5e-6,
                "warmup_steps": 500,
                "save_interval": 5000,
                "log_interval": 100,
                "num_workers": 16,
                "fsdp_devices": 1,
                "seed": 42,
                "ema": True,
                "ema_decay": 0.99,
                "wandb": True,
                "wandb_project": "airoa-0-post-training",
            },
            "visual_budget": {
                "enabled": True,
                "budget": 0.9,
                "tau": 0.805,
                "floor": 0.1,
                "floor_end": 0.02,
                "schedule_steps": 20000,
                "attention_warmup_steps": 500,
                "budget_weight": 0.1,
                "anchor_weight": 0.0175,
                "anchor_delta": 0.2,
            },
            "proprio_history": {
                "slots": 10,
                "rates_hz": [5.0, 8.0, 10.0, 12.0, 15.0],
                "dropout": 0.25,
                "seed": 42,
                "memory_tokens": 4,
                "hidden_dim": 256,
                "prefix_scale": 0.1,
            },
            "auxiliary": {"weight": 0.01, "max_tokens": 32},
            "stage": {"enabled": True, "weight": 0.1, "max_tokens": 48},
            "labels": {"task_instruction": "", "stages": [], "pa_to_stage": {}},
        }
        for section, values in defaults.items():
            raw.setdefault(section, {})
            for key, value in values.items():
                raw[section].setdefault(key, value)
        raw["data"].setdefault("history_valid_key", "")
        raw["data"].setdefault("history_time_key", "timestamp")
        allowed = {
            "name",
            "init_params",
            "assets_dir",
            "checkpoint_dir",
            "data",
            "model",
            "training",
            "visual_budget",
            "auxiliary",
            "stage",
            "labels",
            "proprio_history",
        }
        if set(raw) - allowed:
            raise ValueError(f"Unknown top-level settings: {set(raw) - allowed}")
        if not re.fullmatch(r"[A-Za-z0-9_-]+", raw["name"]):
            raise ValueError(
                "name must contain only letters, numbers, underscores or hyphens"
            )
        sections = {
            "data": {
                "repo_id",
                "state_key",
                "action_key",
                "state_dim",
                "action_dim",
                "prompt_source",
                "instruction",
                "state_representation",
                "action_representation",
                "cameras",
                "history_valid_key",
                "history_time_key",
            },
            "model": {
                "action_horizon",
                "max_token_len",
                "nuf_chunk_size",
                "nuf_buffer_size",
                "nuf_flow_times",
            },
            "training": {
                "batch_size",
                "steps",
                "learning_rate",
                "warmup_steps",
                "save_interval",
                "log_interval",
                "num_workers",
                "fsdp_devices",
                "seed",
                "ema",
                "ema_decay",
                "wandb",
                "wandb_project",
            },
            "proprio_history": {
                "slots",
                "rates_hz",
                "dropout",
                "seed",
                "memory_tokens",
                "hidden_dim",
                "prefix_scale",
            },
        }
        sections.update(
            visual_budget={
                "enabled",
                "budget",
                "tau",
                "floor",
                "attention_warmup_steps",
                "budget_weight",
                "floor_end",
                "schedule_steps",
                "anchor_weight",
                "anchor_delta",
            },
            auxiliary={"weight", "max_tokens"},
            stage={"enabled", "weight", "max_tokens"},
            labels={"task_instruction", "stages", "pa_to_stage"},
        )
        for section, keys in sections.items():
            if set(raw[section]) - keys:
                raise ValueError(
                    f"Unknown {section} settings: {set(raw[section]) - keys}"
                )
        data, model, training = (raw[k] for k in ("data", "model", "training"))
        history = raw["proprio_history"]
        if type(history["slots"]) is not int or history["slots"] < 0:
            raise ValueError("proprio_history.slots must be a nonnegative integer")
        if (
            any(
                type(history[k]) is not int or history[k] < 1
                for k in ("memory_tokens", "hidden_dim")
            )
            or history["hidden_dim"] % 2
        ):
            raise ValueError(
                "History needs positive memory_tokens and a positive even hidden_dim"
            )
        if type(history["seed"]) is not int:
            raise ValueError("proprio_history.seed must be an integer")
        import math

        rates = history["rates_hz"]
        if (
            not isinstance(rates, list)
            or not rates
            or any(
                type(rate) not in (int, float)
                or not math.isfinite(rate)
                or not 0 < rate <= 30
                for rate in rates
            )
        ):
            raise ValueError(
                "History rates_hz must contain finite frequencies in (0, 30]"
            )
        if (
            type(history["dropout"]) not in (int, float)
            or not math.isfinite(history["dropout"])
            or not 0 <= history["dropout"] < 1
        ):
            raise ValueError("History dropout must be finite and in [0, 1)")
        if (
            type(history["prefix_scale"]) not in (int, float)
            or not math.isfinite(history["prefix_scale"])
            or history["prefix_scale"] < 0
        ):
            raise ValueError("History prefix_scale must be finite and nonnegative")
        if (
            not isinstance(data["history_valid_key"], str)
            or not isinstance(data["history_time_key"], str)
            or not data["history_time_key"].strip()
        ):
            raise ValueError(
                "History feature keys must be strings and history_time_key must be nonempty"
            )
        for key in ("state_dim", "action_dim"):
            if type(data[key]) is not int or not 1 <= data[key] <= 32:
                raise ValueError(f"{key} must be an integer in [1, 32]")
        if data["prompt_source"] not in ("task", "fixed"):
            raise ValueError("prompt_source must be task or fixed")
        if data["prompt_source"] == "fixed" and not data["instruction"].strip():
            raise ValueError("A non-empty instruction is required for a fixed prompt")
        views = {"base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb"}
        if not data["cameras"] or set(data["cameras"]) - views:
            raise ValueError(
                f"Configure at least one camera, with model names in {views}"
            )
        if len(set(data["cameras"].values())) != len(data["cameras"]):
            raise ValueError(
                "Do not silently duplicate a source camera into multiple views"
            )
        for key in ("action_horizon", "max_token_len"):
            if type(model[key]) is not int or model[key] < 1:
                raise ValueError(f"{key} must be a positive integer")
        for key in (
            "batch_size",
            "steps",
            "save_interval",
            "log_interval",
            "fsdp_devices",
        ):
            if type(training[key]) is not int or training[key] < 1:
                raise ValueError(f"{key} must be a positive integer")
        if training["warmup_steps"] < 0 or training["num_workers"] < 0:
            raise ValueError("warmup_steps and num_workers must be nonnegative")
        import math

        if (
            not math.isfinite(training["learning_rate"])
            or training["learning_rate"] <= 0
        ):
            raise ValueError("learning_rate must be finite and positive")
        if not 0 <= training["ema_decay"] < 1:
            raise ValueError("ema_decay must be in [0, 1)")
        if any(type(training[k]) is not bool for k in ("ema", "wandb")):
            raise ValueError("ema and wandb must be TOML booleans")
        visual_budget, aux, stage, labels = (
            raw[k] for k in ("visual_budget", "auxiliary", "stage", "labels")
        )
        if type(visual_budget["enabled"]) is not bool or type(stage["enabled"]) is not bool:
            raise ValueError("visual_budget.enabled and stage.enabled must be booleans")
        for x in (
            visual_budget["budget"],
            visual_budget["tau"],
            visual_budget["floor"],
            visual_budget["budget_weight"],
            visual_budget["anchor_weight"],
            visual_budget["anchor_delta"],
            aux["weight"],
            stage["weight"],
        ):
            if not math.isfinite(x):
                raise ValueError("Objective parameters must be finite")
        if (
            not 0 < visual_budget["budget"] < 1
            or not 0 <= visual_budget["floor"] < 1
            or visual_budget["tau"] <= 0
            or visual_budget["budget_weight"] < 0
            or visual_budget["anchor_weight"] < 0
            or visual_budget["anchor_delta"] < 0
        ):
            raise ValueError("Invalid visual-budget ratio, floor, temperature or weight")
        if (
            not 0 <= visual_budget["floor_end"] < 1
            or visual_budget["schedule_steps"] < 1
            or visual_budget["attention_warmup_steps"] < 1
            or aux["max_tokens"] < 2
            or stage["max_tokens"] < 2
            or aux["weight"] < 0
            or stage["weight"] <= 0
        ):
            raise ValueError("Invalid auxiliary/stage dimensions or loss settings")
        if aux["weight"] or stage["enabled"] or labels["task_instruction"]:
            if data["prompt_source"] != "task":
                raise ValueError("Auxiliary/stage require per-frame PA task labels")
            if not labels["task_instruction"].strip() or not labels["pa_to_stage"]:
                raise ValueError(
                    "Provide task_instruction and reviewed pa_to_stage labels"
                )
            if labels["task_instruction"] in labels["pa_to_stage"]:
                raise ValueError("Full task instruction must differ from PA labels")
            count = len(labels["stages"])
            if stage["enabled"] and count < 2:
                raise ValueError("Stage prediction needs at least two stages")
            if any(
                not isinstance(k, str)
                or not k.strip()
                or type(v) is not int
                or v < 0
                or (stage["enabled"] and v >= count)
                for k, v in labels["pa_to_stage"].items()
            ):
                raise ValueError(
                    "PA mapping needs nonempty strings and valid zero-based stage IDs"
                )
            if stage["enabled"] and set(labels["pa_to_stage"].values()) != set(
                range(count)
            ):
                raise ValueError("Every stage must have at least one PA")
            if stage["enabled"] and (
                any(not isinstance(t, str) or not t.strip() for t in labels["stages"])
                or len(set(labels["stages"])) != count
            ):
                raise ValueError("Stage texts must be nonempty and unique")
        if visual_budget["enabled"] is not True:
            raise ValueError(
                "visual_budget.enabled must be true"
            )
        c, n = model["nuf_chunk_size"], model["nuf_buffer_size"]
        if type(c) is not int or type(n) is not int:
            raise ValueError("NUF chunk and buffer sizes must be integers")
        if (c, n) != (0, 0) and not (
            c > 0 and n == 2 and model["action_horizon"] == c * n
        ):
            raise ValueError(
                "Disable NUF with sizes 0/0, or use positive chunk size, buffer size 2 and horizon=chunk*buffer"
            )
        levels = model["nuf_flow_times"]
        if levels is not None:
            if not c or not isinstance(levels, list) or len(levels) != n:
                raise ValueError(
                    "NUF flow times need one level per enabled buffer slot"
                )
            if any(
                type(v) not in (int, float) or not math.isfinite(v) or v <= 0
                for v in levels
            ):
                raise ValueError("NUF flow times must be positive finite numbers")
            import struct

            rounded = [struct.unpack("<f", struct.pack("<f", v))[0] for v in levels]
            if levels[-1] != 1.0 or any(b <= a for a, b in zip(rounded, rounded[1:])):
                raise ValueError("NUF flow times must increase in float32 and end at 1")
        return cls(path, raw)

    def resolve(self, key):
        path = Path(self.raw[key]).expanduser()
        return (
            (self.path.parent / path).resolve()
            if not path.is_absolute()
            else path.resolve()
        )

    def initialization_params(self):
        path = self.resolve("init_params")
        if (path / "_METADATA").is_file() or path.name == "params":
            return path
        return path / "params"

    def require_data_contract(self):
        for key in ("state_representation", "action_representation"):
            value = self.raw["data"].get(key, "").strip()
            if not value or value.startswith("REPLACE:"):
                raise ValueError(
                    f"Document data.{key} before training; no coordinate conversion is implicit"
                )
        if self.raw["data"]["repo_id"] == "fake":
            raise ValueError(
                "Synthetic datasets are not supported by this training entrypoint"
            )

    @property
    def data_signature(self):
        contract = {
            "data": self.raw["data"],
            "model": self.raw["model"],
            "labels": self.raw["labels"],
            "auxiliary": self.raw["auxiliary"],
            "stage": self.raw["stage"],
            "proprio_history": self.raw["proprio_history"],
        }
        return hashlib.sha256(json.dumps(contract, sort_keys=True).encode()).hexdigest()

    @property
    def stats_dir(self):
        return self.resolve("assets_dir") / self.raw["name"] / "dataset"
