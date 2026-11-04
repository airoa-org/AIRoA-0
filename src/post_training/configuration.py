# Copyright 2026 AI Robot Association (AIRoA)
# SPDX-License-Identifier: Apache-2.0
"""Train AIRoA-0 with visual-budget, PA and stage supervision from released parameters."""

import dataclasses
import logging
import numpy as np
import optax
from flax.traverse_util import flatten_dict, unflatten_dict
from openpi.models import model as model_lib
from post_training.language import Labels, stage_tokens, tokenizer_for
from openpi.models.pi0_config import Pi0Config
from openpi.training import config as upstream, weight_loaders
from post_training.data import Inputs, LeRobotData


@dataclasses.dataclass(frozen=True)
class WarmupConstant:
    peak_lr: float = 5e-6
    warmup_steps: int = 500

    def create(self):
        if self.warmup_steps == 0:
            return optax.constant_schedule(self.peak_lr)
        return optax.join_schedules(
            [
                optax.linear_schedule(
                    self.peak_lr / (self.warmup_steps + 1),
                    self.peak_lr,
                    self.warmup_steps,
                ),
                optax.constant_schedule(self.peak_lr),
            ],
            [self.warmup_steps],
        )


@dataclasses.dataclass(frozen=True)
class CoreWeights(weight_loaders.CheckpointWeightLoader):
    """Require core/visual-gate arrays; explicitly initialize task/history additions."""

    allow_new_history: bool = False

    def load(self, params):
        path = self.params_path
        from pathlib import Path

        local = Path(path).expanduser().resolve()
        if (local / "params" / "_METADATA").is_file():
            local = local / "params"
        if not (local / "_METADATA").is_file():
            raise FileNotFoundError(
                "Initialization must be an AIRoA-0 release root or its complete params directory"
            )
        loaded = model_lib.restore_params(local, restore_type=np.ndarray)
        expected, actual = flatten_dict(params), flatten_dict(loaded)
        history_roots = {
            "temporal_prompt_proj",
            "proprio_memory_in_proj",
            "proprio_memory_context_proj",
            "proprio_memory_compressor",
            "temporal_proprio_prefix_gate_raw",
        }
        history_keys = {key for key in expected if key[0] in history_roots}
        missing_history = history_keys - actual.keys()
        actual_history = {key for key in actual if key[0] in history_roots}
        if missing_history and actual_history:
            raise ValueError("Checkpoint contains an incomplete proprio history branch")
        if missing_history and not self.allow_new_history:
            raise ValueError(
                "New proprio history parameters require explicit initialization"
            )
        gate_keys = {key for key in actual if key[0] == "visual_budget_gate_heads"}
        if len(gate_keys) != 8:
            raise ValueError(
                "AIRoA-0 initialization requires all eight pretrained visual-gate arrays"
            )
        output = {}
        for key, ref in expected.items():
            if key not in actual:
                if key[0] == "stage_classifier":
                    output[key] = ref  # Newly initialized task classifier only.
                    continue
                if key in missing_history:
                    output[key] = ref
                    continue
                raise ValueError(
                    f"Checkpoint is missing required core/visual-gate parameter: {key}"
                )
            value = actual[key]
            if value.shape != ref.shape or not np.isfinite(value).all():
                raise ValueError(f"Invalid parameter: {key}")
            output[key] = value.astype(ref.dtype)
        logging.info(
            "Loaded %d arrays; initialized %d new task/history arrays",
            sum(k in actual for k in expected),
            sum(k not in actual for k in expected),
        )
        return unflatten_dict(output)


def build(settings, run_name, *, resume=False):
    r = settings.raw
    d, t, m = (r[k] for k in ("data", "training", "model"))
    visual_budget, aux, stage, labels = (r[k] for k in ("visual_budget", "auxiliary", "stage", "labels"))
    history = r["proprio_history"]
    label_transform = (
        Labels(
            labels["task_instruction"],
            tuple(labels["pa_to_stage"].items()),
            bool(aux["weight"]),
            stage["enabled"],
            aux["max_tokens"],
        )
        if labels["task_instruction"]
        else None
    )
    model_kwargs = dict(
        pi05=True,
        action_horizon=m["action_horizon"],
        max_token_len=m["max_token_len"],
        use_visual_budget_gate=True,
        visual_budget_gate_ignore_priors=True,
        visual_budget_mpa_enabled=True,
        visual_budget_s2_budget_enabled=True,
        visual_budget_objective_mode="visual_budget",
        visual_budget_gate_tau_start=visual_budget["tau"],
        visual_budget_gate_tau_end=visual_budget["tau"],
        visual_budget_mpa_tau=visual_budget["tau"],
        visual_budget_gate_floor=visual_budget["floor"],
        visual_budget_gate_floor_end=visual_budget["floor_end"],
        visual_budget_s2_alpha_warmup_steps=visual_budget["attention_warmup_steps"],
        visual_budget_gate_keep_ratio_base=visual_budget["budget"],
        visual_budget_gate_keep_ratio_wrist=visual_budget["budget"],
        visual_budget_gate_lambda_budget_base=visual_budget["budget_weight"],
        visual_budget_gate_lambda_budget_wrist=visual_budget["budget_weight"],
        visual_budget_gate_lambda_anchor_start=visual_budget["anchor_weight"],
        visual_budget_gate_lambda_anchor_end=visual_budget["anchor_weight"],
        visual_budget_gate_anchor_delta=visual_budget["anchor_delta"],
        pa_aux_loss_weight=aux["weight"],
        pa_aux_max_token_len=aux["max_tokens"],
        stage_num_classes=len(labels["stages"]) if stage["enabled"] else 0,
        stage_loss_weight=stage["weight"],
        stage_pa_conditioning=stage["enabled"],
        stage_pa_token_ids=stage_tokens(
            tokenizer_for(m["max_token_len"]), labels["stages"], stage["max_tokens"]
        )
        if stage["enabled"]
        else (),
    )
    if m["nuf_chunk_size"]:
        model_kwargs.update(
            flash_enabled=True,
            flash_chunk_size=m["nuf_chunk_size"],
            flash_buffer_size=m["nuf_buffer_size"],
            flash_flow_times=tuple(m["nuf_flow_times"])
            if m["nuf_flow_times"] is not None
            else None,
        )
    if history["slots"]:
        model_kwargs.update(
            use_temporal_proprio_memory=True,
            temporal_proprio_history_lags=tuple(range(1, history["slots"] + 1)),
            temporal_proprio_memory_tokens=history["memory_tokens"],
            temporal_hidden_dim=history["hidden_dim"],
            temporal_proprio_use_delta=False,
            temporal_proprio_tokens_in_prefix=True,
            temporal_proprio_prefix_token_scale=history["prefix_scale"],
            temporal_proprio_prefix_gate_warmup_steps=0,
            proprio_frequency_enabled=True,
            proprio_frequency_rates_hz=tuple(history["rates_hz"]),
            proprio_frequency_dropout=history["dropout"],
            proprio_frequency_seed=history["seed"],
            proprio_frequency_interpolate=False,
        )
    model = Pi0Config(**model_kwargs)
    adapter = Inputs(
        d["state_key"],
        d["action_key"],
        d["state_dim"],
        d["action_dim"],
        tuple(d["cameras"].items()),
        d["prompt_source"],
        d["instruction"],
        history_slots=history["slots"],
        history_valid_key=d["history_valid_key"],
        history_time_key=d["history_time_key"],
    )
    data = LeRobotData(
        repo_id=d["repo_id"],
        assets=upstream.AssetsConfig(asset_id="dataset"),
        adapter=adapter,
        label_transform=label_transform,
        runtime_contract={
            **{
                k: r[k]
                for k in (
                    "model",
                    "visual_budget",
                    "auxiliary",
                    "stage",
                    "labels",
                    "proprio_history",
                )
            },
            "data": {k: v for k, v in d.items() if k != "repo_id"},
        },
    )
    return upstream.TrainConfig(
        name=r["name"],
        exp_name=run_name,
        project_name=t["wandb_project"],
        model=model,
        weight_loader=CoreWeights(
            str(settings.initialization_params()),
            allow_new_history=bool(history["slots"]),
        ),
        data=data,
        assets_base_dir=str(settings.resolve("assets_dir")),
        checkpoint_base_dir=str(settings.resolve("checkpoint_dir")),
        lr_schedule=WarmupConstant(t["learning_rate"], t["warmup_steps"]),
        ema_decay=t["ema_decay"] if t["ema"] else None,
        batch_size=t["batch_size"],
        num_workers=t["num_workers"],
        num_train_steps=t["steps"],
        log_interval=t["log_interval"],
        save_interval=t["save_interval"],
        keep_period=t["save_interval"],
        seed=t["seed"],
        fsdp_devices=t["fsdp_devices"],
        wandb_enabled=t["wandb"],
        resume=resume,
        overwrite=False,
        policy_metadata={
            "s2_schedule_steps": visual_budget["schedule_steps"],
            "s2_enabled": visual_budget["enabled"],
            "stage_texts": labels["stages"] if stage["enabled"] else [],
            "action_dim": d["action_dim"],
            "state_dim": d["state_dim"],
            "state_representation": d["state_representation"],
            "action_representation": d["action_representation"],
        },
    )
