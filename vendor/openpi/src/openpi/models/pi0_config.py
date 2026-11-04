# Modified by AI Robot Association (AIRoA) in 2026 for AIRoA-0 (A-0) post-training.
# Derived from Physical Intelligence openpi (Apache-2.0), revision 215abfb.
import dataclasses
from typing import Literal
import flax.nnx as nnx
import jax
import jax.numpy as jnp
from openpi.models import model as _model
from openpi.models import gemma as _gemma
from openpi.shared import array_typing as at


@dataclasses.dataclass(frozen=True)
class Pi0Config(_model.BaseModelConfig):
    action_expert_variant: _gemma.Variant = "gemma_300m"
    base_view_names: tuple[str, ...] = ("base_0_rgb",)
    discrete_state_input: bool = True
    dtype: str = "bfloat16"
    eval_photometric_enabled: bool = False
    fine_progress_enabled: bool = False
    flash_buffer_size: int = 4
    flash_chunk_size: int = 5
    flash_enabled: bool = False
    flash_flow_substeps: tuple[tuple[float, ...], ...] | None = None
    flash_flow_times: tuple[float, ...] | None = None
    image_prior_floor: float = 0.25
    pa_aux_logit_chunk_size: int = 8
    pa_aux_loss_weight: float = 0.0
    pa_aux_max_token_len: int = 32
    paligemma_variant: _gemma.Variant = "gemma_2b"
    pi05: bool = True
    previous_plan_conditioning_enabled: bool = False
    siglip_variant: str = "So400m/14"
    stage_action_conditioning: bool = False
    stage_loss_weight: float = 0.1
    stage_num_classes: int = 0
    stage_pa_conditioning: bool = False
    stage_pa_token_ids: tuple[tuple[int, ...], ...] = ()
    temporal_hidden_dim: int | None = None
    use_temporal_proprio_memory: bool = False
    temporal_proprio_history_lags: tuple[int, ...] = ()
    temporal_proprio_memory_tokens: int = 4
    temporal_proprio_use_delta: bool = False
    temporal_proprio_tokens_in_prefix: bool = True
    temporal_proprio_prefix_token_scale: float = 0.1
    temporal_proprio_prefix_gate_warmup_steps: int = 0
    proprio_frequency_enabled: bool = False
    proprio_frequency_rates_hz: tuple[float, ...] = (5.0, 8.0, 10.0, 12.0, 15.0)
    proprio_frequency_interpolate: bool = True
    proprio_frequency_dropout: float = 0.25
    proprio_frequency_seed: int = 42
    tt_rtc_enabled: bool = False
    use_visual_budget_gate: bool = True
    visual_budget_gate_anchor_delta: float = 0.2
    visual_budget_gate_floor: float = 0.1
    visual_budget_gate_floor_end: float | None = None
    visual_budget_gate_hidden_dim: int = 256
    visual_budget_gate_ignore_priors: bool = True
    visual_budget_gate_inference_mode: Literal["off", "soft"] = "soft"
    visual_budget_gate_keep_ratio_base: float = 0.4
    visual_budget_gate_keep_ratio_wrist: float = 0.6
    visual_budget_gate_lambda_anchor_end: float = 0.0
    visual_budget_gate_lambda_anchor_start: float = 0.05
    visual_budget_gate_lambda_budget_base: float = 0.1
    visual_budget_gate_lambda_budget_wrist: float = 0.1
    visual_budget_gate_lambda_gated: float = 1.0
    visual_budget_gate_prior_bias_end: float = 0.0
    visual_budget_gate_prior_bias_start: float = 0.5
    visual_budget_gate_tau_end: float = 0.7
    visual_budget_gate_tau_start: float = 1.0
    visual_budget_mpa_enabled: bool = True
    visual_budget_mpa_log_eps: float = 1e-06
    visual_budget_mpa_logit_clip: float = 4.0
    visual_budget_mpa_tau: float | None = None
    visual_budget_objective_mode: Literal["visual_budget"] = "visual_budget"
    visual_budget_s2_alpha_warmup_steps: int | None = None
    visual_budget_s2_budget_enabled: bool = True
    white_balance_probability: float = 0.7
    white_balance_strength: float = 0.0
    wrist_view_names: tuple[str, ...] = ("left_wrist_0_rgb", "right_wrist_0_rgb")
    action_dim: int = 32
    action_horizon: int = 20
    max_token_len: int = 200

    def __post_init__(self):
        if (
            not self.pi05
            or self.visual_budget_objective_mode != "visual_budget"
            or (not self.use_visual_budget_gate)
            or (not self.visual_budget_mpa_enabled)
            or (not self.visual_budget_s2_budget_enabled)
        ):
            raise ValueError("AIRoA-0 requires its pretrained visual-evidence architecture")
        if (
            self.eval_photometric_enabled
            or self.fine_progress_enabled
            or self.tt_rtc_enabled
            or self.stage_action_conditioning
            or (not self.visual_budget_gate_ignore_priors)
        ):
            raise ValueError(
                "Unsupported experimental feature in the training-only runtime"
            )
        if self.use_temporal_proprio_memory:
            if not self.temporal_proprio_history_lags or any(
                type(lag) is not int or lag < 1
                for lag in self.temporal_proprio_history_lags
            ):
                raise ValueError("Proprio history requires positive past-only lags")
            if (
                not self.proprio_frequency_enabled
                or self.temporal_proprio_use_delta
                or not self.temporal_proprio_tokens_in_prefix
            ):
                raise ValueError(
                    "Proprio history requires the elapsed-time prefix-memory recipe"
                )
            if (
                self.temporal_hidden_dim is None
                or self.temporal_hidden_dim <= 0
                or self.temporal_hidden_dim % 2
            ):
                raise ValueError("Proprio hidden width must be a positive even number")
            if (
                self.temporal_proprio_memory_tokens <= 0
                or self.temporal_proprio_prefix_token_scale < 0
            ):
                raise ValueError("Invalid proprio memory size or scale")
            if self.temporal_proprio_prefix_gate_warmup_steps < 0:
                raise ValueError("Proprio gate warmup must be nonnegative")
            if not self.proprio_frequency_rates_hz or any(
                not 0 < x <= 30 for x in self.proprio_frequency_rates_hz
            ):
                raise ValueError("Proprio rates must be in (0, 30] Hz")
            if not 0 <= self.proprio_frequency_dropout < 1:
                raise ValueError("Proprio dropout must be in [0, 1)")
        elif self.temporal_proprio_history_lags or self.proprio_frequency_enabled:
            raise ValueError(
                "Disabled proprio history must have no lags or frequency sampling"
            )
        if self.previous_plan_conditioning_enabled:
            raise ValueError("Previous-action conditioning is not supported")
        if self.action_dim != 32 or self.max_token_len <= 0 or self.action_horizon <= 0:
            raise ValueError("Invalid model dimensions")
        if self.flash_enabled and self.action_horizon != 2 * self.flash_chunk_size:
            raise ValueError("NUF requires a two-chunk target horizon")
        if (
            self.stage_num_classes
            and len(self.stage_pa_token_ids) != self.stage_num_classes
        ):
            raise ValueError("Provide one canonical PA token block per stage")

    @property
    def image_view_names(self):
        return (*self.base_view_names, *self.wrist_view_names)

    @property
    def model_type(self):
        return _model.ModelType.PI05

    def create(self, rng):
        from openpi.models.pi0 import Pi0

        return Pi0(self, rngs=nnx.Rngs(rng))

    def inputs_spec(self, *, batch_size=1):
        with at.disable_typechecking():
            image = jax.ShapeDtypeStruct((batch_size, 224, 224, 3), jnp.float32)
            mask = jax.ShapeDtypeStruct((batch_size,), jnp.bool_)
            observation = _model.Observation(
                images={k: image for k in self.image_view_names},
                image_masks={k: mask for k in self.image_view_names},
                state=jax.ShapeDtypeStruct((batch_size, self.action_dim), jnp.float32),
                state_history=(
                    jax.ShapeDtypeStruct(
                        (
                            batch_size,
                            len(self.temporal_proprio_history_lags),
                            self.action_dim,
                        ),
                        jnp.float32,
                    )
                    if self.use_temporal_proprio_memory
                    else None
                ),
                state_history_mask=(
                    jax.ShapeDtypeStruct(
                        (batch_size, len(self.temporal_proprio_history_lags)), jnp.bool_
                    )
                    if self.use_temporal_proprio_memory
                    else None
                ),
                state_history_age=(
                    jax.ShapeDtypeStruct(
                        (batch_size, len(self.temporal_proprio_history_lags)),
                        jnp.float32,
                    )
                    if self.use_temporal_proprio_memory
                    else None
                ),
                tokenized_prompt=jax.ShapeDtypeStruct(
                    (batch_size, self.max_token_len), jnp.int32
                ),
                tokenized_prompt_mask=jax.ShapeDtypeStruct(
                    (batch_size, self.max_token_len), jnp.bool_
                ),
            )
            return (
                observation,
                jax.ShapeDtypeStruct(
                    (batch_size, self.action_horizon, self.action_dim), jnp.float32
                ),
            )

    def get_freeze_filter(self):
        return nnx.Nothing
