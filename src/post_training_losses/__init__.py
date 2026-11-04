# Copyright 2026 AI Robot Association (AIRoA)
# SPDX-License-Identifier: Apache-2.0
"""Model-independent post-training objectives."""

from .auxiliary import pa_attention_mask, pa_loss, pa_token_cross_entropy, stage_loss
from .visual_budget import (
    attention_log_prior,
    attention_strength,
    visual_budget_statistics,
    embedding_gates,
    linear_schedule,
    project_attention_probabilities,
)
from .flow import flow_matching_inputs, flow_matching_loss
from .nuf import nuf_loss, packed_causal_attention, training_inputs
from .objective import post_training_loss

__all__ = [
    "attention_log_prior", "attention_strength", "visual_budget_statistics", "embedding_gates",
    "flow_matching_inputs", "flow_matching_loss", "linear_schedule", "nuf_loss",
    "packed_causal_attention", "pa_attention_mask", "pa_loss", "pa_token_cross_entropy",
    "post_training_loss", "project_attention_probabilities", "stage_loss", "training_inputs",
]
