# Post-training objectives

The training entry point combines an action objective, PA text supervision,
stage classification and visual-budget regularization. Helpers in
`post_training_losses` operate on JAX arrays with explicit targets and masks.

## Action flow matching

The time convention is clean at zero and Gaussian noise at one:

\[
x_t=(1-t)a+t\epsilon,\qquad u_t=\epsilon-a.
\]

`flow_matching_loss` averages squared velocity error over valid physical action
dimensions, then over valid action times across the batch. Dimension masks are
padded with false validity to the model width. Finite, fully masked inputs
produce zero. Sample weights scale the numerator while the denominator counts
valid action times. Supply finite targets and model outputs, including masked
positions, because the reduction uses multiplication by validity masks.

## NUF packed action objective

For chunk length `C` and buffer size `N`, provide actions of shape `[B, N*C, D]`.
`training_inputs` constructs independent cold-start buffers containing the
first one, two, through `N` chunks, right-aligned in each buffer. It returns noisy
actions, velocity targets, flow times, time validity and optional dimension
validity. The packed action length is `N*N*C`.

The supplied random key is split into time and noise keys. Time draws use
Beta(1.5, 1), and Gaussian noise covers the complete packed shape.

- `levels=None` selects the uniform stratified clock.
- Ascending pre-step levels ending at one select the nonuniform clock.
- Optional per-slot `substeps` sample each subinterval with equal exposure.
- `packed_causal_attention` separates independent buffers, permits causal
  attention across chunks, and bidirectional attention within each chunk.
- Source masks sanitize invalid actions before interpolation and accompany the
  packed targets into the loss.

The first target chunk appears `N` times, the second `N-1`, and the last once.
`nuf_loss` accounts for this exposure through the packed targets and the
valid-time denominator.

## PA text supervision

The text decoder uses teacher forcing: BOS followed by preceding target tokens,
with EOS included in the token-loss mask. `pa_attention_mask` separates the
observation/instruction prefix from PA answer tokens and gives the decoder
causal attention over its shifted inputs.

`pa_token_cross_entropy` computes full-vocabulary cross-entropy in sequence
chunks with backward rematerialization. It averages supervised tokens within
each example. `pa_loss` then averages over examples with supervised tokens;
examples without annotations contribute zero. The default weight is `0.01`.

## Stage supervision

`stage_loss` averages categorical cross-entropy over the batch with a default
weight of `0.1`. Integer stage IDs must match the configured class vocabulary.
The stage head classifies progress from the observation.

## Visual-budget regularization

The same visual budget gate logits feed two interfaces:

1. `embedding_gates` returns differentiable sigmoid probabilities and floored
   gates. These gates multiply image embeddings before the action forward.
2. `attention_log_prior` uses a detached copy of the logits to cap each valid
   view's mean probability at the configured budget. A nonnegative logit shift
   is found with 25 FP32 bisection steps; probabilities already below the budget
   keep their original values. The attention prior is
   `log(exp(-4) + (1-exp(-4))*p)`, clipped to `[-4, 0]` at the default clip value,
   and added to attention key logits with the warmup from `attention_strength`.

These are soft gates over the existing token sequence. The budget controls gate
probabilities rather than a token count or a promised compute reduction.

`visual_budget_statistics` computes regularization from the unprojected
sigmoid gates. Budget error is the squared difference between each sample-view's
mean valid gate and its target ratio. Anchor error penalizes gates below
`anchor_delta`, weighted by token priors. Base and wrist terms are aggregated
by their counts of valid sample-views. Anchor terms use views with positive
prior mass. Both token padding and whole-view validity affect these statistics.

Use finite logits and priors on matching token grids. Zero priors give zero
anchor loss. Embedding gates and statistics follow the image embedding dtype;
attention projection uses FP32.

## Total objective

`post_training_loss` assembles:

\[
L=\lambda_aL_a+\lambda_bL_{budget,base}
+\lambda_wL_{budget,wrist}
+\lambda_{anchor}(L_{anchor,base}+L_{anchor,wrist})
+L_{PA,weighted}+L_{stage,weighted}.
\]

The action term uses one gated action forward. Compute PA and stage losses once
per observation batch and supply their weighted scalars to the assembler.

Default settings are action weight `1`, base/wrist budget weights `0.1/0.1`,
target ratios `0.9/0.9`, gate/attention temperature `0.805`, anchor weight
`0.0175`, anchor threshold `0.2` and attention warmup `500`. The embedding floor
moves from `0.1` to `0.02` over `visual_budget.schedule_steps`.

See `examples/losses.py` for a runnable example with synthetic tensors.
