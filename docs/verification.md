# Verification

Run the test suite with:

```sh
python -m pytest -q
```

The suite covers:

- Flow-matching reductions, validity masks, sample weights and gradients.
- NUF packing, chunk exposure, clock intervals and attention masks.
- PA cross-entropy, token masks and stage classification.
- Visual-budget embedding gates, attention projection, regularization and gradients.
- Dataset feature selection, current-image decoding and normalization.
- Proprio-history timing, dropout, episode boundaries, validity and padding.
- Pretrained parameter loading and initialization of new task/history parameters.

Compact CPU models exercise action loss and gradients with PA, stage and
proprio history enabled. Checkpoint tests save and restore working parameters,
EMA and optimizer state, and compare the next update on the same batch.
Numerical unit tests use synthetic inputs; model quality is assessed separately
through task evaluation. Data sampling restarts when a training run resumes.
