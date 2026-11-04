# Third-party notices

AIRoA-0 (A-0) post-training is licensed under the Apache License 2.0
([LICENSE](LICENSE)); see [NOTICE](NOTICE) for required attributions.

## OpenPI

The model and training runtime in `vendor/openpi/` derives from
[Physical Intelligence OpenPI](https://github.com/Physical-Intelligence/openpi),
based on revision `215abfb217dbac7d5f1273282331b9b1866c0479`, licensed under the
Apache License 2.0 ([vendor/openpi/LICENSE](vendor/openpi/LICENSE)), the reference
implementation of [π0.5](https://www.pi.website/blog/pi05)
([paper](https://arxiv.org/abs/2504.16054)). AIRoA modified
it for the AIRoA-0 post-training objectives and task labels; each modified file
carries a notice at the top, and files added by AIRoA carry an AIRoA copyright
header.

## Big Vision

`vendor/openpi/src/openpi/models/gemma.py`, `siglip.py`, and functions marked
"Adapted from big_vision" derive from
[Google Research Big Vision](https://github.com/google-research/big_vision),
Copyright 2024 Big Vision Authors, licensed under the Apache License 2.0. The
original copyright headers are retained.

## Gemma

AIRoA-0 builds on the PaliGemma backbone. The PaliGemma tokenizer (downloaded on
first use) and model weights derived from Gemma are subject to the
[Gemma Terms of Use](https://ai.google.dev/gemma/terms), a copy of which is in
[vendor/openpi/LICENSE_GEMMA.txt](vendor/openpi/LICENSE_GEMMA.txt), including the
use restrictions in Section 3.2 and the
[Gemma Prohibited Use Policy](https://ai.google.dev/gemma/prohibited_use_policy).

Gemma is provided under and subject to the Gemma Terms of Use found at
ai.google.dev/gemma/terms.

## LeRobot

The LeRobot data reader is an external dependency at the version specified in
`pyproject.toml`. It is not redistributed here; its source and license remain
with the upstream project.
