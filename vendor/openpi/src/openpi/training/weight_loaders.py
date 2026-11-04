# Modified by AI Robot Association (AIRoA) in 2026 for AIRoA-0 (A-0) post-training.
# Derived from Physical Intelligence openpi (Apache-2.0), revision 215abfb.
from dataclasses import dataclass
from typing import Protocol


class WeightLoader(Protocol):
    def load(self, params): ...


@dataclass(frozen=True)
class NoOpWeightLoader:
    def load(self, params):
        raise ValueError("An AIRoA-0 pretrained checkpoint is required")


@dataclass(frozen=True)
class CheckpointWeightLoader:
    params_path: str

    def load(self, params):
        raise NotImplementedError("Use the strict AIRoA-0 core/visual-gate checkpoint loader")
