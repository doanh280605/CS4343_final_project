"""Explicit, validated run configuration; paths are relative to the working directory."""

from dataclasses import asdict, dataclass
from pathlib import Path

import yaml


@dataclass
class Config:
    manifest: str = "data/prepared/manifest.json"
    splits: str = "data/prepared/splits.json"
    output: str = "outputs/baseline"
    model: str = "compact"
    initialization: str = "scratch"
    input_type: str = "rgb"
    rgb_source: str = "ms"
    fraction: float = 1.0
    seed: int = 42
    steps: int = 1000
    eval_every: int = 100
    batch_size: int = 64
    learning_rate: float = 0.001
    pretrained_backbone_learning_rate: float | None = None
    scheduler: str = "constant"
    warmup_steps: int = 0
    measure_train_metrics: bool = False
    weight_decay: float = 0.0001
    dropout: float = 0.3
    augmentation: bool = True
    device: str = "auto"
    workers: int = 0
    threads: int = 4
    deterministic: bool = True

    def validate(self):
        if self.model not in {"compact", "resnet18"}:
            raise ValueError("model must be compact or resnet18")
        if self.initialization not in {"scratch", "pretrained"}:
            raise ValueError("invalid initialization")
        if self.model == "compact" and self.initialization != "scratch":
            raise ValueError("compact has no pretrained weights")
        if self.input_type not in {"rgb", "ms"} or self.rgb_source not in {"ms", "jpg"}:
            raise ValueError("invalid input type or RGB source")
        if self.fraction not in {1.0, 0.1, 0.05, 0.01}:
            raise ValueError("fraction must be 1, .1, .05, or .01")
        if min(self.steps, self.eval_every, self.batch_size, self.threads) < 1:
            raise ValueError("steps, eval_every, batch_size and threads must be positive")
        if self.batch_size < 2 or self.workers < 0 or not 0 <= self.dropout < 1:
            raise ValueError("batch size >=2, workers >=0, dropout in [0,1) required")
        if self.learning_rate <= 0 or self.weight_decay < 0:
            raise ValueError("invalid optimization parameters")
        if (
            self.pretrained_backbone_learning_rate is not None
            and self.pretrained_backbone_learning_rate <= 0
        ):
            raise ValueError("pretrained backbone learning rate must be positive")
        if self.scheduler not in {"constant", "cosine"}:
            raise ValueError("scheduler must be constant or cosine")
        if not 0 <= self.warmup_steps < self.steps:
            raise ValueError("warmup_steps must be nonnegative and smaller than steps")
        if self.scheduler == "constant" and self.warmup_steps:
            raise ValueError("warmup requires the cosine scheduler")
        if self.device not in {"auto", "cpu", "cuda", "mps"}:
            raise ValueError("unsupported device")
        return self

    @classmethod
    def load(cls, path):
        return cls(**yaml.safe_load(Path(path).read_text())).validate()

    def save(self, path):
        Path(path).write_text(yaml.safe_dump(asdict(self), sort_keys=True))
