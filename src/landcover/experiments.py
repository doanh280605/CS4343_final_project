"""Generate the predeclared 48-run grid without executing it."""

import itertools
import shlex
from dataclasses import replace
from pathlib import Path

from landcover.config import Config
from landcover.data import FRACTIONS, dump_json


def grid(output, base=None):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    base = base or Config.load("configs/baseline.yaml")
    names = []
    for initialization, input_type, fraction, seed in itertools.product(
        ("pretrained", "scratch"), ("rgb", "ms"), FRACTIONS, (42, 43, 44)
    ):
        name = f"resnet18-{initialization}-{input_type}-f{fraction:g}-s{seed}"
        config = replace(
            base,
            model="resnet18",
            initialization=initialization,
            input_type=input_type,
            rgb_source="ms",
            fraction=fraction,
            seed=seed,
            output=f"outputs/runs/{name}",
        )
        config.validate().save(output / f"{name}.yaml")
        names.append(name)
    dump_json(output / "index.json", {"count": len(names), "runs": names})
    (output / "run.sh").write_text(
        "#!/usr/bin/env bash\nset -euo pipefail\n"
        + "".join(
            f"uv run landcover train --config {shlex.quote(str(output / (name + '.yaml')))}\n"
            for name in names
        )
    )
    return names
