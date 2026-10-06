"""Command-line entry points. Run `landcover --help` for the complete interface."""

import argparse
import json
from pathlib import Path

from landcover.config import Config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    p = commands.add_parser("download", help="download/check/extract official EuroSAT archives")
    p.add_argument("--kind", choices=["rgb", "ms"], required=True)
    p.add_argument("--root", default="data/raw")
    p.add_argument("--archive", help="verify/extract an already downloaded ZIP")
    p = commands.add_parser("prepare", help="persist fixed aligned splits and nested subsets")
    p.add_argument("--rgb-root")
    p.add_argument("--ms-root")
    p.add_argument("--output", default="data/prepared")
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument(
        "--limit-per-class", type=int, help="smoke fixtures only, not the experiment dataset"
    )
    p = commands.add_parser("audit", help="verify sample contents, shapes, identities and splits")
    p.add_argument("--manifest", default="data/prepared/manifest.json")
    p.add_argument("--splits", default="data/prepared/splits.json")
    p = commands.add_parser("train")
    p.add_argument("--config", required=True)
    p = commands.add_parser("evaluate")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--split", choices=["val", "test"], default="val")
    p.add_argument("--allow-test", action="store_true")
    p.add_argument("--output")
    p.add_argument("--manifest")
    p.add_argument("--splits")
    p = commands.add_parser("grid")
    p.add_argument("--output", default="outputs/grid")
    p.add_argument("--base-config")
    p = commands.add_parser("study-prepare", help="freeze the 63-run study without training")
    p.add_argument("--output", default="outputs/final-study")
    p.add_argument("--base-config", default="configs/final-study.yaml")
    p = commands.add_parser("study-run", help="run frozen study and generate validation report")
    p.add_argument("--study", default="outputs/final-study")
    p.add_argument("--retry-incomplete", action="store_true")
    p = commands.add_parser("study-report", help="regenerate complete study tables and figures")
    p.add_argument("--study", default="outputs/final-study")
    p.add_argument("--split", choices=["val", "test"], default="val")
    p = commands.add_parser(
        "study-test", help="explicit final test phase after validation selection"
    )
    p.add_argument("--study", default="outputs/final-study")
    p.add_argument("--allow-test", action="store_true")
    p = commands.add_parser("smoke", help="synthetic paired data, training, reload and validation")
    p.add_argument("--output", default="outputs/synthetic-smoke")
    p = commands.add_parser("nepal-acquire")
    p.add_argument("--config", default="configs/nepal.yaml")
    p.add_argument("--output", required=True)
    p.add_argument("--submit", action="store_true", help="submit Earth Engine Drive exports")
    p = commands.add_parser("nepal-align")
    p.add_argument("--reference", required=True)
    p.add_argument("--source", required=True)
    p.add_argument("--output", required=True)
    p = commands.add_parser("nepal-patches")
    p.add_argument("--pre", required=True)
    p.add_argument("--post", required=True)
    p.add_argument("--output", required=True)
    p = commands.add_parser("nepal-infer")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--patches", required=True)
    p.add_argument("--period", choices=["pre", "post"], required=True)
    p.add_argument("--labels")
    p.add_argument("--output", required=True)
    p = commands.add_parser("nepal-change")
    p.add_argument("--pre", required=True)
    p.add_argument("--post", required=True)
    p.add_argument("--patches", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--threshold", type=float, default=0.7)
    args = parser.parse_args()
    if args.command == "download":
        from landcover.download import download

        print(download(args.kind, args.root, args.archive))
    elif args.command == "prepare":
        from landcover.data import prepare

        manifest, _ = prepare(
            args.output, args.rgb_root, args.ms_root, args.seed, args.limit_per_class
        )
        print(f"Prepared {len(manifest['samples'])} samples in {args.output}")
    elif args.command == "audit":
        from landcover.data import audit

        print(json.dumps(audit(args.manifest, args.splits), indent=2))
    elif args.command == "train":
        from landcover.engine import train

        print(train(Config.load(args.config)))
    elif args.command == "evaluate":
        from landcover.engine import evaluate

        result = evaluate(
            args.checkpoint, args.split, args.output, args.allow_test, args.manifest, args.splits
        )
        print(json.dumps({k: result[k] for k in ("accuracy", "macro_f1")}, indent=2))
    elif args.command == "grid":
        from landcover.experiments import grid

        base = Config.load(args.base_config) if args.base_config else None
        count = len(grid(args.output, base))
        print(f"Generated {count} configs; no training launched")
    elif args.command.startswith("study-"):
        from landcover.study import prepare_study, run_study, test_study
        from landcover.study_report import report

        if args.command == "study-prepare":
            count = prepare_study(args.output, Config.load(args.base_config))
            print(f"Frozen {count} runs; no training launched")
        elif args.command == "study-run":
            print(run_study(args.study, args.retry_incomplete))
        elif args.command == "study-test":
            print(test_study(args.study, args.allow_test))
        else:
            print(report(args.study, args.split))
    elif args.command == "smoke":
        smoke(args.output)
    else:
        from landcover import nepal

        if args.command == "nepal-acquire":
            print(nepal.acquire(args.config, args.output, args.submit))
        elif args.command == "nepal-align":
            nepal.align(args.reference, args.source, args.output)
        elif args.command == "nepal-patches":
            print(nepal.patches(args.pre, args.post, args.output))
        elif args.command == "nepal-infer":
            print(nepal.infer(args.checkpoint, args.patches, args.period, args.output, args.labels))
        elif args.command == "nepal-change":
            nepal.change_map(args.pre, args.post, args.patches, args.output, args.threshold)


def smoke(output):
    from landcover.data import audit, dump_json, synthetic
    from landcover.engine import evaluate, train

    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    synthetic(output / "data")
    prepared = output / "data/prepared"
    checks = audit(prepared / "manifest.json", prepared / "splits.json")
    for input_type in ("rgb", "ms"):
        config = Config(
            manifest=str(prepared / "manifest.json"),
            splits=str(prepared / "splits.json"),
            output=str(output / input_type),
            input_type=input_type,
            steps=2,
            eval_every=1,
            batch_size=4,
            threads=2,
            device="cpu",
        )
        checkpoint = train(config)
        evaluate(checkpoint)
    dump_json(
        output / "verification.json",
        {
            "data_kind": "synthetic",
            "checks": checks,
            "note": "Pipeline validation only; not a land-cover benchmark.",
        },
    )
    print(f"Synthetic smoke passed: {output}")


if __name__ == "__main__":
    main()
