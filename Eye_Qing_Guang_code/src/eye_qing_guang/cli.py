"""Shared command-line parsing; scripts remain thin executable adapters."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

from eye_qing_guang.utils.paths import PROJECT_ROOT, resolve_path


def build_parser():
    parser = argparse.ArgumentParser(description="Glaucoma binary-classification experiments")
    commands = parser.add_subparsers(dest="command", required=True)
    for name, description in (
        ("audit", "Audit raw tables and image paths"),
        ("manifest", "Build normalized manifests"),
        ("development", "Build B0 internal fit/development manifests from official train only"),
        ("test-manifest", "Build a separate unlabeled official-test manifest for inference only"),
        ("preflight", "Run budget-limited B0 CUDA updates and an FP32 gradient reference"),
        ("train", "Train and select a validation checkpoint"),
        ("evaluate", "Evaluate a checkpoint on labeled data"),
        ("summarize", "Summarize experiment metric reports"),
        ("predict", "Predict unlabeled test probabilities"),
        ("submit", "Build a submission with explicit format settings"),
        ("curves", "Plot training curves"),
        ("confusion", "Plot a recorded confusion matrix"),
        ("gradcam", "Generate an image Grad-CAM overlay"),
    ):
        command = commands.add_parser(name, help=description, description=description)
        command.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
        if name in {"audit", "manifest", "train", "submit", "development", "preflight", "test-manifest"}:
            command.add_argument("--config", default="configs/resnet18.yaml")
            if name in {"development", "preflight", "test-manifest"}:
                command.set_defaults(config="configs/b0_resnet50_512_dev_v1.yaml")
        if name == "audit":
            command.add_argument("--output", required=True)
            command.add_argument("--decode-images", action="store_true")
        elif name == "manifest":
            command.add_argument("--output-dir", help="New directory; default: data.manifest_dir")
        elif name == "test-manifest":
            command.add_argument("--output-dir", required=True, help="New directory, separate from development manifests")
            command.add_argument("--decode-images", action="store_true")
        elif name == "preflight":
            command.add_argument("--name", default="b0_resnet50_512_preflight_s42_v1")
        elif name == "train":
            command.add_argument("--name", help="Unique experiment directory name")
        elif name in {"evaluate", "predict", "gradcam"}:
            command.add_argument("--checkpoint", required=True)
            command.add_argument("--device", default="auto")
            if name in {"evaluate", "predict"}:
                command.add_argument("--manifest")
                command.add_argument("--output-dir", required=True)
            if name == "evaluate":
                command.add_argument("--split", choices=["train", "valid"], default="valid")
                command.add_argument("--evaluation-source", choices=["internal_fit", "internal_development", "official_valid", "external_labeled"])
            if name == "gradcam":
                command.add_argument("--image", required=True)
                command.add_argument("--age", type=float)
                command.add_argument("--class-index", type=int, choices=[0, 1], default=1)
                command.add_argument("--output", required=True)
        elif name == "summarize":
            command.add_argument("--experiments", nargs="+", required=True)
            command.add_argument("--output", required=True)
        elif name == "submit":
            command.add_argument("--predictions", required=True)
            command.add_argument("--test-csv")
            command.add_argument("--id-column")
            command.add_argument("--value-column")
            command.add_argument("--label-column")
            command.add_argument("--mode", choices=["competition", "probability", "label"],
                                 help="competition: fixed three-column submission.csv; other modes: legacy two-column export")
            command.add_argument("--output", required=True)
        elif name == "curves":
            command.add_argument("--history", required=True)
            command.add_argument("--metric", default="roc_auc")
            command.add_argument("--output", required=True)
        elif name == "confusion":
            command.add_argument("--metrics", required=True)
            command.add_argument("--output", required=True)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    root = args.project_root.resolve()

    def path(value):
        return resolve_path(value, root)

    if hasattr(args, "config"):
        from eye_qing_guang.utils.experiment_io import load_config

        config = load_config(path(args.config))
    if args.command == "audit":
        from eye_qing_guang.datasets.metadata import collect_records
        from eye_qing_guang.utils.experiment_io import run_manifest, write_json

        output = path(args.output)
        if output.exists():
            raise FileExistsError(output)
        try:
            audit_config = copy.deepcopy(config)
            if config["experiment"].get("protocol") == "b0_dev_v1":
                audit_config["data"]["splits"] = {"train": config["data"]["splits"]["train"]}
            _, summary = collect_records(audit_config, root, args.decode_images)
        except Exception as exc:
            write_json(output, run_manifest("failed", error=f"{type(exc).__name__}: {exc}"))
            raise
        write_json(output, run_manifest("completed", splits=summary))
        print(json.dumps(summary, indent=2))
    elif args.command == "manifest":
        if config["experiment"].get("protocol") == "b0_dev_v1":
            raise ValueError("Use the development command for B0, never the official-split manifest command.")
        from eye_qing_guang.datasets.metadata import build_manifests

        summary = build_manifests(config, root, path(args.output_dir or config["data"]["manifest_dir"]))
        print(json.dumps(summary, indent=2))
    elif args.command == "test-manifest":
        from eye_qing_guang.datasets.metadata import build_test_manifest

        output = path(args.output_dir)
        if output == path(config["data"]["manifest_dir"]):
            raise ValueError("Official-test manifests require a separate output directory.")
        print(json.dumps(build_test_manifest(config, root, output, args.decode_images), indent=2))
    elif args.command == "development":
        if config["experiment"].get("protocol") != "b0_dev_v1":
            raise ValueError("development requires a B0-dev configuration.")
        from eye_qing_guang.datasets.development import build_development_manifests

        print(json.dumps(build_development_manifests(config, root, path(config["data"]["manifest_dir"])), indent=2))
    elif args.command == "preflight":
        from eye_qing_guang.pipelines.preflight import preflight

        directory = preflight(config, root, args.name)
        print(directory)
        with (directory / "run_manifest.json").open(encoding="utf-8") as handle:
            status = json.load(handle)["status"]
        if status != "completed":
            # An inconclusive preflight must not let a shell command chain
            # start formal training automatically.
            raise SystemExit(2)
    elif args.command == "train":
        from eye_qing_guang.pipelines.training import train

        print(train(config, root, args.name))
    elif args.command == "evaluate":
        from eye_qing_guang.pipelines.evaluation import evaluate_checkpoint

        result = evaluate_checkpoint(path(args.checkpoint), root, path(args.output_dir), args.split,
                                     args.device, path(args.manifest) if args.manifest else None, args.evaluation_source)
        print(json.dumps(result, indent=2))
    elif args.command == "predict":
        from eye_qing_guang.pipelines.inference import predict_test

        print(predict_test(path(args.checkpoint), root, path(args.output_dir), args.device,
                           path(args.manifest) if args.manifest else None))
    elif args.command == "submit":
        from eye_qing_guang.pipelines.inference import build_submission

        from eye_qing_guang.contracts.config import submission_options

        overrides = {key: value for key, value in {
            "id_column": args.id_column, "value_column": args.value_column,
            "label_column": args.label_column, "mode": args.mode,
        }.items() if value is not None}
        submission = submission_options({"submission": {**config.get("submission", {}), **overrides}})
        data = config["data"]
        source = (path(args.test_csv) if args.test_csv else
                  path(data["raw_root"]) / data["splits"]["test"]["csv"])
        print(build_submission(
            path(args.predictions), source, path(args.output),
            submission["id_column"], submission["value_column"], data["image_column"],
            submission["mode"], float(config["evaluation"]["threshold"]), submission["label_column"],
        ))
    elif args.command == "summarize":
        from eye_qing_guang.metrics.aggregation import summarize_experiments

        rows = summarize_experiments([path(value) for value in args.experiments], path(args.output))
        print(f"Summarized {len(rows)} metric reports: {path(args.output)}")
    elif args.command == "curves":
        from eye_qing_guang.visualization.training_curves import plot_training_curves

        plot_training_curves(path(args.history), path(args.output), args.metric)
        print(path(args.output))
    elif args.command == "confusion":
        from eye_qing_guang.visualization.confusion_matrix import plot_confusion_matrix

        plot_confusion_matrix(path(args.metrics), path(args.output))
        print(path(args.output))
    elif args.command == "gradcam":
        from eye_qing_guang.visualization.gradcam import generate_gradcam

        generate_gradcam(path(args.checkpoint), path(args.image), path(args.output),
                         args.age, args.device, args.class_index)
        print(path(args.output))


if __name__ == "__main__":
    main()
