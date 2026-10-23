"""Official timm ConvNeXt-Tiny training -> validation -> test -> submission."""

import argparse
import os
import sys
from pathlib import Path

# Limit per-worker BLAS threads before importing torch/numpy in spawned workers.
for variable in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(variable, "1")
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--config", default="configs/timm_convnext_tiny_fb_in22k_ft_in1k_512_pw5_official.yaml")
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--skip-preflight", action="store_true",
                        help="Skip the bounded target-batch preflight and start the formal run directly")
    parser.add_argument("--preflight-version", default="pf_v1", help="Explicit independent preflight output suffix; never auto-renamed")
    parser.add_argument("--max-sample-instances", type=int, help="Explicitly authorized global real-instance preflight budget; default 64")
    args = parser.parse_args(argv)
    if args.preflight_only and args.skip_preflight:
        parser.error("--preflight-only and --skip-preflight cannot be combined")
    # Configuration is validated before importing a model, loading images or creating outputs.
    from eye_qing_guang.contracts.convnext_config import UnresolvedRecipeError, load_convnext_config
    from eye_qing_guang.utils.paths import resolve_path

    try:
        load_convnext_config(resolve_path(args.config, args.project_root))
    except UnresolvedRecipeError as exc:
        parser.error(str(exc))
    from eye_qing_guang.pipelines.convnext_official import run_convnext_experiment

    result = run_convnext_experiment(args.config, args.project_root, preflight_only=args.preflight_only,
                                 preflight_version=args.preflight_version, max_sample_instances=args.max_sample_instances,
                                 skip_preflight=args.skip_preflight)
    if os.environ.get("RANK", "0") == "0":
        print(result, flush=True)


if __name__ == "__main__":
    main()
