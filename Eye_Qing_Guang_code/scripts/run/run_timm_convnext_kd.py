"""Run one frozen ConvNeXt KD variant, or its explicit bounded preflight."""

import argparse
import os
import sys
from pathlib import Path

for variable in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(variable, "1")
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--config", required=True)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--preflight-output", type=Path)
    parser.add_argument("--max-preflight-sample-instances", type=int)
    args = parser.parse_args()
    if args.preflight_only:
        if args.preflight_output is None or args.max_preflight_sample_instances is None:
            parser.error("Preflight requires an independent --preflight-output and explicit --max-preflight-sample-instances.")
    elif args.preflight_output is not None or args.max_preflight_sample_instances is not None:
        parser.error("Preflight arguments require --preflight-only.")
    from eye_qing_guang.contracts.convnext_kd_config import load_kd_config
    from eye_qing_guang.utils.paths import resolve_path
    root = args.project_root.expanduser().resolve()
    load_kd_config(resolve_path(args.config, root))
    from eye_qing_guang.pipelines.convnext_kd import run_kd_experiment, run_kd_preflight
    print(run_kd_preflight(args.config, root, args.preflight_output, args.max_preflight_sample_instances)
          if args.preflight_only else run_kd_experiment(args.config, root), flush=True)


if __name__ == "__main__":
    main()
