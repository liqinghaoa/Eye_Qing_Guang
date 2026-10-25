"""Build a separate deterministic 288x288 cache for the official timm experiments."""

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
    parser.add_argument(
        "--config",
        default="configs/timm_v2s_in21k_ft_in1k_288_pw5_official_single3090_cache_v1.yaml",
        help="288x288 V2-S or ConvNeXt cache-enabled configuration",
    )
    parser.add_argument(
        "--workers", type=int, default=4,
        help="CPU cache-build workers; 0 runs serially.",
    )
    args = parser.parse_args()
    if any(os.environ.get(k, "1") != "1" for k in ("WORLD_SIZE", "LOCAL_WORLD_SIZE")):
        parser.error("Run with ordinary Python, not a multi-rank torchrun launcher.")
    import yaml
    from eye_qing_guang.datasets.offline_cache import build_offline_cache
    from eye_qing_guang.utils.paths import resolve_path

    root = args.project_root.expanduser().resolve()
    config_path = resolve_path(args.config, root)
    with config_path.open(encoding="utf-8") as handle:
        protocol = yaml.safe_load(handle).get("experiment", {}).get("protocol")
    if protocol == "timm_convnext_official_trainvalid_v1":
        from eye_qing_guang.contracts.convnext_config import load_convnext_config
        config = load_convnext_config(config_path)
    elif protocol == "timm_official_trainvalid_v1":
        from eye_qing_guang.contracts.timm_config import load_timm_config
        config = load_timm_config(config_path)
    else:
        parser.error("Only official V2-S or ConvNeXt experiment configurations are supported")
    if config["data"]["image_size"] != 288:
        parser.error("This entry point accepts only configurations with data.image_size=288")
    build_offline_cache(config, root, workers=args.workers)


if __name__ == "__main__":
    main()
