"""Fixed-epoch ConvNeXt train+valid final fit followed by one official-test inference."""

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
    parser.add_argument("--config", default="configs/timm_convnext_tiny_fb_in1k_512_pw5_trainvalid_final_single3090_s42_v1.yaml")
    args = parser.parse_args()
    from eye_qing_guang.contracts.convnext_final_config import load_convnext_final_config
    from eye_qing_guang.pipelines.convnext_final_fit import run_convnext_final_fit
    from eye_qing_guang.utils.paths import resolve_path

    load_convnext_final_config(resolve_path(args.config, args.project_root))
    result = run_convnext_final_fit(args.config, args.project_root)
    if os.environ.get("RANK", "0") == "0":
        print(result, flush=True)


if __name__ == "__main__":
    main()
