"""Predict using a format-4 KD student, without teacher or ImageNet weight files."""

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
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--manifest", required=True, help="Unlabeled manifest with current raw image paths.")
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    from eye_qing_guang.pipelines.convnext_kd import predict_kd_student
    from eye_qing_guang.utils.reproducibility import choose_device
    print(predict_kd_student(args.checkpoint, args.manifest, args.project_root, args.output,
                             choose_device(args.device)), flush=True)


if __name__ == "__main__":
    main()
