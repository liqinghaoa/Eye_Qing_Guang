"""Bounded target-device B0 preflight; does not dispatch formal training."""

import sys

from eye_qing_guang.cli import main

if __name__ == "__main__":
    main(["preflight", *sys.argv[1:]])
