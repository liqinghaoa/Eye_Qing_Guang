"""Predict unlabeled test scores; see --help for options."""

import sys

from eye_qing_guang.cli import main

if __name__ == "__main__":
    main(["predict", *sys.argv[1:]])
