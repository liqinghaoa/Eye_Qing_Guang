"""Evaluate a saved checkpoint; see --help for options."""

import sys

from eye_qing_guang.cli import main

if __name__ == "__main__":
    main(["evaluate", *sys.argv[1:]])
