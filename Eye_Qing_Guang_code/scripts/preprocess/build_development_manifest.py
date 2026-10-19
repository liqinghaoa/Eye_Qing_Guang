"""Build the fixed B0 internal development split from official train only."""

import sys

from eye_qing_guang.cli import main

if __name__ == "__main__":
    main(["development", *sys.argv[1:]])
