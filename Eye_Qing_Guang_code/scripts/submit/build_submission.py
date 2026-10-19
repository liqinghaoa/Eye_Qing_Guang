"""Format test predictions for submission; see --help for options."""

import sys

from eye_qing_guang.cli import main

if __name__ == "__main__":
    main(["submit", *sys.argv[1:]])
