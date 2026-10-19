"""Generate a Grad-CAM overlay; see --help for options."""

import sys

from eye_qing_guang.cli import main

if __name__ == "__main__":
    main(["gradcam", *sys.argv[1:]])
