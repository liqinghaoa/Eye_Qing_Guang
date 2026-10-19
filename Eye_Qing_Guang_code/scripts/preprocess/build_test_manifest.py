"""Build official-test metadata separately from development data; never run inference."""

import sys

from eye_qing_guang.cli import main

if __name__ == "__main__":
    main(["test-manifest", *sys.argv[1:]])
