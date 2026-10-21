"""Run pure Scheme-B functional-signal unlearning on TinyImageNet with CLIP."""

import sys
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from normal_unlearning_subliminal_core import main


if __name__ == "__main__":
    main("tinyimagenet")
