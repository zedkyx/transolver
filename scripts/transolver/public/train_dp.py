from __future__ import annotations

import os
import sys

import matplotlib

matplotlib.use("Agg")

# Keep direct file execution working from scripts/transolver/public/.
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from scripts.transolver.core.argparser import build_argparser, load_yaml_config
from scripts.transolver.train.trainer import run


def main():
    """
    DataParallel (DP) entry for cache training.

    Notes:
    - This script is DP-only: it forces args.ddp = False.
    - Use --gpu "0,1,2,3" (comma-separated) to choose multiple GPUs.
    """
    parser = build_argparser()
    args = parser.parse_args()
    args = load_yaml_config(args, parser=parser)

    # DP-only
    args.ddp = False
    run(args)


if __name__ == "__main__":
    main()

