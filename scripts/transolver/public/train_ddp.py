from __future__ import annotations

import os
import sys

# Keep direct file execution working from scripts/transolver/public/.
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from scripts.transolver.core.argparser import build_argparser, load_yaml_config
from scripts.transolver.train.trainer import run


def main():
    parser = build_argparser()
    args = parser.parse_args()
    args = load_yaml_config(args, parser=parser)
    # train_ddp.py is a DDP-only entry: force enable DDP so users don't need to pass --ddp
    args.ddp = True
    run(args)


if __name__ == "__main__":
    main()

