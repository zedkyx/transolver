# Public entrypoints

This directory collects the stable, user-facing Transolver launchers.

Preferred commands:

- `python -m scripts.transolver.public.train_dp`
- `python -m scripts.transolver.public.train_ddp`
- `python -m scripts.transolver.public.evaluate`

Internal helpers live in `scripts.transolver.core`, while implementation
packages remain under `train`, `eval`, and `viz`.
