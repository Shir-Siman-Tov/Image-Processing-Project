"""Save README-ready result tables.

Mirrors viz.plotting.save_figure(): notebooks already persist a cross-notebook
working copy of each metrics table under config.RESULTS_ROOT (Drive/local,
gitignored - so later notebooks can read it back). save_results_csv() writes
an additional copy under config.REPORTS_ROOT, which lives inside the repo and
is NOT gitignored, so tables can be embedded/linked from the README the same
way figures are (see colab_sync.py for how it gets from a Colab run onto
GitHub).
"""
from pathlib import Path

import pandas as pd

from ipproj import config


def save_results_csv(df: pd.DataFrame, relative_path: str) -> Path:
    out_path = config.REPORTS_ROOT / relative_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    return out_path
