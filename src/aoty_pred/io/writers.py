"""Output writers (stub)."""

import pandas as pd


def write_csv(df: pd.DataFrame, path: str) -> None:
    df.to_csv(path, index=False)