"""Saving of ranking results to CSV files.

Can also be run as a script to compute and save all results::

    python -m src.export
"""

import sys
from pathlib import Path

import pandas as pd

from src import config
from src.analysis import summarize_by_column, summarize_by_tags
from src.data_loader import DataLoadError, load_interactions, load_recipes
from src.preprocessing import build_recipe_table
from src.ranking import METHODS, add_scores, compare_methods, top_recipes

TOP_COLUMNS = ["recipe_id", "name", "n_ratings", "mean_rating", "minutes",
               "n_ingredients"]


class DataSaveError(Exception):
    """Raised when a result file cannot be written."""


def to_csv_bytes(table: pd.DataFrame, index: bool = False) -> bytes:
    """Convert a table to CSV bytes (e.g. for a download button).

    Parameters
    ----------
    table : pandas.DataFrame
        Table to convert.
    index : bool, optional
        Whether to include the index.

    Returns
    -------
    bytes
        UTF-8 encoded CSV content.
    """
    return table.to_csv(index=index, float_format="%.4f").encode("utf-8")


def save_table(
    table: pd.DataFrame,
    filename: str,
    output_dir: Path = config.OUTPUT_DIR,
    index: bool = False,
) -> Path:
    """Save a table as a CSV file, creating the folder if needed.

    Parameters
    ----------
    table : pandas.DataFrame
        Table to save.
    filename : str
        Name of the file, e.g. ``"top_lower_bound.csv"``.
    output_dir : pathlib.Path, optional
        Destination folder. Defaults to :data:`src.config.OUTPUT_DIR`.
    index : bool, optional
        Whether to write the index.

    Returns
    -------
    pathlib.Path
        Path of the written file.

    Raises
    ------
    DataSaveError
        If the folder or the file cannot be written.
    """
    path = Path(output_dir) / filename
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        table.to_csv(path, index=index, float_format="%.4f")
    except OSError as exc:
        raise DataSaveError(f"Could not write {path}: {exc}") from exc
    return path


def _slug(text: str) -> str:
    """Make a file-name friendly version of a label.

    Parameters
    ----------
    text : str
        Label like ``"Main ingredient"``.

    Returns
    -------
    str
        Lower-case text with underscores, like ``"main_ingredient"``.
    """
    return text.lower().replace(" ", "_")


def export_results(
    scored: pd.DataFrame,
    output_dir: Path = config.OUTPUT_DIR,
    top_n: int = config.TOP_N_EXPORT,
) -> list[Path]:
    """Save top lists, method comparison and category summaries.

    Parameters
    ----------
    scored : pandas.DataFrame
        Recipe table from :func:`src.ranking.add_scores`.
    output_dir : pathlib.Path, optional
        Destination folder.
    top_n : int, optional
        Number of recipes in each top list.

    Returns
    -------
    list of pathlib.Path
        Paths of all written files.

    Raises
    ------
    DataSaveError
        If a file cannot be written.
    """
    paths = []

    for method, score_col in METHODS.items():
        rank_col = score_col.replace("_score", "_rank")
        top = top_recipes(scored, method, top_n)
        table = top[TOP_COLUMNS + [score_col, rank_col]]
        paths.append(save_table(table, f"top_{_slug(method)}.csv",
                                output_dir))

    paths.append(save_table(compare_methods(scored),
                            "method_comparison.csv", output_dir))

    for column in ["time_category", "ingredient_category"]:
        summary = summarize_by_column(scored, column)
        paths.append(save_table(summary, f"summary_{column}.csv",
                                output_dir, index=True))

    for group, tags in config.TAG_GROUPS.items():
        summary = summarize_by_tags(scored, tags)
        paths.append(save_table(summary, f"summary_{_slug(group)}.csv",
                                output_dir, index=True))

    return paths


def main() -> int:
    """Load the data, rank recipes and save all results.

    Returns
    -------
    int
        Exit code: 0 on success, 1 on error.
    """
    try:
        recipes = build_recipe_table(load_recipes(), load_interactions())
        paths = export_results(add_scores(recipes))
    except (DataLoadError, DataSaveError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    print(f"Saved {len(paths)} files to {config.OUTPUT_DIR}:")
    for path in paths:
        print(f"  {path.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
