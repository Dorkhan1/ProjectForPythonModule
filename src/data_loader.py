"""Loading of the raw Food.com CSV files."""

from pathlib import Path

import pandas as pd

from src import config


class DataLoadError(Exception):
    """Raised when a dataset file is missing, unreadable or malformed."""


def _check_file_exists(path: Path) -> None:
    """Check that a data file exists.

    Parameters
    ----------
    path : pathlib.Path
        Location of the CSV file.

    Raises
    ------
    DataLoadError
        If the file does not exist, with download instructions.
    """
    if not path.is_file():
        raise DataLoadError(
            f"File not found: {path}\n"
            f"Download the dataset from {config.DATASET_URL} and put "
            f"'{path.name}' into the '{config.DATA_DIR.name}/' folder."
        )


def _check_columns(path: Path, required: list[str]) -> None:
    """Check that a CSV file has all required columns (reads header only).

    Parameters
    ----------
    path : pathlib.Path
        Location of the CSV file.
    required : list of str
        Column names that must be present.

    Raises
    ------
    DataLoadError
        If the file is empty, cannot be parsed, or lacks some columns.
    """
    try:
        header = pd.read_csv(path, nrows=0).columns
    except pd.errors.EmptyDataError as exc:
        raise DataLoadError(f"File is empty: {path}") from exc
    except pd.errors.ParserError as exc:
        raise DataLoadError(f"File is not a valid CSV: {path}") from exc

    missing = sorted(set(required) - set(header))
    if missing:
        raise DataLoadError(
            f"File {path.name} is missing columns: {', '.join(missing)}"
        )


def _read_csv(
    path: Path,
    dtypes: dict[str, str],
    date_columns: list[str],
) -> pd.DataFrame:
    """Read selected columns of a CSV file with explicit dtypes.

    Parameters
    ----------
    path : pathlib.Path
        Location of the CSV file.
    dtypes : dict of str to str
        Mapping ``column name -> pandas dtype`` for non-date columns.
    date_columns : list of str
        Columns to parse as ``datetime64``.

    Returns
    -------
    pandas.DataFrame
        DataFrame containing exactly the requested columns.

    Raises
    ------
    DataLoadError
        If the file is missing, malformed, or a value cannot be
        converted to its declared dtype.
    """
    columns = list(dtypes) + date_columns
    _check_file_exists(path)
    _check_columns(path, columns)

    try:
        return pd.read_csv(
            path,
            usecols=columns,
            dtype=dtypes,
            parse_dates=date_columns,
        )
    except (ValueError, pd.errors.ParserError) as exc:
        raise DataLoadError(f"Could not parse {path.name}: {exc}") from exc


def load_recipes(path: Path = config.RECIPES_PATH) -> pd.DataFrame:
    """Load recipe metadata from ``RAW_recipes.csv``.

    Parameters
    ----------
    path : pathlib.Path, optional
        Location of the recipes file. Defaults to
        :data:`src.config.RECIPES_PATH`.

    Returns
    -------
    pandas.DataFrame
        One row per recipe with columns ``id``, ``name``, ``minutes``,
        ``tags``, ``n_steps``, ``n_ingredients`` and ``submitted``.

    Raises
    ------
    DataLoadError
        If the file cannot be loaded (see :func:`_read_csv`).
    """
    return _read_csv(path, config.RECIPE_DTYPES, config.RECIPE_DATE_COLUMNS)


def load_interactions(
    path: Path = config.INTERACTIONS_PATH,
) -> pd.DataFrame:
    """Load user ratings from ``RAW_interactions.csv`` (without review text).

    Parameters
    ----------
    path : pathlib.Path, optional
        Location of the interactions file. Defaults to
        :data:`src.config.INTERACTIONS_PATH`.

    Returns
    -------
    pandas.DataFrame
        One row per review with columns ``user_id``, ``recipe_id``,
        ``rating`` (0-5, where 0 means "no rating given") and ``date``.

    Raises
    ------
    DataLoadError
        If the file cannot be loaded (see :func:`_read_csv`).
    """
    return _read_csv(
        path, config.INTERACTION_DTYPES, config.INTERACTION_DATE_COLUMNS
    )
