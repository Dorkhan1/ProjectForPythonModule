"""Recipe scoring and ranking methods."""

import numpy as np
import pandas as pd

from src import config


def global_mean_rating(recipes: pd.DataFrame) -> float:
    """Compute the average of all ratings on the site.

    Every single rating has the same weight, so popular recipes
    contribute more than recipes with few ratings.

    Parameters
    ----------
    recipes : pandas.DataFrame
        Recipe table with ``stars_1`` ... ``stars_5`` columns.

    Returns
    -------
    float
        Mean rating over all ratings.
    """
    totals = recipes[config.STAR_COLUMNS].to_numpy().sum(axis=0)
    return float(np.average(config.STAR_VALUES, weights=totals))


def naive_score(recipes: pd.DataFrame) -> pd.Series:
    """Score recipes by their plain average rating.

    Parameters
    ----------
    recipes : pandas.DataFrame
        Recipe table with a ``mean_rating`` column.

    Returns
    -------
    pandas.Series
        Score of each recipe, aligned with ``recipes``.
    """
    return recipes["mean_rating"].rename("naive_score")


def rank_scores(scores: pd.Series) -> pd.Series:
    """Turn scores into rank positions (1 = best).

    Equal scores share the same (best) position.

    Parameters
    ----------
    scores : pandas.Series
        Score of each recipe, higher is better.

    Returns
    -------
    pandas.Series
        Integer rank of each recipe.
    """
    return scores.rank(ascending=False, method="min").astype("int64")
