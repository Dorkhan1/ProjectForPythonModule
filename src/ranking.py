"""Recipe scoring and ranking methods."""

import numpy as np
import pandas as pd
from scipy import stats

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


def default_prior_weight(recipes: pd.DataFrame) -> float:
    """Default prior weight for :func:`bayesian_score`.

    Equal to the :data:`src.config.PRIOR_WEIGHT_QUANTILE` quantile of
    the number of ratings per recipe.

    Parameters
    ----------
    recipes : pandas.DataFrame
        Recipe table with an ``n_ratings`` column.

    Returns
    -------
    float
        Prior weight (number of "virtual" ratings).
    """
    return float(
        np.quantile(recipes["n_ratings"], config.PRIOR_WEIGHT_QUANTILE)
    )


def bayesian_score(
    recipes: pd.DataFrame,
    prior_weight: float | None = None,
    prior_mean: float | None = None,
) -> pd.Series:
    """Score recipes by the Bayesian average rating.

    Each recipe gets ``prior_weight`` virtual ratings equal to
    ``prior_mean``::

        score = (prior_weight * prior_mean + n * mean) / (prior_weight + n)

    Parameters
    ----------
    recipes : pandas.DataFrame
        Recipe table with ``n_ratings`` and ``mean_rating`` columns
        (and star columns if ``prior_mean`` is not given).
    prior_weight : float, optional
        Number of virtual ratings. Defaults to
        :func:`default_prior_weight`. With 0 the score equals the
        naive average.
    prior_mean : float, optional
        Value of the virtual ratings. Defaults to
        :func:`global_mean_rating`.

    Returns
    -------
    pandas.Series
        Score of each recipe, aligned with ``recipes``.

    Raises
    ------
    ValueError
        If ``prior_weight`` is negative.
    """
    if prior_weight is None:
        prior_weight = default_prior_weight(recipes)
    if prior_weight < 0:
        raise ValueError("prior_weight must be non-negative")
    if prior_mean is None:
        prior_mean = global_mean_rating(recipes)

    n = recipes["n_ratings"]
    rating_sum = recipes["mean_rating"] * n
    score = (prior_weight * prior_mean + rating_sum) / (prior_weight + n)
    return score.rename("bayesian_score")


def lower_bound_score(
    recipes: pd.DataFrame,
    confidence: float = config.DEFAULT_CONFIDENCE,
) -> pd.Series:
    """Score recipes by the lower bound of a confidence interval.

    One virtual rating of each star value is added to every recipe
    (Dirichlet prior), then the score is
    ``mean - z * sqrt(variance / (N + 1))``, where ``N`` is the number
    of real and virtual ratings. Based on Evan Miller, "Ranking Items
    With Star Ratings".

    Parameters
    ----------
    recipes : pandas.DataFrame
        Recipe table with ``stars_1`` ... ``stars_5`` columns.
    confidence : float, optional
        Confidence level of the two-sided interval, between 0 and 1.
        Defaults to :data:`src.config.DEFAULT_CONFIDENCE`.

    Returns
    -------
    pandas.Series
        Score of each recipe, aligned with ``recipes``.

    Raises
    ------
    ValueError
        If ``confidence`` is not strictly between 0 and 1.
    """
    if not 0 < confidence < 1:
        raise ValueError("confidence must be between 0 and 1")

    stars = np.array(config.STAR_VALUES, dtype="float64")
    counts = recipes[config.STAR_COLUMNS].to_numpy() + 1
    total = counts.sum(axis=1)
    probs = counts / total[:, np.newaxis]

    mean = probs @ stars
    variance = probs @ stars**2 - mean**2
    std_error = np.sqrt(variance / (total + 1))
    z = stats.norm.ppf(1 - (1 - confidence) / 2)

    return pd.Series(
        mean - z * std_error, index=recipes.index, name="lower_bound_score"
    )


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
