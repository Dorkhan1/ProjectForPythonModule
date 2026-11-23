"""Recipe scoring and ranking methods."""

from itertools import combinations

import numpy as np
import pandas as pd
from scipy import stats

from src import config

# method name -> score column
METHODS = {
    "Naive average": "naive_score",
    "Bayesian average": "bayesian_score",
    "Lower bound": "lower_bound_score",
}


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


def add_scores(
    recipes: pd.DataFrame,
    prior_weight: float | None = None,
    confidence: float = config.DEFAULT_CONFIDENCE,
) -> pd.DataFrame:
    """Add the score and rank of every method to the recipe table.

    Parameters
    ----------
    recipes : pandas.DataFrame
        Recipe table from :func:`src.preprocessing.build_recipe_table`.
    prior_weight : float, optional
        Passed to :func:`bayesian_score`.
    confidence : float, optional
        Passed to :func:`lower_bound_score`.

    Returns
    -------
    pandas.DataFrame
        Copy of the input with ``<method>_score`` and ``<method>_rank``
        columns for each method in :data:`METHODS`.
    """
    recipes = recipes.copy()
    recipes["naive_score"] = naive_score(recipes)
    recipes["bayesian_score"] = bayesian_score(recipes, prior_weight)
    recipes["lower_bound_score"] = lower_bound_score(recipes, confidence)
    for score_col in METHODS.values():
        rank_col = score_col.replace("_score", "_rank")
        recipes[rank_col] = rank_scores(recipes[score_col])
    return recipes


def top_recipes(
    scored: pd.DataFrame, method: str, top_n: int = 10
) -> pd.DataFrame:
    """Return the best recipes according to one method.

    Parameters
    ----------
    scored : pandas.DataFrame
        Recipe table from :func:`add_scores`.
    method : str
        Key of :data:`METHODS`.
    top_n : int, optional
        Number of recipes to return.

    Returns
    -------
    pandas.DataFrame
        ``top_n`` rows sorted from best to worst. Ties are kept in
        table order.

    Raises
    ------
    KeyError
        If ``method`` is not in :data:`METHODS`.
    """
    if method not in METHODS:
        raise KeyError(f"Unknown method: {method}")
    return scored.nlargest(top_n, METHODS[method], keep="first")


def compare_methods(scored: pd.DataFrame, top_n: int = 100) -> pd.DataFrame:
    """Compare every pair of ranking methods.

    Parameters
    ----------
    scored : pandas.DataFrame
        Recipe table from :func:`add_scores`.
    top_n : int, optional
        Size of the top list used for the overlap.

    Returns
    -------
    pandas.DataFrame
        One row per pair with columns ``method_a``, ``method_b``,
        ``spearman`` (rank correlation over all recipes) and
        ``top_overlap`` (share of common recipes in both top lists).
    """
    rows = []
    for method_a, method_b in combinations(METHODS, 2):
        rho = stats.spearmanr(
            scored[METHODS[method_a]], scored[METHODS[method_b]]
        ).statistic
        top_a = set(top_recipes(scored, method_a, top_n)["recipe_id"])
        top_b = set(top_recipes(scored, method_b, top_n)["recipe_id"])
        rows.append({
            "method_a": method_a,
            "method_b": method_b,
            "spearman": rho,
            "top_overlap": len(top_a & top_b) / top_n,
        })
    return pd.DataFrame(rows)


def summarize_top(scored: pd.DataFrame, top_n: int = 100) -> pd.DataFrame:
    """Describe the top recipes of each method.

    Parameters
    ----------
    scored : pandas.DataFrame
        Recipe table from :func:`add_scores`.
    top_n : int, optional
        Size of the top list.

    Returns
    -------
    pandas.DataFrame
        Indexed by method, with the minimum and median number of
        ratings and the mean rating of its top recipes.
    """
    rows = {}
    for method in METHODS:
        top = top_recipes(scored, method, top_n)
        rows[method] = {
            "min_ratings": top["n_ratings"].min(),
            "median_ratings": top["n_ratings"].median(),
            "mean_rating": top["mean_rating"].mean(),
        }
    return pd.DataFrame.from_dict(rows, orient="index")


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
