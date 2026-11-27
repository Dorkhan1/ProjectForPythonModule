"""Ranking analysis across categories (tags, time, ingredients)."""

import pandas as pd
from scipy import stats

from src import config
from src.ranking import DEFAULT_METHOD, global_mean_rating, top_recipes


def filter_recipes(
    scored: pd.DataFrame,
    tag: str | None = None,
    time_categories: list[str] | None = None,
    ingredient_categories: list[str] | None = None,
    min_ratings: int = 1,
) -> pd.DataFrame:
    """Select recipes matching all given filters.

    Parameters
    ----------
    scored : pandas.DataFrame
        Recipe table from :func:`src.ranking.add_scores`.
    tag : str, optional
        Keep only recipes with this tag. None keeps all.
    time_categories : list of str, optional
        Allowed values of ``time_category``. None keeps all, including
        recipes with unknown time.
    ingredient_categories : list of str, optional
        Allowed values of ``ingredient_category``. None keeps all.
    min_ratings : int, optional
        Minimum number of ratings.

    Returns
    -------
    pandas.DataFrame
        Matching rows of ``scored``.
    """
    mask = scored["n_ratings"] >= min_ratings
    if tag is not None:
        mask &= scored["tags"].map(lambda tags: tag in tags)
    if time_categories is not None:
        mask &= scored["time_category"].isin(time_categories)
    if ingredient_categories is not None:
        mask &= scored["ingredient_category"].isin(ingredient_categories)
    return scored[mask]


def _mark_top(scored: pd.DataFrame, method: str, top_n: int) -> pd.DataFrame:
    """Add a boolean ``in_top`` column for the overall top recipes.

    Parameters
    ----------
    scored : pandas.DataFrame
        Recipe table from :func:`src.ranking.add_scores`.
    method : str
        Key of :data:`src.ranking.METHODS`.
    top_n : int
        Size of the overall top list.

    Returns
    -------
    pandas.DataFrame
        Copy of the input with an ``in_top`` column.
    """
    top_ids = top_recipes(scored, method, top_n)["recipe_id"]
    scored = scored.copy()
    scored["in_top"] = scored["recipe_id"].isin(top_ids)
    return scored


def _summarize(
    data: pd.DataFrame,
    by: str,
    n_recipes_total: int,
    top_n: int,
) -> pd.DataFrame:
    """Aggregate ranking statistics for each category.

    Parameters
    ----------
    data : pandas.DataFrame
        Recipes with an ``in_top`` column and a category column ``by``.
        A recipe may appear in several categories.
    by : str
        Name of the category column.
    n_recipes_total : int
        Number of recipes in the whole table.
    top_n : int
        Size of the overall top list.

    Returns
    -------
    pandas.DataFrame
        Indexed by category, see :func:`summarize_by_column`.
    """
    summary = data.groupby(by, observed=True).agg(
        n_recipes=("recipe_id", "size"),
        n_ratings=("n_ratings", "sum"),
        n_in_top=("in_top", "sum"),
        **{col: (col, "sum") for col in config.STAR_COLUMNS},
    )
    summary["ratings_per_recipe"] = summary["n_ratings"] / summary[
        "n_recipes"
    ]
    star_totals = summary[config.STAR_COLUMNS].to_numpy()
    summary["mean_rating"] = star_totals @ config.STAR_VALUES / summary[
        "n_ratings"
    ]
    summary["recipe_share"] = summary["n_recipes"] / n_recipes_total
    summary["top_share"] = summary["n_in_top"] / top_n
    summary["lift"] = summary["top_share"] / summary["recipe_share"]
    return summary.drop(columns=config.STAR_COLUMNS)


def summarize_by_column(
    scored: pd.DataFrame,
    column: str,
    method: str = DEFAULT_METHOD,
    top_n: int = config.TOP_N_ANALYSIS,
) -> pd.DataFrame:
    """Summarize the ranking for each value of a category column.

    Parameters
    ----------
    scored : pandas.DataFrame
        Recipe table from :func:`src.ranking.add_scores`.
    column : str
        Category column, e.g. ``time_category``.
    method : str, optional
        Key of :data:`src.ranking.METHODS`.
    top_n : int, optional
        Size of the overall top list.

    Returns
    -------
    pandas.DataFrame
        Indexed by category, with columns:

        - ``n_recipes`` : number of recipes
        - ``n_ratings`` : total number of ratings
        - ``n_in_top`` : recipes in the overall top ``top_n``
        - ``ratings_per_recipe`` : average number of ratings per recipe
        - ``mean_rating`` : average of all ratings in the category
        - ``recipe_share`` : share of all recipes
        - ``top_share`` : share of the overall top
        - ``lift`` : ``top_share / recipe_share`` (above 1 means the
          category is over-represented in the top)
    """
    data = _mark_top(scored, method, top_n)
    return _summarize(data, column, len(scored), top_n)


def summarize_by_tags(
    scored: pd.DataFrame,
    tags: list[str],
    method: str = DEFAULT_METHOD,
    top_n: int = config.TOP_N_ANALYSIS,
) -> pd.DataFrame:
    """Summarize the ranking for each tag of a list.

    A recipe with several of the tags is counted in each of them.

    Parameters
    ----------
    scored : pandas.DataFrame
        Recipe table from :func:`src.ranking.add_scores`.
    tags : list of str
        Tags to compare, e.g. one group of :data:`src.config.TAG_GROUPS`.
    method : str, optional
        Key of :data:`src.ranking.METHODS`.
    top_n : int, optional
        Size of the overall top list.

    Returns
    -------
    pandas.DataFrame
        Indexed by tag, same columns as :func:`summarize_by_column`.
    """
    columns = ["recipe_id", "tags", "n_ratings", "in_top",
               *config.STAR_COLUMNS]
    data = _mark_top(scored, method, top_n)[columns]
    data = data.explode("tags").rename(columns={"tags": "tag"})
    data = data[data["tag"].isin(tags)]
    return _summarize(data, "tag", len(scored), top_n)


def summarize_tag_groups(
    scored: pd.DataFrame,
    method: str = DEFAULT_METHOD,
    top_n: int = config.TOP_N_ANALYSIS,
) -> pd.DataFrame:
    """Summarize every tag of :data:`src.config.TAG_GROUPS` in one table.

    Parameters
    ----------
    scored : pandas.DataFrame
        Recipe table from :func:`src.ranking.add_scores`.
    method : str, optional
        Key of :data:`src.ranking.METHODS`.
    top_n : int, optional
        Size of the overall top list.

    Returns
    -------
    pandas.DataFrame
        Same columns as :func:`summarize_by_tags` plus ``group``,
        indexed by tag.
    """
    summaries = [
        summarize_by_tags(scored, tags, method, top_n).assign(group=group)
        for group, tags in config.TAG_GROUPS.items()
    ]
    return pd.concat(summaries)


def lift_correlations(summary: pd.DataFrame) -> pd.Series:
    """Spearman correlation of lift with popularity and with rating.

    Parameters
    ----------
    summary : pandas.DataFrame
        Category summary with ``lift``, ``ratings_per_recipe`` and
        ``mean_rating`` columns.

    Returns
    -------
    pandas.Series
        Correlation of ``lift`` with ``ratings_per_recipe`` and with
        ``mean_rating``.
    """
    return pd.Series({
        col: stats.spearmanr(summary["lift"], summary[col]).statistic
        for col in ["ratings_per_recipe", "mean_rating"]
    })


def dataset_facts(
    recipes: pd.DataFrame, interactions: pd.DataFrame
) -> dict[str, float]:
    """Key numbers about the data and the naive ranking.

    Parameters
    ----------
    recipes : pandas.DataFrame
        Table from :func:`src.preprocessing.build_recipe_table`.
    interactions : pandas.DataFrame
        Raw interactions from :func:`src.data_loader.load_interactions`.

    Returns
    -------
    dict of str to float
        ``n_reviews`` (all interactions), ``n_recipes_reviewed`` (recipes
        with any review), ``n_recipes`` (recipes with star ratings),
        ``n_ratings``, ``n_unrated_reviews``, ``n_invalid_minutes``,
        ``n_without_tags``, ``share_one_rating`` (share of recipes with
        a single rating), ``n_perfect_mean`` (recipes with an average of
        exactly 5) and ``global_mean``.
    """
    top_star = max(config.STAR_VALUES)
    return {
        "n_reviews": len(interactions),
        "n_recipes_reviewed": int(interactions["recipe_id"].nunique()),
        "n_recipes": len(recipes),
        "n_ratings": int(recipes["n_ratings"].sum()),
        "n_unrated_reviews": int(
            (interactions["rating"] == config.NO_RATING).sum()
        ),
        "n_invalid_minutes": int(recipes["minutes"].isna().sum()),
        "n_without_tags": int((recipes["tags"].map(len) == 0).sum()),
        "share_one_rating": float((recipes["n_ratings"] == 1).mean()),
        "n_perfect_mean": int((recipes["mean_rating"] == top_star).sum()),
        "global_mean": global_mean_rating(recipes),
    }
