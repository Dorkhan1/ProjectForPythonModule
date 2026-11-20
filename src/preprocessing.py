"""Cleaning and preparation of recipes and ratings."""

import ast

import pandas as pd

from src import config


def remove_unrated(interactions: pd.DataFrame) -> pd.DataFrame:
    """Remove reviews that have no star rating.

    Parameters
    ----------
    interactions : pandas.DataFrame
        Raw interactions with a ``rating`` column.

    Returns
    -------
    pandas.DataFrame
        Copy of the input containing only ratings from 1 to 5.
    """
    return interactions[interactions["rating"] != config.NO_RATING].copy()


def aggregate_ratings(ratings: pd.DataFrame) -> pd.DataFrame:
    """Compute rating statistics for each recipe.

    Parameters
    ----------
    ratings : pandas.DataFrame
        Ratings with ``recipe_id`` and ``rating`` columns, without zeros
        (see :func:`remove_unrated`).

    Returns
    -------
    pandas.DataFrame
        Indexed by ``recipe_id``, with columns:

        - ``n_ratings`` : number of ratings
        - ``mean_rating`` : average rating
        - ``stars_1`` ... ``stars_5`` : number of ratings of each value
    """
    star_counts = (
        ratings.groupby(["recipe_id", "rating"])
        .size()
        .unstack(fill_value=0)
        .reindex(columns=config.STAR_VALUES, fill_value=0)
    )
    star_counts.columns = config.STAR_COLUMNS

    stats = ratings.groupby("recipe_id")["rating"].agg(
        n_ratings="size", mean_rating="mean"
    )
    return stats.join(star_counts)


def clean_minutes(recipes: pd.DataFrame) -> pd.DataFrame:
    """Replace invalid preparation times with NaN.

    A time is invalid if it equals 0 or exceeds
    :data:`src.config.MAX_MINUTES`.

    Parameters
    ----------
    recipes : pandas.DataFrame
        Recipes with a ``minutes`` column.

    Returns
    -------
    pandas.DataFrame
        Copy of the input where ``minutes`` is float with NaN for
        invalid values.
    """
    recipes = recipes.copy()
    minutes = recipes["minutes"].astype("float64")
    invalid = (minutes <= 0) | (minutes > config.MAX_MINUTES)
    recipes["minutes"] = minutes.mask(invalid)
    return recipes


def add_categories(recipes: pd.DataFrame) -> pd.DataFrame:
    """Add binned preparation time and number of ingredients.

    Parameters
    ----------
    recipes : pandas.DataFrame
        Recipes with ``minutes`` (cleaned, see :func:`clean_minutes`)
        and ``n_ingredients`` columns.

    Returns
    -------
    pandas.DataFrame
        Copy of the input with two new ordered categorical columns:
        ``time_category`` and ``ingredient_category``.
    """
    recipes = recipes.copy()
    recipes["time_category"] = pd.cut(
        recipes["minutes"], bins=config.TIME_BINS, labels=config.TIME_LABELS
    )
    recipes["ingredient_category"] = pd.cut(
        recipes["n_ingredients"],
        bins=config.INGREDIENT_BINS,
        labels=config.INGREDIENT_LABELS,
    )
    return recipes


def _parse_tag_list(text: str) -> list[str]:
    """Convert a stringified list like ``"['a', 'b']"`` into a list.

    Parameters
    ----------
    text : str
        Tags as stored in the CSV file.

    Returns
    -------
    list of str
        Non-empty tags. Empty list if the text cannot be parsed.
    """
    try:
        tags = ast.literal_eval(text)
    except (ValueError, SyntaxError):
        return []
    return [tag for tag in tags if tag]


def parse_tags(recipes: pd.DataFrame) -> pd.DataFrame:
    """Convert the ``tags`` column from strings to lists of tags.

    Parameters
    ----------
    recipes : pandas.DataFrame
        Recipes with a ``tags`` column of stringified lists.

    Returns
    -------
    pandas.DataFrame
        Copy of the input where ``tags`` contains Python lists.
    """
    recipes = recipes.copy()
    recipes["tags"] = recipes["tags"].fillna("[]").map(_parse_tag_list)
    return recipes


def build_recipe_table(
    recipes: pd.DataFrame, interactions: pd.DataFrame
) -> pd.DataFrame:
    """Build the final table: one row per rated recipe with its stats.

    Recipes without any star rating are dropped.

    Parameters
    ----------
    recipes : pandas.DataFrame
        Raw recipes from :func:`src.data_loader.load_recipes`.
    interactions : pandas.DataFrame
        Raw interactions from :func:`src.data_loader.load_interactions`.

    Returns
    -------
    pandas.DataFrame
        Recipe columns (``recipe_id``, ``name``, ``minutes``, ``tags``,
        ``n_steps``, ``n_ingredients``, ``submitted``, ``time_category``,
        ``ingredient_category``) joined with the rating columns from
        :func:`aggregate_ratings`.
    """
    ratings = aggregate_ratings(remove_unrated(interactions))

    recipes = parse_tags(add_categories(clean_minutes(recipes)))
    recipes = recipes.rename(columns={"id": "recipe_id"})
    recipes["name"] = recipes["name"].str.replace(r"\s+", " ", regex=True)

    return recipes.merge(
        ratings,
        left_on="recipe_id",
        right_index=True,
        how="inner",
        validate="one_to_one",
    ).reset_index(drop=True)
