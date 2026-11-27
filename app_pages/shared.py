"""Data loading shared by all pages of the Streamlit app."""

import pandas as pd
import streamlit as st

from src import config
from src.data_loader import DataLoadError, load_interactions, load_recipes
from src.preprocessing import build_recipe_table
from src.ranking import add_scores


@st.cache_data(show_spinner="Loading and preparing the data...")
def _load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load the CSV files and build the recipe table (cached).

    Returns
    -------
    tuple of pandas.DataFrame
        ``(recipes, interactions)``: the table from
        :func:`src.preprocessing.build_recipe_table` and the raw
        interactions.

    Raises
    ------
    DataLoadError
        If the data files cannot be loaded.
    """
    interactions = load_interactions()
    recipes = build_recipe_table(load_recipes(), interactions)
    return recipes, interactions


def get_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return the data, or show an error and stop the page.

    Returns
    -------
    tuple of pandas.DataFrame
        ``(recipes, interactions)``, see :func:`_load_data`.
    """
    try:
        return _load_data()
    except DataLoadError as exc:
        st.error(f"**The data could not be loaded.**\n\n{exc}")
        st.stop()


@st.cache_data(show_spinner="Ranking the recipes...")
def _score(
    _recipes: pd.DataFrame,
    prior_weight: float | None,
    confidence: float,
) -> pd.DataFrame:
    """Add scores and ranks of all methods (cached per parameters).

    Parameters
    ----------
    _recipes : pandas.DataFrame
        Recipe table (not hashed by Streamlit, it never changes).
    prior_weight : float or None
        Passed to :func:`src.ranking.add_scores`.
    confidence : float
        Passed to :func:`src.ranking.add_scores`.

    Returns
    -------
    pandas.DataFrame
        Recipe table with score and rank columns.
    """
    return add_scores(_recipes, prior_weight, confidence)


def get_scored(
    prior_weight: float | None = None,
    confidence: float = config.DEFAULT_CONFIDENCE,
) -> pd.DataFrame:
    """Return the recipe table with scores and ranks.

    Parameters
    ----------
    prior_weight : float, optional
        Prior weight of the Bayesian average. Default is data-driven.
    confidence : float, optional
        Confidence level of the lower bound.

    Returns
    -------
    pandas.DataFrame
        Recipe table from :func:`src.ranking.add_scores`.
    """
    recipes, _ = get_data()
    return _score(recipes, prior_weight, confidence)
