"""Plots for the analysis. Every function returns a matplotlib Figure."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.figure import Figure

from src import config

COLORS = {
    "primary": "#257de0",
    "secondary": "#f47230",
    "muted": "#b3b3b3",
    "text": "#191919",
    "text_secondary": "#666666",
    "grid": "#e6e6e6",
}

# buckets for the number of ratings per recipe
RATING_COUNT_BINS = [0, 1, 2, 5, 10, 50, 100, np.inf]
RATING_COUNT_LABELS = ["1", "2", "3-5", "6-10", "11-50", "51-100", "100+"]


def _new_figure(figsize: tuple[float, float] = (8, 4.5)):
    """Create a figure with one axis in the project style.

    Parameters
    ----------
    figsize : tuple of float, optional
        Width and height in inches.

    Returns
    -------
    tuple
        ``(fig, ax)`` - the matplotlib Figure and Axes.
    """
    fig, ax = plt.subplots(figsize=figsize, layout="constrained")
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.spines["bottom"].set_color(COLORS["muted"])
    ax.tick_params(colors=COLORS["text_secondary"], length=0)
    ax.grid(axis="y", color=COLORS["grid"], linewidth=0.8)
    ax.set_axisbelow(True)
    return fig, ax


def _set_titles(ax, title: str, xlabel: str, ylabel: str) -> None:
    """Set the title and axis labels.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axis to label.
    title, xlabel, ylabel : str
        Texts to show.
    """
    ax.set_title(title, loc="left", fontsize=13, color=COLORS["text"],
                 pad=12)
    ax.set_xlabel(xlabel, color=COLORS["text_secondary"])
    ax.set_ylabel(ylabel, color=COLORS["text_secondary"])


def _thousands(value: float, _position: int) -> str:
    """Format an axis value like ``200k`` (tick formatter).

    Parameters
    ----------
    value : float
        Tick value.
    _position : int
        Tick position (required by matplotlib, unused).

    Returns
    -------
    str
        Formatted label.
    """
    return "0" if value == 0 else f"{value / 1000:.0f}k"


def _label_bars(ax, shares: pd.Series) -> None:
    """Write a percentage above each bar.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axis with one bar per value of ``shares``.
    shares : pandas.Series
        Share (0-1) shown above each bar, in bar order.
    """
    for patch, share in zip(ax.patches, shares):
        ax.annotate(
            f"{share:.1%}",
            (patch.get_x() + patch.get_width() / 2, patch.get_height()),
            ha="center", va="bottom", xytext=(0, 3),
            textcoords="offset points", fontsize=9,
            color=COLORS["text_secondary"],
        )


def plot_rating_distribution(interactions: pd.DataFrame) -> Figure:
    """Bar chart of how often each rating value is given.

    Ratings of 0 (reviews without stars) are shown in gray.

    Parameters
    ----------
    interactions : pandas.DataFrame
        Raw interactions with a ``rating`` column.

    Returns
    -------
    matplotlib.figure.Figure
        The plot.
    """
    values = [config.NO_RATING] + config.STAR_VALUES
    counts = interactions["rating"].value_counts().reindex(values,
                                                           fill_value=0)
    labels = ["no stars"] + [f"{star}★" for star in config.STAR_VALUES]
    colors = [COLORS["muted"]] + [COLORS["primary"]] * len(config.STAR_VALUES)

    fig, ax = _new_figure()
    sns.barplot(x=labels, y=counts.to_numpy(), hue=labels, palette=colors,
                legend=False, width=0.7, saturation=1, ax=ax)
    _label_bars(ax, counts / counts.sum())
    ax.yaxis.set_major_formatter(_thousands)
    _set_titles(ax, "Most reviews give 5 stars", "Rating",
                "Number of reviews")
    return fig


def plot_ratings_per_recipe(recipes: pd.DataFrame) -> Figure:
    """Bar chart of recipes grouped by their number of ratings.

    Parameters
    ----------
    recipes : pandas.DataFrame
        Recipe table with an ``n_ratings`` column.

    Returns
    -------
    matplotlib.figure.Figure
        The plot.
    """
    buckets = pd.cut(recipes["n_ratings"], bins=RATING_COUNT_BINS,
                     labels=RATING_COUNT_LABELS)
    counts = buckets.value_counts(sort=False)

    fig, ax = _new_figure()
    sns.barplot(x=counts.index.astype(str), y=counts.to_numpy(),
                color=COLORS["primary"], width=0.7, saturation=1, ax=ax)
    _label_bars(ax, counts / counts.sum())
    ax.yaxis.set_major_formatter(_thousands)
    _set_titles(ax, "Most recipes have only one or two ratings",
                "Number of ratings per recipe", "Number of recipes")
    return fig
