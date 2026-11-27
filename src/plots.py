"""Plots for the analysis. Every function returns a matplotlib Figure."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.figure import Figure

from src import config
from src.ranking import METHODS, top_recipes

COLORS = {
    "primary": "#257de0",
    "secondary": "#f47230",
    "muted": "#b3b3b3",
    "text": "#191919",
    "text_secondary": "#666666",
    "grid": "#e6e6e6",
}
# categorical colors, used in this order
SERIES_COLORS = ["#257de0", "#f47230", "#00b894", "#f2a600"]

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


def plot_score_vs_ratings(
    scored: pd.DataFrame,
    methods: tuple[str, str] = ("Naive average", "Lower bound"),
) -> Figure:
    """Scatter plots of score against number of ratings, one per method.

    Parameters
    ----------
    scored : pandas.DataFrame
        Recipe table from :func:`src.ranking.add_scores`.
    methods : tuple of str, optional
        Two keys of :data:`src.ranking.METHODS`.

    Returns
    -------
    matplotlib.figure.Figure
        The plot.
    """
    fig, axes = plt.subplots(1, len(methods), figsize=(11, 4.5),
                             sharey=True, layout="constrained")
    for ax, method in zip(axes, methods):
        ax.scatter(scored["n_ratings"], scored[METHODS[method]], s=4,
                   alpha=0.08, color=COLORS["primary"], linewidths=0,
                   rasterized=True)
        ax.set_xscale("log")
        ax.spines[["top", "right"]].set_visible(False)
        ax.spines[["left", "bottom"]].set_color(COLORS["muted"])
        ax.tick_params(colors=COLORS["text_secondary"])
        ax.grid(color=COLORS["grid"], linewidth=0.8)
        ax.set_axisbelow(True)
        _set_titles(ax, method, "Number of ratings (log scale)",
                    "Score" if ax is axes[0] else "")
    fig.suptitle("Few ratings: the naive score is extreme, the lower "
                 "bound is cautious", x=0.01, ha="left", fontsize=14,
                 color=COLORS["text"])
    return fig


def plot_top_profile(
    scored: pd.DataFrame, top_n: int = config.TOP_N_EXPORT
) -> Figure:
    """Box plots of the number of ratings of each method's top recipes.

    Parameters
    ----------
    scored : pandas.DataFrame
        Recipe table from :func:`src.ranking.add_scores`.
    top_n : int, optional
        Size of each top list.

    Returns
    -------
    matplotlib.figure.Figure
        The plot.
    """
    tops = pd.concat(
        top_recipes(scored, method, top_n).assign(method=method)
        for method in METHODS
    )

    fig, ax = _new_figure()
    sns.boxplot(data=tops, x="n_ratings", y="method", log_scale=True,
                color=COLORS["primary"], saturation=1, width=0.5,
                fliersize=3, linecolor=COLORS["text"], linewidth=1, ax=ax)
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", color=COLORS["grid"], linewidth=0.8)
    for i, method in enumerate(METHODS):
        median = tops.loc[tops["method"] == method, "n_ratings"].median()
        ax.annotate(f"median {median:.0f}", (median, i - 0.27),
                    ha="center", va="bottom", fontsize=9,
                    color=COLORS["text_secondary"])
    _set_titles(ax, f"How many ratings the top {top_n} recipes have",
                "Number of ratings (log scale)", "")
    return fig


def _spread_log_positions(
    values: np.ndarray, min_gap: float = 0.18
) -> np.ndarray:
    """Move label positions apart so they do not overlap on a log axis.

    Parameters
    ----------
    values : numpy.ndarray
        Positive positions of the labels.
    min_gap : float, optional
        Minimum distance between labels, in powers of ten.

    Returns
    -------
    numpy.ndarray
        New positions, in the same order as ``values``.
    """
    logs = np.log10(values)
    order = np.argsort(logs)
    spread = logs[order].copy()
    for i in range(1, len(spread)):
        spread[i] = max(spread[i], spread[i - 1] + min_gap)
    result = np.empty_like(spread)
    result[order] = spread
    return 10**result


def plot_rank_changes(
    scored: pd.DataFrame,
    recipe_ids: list[int] = config.EXAMPLE_RECIPE_IDS,
) -> Figure:
    """Slope chart of the rank of some recipes under each method.

    Parameters
    ----------
    scored : pandas.DataFrame
        Recipe table from :func:`src.ranking.add_scores`.
    recipe_ids : list of int, optional
        Recipes to show (at most 4).

    Returns
    -------
    matplotlib.figure.Figure
        The plot.
    """
    rank_cols = [col.replace("_score", "_rank") for col in METHODS.values()]
    examples = scored.set_index("recipe_id").loc[recipe_ids]
    x = np.arange(len(METHODS))

    fig, ax = _new_figure(figsize=(9, 5))
    for color, (_, recipe) in zip(SERIES_COLORS, examples.iterrows()):
        n = recipe["n_ratings"]
        label = f"{recipe['name']} ({n} rating{'s' if n != 1 else ''})"
        ax.plot(x, recipe[rank_cols].to_numpy(dtype=float), color=color,
                linewidth=2, marker="o", markersize=8, label=label)

    for xi, col in zip(x, rank_cols):
        ranks = examples[col].to_numpy(dtype=float)
        label_y = _spread_log_positions(ranks)
        for rank, y in zip(ranks, label_y):
            ax.annotate(f"{rank:,.0f}", (xi, rank), xytext=(xi + 0.05, y),
                        textcoords="data", va="center", fontsize=9,
                        color=COLORS["text_secondary"])

    ax.set_yscale("log")
    ax.set_ylim(examples[rank_cols].to_numpy().max() * 4, 0.7)
    ax.yaxis.set_major_formatter(lambda y, _: f"{y:,.0f}")
    ax.yaxis.set_minor_locator(plt.NullLocator())
    ax.set_xticks(x, list(METHODS))
    ax.set_xlim(-0.3, len(METHODS) - 0.5)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=2,
              frameon=False, fontsize=9)
    _set_titles(ax, "Rank of example recipes under each method "
                "(1 = best)", "", "Rank (log scale)")
    return fig


def plot_category_lift(
    summary: pd.DataFrame, title: str, sort: bool = True
) -> Figure:
    """Horizontal bars of the lift of each category.

    Bars above 1 (over-represented in the top) are blue, the others gray.

    Parameters
    ----------
    summary : pandas.DataFrame
        Category summary from :mod:`src.analysis` with a ``lift`` column.
    title : str
        Plot title.
    sort : bool, optional
        Sort categories by lift. Use False for ordered categories
        such as time intervals.

    Returns
    -------
    matplotlib.figure.Figure
        The plot.
    """
    lift = summary["lift"]
    if sort:
        lift = lift.sort_values(ascending=False)
    labels = lift.index.astype(str)
    colors = [COLORS["primary"] if value >= 1 else COLORS["muted"]
              for value in lift]

    fig, ax = _new_figure(figsize=(8, 0.4 * len(lift) + 1.5))
    sns.barplot(x=lift.to_numpy(), y=labels, hue=labels, palette=colors,
                legend=False, saturation=1, width=0.7, ax=ax)
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", color=COLORS["grid"], linewidth=0.8)
    ax.axvline(1, color=COLORS["text"], linewidth=1, linestyle="--")
    for patch, value in zip(ax.patches, lift):
        # labels of short bars go right of the reference line
        x = max(value, 1)
        y = patch.get_y() + patch.get_height() / 2
        ax.annotate(f"{value:.2f}×", (x, y), xytext=(4, 0),
                    textcoords="offset points", va="center", fontsize=9,
                    color=COLORS["text_secondary"])
    _set_titles(ax, title, "Lift (share in top / share in all recipes)", "")
    return fig


def plot_lift_vs_popularity(tag_summary: pd.DataFrame) -> Figure:
    """Scatter plot of lift against ratings per recipe for all tags.

    Parameters
    ----------
    tag_summary : pandas.DataFrame
        Table from :func:`src.analysis.summarize_tag_groups`.

    Returns
    -------
    matplotlib.figure.Figure
        The plot.
    """
    fig, ax = _new_figure(figsize=(9, 5.5))
    ax.grid(axis="x", color=COLORS["grid"], linewidth=0.8)
    groups = tag_summary["group"].unique()
    for color, group in zip(SERIES_COLORS, groups):
        part = tag_summary[tag_summary["group"] == group]
        ax.scatter(part["ratings_per_recipe"], part["lift"], s=60,
                   color=color, edgecolors="white", linewidths=1.5,
                   label=group, zorder=3)

    extremes = pd.concat([tag_summary.nlargest(3, "lift"),
                          tag_summary.nsmallest(3, "lift")])
    for tag, row in extremes.iterrows():
        ax.annotate(tag, (row["ratings_per_recipe"], row["lift"]),
                    xytext=(7, 0), textcoords="offset points", va="center",
                    fontsize=9, color=COLORS["text_secondary"])

    ax.axhline(1, color=COLORS["text"], linewidth=1, linestyle="--")
    ax.legend(frameon=False, fontsize=9, loc="upper left")
    _set_titles(ax, "Tags with more ratings per recipe reach the top "
                "more often", "Average ratings per recipe", "Lift")
    return fig
