"""Page: how the ranking changes across tags, time and ingredients."""

import pandas as pd
import streamlit as st

from app_pages.shared import get_scored
from src import config
from src.analysis import (
    filter_recipes,
    lift_correlations,
    summarize_by_column,
    summarize_by_tags,
    summarize_tag_groups,
)
from src.export import to_csv_bytes
from src.plots import plot_category_lift, plot_lift_vs_popularity
from src.ranking import DEFAULT_METHOD, top_recipes

# dimension name -> recipe column (None for tag groups)
COLUMN_DIMENSIONS = {
    "Preparation time": "time_category",
    "Number of ingredients": "ingredient_category",
}


@st.cache_data(show_spinner="Summarizing all tags...")
def _all_tags_summary(_scored: pd.DataFrame) -> pd.DataFrame:
    """Summary of every tag group (cached).

    Parameters
    ----------
    _scored : pandas.DataFrame
        Recipe table with default scores (not hashed).

    Returns
    -------
    pandas.DataFrame
        Table from :func:`src.analysis.summarize_tag_groups`.
    """
    return summarize_tag_groups(_scored)


st.title("Categories")
top_n = config.TOP_N_ANALYSIS
st.markdown(
    f"Which kinds of recipes reach the top? For each category we compare "
    f"its share among the **top {top_n:,}** recipes ({DEFAULT_METHOD} "
    "method) with its share among all recipes. Their ratio is the "
    "**lift**: above 1 the category is over-represented in the top, below "
    "1 it is under-represented."
)

scored = get_scored()

dimension = st.selectbox(
    "Compare categories by",
    [*config.TAG_GROUPS, *COLUMN_DIMENSIONS],
)
if dimension in COLUMN_DIMENSIONS:
    column = COLUMN_DIMENSIONS[dimension]
    summary = summarize_by_column(scored, column)
else:
    column = None
    summary = summarize_by_tags(scored, config.TAG_GROUPS[dimension])

st.pyplot(plot_category_lift(summary, dimension,
                             sort=column is None),
          clear_figure=True)

with st.expander("Summary table"):
    st.dataframe(
        summary,
        column_config={
            "n_recipes": "Recipes",
            "n_ratings": "Ratings",
            "n_in_top": f"In top {top_n:,}",
            "ratings_per_recipe": st.column_config.NumberColumn(
                "Ratings per recipe", format="%.2f"),
            "mean_rating": st.column_config.NumberColumn(
                "Mean rating", format="%.2f"),
            "recipe_share": st.column_config.NumberColumn(
                "Share of recipes", format="percent"),
            "top_share": st.column_config.NumberColumn(
                "Share of top", format="percent"),
            "lift": st.column_config.NumberColumn("Lift", format="%.2f"),
        },
    )
    st.download_button(
        "Download CSV",
        data=to_csv_bytes(summary, index=True),
        file_name=f"summary_{dimension.lower().replace(' ', '_')}.csv",
        mime="text/csv",
        icon=":material/download:",
    )

st.subheader("Best recipes in a category")
category = st.selectbox("Category", summary.index.astype(str))
if column is None:
    in_category = filter_recipes(scored, tag=category)
else:
    in_category = scored[scored[column] == category]
best = top_recipes(in_category, DEFAULT_METHOD, 5)
st.dataframe(
    best[["name", "n_ratings", "mean_rating", "lower_bound_rank"]],
    hide_index=True,
    column_config={
        "name": "Recipe",
        "n_ratings": "Ratings",
        "mean_rating": st.column_config.NumberColumn("Average",
                                                     format="%.2f"),
        "lower_bound_rank": "Overall rank",
    },
)

st.subheader("Is it about quality or popularity?")
tags_summary = _all_tags_summary(scored)
correlations = lift_correlations(tags_summary)
st.pyplot(plot_lift_vs_popularity(tags_summary), clear_figure=True)

col_pop, col_rating = st.columns(2)
col_pop.metric("Lift vs ratings per recipe",
               f"{correlations['ratings_per_recipe']:.2f}",
               help="Spearman correlation over all tags.")
col_rating.metric("Lift vs mean rating",
                  f"{correlations['mean_rating']:.2f}",
                  help="Spearman correlation over all tags.")

low, high = tags_summary["mean_rating"].min(), tags_summary[
    "mean_rating"].max()
st.markdown(
    f"""
The mean rating is almost the same in every tag (from {low:.2f} to
{high:.2f}), but the lift varies a lot. The lift follows the number of
ratings per recipe much more closely than the mean rating. The lower bound
needs evidence, and recipes in popular categories collect more ratings, so
they are easier to prove good.

So the differences between categories mostly show **how popular** a kind
of recipe is on Food.com, not that users like it more. A great recipe in a
less popular category (for example a beverage) needs more time to collect
enough ratings to reach the top.
"""
)
