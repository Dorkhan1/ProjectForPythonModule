"""Page: the dataset, the cleaning steps and the problem."""

import streamlit as st

from app_pages.shared import get_data
from src import config
from src.analysis import dataset_facts
from src.plots import plot_rating_distribution, plot_ratings_per_recipe

PREVIEW_COLUMNS = ["recipe_id", "name", "n_ratings", "mean_rating",
                   *config.STAR_COLUMNS, "minutes", "time_category",
                   "n_ingredients", "ingredient_category", "tags"]

st.title("Data")
recipes, interactions = get_data()
facts = dataset_facts(recipes, interactions)

st.markdown(
    f"The [Food.com Recipes and Interactions]({config.DATASET_URL}) "
    "dataset has two files:"
)
col_recipes, col_reviews = st.columns(2)
with col_recipes:
    st.markdown(
        "**RAW_recipes.csv** - one row per recipe: name, preparation "
        "time, number of ingredients and steps, tags."
    )
with col_reviews:
    st.markdown(
        f"**RAW_interactions.csv** - one row per review "
        f"({facts['n_reviews']:,} in total): user, recipe, date and a "
        "rating from 0 to 5."
    )
st.markdown(
    "A recipe can have many reviews. The reviews are aggregated per "
    "recipe and then joined with the recipe information."
)

st.subheader("Cleaning steps")
st.markdown(
    f"""
1. **Reviews without stars removed.** A rating of 0 means the user wrote
   a comment (often a question) without giving stars:
   {facts['n_unrated_reviews']:,} such reviews were dropped.
2. **Recipes without star ratings removed.**
   {facts['n_recipes_reviewed'] - facts['n_recipes']:,} recipes had only
   comments, so there is nothing to rank them by.
3. **Invalid preparation times set to missing.** Times of 0 minutes or
   more than one year ({facts['n_invalid_minutes']:,} recipes) are not
   used in the time analysis.
4. **Tags parsed.** Tags are stored as text like `"['easy', 'dinner']"`
   and were converted to lists; empty tags were removed
   ({facts['n_without_tags']:,} recipes have no tags left).
5. **Categories added.** Preparation time and number of ingredients were
   grouped into intervals.
"""
)

st.subheader("Ratings are very positive")
st.pyplot(plot_rating_distribution(interactions), clear_figure=True)
st.markdown(
    f"Almost three quarters of all reviews give 5 stars and the average "
    f"rating on the site is **{facts['global_mean']:.2f}**. Differences "
    "between good and very good recipes are therefore small."
)

st.subheader("Most recipes have very few ratings")
st.pyplot(plot_ratings_per_recipe(recipes), clear_figure=True)
st.markdown(
    f"**{facts['share_one_rating']:.0%}** of recipes have a single rating. "
    "Their average is just that one rating, so "
    f"**{facts['n_perfect_mean']:,} recipes** have a perfect 5.0 average "
    "and would all share first place in a naive ranking. The "
    "[ranking methods](/methods) page shows how to fix this."
)

st.subheader("Prepared table")
st.caption(f"First rows of the final table ({len(recipes):,} recipes).")
st.dataframe(recipes[PREVIEW_COLUMNS].head(100), hide_index=True)
