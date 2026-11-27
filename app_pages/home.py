"""Page: project overview and key numbers."""

import streamlit as st

from app_pages.shared import get_data, get_scored
from src import config
from src.analysis import dataset_facts

st.title("Food.com Recipe Ranking")
st.markdown(
    "Food.com users rate recipes from 1 to 5 stars. This app ranks the "
    "recipes by how much users like them. A plain average rating is not "
    "enough for this, because most recipes have only one or two ratings. "
    "The ranking used here also takes into account **how many** ratings a "
    "recipe has and **how much they agree**."
)

recipes, interactions = get_data()
facts = dataset_facts(recipes, interactions)

st.subheader("Key numbers")
cols = st.columns(4)
cols[0].metric("Rated recipes", f"{facts['n_recipes']:,}")
cols[1].metric("Star ratings", f"{facts['n_ratings']:,}",
               help=f"{facts['n_unrated_reviews']:,} reviews without stars "
                    "were left out.")
cols[2].metric("Recipes with a single rating",
               f"{facts['share_one_rating']:.0%}")
cols[3].metric("Recipes with a perfect 5.0 average",
               f"{facts['n_perfect_mean']:,}",
               help="With the plain average they all share first place.")

st.subheader("The problem in one example")
scored = get_scored()
recipe = scored.set_index("recipe_id").loc[config.HIGHLIGHT_RECIPE_ID]
st.markdown(
    f"**{recipe['name']}** has **{recipe['n_ratings']:,} ratings** with an "
    f"average of **{recipe['mean_rating']:.2f}**. Here is its position "
    "under each ranking method:"
)
cols = st.columns(3)
cols[0].metric("Naive average", f"#{recipe['naive_rank']:,}",
               help="Behind every recipe with a perfect score, "
                    "even those with one rating.")
cols[1].metric("Bayesian average", f"#{recipe['bayesian_rank']:,}")
cols[2].metric("Lower bound (main method)",
               f"#{recipe['lower_bound_rank']:,}")

st.subheader("What is in this app")
st.page_link("app_pages/data.py", label="Data",
             icon=":material/database:",
             help="The dataset, the cleaning steps and the problem.")
st.page_link("app_pages/methods.py", label="Ranking methods",
             icon=":material/functions:",
             help="How the three methods work and how they compare.")
st.page_link("app_pages/top_recipes.py", label="Top recipes",
             icon=":material/trophy:",
             help="Interactive ranking with filters and CSV download.")
st.page_link("app_pages/categories.py", label="Categories",
             icon=":material/category:",
             help="How the ranking changes across tags, time and "
                  "ingredients.")

st.caption(
    f"Data: [Food.com Recipes and Interactions]({config.DATASET_URL}) "
    "(Kaggle)."
)
