"""Page: interactive ranking with filters and CSV download."""

import streamlit as st

from app_pages.shared import get_scored
from src import config
from src.analysis import filter_recipes
from src.export import to_csv_bytes
from src.ranking import DEFAULT_METHOD, METHODS, top_recipes

ANY = "Any"

st.title("Top recipes")
scored = get_scored()

with st.sidebar:
    st.header("Filters")
    group = st.selectbox("Tag group", [ANY, *config.TAG_GROUPS])
    tag = None
    if group != ANY:
        tag = st.selectbox("Tag", config.TAG_GROUPS[group])
    times = st.multiselect("Preparation time", config.TIME_LABELS,
                           default=config.TIME_LABELS)
    ingredients = st.multiselect("Number of ingredients",
                                 config.INGREDIENT_LABELS,
                                 default=config.INGREDIENT_LABELS)
    min_ratings = st.number_input("Minimum number of ratings",
                                  min_value=1, max_value=500, value=1)

col_method, col_size = st.columns([2, 1])
method = col_method.segmented_control(
    "Ranking method", list(METHODS), default=DEFAULT_METHOD,
    required=True,
)
top_n = col_size.slider("Number of recipes", 10, 100, 25, step=5)

# all intervals selected = no filter (keeps recipes with unknown time)
filtered = filter_recipes(
    scored,
    tag=tag,
    time_categories=None if len(times) == len(config.TIME_LABELS)
    else times,
    ingredient_categories=None
    if len(ingredients) == len(config.INGREDIENT_LABELS) else ingredients,
    min_ratings=min_ratings,
)

st.caption(f"{len(filtered):,} of {len(scored):,} recipes match the "
           "filters.")
if filtered.empty:
    st.warning("No recipes match the filters. Try removing some of them.")
    st.stop()

score_col = METHODS[method]
rank_col = score_col.replace("_score", "_rank")
top = top_recipes(filtered, method, top_n)
table = top.assign(position=range(1, len(top) + 1))[
    ["position", "name", "n_ratings", "mean_rating", score_col, rank_col,
     "minutes", "n_ingredients", "tags"]
]

st.dataframe(
    table,
    hide_index=True,
    column_config={
        "position": st.column_config.NumberColumn("#", width="small"),
        "name": st.column_config.TextColumn("Recipe", width="large"),
        "n_ratings": "Ratings",
        "mean_rating": st.column_config.NumberColumn("Average",
                                                     format="%.2f"),
        score_col: st.column_config.NumberColumn("Score", format="%.3f"),
        rank_col: st.column_config.NumberColumn(
            "Overall rank", help="Rank among all recipes, without filters."
        ),
        "minutes": st.column_config.NumberColumn("Minutes", format="%.0f"),
        "n_ingredients": "Ingredients",
        "tags": st.column_config.ListColumn("Tags"),
    },
)

st.download_button(
    "Download CSV",
    data=to_csv_bytes(table.drop(columns="tags")),
    file_name=f"top_{top_n}_{score_col.removesuffix('_score')}.csv",
    mime="text/csv",
    icon=":material/download:",
)
st.caption("Bayesian average and lower bound use the default parameters "
           "(see the Ranking methods page).")
