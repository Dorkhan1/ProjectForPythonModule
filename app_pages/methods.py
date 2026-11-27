"""Page: the three ranking methods and how they compare."""

import streamlit as st

from app_pages.shared import get_data, get_scored
from src import config
from src.ranking import (
    METHODS,
    compare_methods,
    default_prior_weight,
    global_mean_rating,
    summarize_top,
)
from src.plots import (
    plot_rank_changes,
    plot_score_vs_ratings,
    plot_top_profile,
)

CONFIDENCE_LEVELS = [0.80, 0.90, 0.95, 0.99]

st.title("Ranking methods")
st.markdown(
    "Each method turns the ratings of a recipe into one **score**; recipes "
    "are then sorted by score. The methods differ in how much they trust "
    "recipes with few ratings."
)

recipes, _ = get_data()
global_mean = global_mean_rating(recipes)
default_weight = default_prior_weight(recipes)

with st.container(border=True):
    st.markdown("**Method parameters**")
    col_weight, col_conf = st.columns(2)
    prior_weight = col_weight.slider(
        "Bayesian average: prior weight C",
        min_value=0, max_value=50, value=int(default_weight),
        help=f"Number of virtual ratings equal to the site average. "
             f"Default {default_weight:g} = "
             f"{config.PRIOR_WEIGHT_QUANTILE:.0%} quantile of the number "
             "of ratings. With 0 it is the naive average.",
    )
    confidence = col_conf.select_slider(
        "Lower bound: confidence level",
        options=CONFIDENCE_LEVELS, value=config.DEFAULT_CONFIDENCE,
        format_func=lambda level: f"{level:.0%}",
        help="Higher confidence gives a wider interval and a more "
             "cautious score.",
    )

scored = get_scored(prior_weight, confidence)

st.subheader("How the methods work")
tab_naive, tab_bayes, tab_bound = st.tabs(list(METHODS))
with tab_naive:
    st.latex(r"\text{score} = \bar{r} = \frac{1}{n}\sum_{i=1}^{n} r_i")
    st.markdown(
        "The plain average of the $n$ ratings. It ignores how many "
        "ratings there are: one 5-star rating gives a perfect score."
    )
with tab_bayes:
    st.latex(r"\text{score} = \frac{C \cdot m + n \cdot \bar{r}}{C + n}")
    st.markdown(
        f"Every recipe gets $C = {prior_weight}$ virtual ratings equal to "
        f"the site average $m = {global_mean:.2f}$. With few real ratings "
        "the score stays close to $m$; with many it approaches the "
        "recipe's own average."
    )
with tab_bound:
    st.latex(
        r"p_k = \frac{n_k + 1}{n + 5}, \quad"
        r"\mu = \sum_{k=1}^{5} k \, p_k, \quad"
        r"\sigma^2 = \sum_{k=1}^{5} k^2 p_k - \mu^2"
    )
    st.latex(r"\text{score} = \mu - z \sqrt{\frac{\sigma^2}{n + 6}}")
    st.markdown(
        "One virtual rating of each star value is added ($n_k$ is the "
        "number of $k$-star ratings). The score is the lower end of the "
        f"{confidence:.0%} confidence interval of the true average "
        "rating: it is high only if the ratings are high, numerous and "
        "consistent. Based on Evan Miller, *Ranking Items With Star "
        "Ratings*."
    )

st.subheader("Score and number of ratings")
other_method = st.radio(
    "Compare the naive average with", ["Lower bound", "Bayesian average"],
    horizontal=True,
)
st.pyplot(plot_score_vs_ratings(scored, ("Naive average", other_method)),
          clear_figure=True)
st.markdown(
    "Each dot is a recipe. With one or two ratings the naive score can be "
    "anything from 1 to 5. The other methods keep these recipes away from "
    "the top until there is enough evidence."
)

st.subheader("Who gets into the top")
top_n = config.TOP_N_EXPORT
st.pyplot(plot_top_profile(scored, top_n), clear_figure=True)
profile = summarize_top(scored, top_n)
st.dataframe(
    profile,
    column_config={
        "min_ratings": "Min ratings",
        "median_ratings": st.column_config.NumberColumn("Median ratings",
                                                        format="%.0f"),
        "mean_rating": st.column_config.NumberColumn("Mean rating",
                                                     format="%.2f"),
    },
)

st.subheader("How much the methods agree")
comparison = compare_methods(scored, top_n)
st.dataframe(
    comparison,
    hide_index=True,
    column_config={
        "method_a": "Method A",
        "method_b": "Method B",
        "spearman": st.column_config.NumberColumn(
            "Spearman correlation", format="%.2f",
            help="Agreement of the full ranking (1 = same order).",
        ),
        "top_overlap": st.column_config.NumberColumn(
            f"Common recipes in top {top_n}", format="percent",
        ),
    },
)
st.pyplot(plot_rank_changes(scored), clear_figure=True)

st.subheader("Why the lower bound is the main method")
many_id, one_id = config.WEAKNESS_EXAMPLE_IDS
examples = scored.set_index("recipe_id")
many, one = examples.loc[many_id], examples.loc[one_id]
st.markdown(
    f"""
The Bayesian average fixes the top of the ranking but ignores how
**certain** a score is. **{many['name']}** has {many['n_ratings']:,}
ratings with an average of {many['mean_rating']:.2f}, close to the site
average. **{one['name']}** has a single 5-star rating. The Bayesian average
puts the second one higher (#{one['bayesian_rank']:,} vs
#{many['bayesian_rank']:,}), because an unknown recipe is expected to be
slightly above average.

The lower bound ranks them #{many['lower_bound_rank']:,} and
#{one['lower_bound_rank']:,}: a well-tested recipe is placed above one
nobody has really tried. It also uses the spread of the ratings, so a
recipe that splits opinions is ranked below a consistent one with the
same average.
"""
)
