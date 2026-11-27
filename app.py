"""Streamlit app: ranking Food.com recipes by user liking.

Run with::

    streamlit run app.py
"""

import streamlit as st

st.set_page_config(
    page_title="Food.com Recipe Ranking",
    page_icon="🍳",
    layout="wide",
)

pages = [
    st.Page("app_pages/home.py", title="Overview", icon=":material/home:",
            default=True),
    st.Page("app_pages/data.py", title="Data",
            icon=":material/database:"),
    st.Page("app_pages/methods.py", title="Ranking methods",
            icon=":material/functions:"),
    st.Page("app_pages/top_recipes.py", title="Top recipes",
            icon=":material/trophy:"),
    st.Page("app_pages/categories.py", title="Categories",
            icon=":material/category:"),
]

st.navigation(pages).run()
