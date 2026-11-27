"""Page: Food.com Recipe Ranking."""

import streamlit as st

from app_pages.shared import get_data

st.title("Food.com Recipe Ranking")
recipes, interactions = get_data()
st.info("Work in progress.")
