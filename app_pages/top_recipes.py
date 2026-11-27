"""Page: Top recipes."""

import streamlit as st

from app_pages.shared import get_data

st.title("Top recipes")
recipes, interactions = get_data()
st.info("Work in progress.")
