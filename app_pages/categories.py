"""Page: Categories."""

import streamlit as st

from app_pages.shared import get_data

st.title("Categories")
recipes, interactions = get_data()
st.info("Work in progress.")
