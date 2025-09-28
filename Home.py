# Home.py
import streamlit as st

st.set_page_config(page_title="ML Models App", layout="wide")
st.title("Machine Learning Models App")
st.write("Welcome — select a model from the sidebar (or the Pages menu) to explore demos and run small experiments.")
st.markdown("""
**This repo**: separate pages implement single models.  
Upload your CSV or use the sample datasets shipped in the pages.
""")