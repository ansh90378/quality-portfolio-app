import streamlit as st
import pandas as pd
import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
from utils.plots import plot_classification_results, plot_regression_results

st.title("Linear Regression (Demo)")

# Data options
use_sample = st.checkbox("Use sample sklearn diabetes dataset", value=True)

if use_sample:
    ds = load_diabetes(as_frame=True)
    data = ds.frame
    st.caption("Loaded `sklearn.datasets.load_diabetes()`")
else:
    uploaded = st.file_uploader("Upload CSV (first row header)", type=["csv"])
    if uploaded:
        data = pd.read_csv(uploaded)
    else:
        st.info("Upload a CSV or check 'Use sample' to proceed.")
        st.stop()

st.write("Data preview:")
st.dataframe(data.head())

cols = list(data.columns)
target = st.selectbox("Select target column", cols, index=len(cols)-1)
features = st.multiselect("Select feature columns (at least 1)", [c for c in cols if c != target], default=[c for c in cols if c != target][:3])

test_size = st.slider("Test fraction", 0.1, 0.5, 0.2)

if st.button("Train Linear Regression"):
    X = data[features].values
    y = data[target].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    st.write(f"**MSE:** {mse:.4f}")
    st.write(f"**R²:** {r2:.4f}")

    plot_regression_results(model, X_train, X_test, y_test)
