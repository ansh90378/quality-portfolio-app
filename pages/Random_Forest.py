import streamlit as st
import pandas as pd
import numpy as np
import shap
import joblib

from sklearn.model_selection import train_test_split,  cross_val_score
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (
    accuracy_score, confusion_matrix, classification_report, roc_auc_score, roc_curve,
    mean_squared_error, r2_score
)
from utils.plots import plot_classification_results, plot_regression_results
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="Random Forest", layout="wide")
st.title("Random Forest — Classification & Regression")

st.markdown("""
Random Forest builds many decision trees and averages their predictions for better generalization.
Use the left sidebar to tweak hyperparameters and the main area to upload or preview data.
""")

# -----------------------
# Choose task & data
# -----------------------
task = st.radio("Task type", ("Classification", "Regression"))

use_sample = st.checkbox("Use sample dataset", value=True)

if use_sample:
    if task == "Classification":
        from sklearn.datasets import load_breast_cancer
        ds = load_breast_cancer()
        data = pd.DataFrame(ds.data, columns=ds.feature_names)
        data["target"] = ds.target
        st.caption("Loaded `sklearn.datasets.load_breast_cancer()` (binary classification)")
    else:
        from sklearn.datasets import load_diabetes
        ds = load_diabetes()
        data = pd.DataFrame(ds.data, columns=ds.feature_names)
        data["target"] = ds.target
        st.caption("Loaded `sklearn.datasets.load_diabetes()` (regression)")
else:
    uploaded = st.file_uploader("Upload CSV (first row header)", type=["csv"])
    if uploaded:
        try:
            data = pd.read_csv(uploaded)
        except Exception as e:
            st.error(f"Error reading CSV: {e}")
            st.stop()
    else:
        st.info("Upload a CSV or check 'Use sample' to proceed.")
        st.stop()

st.write("Data preview:")
st.dataframe(data.head())

# -----------------------
# Column selection
# -----------------------
cols = data.columns.tolist()
target_col = st.selectbox("Select target column", cols, index=len(cols)-1)
feature_cols = st.multiselect(
    "Select feature columns (at least 1)",
    [c for c in cols if c != target_col],
    default=[c for c in cols if c != target_col][:6]
)

if not feature_cols:
    st.warning("Please select at least one feature column.")
    st.stop()

X = data[feature_cols]
y = data[target_col]

# -----------------------
# Train/test split
# -----------------------
test_pct = st.slider("Test size (%)", 5, 50, 20)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_pct/100, random_state=42)

# -----------------------
# Hyperparameters UI
# -----------------------
st.sidebar.header("Hyperparameters")
n_estimators = st.sidebar.slider("n_estimators", 10, 500, 100, step=10)
max_depth_none = st.sidebar.checkbox("No max depth (grow full trees)", value=False)
if not max_depth_none:
    max_depth = st.sidebar.slider("max_depth", 1, 50, 12)
else:
    max_depth = None
min_samples_split = st.sidebar.slider("min_samples_split", 2, 50, 2)
max_features = st.sidebar.selectbox("max_features", ["sqrt", "log2", None])

## New features
balance = st.sidebar.checkbox("Use class_weight='balanced' (only Classification)")
use_cv = st.sidebar.checkbox("Use Cross-Validation")
save_model = st.sidebar.checkbox("Save trained model")

# -----------------------
# Train the model
# -----------------------
if st.button("Train Random Forest"):
    with st.spinner("Training..."):
        try:
            if task == "Classification":
                model = RandomForestClassifier(
                    n_estimators=n_estimators,
                    max_depth=max_depth,
                    min_samples_split=min_samples_split,
                    max_features=max_features,
                    class_weight="balanced" if balance else None,
                    random_state=42,
                    n_jobs=-1
                )
            else:
                model = RandomForestRegressor(
                    n_estimators=n_estimators,
                    max_depth=max_depth,
                    min_samples_split=min_samples_split,
                    max_features=max_features,
                    random_state=42,
                    n_jobs=-1
                )

            model.fit(X_train, y_train)
        except Exception as e:
            st.error(f"Training failed: {e}")
            st.stop()

    st.success("Model trained ✅")

    # -----------------------
    # Evaluate
    # -----------------------
    st.subheader("Model Evaluation")

    if use_cv:
        scoring = "accuracy" if task == "Classification" else "r2"
        scores = cross_val_score(model, X, y, cv=5, scoring=scoring)
        st.write("Cross-validation scores:", scores)
        st.write("Mean CV Score:", scores.mean())
    else:
        y_pred = model.predict(X_test)
        if task == "Classification":
            st.write("Accuracy:", accuracy_score(y_test, y_pred))
        else:
            st.write("R² Score:", r2_score(y_test, y_pred))

    # -----------------------
    # Predictions & metrics
    # -----------------------
    y_pred = model.predict(X_test)

    if task == "Classification":
        acc = accuracy_score(y_test, y_pred)
        st.write(f"**Accuracy:** {acc:.4f}")
        st.text("Classification report:")
        st.text(classification_report(y_test, y_pred))

        plot_classification_results(model, X_train, X_test, y_test)
    else:
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, y_pred)
        st.write(f"**RMSE:** {rmse:.4f}")
        st.write(f"**R²:** {r2:.4f}")

        plot_regression_results(model, X_train, X_test, y_test)

    # -----------------------
    # Save model
    # -----------------------
    if save_model:
        joblib.dump(model, "random_forest_model.pkl")
        with open("random_forest_model.pkl", "rb") as f:
            st.download_button(
                "Download trained model",
                f,
                file_name="random_forest_model.pkl",
            )