import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.metrics import (
    accuracy_score, confusion_matrix, classification_report, roc_auc_score, roc_curve,
    mean_squared_error, r2_score
)
from utils.plots import plot_classification_results, plot_regression_results
import matplotlib.pyplot as plt
import seaborn as sns
import io

st.set_page_config(page_title="Decision Tree", layout="wide")
st.title("Decision Tree — Classification & Regression")

st.markdown("""
**What this page does:**  
Choose whether to run a Decision Tree for **classification** or **regression**.  
You can use a sample dataset or upload your own CSV.
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
max_depth_none = st.sidebar.checkbox("No max depth (grow full tree)", value=False)
if not max_depth_none:
    max_depth = st.sidebar.slider("max_depth", 1, 50, 6)
else:
    max_depth = None

min_samples_split = st.sidebar.slider("min_samples_split", 2, 50, 2)
if task == "Classification":
    criterion = st.sidebar.selectbox("Criterion", ["gini", "entropy"])
else:
    criterion = st.sidebar.selectbox("Criterion", ["squared_error", "friedman_mse", "absolute_error"])

# -----------------------
# Train the model
# -----------------------
if st.button("Train Decision Tree"):
    with st.spinner("Training..."):
        try:
            if task == "Classification":
                model = DecisionTreeClassifier(
                    criterion=criterion,
                    max_depth=max_depth,
                    min_samples_split=min_samples_split,
                    random_state=42
                )
            else:
                model = DecisionTreeRegressor(
                    criterion=criterion,
                    max_depth=max_depth,
                    min_samples_split=min_samples_split,
                    random_state=42
                )

            model.fit(X_train, y_train)
        except Exception as e:
            st.error(f"Training failed: {e}")
            st.stop()

    st.success("Model trained ✅")

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
        st.write(f"**MSE:** {mse:.4f}")
        st.write(f"**RMSE:** {rmse:.4f}")
        st.write(f"**R²:** {r2:.4f}")

        plot_regression_results(model, X_train, X_test, y_test)