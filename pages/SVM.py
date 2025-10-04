import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.svm import SVC, SVR
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, r2_score, mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns

st.title("⚡ Support Vector Machine (SVM)")

task = st.sidebar.radio("Select task", ["Classification", "Regression"])

use_sample = st.sidebar.checkbox("Use sample dataset")
if use_sample:
    if task == "Classification":
        from sklearn.datasets import load_breast_cancer
        ds = load_breast_cancer(as_frame=True)
        data = ds.frame
        st.caption("Loaded `sklearn.datasets.load_breast_cancer()` (Classification dataset)")
    else:
        from sklearn.datasets import fetch_california_housing
        ds = fetch_california_housing(as_frame=True)
        data = ds.frame
        st.caption("Loaded `sklearn.datasets.fetch_california_housing()` (Regression dataset)")
else:
    uploaded = st.file_uploader("Upload CSV (first row header)", type=["csv"])
    if uploaded:
        data = pd.read_csv(uploaded)
    else:
        st.info("Upload a CSV or check 'Use sample' to proceed.")
        st.stop()

target = st.sidebar.selectbox("Select target column", data.columns)
X = data.drop(columns=[target])
y = data[target]

if task == "Classification" and y.dtype.kind in "fc":  # float/continuous
    st.error("❌ Target looks continuous. Choose 'Regression' instead.")
    st.stop()

if task == "Regression" and y.dtype.kind in "biu":  # int/bool
    st.error("❌ Target looks categorical. Choose 'Classification' instead.")
    st.stop()

test_size = st.sidebar.slider("Test size (for train/test split)", 0.1, 0.5, 0.2)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

if task == "Classification":
    kernel = st.sidebar.selectbox("Kernel", ["linear", "rbf", "poly", "sigmoid"])
    C = st.sidebar.slider("Regularization (C)", 0.01, 10.0, 1.0)
    model = SVC(kernel=kernel, C=C, probability=True, random_state=42)
else:
    kernel = st.sidebar.selectbox("Kernel", ["linear", "rbf", "poly", "sigmoid"])
    C = st.sidebar.slider("Regularization (C)", 0.01, 10.0, 1.0)
    epsilon = st.sidebar.slider("Epsilon (tolerance)", 0.01, 1.0, 0.1)
    model = SVR(kernel=kernel, C=C, epsilon=epsilon)

if st.button("Train Model"):
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    if task == "Classification":
        acc = accuracy_score(y_test, preds)
        st.write(f"✅ Accuracy: {acc:.4f}")

        st.text("Classification Report:")
        st.text(classification_report(y_test, preds))

        cm = confusion_matrix(y_test, preds)
        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
        ax.set_title("Confusion Matrix")
        st.pyplot(fig)

    else:  # Regression
        r2 = r2_score(y_test, preds)
        mse = mean_squared_error(y_test, preds)
        st.write(f"✅ R² Score: {r2:.4f}")
        st.write(f"✅ MSE: {mse:.4f}")

        fig, ax = plt.subplots()
        ax.scatter(y_test, preds, alpha=0.6)
        ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
        ax.set_xlabel("Actual")
        ax.set_ylabel("Predicted")
        st.pyplot(fig)
