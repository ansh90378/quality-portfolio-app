import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris, load_diabetes
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    r2_score, mean_absolute_error, mean_squared_error
)

st.title("📊 Model Comparison Dashboard (with Cross-Validation)")

# -----------------------------
# Step 1: Choose Task
# -----------------------------
task = st.radio("Select task type:", ["Classification", "Regression"])

# -----------------------------
# Step 2: Load Dataset
# -----------------------------
if task == "Classification":
    use_sample = st.checkbox("Use Iris dataset (built-in)", value=True)
    if use_sample:
        ds = load_iris(as_frame=True)
        data = ds.frame
        X = data.drop(columns=["target"])
        y = data["target"]
        st.caption("Loaded `sklearn.datasets.load_iris()`")
    else:
        uploaded = st.file_uploader("Upload CSV (last column = target)", type=["csv"])
        if uploaded:
            data = pd.read_csv(uploaded)
            X = data.iloc[:, :-1]
            y = data.iloc[:, -1]
        else:
            st.stop()
else:
    use_sample = st.checkbox("Use Diabetes dataset (built-in)", value=True)
    if use_sample:
        ds = load_diabetes(as_frame=True)
        data = ds.frame
        X = data.drop(columns=["target"])
        y = data["target"]
        st.caption("Loaded `sklearn.datasets.load_diabetes()`")
    else:
        uploaded = st.file_uploader("Upload CSV (last column = target)", type=["csv"])
        if uploaded:
            data = pd.read_csv(uploaded)
            X = data.iloc[:, :-1]
            y = data.iloc[:, -1]
        else:
            st.stop()

# -----------------------------
# Step 3: Train/Test Split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -----------------------------
# Step 4: Model Options
# -----------------------------
if task == "Classification":
    model_options = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Decision Tree": DecisionTreeClassifier(),
        "Random Forest": RandomForestClassifier(),
    }
    default_models = ["Logistic Regression", "Decision Tree"]
else:
    model_options = {
        "Linear Regression": LinearRegression(),
        "Decision Tree Regressor": DecisionTreeRegressor(),
        "Random Forest Regressor": RandomForestRegressor(),
    }
    default_models = ["Linear Regression", "Decision Tree Regressor"]

selected_models = st.multiselect(
    "Choose models to train & evaluate",
    options=list(model_options.keys()),
    default=default_models
)

# -----------------------------
# Step 5: Train, Cross-Validate & Evaluate
# -----------------------------
results = []

for name in selected_models:
    model = model_options[name]

    if task == "Classification":
        # 5-fold CV accuracy
        cv_scores = cross_val_score(model, X, y, cv=5, scoring="accuracy")
        mean_acc = np.mean(cv_scores)

        # Final fit for reports
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        results.append({"Model": name, "CV Accuracy": mean_acc})
    else:
        # 5-fold CV R²
        cv_scores = cross_val_score(model, X, y, cv=5, scoring="r2")
        mean_r2 = np.mean(cv_scores)

        # Final fit for reports
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))

        results.append({"Model": name, "CV R²": mean_r2, "MAE": mae, "RMSE": rmse})

# -----------------------------
# Step 6: Show Results
# -----------------------------
if results:
    df_results = pd.DataFrame(results)
    st.subheader("📈 Cross-Validation Results")
    st.dataframe(df_results)

    # -------------------------
    # Bar Chart of Key Metric
    # -------------------------
    st.subheader("📊 Performance Comparison (Cross-Validation)")

    if task == "Classification":
        fig, ax = plt.subplots()
        ax.bar(df_results["Model"], df_results["CV Accuracy"], color="skyblue")
        ax.set_ylabel("CV Accuracy")
        ax.set_ylim(0, 1)
        plt.xticks(rotation=30)
        st.pyplot(fig)
    else:
        fig, ax = plt.subplots()
        ax.bar(df_results["Model"], df_results["CV R²"], color="salmon")
        ax.set_ylabel("CV R² Score")
        ax.set_ylim(0, 1)
        plt.xticks(rotation=30)
        st.pyplot(fig)

from utils.plots import plot_classification_results, plot_regression_results

# After training
if task == "Classification":
    plot_classification_results(model, X_train, X_test, y_test)

elif task == "Regression":
    plot_regression_results(model, X_train, X_test, y_test)