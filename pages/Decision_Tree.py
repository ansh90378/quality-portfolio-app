# pages/3_Decision_Tree.py
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.metrics import (
    accuracy_score, confusion_matrix, classification_report, roc_auc_score, roc_curve,
    mean_squared_error, r2_score
)
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

        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        fig_cm, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt="d", ax=ax)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.set_title("Confusion Matrix")
        st.pyplot(fig_cm)

        # ROC AUC (only for binary)
        unique_classes = np.unique(y_test)
        if unique_classes.size == 2 and hasattr(model, "predict_proba"):
            y_prob = model.predict_proba(X_test)[:, 1]
            try:
                auc = roc_auc_score(y_test, y_prob)
                st.write(f"**ROC AUC:** {auc:.4f}")

                fpr, tpr, _ = roc_curve(y_test, y_prob)
                fig_roc, ax2 = plt.subplots()
                ax2.plot(fpr, tpr)
                ax2.plot([0, 1], [0, 1], "--")
                ax2.set_xlabel("FPR")
                ax2.set_ylabel("TPR")
                ax2.set_title("ROC Curve")
                st.pyplot(fig_roc)
            except Exception:
                pass

    else:
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, y_pred)
        st.write(f"**MSE:** {mse:.4f}")
        st.write(f"**RMSE:** {rmse:.4f}")
        st.write(f"**R²:** {r2:.4f}")

        # Actual vs Predicted
        fig_scatter, ax3 = plt.subplots()
        ax3.scatter(y_test, y_pred, alpha=0.7)
        ax3.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "--")
        ax3.set_xlabel("Actual")
        ax3.set_ylabel("Predicted")
        ax3.set_title("Actual vs Predicted")
        st.pyplot(fig_scatter)

        # Residuals
        fig_res, ax4 = plt.subplots()
        residuals = y_test - y_pred
        ax4.hist(residuals, bins=30)
        ax4.set_title("Residuals distribution")
        st.pyplot(fig_res)

    # -----------------------
    # Feature importances
    # -----------------------
    try:
        importances = model.feature_importances_
        fi = pd.DataFrame({"feature": feature_cols, "importance": importances})
        fi = fi.sort_values("importance", ascending=False)
        st.subheader("Feature importances")
        st.dataframe(fi)

        fig_fi, ax5 = plt.subplots()
        ax5.barh(fi["feature"].iloc[::-1], fi["importance"].iloc[::-1])
        ax5.set_title("Feature importances")
        st.pyplot(fig_fi)
    except Exception:
        st.info("Feature importances not available for this model.")