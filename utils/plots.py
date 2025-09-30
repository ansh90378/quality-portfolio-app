import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score

def plot_classification_results(model, X_train, X_test, y_test):
    """Plots confusion matrix, ROC curve, and feature importances for classification models."""
    y_pred = model.predict(X_test)

    # ✅ Confusion Matrix
    st.subheader("Confusion Matrix")
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    st.pyplot(fig)

    # ✅ ROC Curve + AUC (binary only)
    if hasattr(model, "predict_proba") and len(set(y_test)) == 2:
        y_prob = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        auc = roc_auc_score(y_test, y_prob)

        st.subheader("ROC Curve")
        fig, ax = plt.subplots()
        ax.plot(fpr, tpr, label=f"AUC = {auc:.2f}")
        ax.plot([0, 1], [0, 1], linestyle="--")
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.legend()
        st.pyplot(fig)

    # ✅ Feature Importances (Tree-based models only)
    if hasattr(model, "feature_importances_"):
        st.subheader("Feature Importances")
        importances = model.feature_importances_
        fig, ax = plt.subplots()
        sns.barplot(x=importances, y=X_train.columns, ax=ax)
        st.pyplot(fig)


def plot_regression_results(model, X_train, X_test, y_test):
    """Plots predicted vs actual, residuals, and feature importances for regression models."""
    y_pred = model.predict(X_test)

    # ✅ Predicted vs Actual
    st.subheader("Predicted vs Actual")
    fig, ax = plt.subplots()
    ax.scatter(y_test, y_pred, alpha=0.6)
    ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
    ax.set_xlabel("Actual")
    ax.set_ylabel("Predicted")
    ax.set_title("Predicted vs Actual")
    st.pyplot(fig)

    # ✅ Residuals Distribution
    st.subheader("Residual Distribution")
    residuals = y_test - y_pred
    fig, ax = plt.subplots()
    sns.histplot(residuals, bins=20, kde=True, ax=ax)
    st.pyplot(fig)

    # ✅ Feature Importances (Tree-based regressors only)
    if hasattr(model, "feature_importances_"):
        st.subheader("Feature Importances")
        importances = model.feature_importances_
        fig, ax = plt.subplots()
        sns.barplot(x=importances, y=X_train.columns, ax=ax)
        st.pyplot(fig)
