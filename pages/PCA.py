import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import time

st.set_page_config(page_title="PCA Demo", layout="wide")
st.title("🧩 Principal Component Analysis (PCA)")

# -------------------------------
# 1. Upload or select dataset
# -------------------------------
st.sidebar.header("📂 Upload or Select Dataset")
dataset_choice = st.sidebar.selectbox(
    "Choose Dataset", ["Iris", "Wine", "Digits", "Upload CSV"]
)

if dataset_choice == "Upload CSV":
    file = st.sidebar.file_uploader("Upload your CSV", type=["csv"])
    if file:
        df = pd.read_csv(file)
else:
    from sklearn import datasets
    if dataset_choice == "Iris":
        data = datasets.load_iris(as_frame=True)
        df = data.frame
    elif dataset_choice == "Wine":
        data = datasets.load_wine(as_frame=True)
        df = data.frame
    else:  # Digits
        data = datasets.load_digits(as_frame=True)
        df = pd.DataFrame(data.data, columns=data.feature_names)
        df["target"] = data.target

st.subheader("📊 Preview Dataset")
st.write(df.head())

# -------------------------------
# 2. Processing Options
# -------------------------------
st.sidebar.header("⚙️ Preprocessing Options")
target_col = st.sidebar.selectbox("Select target column (optional)", [None] + df.columns.tolist())

missing_strategy = st.sidebar.selectbox("Handle Missing Values", ["Drop rows", "Fill with mean", "Fill with median", "Fill with mode"])
cat_strategy = st.sidebar.selectbox("Handle Categorical Columns", ["Drop", "Label Encode"])

# Handle missing values
if missing_strategy == "Drop rows":
    df = df.dropna()
elif missing_strategy == "Fill with mean":
    df = df.fillna(df.mean(numeric_only=True))
elif missing_strategy == "Fill with median":
    df = df.fillna(df.median(numeric_only=True))
elif missing_strategy == "Fill with mode":
    for col in df.columns:
        df[col] = df[col].fillna(df[col].mode()[0])

# Encode categorical columns
cat_cols = df.select_dtypes(include=["object", "category"]).columns
if cat_strategy == "Label Encode":
    le = LabelEncoder()
    for col in cat_cols:
        df[col] = le.fit_transform(df[col].astype(str))
elif cat_strategy == "Drop":
    df = df.drop(columns=cat_cols)

st.subheader("🧹 Processed Data")
st.write(df.head())

# -------------------------------
# 3. PCA
# -------------------------------
if target_col and target_col in df.columns:
    X = df.drop(columns=[target_col])
    y = df[target_col]
else:
    X = df.copy()
    y = None

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

n_components = st.sidebar.slider("Select number of PCA components", 2, min(10, X.shape[1]), 2)
pca = PCA(n_components=n_components)
X_pca = pca.fit_transform(X_scaled)

st.subheader("📉 Explained Variance Ratio")
fig, ax = plt.subplots()
ax.plot(range(1, n_components+1), np.cumsum(pca.explained_variance_ratio_), marker="o")
ax.set_xlabel("Number of Components")
ax.set_ylabel("Cumulative Explained Variance")
st.pyplot(fig)

# -------------------------------
# 4. Scatter Plot with Labels
# -------------------------------
st.subheader("🔍 PCA Projection")
if n_components >= 2:
    fig, ax = plt.subplots(figsize=(8, 6))
    if y is not None:
        unique_labels = np.unique(y)
        for lbl in unique_labels:
            mask = y == lbl
            ax.scatter(X_pca[mask, 0], X_pca[mask, 1], label=str(lbl), alpha=0.6)
            # Annotate with text
            for i in np.where(mask)[0][:30]:  # avoid clutter, only first 30
                ax.text(X_pca[i, 0], X_pca[i, 1], str(lbl), fontsize=7, alpha=0.7)
        ax.legend(title="Classes / Labels")
    else:
        ax.scatter(X_pca[:, 0], X_pca[:, 1], alpha=0.6)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    st.pyplot(fig)

# -------------------------------
# 5. Model Comparison
# -------------------------------
if y is not None:
    st.subheader("⚖️ Model Comparison (Original vs PCA Features)")

    # Choose model interactively
    model_choice = st.selectbox("Select Model", ["Logistic Regression", "SVM"])

    if model_choice == "Logistic Regression":
        from sklearn.linear_model import LogisticRegression
        model = LogisticRegression(max_iter=2000)
    else:
        from sklearn.svm import SVC
        st.caption(
            "⚠️ Note: SVMs can be very slow on large datasets. This demo is included to showcase the trade-off between accuracy and scalability.")

        model = SVC()

    # Train on original data
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )
    start = time.time()
    model.fit(X_train, y_train)
    orig_score = model.score(X_test, y_test)
    orig_time = time.time() - start

    # Train on PCA data
    X_train_pca, X_test_pca, y_train_pca, y_test_pca = train_test_split(
        X_pca, y, test_size=0.2, random_state=42
    )
    start = time.time()
    model.fit(X_train_pca, y_train_pca)
    pca_score = model.score(X_test_pca, y_test_pca)
    pca_time = time.time() - start

    # Show results
    st.write(f"**Original ({model_choice})**: {orig_score:.3f} accuracy (time {orig_time:.2f}s)")
    st.write(f"**PCA ({model_choice})**: {pca_score:.3f} accuracy (time {pca_time:.2f}s)")

# -------------------------------
# 6. Feature Contributions
# -------------------------------
st.subheader("🧠 Feature Contributions to PCs")
loadings = pd.DataFrame(
    pca.components_.T,
    columns=[f"PC{i+1}" for i in range(pca.n_components_)],
    index=X.columns
)
fig, ax = plt.subplots(figsize=(10, 6))
sns.heatmap(loadings, cmap="coolwarm", center=0, annot=False, ax=ax)
st.pyplot(fig)

# Step 3.2: Bar Chart – Variance per Component
# Step 3.2: Bar Chart – Variance per Component
st.subheader("📊 Variance Explained by Each Component")

cum_var = np.cumsum(pca.explained_variance_ratio_)
n_95 = np.argmax(cum_var >= 0.95) + 1  # components needed for 95%

fig, ax = plt.subplots()
ax.bar(range(1, n_components + 1), pca.explained_variance_ratio_, alpha=0.7, color="skyblue", label="Variance per PC")

# Add cumulative variance line
ax.plot(range(1, n_components + 1), cum_var, marker="o", color="orange", label="Cumulative Variance")

# Add threshold
ax.axhline(y=0.95, color="r", linestyle="--", label="95% Threshold")
ax.axvline(x=n_95, color="g", linestyle="--", label=f"{n_95} PCs for 95%")

ax.set_xlabel("Principal Component")
ax.set_ylabel("Variance Ratio")
ax.set_title("Variance per Component + Cumulative Variance")
ax.legend()
st.pyplot(fig)

st.caption(f"✅ {n_95} components are enough to retain 95% variance.")