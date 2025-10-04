import streamlit as st
import pandas as pd
import numpy as np
from sklearn.datasets import load_iris, load_wine, load_digits
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import plotly.express as px

st.set_page_config(page_title="K-Means Clustering", page_icon="🔮")

st.title("🔮 K-Means Clustering Demo")

# -------------------- Dataset selection --------------------
use_sample = st.checkbox("Use sample dataset")

if use_sample:
    dataset_name = st.selectbox("Choose a dataset:", ["Iris", "Wine", "Digits"])
    if dataset_name == "Iris":
        ds = load_iris(as_frame=True)
    elif dataset_name == "Wine":
        ds = load_wine(as_frame=True)
    elif dataset_name == "Digits":
        ds = load_digits(as_frame=True)

    data = pd.DataFrame(ds.data, columns=ds.feature_names)
    st.caption(f"Loaded sklearn.datasets `{dataset_name}`")
else:
    uploaded = st.file_uploader("Upload CSV (first row header)", type=["csv"])
    if uploaded:
        data = pd.read_csv(uploaded)
    else:
        st.info("Upload a CSV or check 'Use sample' to proceed.")
        st.stop()

st.write("📊 Data Preview", data.head())

# -------------------- Handle missing values --------------------
st.sidebar.subheader("⚠️ Missing Value Handling")
missing_strategy = st.sidebar.radio(
    "Choose missing value handling:",
    ["Drop rows", "Fill with mean/median/mode"],
    index=1
)

if missing_strategy == "Drop rows":
    data = data.dropna()
else:
    for col in data.columns:
        if data[col].dtype in [np.float64, np.int64]:
            data[col] = data[col].fillna(data[col].median())
        else:
            data[col] = data[col].fillna(data[col].mode()[0])

# -------------------- Handle categorical columns --------------------
categorical_cols = data.select_dtypes(exclude=[np.number]).columns.tolist()

if categorical_cols:
    st.info(f"Categorical columns detected: {categorical_cols}")

    for col in categorical_cols:
        unique_vals = data[col].nunique()

        # Auto-drop high-cardinality cols (>20 unique values)
        if unique_vals > 20:
            st.warning(f"Dropping '{col}' (too many unique values: {unique_vals})")
            data = data.drop(columns=[col])

        else:
            action = st.sidebar.selectbox(
                f"What to do with {col}? ({unique_vals} unique values)",
                ["Drop", "Label Encode", "One-Hot Encode"],
                key=col
            )

            if action == "Drop":
                data = data.drop(columns=[col])

            elif action == "Label Encode":
                le = LabelEncoder()
                data[col] = le.fit_transform(data[col].astype(str))

            elif action == "One-Hot Encode":
                ohe = pd.get_dummies(data[col], prefix=col)
                data = pd.concat([data.drop(columns=[col]), ohe], axis=1)

# Keep only numeric columns
data = data.select_dtypes(include=[np.number])
if data.empty:
    st.error("❌ No numeric features left after preprocessing. Please adjust options.")
    st.stop()

st.write("✅ Preprocessed Data for Clustering:", data.head())

# -------------------- K-Means clustering --------------------
st.sidebar.subheader("⚙️ K-Means Settings")
max_k = st.sidebar.slider("Max number of clusters (for Elbow)", 2, 15, 10)

# Elbow method
sse = []
for k in range(2, max_k + 1):
    km = KMeans(n_clusters=k, random_state=42, n_init="auto")
    km.fit(data)
    sse.append(km.inertia_)

fig, ax = plt.subplots()
ax.plot(range(2, max_k + 1), sse, marker="o")
ax.set_xlabel("Number of Clusters (k)")
ax.set_ylabel("SSE (Inertia)")
ax.set_title("📉 Elbow Method")
st.pyplot(fig)

# Choose final k
n_clusters = st.slider("Select number of clusters", 2, max_k, 3)

# Fit KMeans
model = KMeans(n_clusters=n_clusters, random_state=42, n_init="auto")
labels = model.fit_predict(data)

st.write("🔹 Cluster Assignments", pd.Series(labels).value_counts())

# Silhouette score
score = silhouette_score(data, labels)
st.metric("Silhouette Score", f"{score:.3f}")

# -------------------- Visualization --------------------
if data.shape[1] >= 2:
    # --- 2D Matplotlib plot ---
    fig, ax = plt.subplots(figsize=(7, 5))

    scatter = ax.scatter(
        data.iloc[:, 0], data.iloc[:, 1],
        c=labels,
        cmap="tab10",
        s=80,
        alpha=0.7,
        edgecolors="k"
    )

    # Centroids
    centers = model.cluster_centers_
    ax.scatter(
        centers[:, 0], centers[:, 1],
        c="white", edgecolors="black",
        marker="o", s=250, linewidths=2,
        label="Centroids"
    )

    ax.set_xlabel(data.columns[0], fontsize=12)
    ax.set_ylabel(data.columns[1], fontsize=12)
    ax.set_title("✨ Cluster Visualization (first 2 features)", fontsize=14, weight="bold")

    handles, _ = scatter.legend_elements(prop="colors", alpha=0.6)
    legend_labels = [f"Cluster {i}" for i in range(n_clusters)]
    ax.legend(handles, legend_labels, title="Clusters", bbox_to_anchor=(1.05, 1), loc="upper left")

    st.pyplot(fig)

# --- Optional 3D Plotly visualization ---
if data.shape[1] >= 3:
    st.subheader("🌐 Interactive 3D Cluster Visualization")

    fig3d = px.scatter_3d(
        data,
        x=data.columns[0],
        y=data.columns[1],
        z=data.columns[2],
        color=labels.astype(str),
        symbol=labels.astype(str),
        opacity=0.7,
        size_max=10,
        color_discrete_sequence=px.colors.qualitative.Set1
    )

    # Add centroids
    centers_df = pd.DataFrame(centers, columns=data.columns[:data.shape[1]])
    fig3d.add_scatter3d(
        x=centers_df.iloc[:, 0],
        y=centers_df.iloc[:, 1],
        z=centers_df.iloc[:, 2],
        mode="markers",
        marker=dict(size=12, color="black", symbol="diamond"),
        name="Centroids"
    )

    st.plotly_chart(fig3d, use_container_width=True)

# -------------------- Add cluster labels to dataset --------------------
data_with_clusters = data.copy()
data_with_clusters["Cluster"] = labels

st.subheader("📥 Download Clustered Dataset")
st.dataframe(data_with_clusters.head())

# Convert to CSV for download
csv = data_with_clusters.to_csv(index=False).encode("utf-8")
st.download_button(
    label="⬇️ Download CSV with Clusters",
    data=csv,
    file_name="clustered_dataset.csv",
    mime="text/csv"
)
