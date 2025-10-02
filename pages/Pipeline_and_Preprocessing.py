import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt

st.title("🔄 ML Pipeline + Preprocessing Demo")

# -------------------------------
# Step 1: Dataset selection
# -------------------------------
use_sample = st.checkbox("Use sample Titanic-like dataset")

if use_sample:
    data = pd.DataFrame({
        "Age": [22, 38, np.nan, 35, np.nan, 28, np.nan, 42],
        "Sex": ["male", "female", "female", "male", "female", "male", "female", "male"],
        "Fare": [7.25, 71.83, 8.05, np.nan, 8.46, 15.50, 23.0, np.nan],
        "Survived": [0, 1, 1, 1, 0, 0, 1, 0]
    })
    st.caption("✅ Loaded built-in Titanic-style dataset with missing values + categorical features")
else:
    uploaded = st.file_uploader("📂 Upload your CSV", type=["csv"])
    if uploaded:
        data = pd.read_csv(uploaded)
    else:
        st.info("Upload a CSV or check 'Use sample' to proceed.")
        st.stop()

st.subheader("🔎 Raw Data Preview")
st.write(data.head())

# -------------------------------
# Step 2: Feature/Target Split
# -------------------------------
target_col = st.selectbox("Select target column", options=data.columns, index=len(data.columns)-1)
X = data.drop(target_col, axis=1)
y = data[target_col]

# Ensure no NaNs in target
if y.isnull().any():
    st.warning("⚠️ Target column contains missing values. They will be dropped.")
    non_null_idx = y.dropna().index
    X = X.loc[non_null_idx]
    y = y.loc[non_null_idx]

num_cols = X.select_dtypes(include=["int64", "float64"]).columns
cat_cols = X.select_dtypes(include=["object", "category"]).columns

st.write("📊 Numeric columns:", list(num_cols))
st.write("🔤 Categorical columns:", list(cat_cols))

# -------------------------------
# Step 3: Define preprocessing
# -------------------------------
num_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="mean")),
    ("scaler", StandardScaler())
])

cat_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1))
])

preprocessor = ColumnTransformer(
    transformers=[
        ("num", num_transformer, num_cols),
        ("cat", cat_transformer, cat_cols)
    ]
)

# -------------------------------
# Step 4: Apply preprocessing immediately
# -------------------------------
if st.button("🧹 Run Preprocessing"):
    transformed = preprocessor.fit_transform(X)

    # Ensure dense array for DataFrame
    # transformed_dense = transformed.toarray() if hasattr(transformed, "toarray") else transformed

    # ohe = preprocessor.named_transformers_["cat"].named_steps["encoder"]
    # encoded_features = ohe.get_feature_names_out(cat_cols) if len(cat_cols) > 0 else []

    # all_features = list(num_cols) + list(encoded_features)
    # clean_df = pd.DataFrame(transformed_dense, columns=all_features)

    clean_df = pd.DataFrame(transformed, columns=list(num_cols) + list(cat_cols))

    # Double-check no NaN left in features
    if clean_df.isnull().any().any():
        st.error("🚨 NaNs still remain in features after preprocessing!")
    else:
        st.success("✅ No NaNs in features, safe to train models.")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**🟡 Raw Data (first 5 rows):**")
        st.write(X.head())
    with col2:
        st.markdown("**🟢 After Preprocessing (first 5 rows):**")
        st.write(clean_df.head())

    # Download button
    csv = clean_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="💾 Download Cleaned Dataset",
        data=csv,
        file_name="cleaned_dataset.csv",
        mime="text/csv"
    )

    st.success("Preprocessing complete ✅ Now you can train a model below.")

# -------------------------------
# Step 5: Train Model (optional)
# -------------------------------
clf = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("model", RandomForestClassifier(random_state=42))
])

if st.button("🚀 Train Model"):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    st.subheader("📈 Model Results")
    st.write("✅ Accuracy:", accuracy_score(y_test, y_pred))
    st.text("Classification Report:\n" + classification_report(y_test, y_pred))

    # Feature Importances
    st.subheader("🌲 Random Forest Feature Importances")
    ohe = clf.named_steps["preprocessor"].named_transformers_["cat"].named_steps["encoder"]
    encoded_features = ohe.get_feature_names_out(cat_cols) if len(cat_cols) > 0 else []
    all_features = list(num_cols) + list(encoded_features)

    importances = clf.named_steps["model"].feature_importances_
    imp_df = pd.DataFrame({"Feature": all_features, "Importance": importances}).sort_values(by="Importance", ascending=False)

    st.dataframe(imp_df)

    fig, ax = plt.subplots()
    ax.barh(imp_df["Feature"], imp_df["Importance"])
    ax.set_xlabel("Importance")
    ax.set_title("Feature Importances")
    st.pyplot(fig)
