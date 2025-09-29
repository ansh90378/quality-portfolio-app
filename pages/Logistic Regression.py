import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

st.title("Logistic Regression Demo")

# --- Choose sample or upload ---
use_sample = st.checkbox("Use sample dataset (Breast Cancer)", value=True)

if use_sample:
    from sklearn.datasets import load_breast_cancer
    ds = load_breast_cancer(as_frame=True)
    data = ds.frame  # pandas DataFrame with features + target
    # In the sklearn frame, the target is separate, so join:
    data['target'] = ds.target
    st.caption("Loaded `sklearn.datasets.load_breast_cancer()` dataset")
else:
    uploaded = st.file_uploader("Upload CSV (first row header)", type=["csv"])
    if uploaded:
        data = pd.read_csv(uploaded)
    else:
        st.info("Upload a CSV or check 'Use sample' to proceed.")
        st.stop()

st.write("Data Preview:", data.head())

all_columns = data.columns.tolist()
target_col = st.selectbox("Select Target Column", all_columns, index=len(all_columns)-1)
feature_cols = st.multiselect("Select Feature Columns", all_columns[:-1], default=all_columns[:-1])

X = data[feature_cols]
y = data[target_col]

test_size = st.slider("Test Size (%)", 10, 50, 20)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size/100, random_state=42)

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

acc = model.score(X_test, y_test)
st.success(f"Accuracy on test data: {acc:.2f}")

st.subheader("Try prediction on new input:")

input_data = []
for col in feature_cols:
    val = st.number_input(
        f"Enter {col}",
        float(X[col].min()),
        float(X[col].max()),
        float(X[col].mean())
    )
    input_data.append(val)

if st.button("Predict"):
    pred = model.predict([input_data])[0]

    if pred == 0:
        answer = "Yes – The model predicts this case **has breast cancer (malignant)**"
    else:
        answer = "No – The model predicts this case **does not have breast cancer (benign)**"

    st.success(answer)
    proba = model.predict_proba([input_data])[0][pred]
    st.caption(f"Confidence: {proba:.1%}")
