[📌 View Interactive Roadmap Checklist](../../issues/2)

# 🧠 Quality Portfolio App  

![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-%E2%9C%A8%20latest-red?logo=streamlit)
![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)
[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://quality-portfolio-app-lavyv9znczkrgjt3xc4h5a.streamlit.app)

A single **Streamlit** application showcasing **machine-learning models from basic to advanced**.  
Each model is implemented as its own page with explanations, training, and visualizations.  
This project is designed to **demonstrate practical ML skills** to interviewers and peers.

---

## 📑 Table of Contents  

- [Project Vision](#-project-vision)  
- [Live Demo](#-live-demo)  
- [Current Pages](#-current-pages)  
- [Theory Covered](#-theory-covered)  
- [Tech Stack](#-tech-stack)  
- [Installation](#-installation-local-run)  
- [Roadmap](#-roadmap-planned-pages)  
- [Contributing](#-contributing)  
- [License](#-license)  
- [Why This Project](#-why-this-project)  

---

## 🗂️ Project Vision  
- A **one-stop ML demo app**  
- Each ML model appears as a **module/page**  
- Users can:
  - Read a **short description** of the model  
  - See **example code** and how it works  
  - **Upload their own CSV dataset** or use a sample dataset  
  - Train & evaluate the model live in the browser  
  - Visualize predictions and metrics  

---

## 🚀 Live Demo  
🌐 **[Open the app on Streamlit Cloud](https://quality-portfolio-app-lavyv9znczkrgjt3xc4h5a.streamlit.app)**  

---

## 📁 Current Pages  
- `Home` – project overview & navigation  
- `Linear Regression` – regression with sample/custom data  
- `Logistic Regression` – binary classification with sample/custom data  
- `Decision Tree & Random Forest` – tree-based models for classification & regression  
- `K-Means Clustering` – unsupervised clustering with Elbow & Silhouette  
- `PCA` – dimensionality reduction with 2D & 3D visualization  

---

## 📖 Theory Covered  

### 🔹 Week 1 – Linear Regression  
Predicts a **continuous target** as a weighted sum of features.  
![Linear Regression](https://media.geeksforgeeks.org/wp-content/uploads/20231129130431/11111111.png)  
Evaluation: **MSE, RMSE, R²**.  

---

### 🔹 Week 2 – Logistic Regression  
Predicts **probabilities** for classification problems.  
![Logistic Regression Curve](https://zd-brightspot.s3.us-east-1.amazonaws.com/wp-content/uploads/2022/04/11040521/46-4-e1715636469361.png)  
Evaluation: **Accuracy, Precision, Recall, F1-Score, Confusion Matrix**.  

---

### 🔹 Week 3–4 – Decision Tree & Random Forest  
Decision Trees: Splits data using **if-else rules** for interpretability.  
![Decision Tree](https://scikit-learn.org/stable/_images/sphx_glr_plot_tree_regression_001.png)  

Random Forest: An **ensemble of trees** for higher accuracy & robustness.  
![Random Forest Ensemble](https://media.geeksforgeeks.org/wp-content/uploads/20240130162938/random.webp)  

---

### 🔹 Week 5 – K-Means Clustering  
Groups data into **K clusters** based on similarity.  
- **Elbow Method** helps choose K.  
- **Silhouette Score** measures separation.  

![K-Means Clustering](https://scikit-learn.org/stable/_images/sphx_glr_plot_kmeans_digits_001.png)  

---

### 🔹 Week 6 – Principal Component Analysis (PCA)  
Reduces dimensionality by projecting data into fewer **principal components**.  
![PCA Components](https://scikit-learn.org/stable/_images/sphx_glr_plot_pca_iris_001.png)  

Bar plot shows **variance explained**, and 2D/3D scatter plots illustrate new transformed features.  

---

## 🛠️ Tech Stack  
- [Python 3.x](https://www.python.org/)  
- [Streamlit](https://streamlit.io/)  
- [scikit-learn](https://scikit-learn.org/)  
- [pandas](https://pandas.pydata.org/)  
- [NumPy](https://numpy.org/)  
- GitHub + Streamlit Cloud for deployment  

---

## 📦 Installation (local run)

```bash
# Clone the repository
git clone https://github.com/<yourusername>/quality-portfolio-app.git
cd quality-portfolio-app

# (Optional) create a virtual environment
python -m venv venv
source venv/bin/activate   # on Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run the app locally
streamlit run Home.py
```


## 🤝 Contributing

Feel free to fork the repo and add your own model pages!  
Contributions are welcome via pull requests.

---

## 📜 License

MIT License – feel free to use and adapt this project with credit.

---

### ✨ Why this project?

> This app demonstrates not only knowledge of **machine-learning algorithms** but also skills in **Python coding, project structuring, version control, and cloud deployment** – everything an interviewer likes to see.


