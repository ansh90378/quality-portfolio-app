# 🧠 Quality Portfolio App  

![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-%E2%9C%A8%20latest-red?logo=streamlit)
![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)
[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://quality-portfolio-app-lavyv9znczkrgjt3xc4h5a.streamlit.app)

A single **Streamlit** application showcasing **machine-learning models from basic to advanced**.  
Each model is implemented as its own page with explanations, training, and visualizations.  
This project is designed to **demonstrate practical ML skills** to interviewers and peers.

[📌 Roadmap Progress (interactive checklist)](../../issues/1)

---

## 📑 Table of Contents  

- [Project Vision](#-project-vision)  
- [Live Demo](#-live-demo)  
- [Current Pages](#-current-pages)  
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
- `Linear Regression` – classic regression on sample or custom data  
- `Logistic Regression` – binary classification on sample or custom data  

*(More models will be added each week.)*

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
Open [http://localhost:8501](http://localhost:8501) in your browser.

## 🗺️ Roadmap (planned pages)

- [x] **Week 1:** Linear Regression – Regression  
- [x] **Week 2:** Logistic Regression – Classification  
- [ ] **Week 3:** Decision Tree / Random Forest – Classification/Regression  
- [ ] **Week 4:** Support Vector Machine (SVM) – Classification  
- [ ] **Week 5:** K-Means / Clustering – Unsupervised  
- [ ] **Week 6:** Principal Component Analysis (PCA) – Dimensionality Reduction  
- [ ] **Week 7:** Gradient Boosting / XGBoost – Ensemble  
- [ ] **Week 8:** Neural Network (basic MLP) – Deep Learning

*(You can tick them off as you add new pages.)*

---

## 🤝 Contributing

Feel free to fork the repo and add your own model pages!  
Contributions are welcome via pull requests.

---

## 📜 License

MIT License – feel free to use and adapt this project with credit.

---

### ✨ Why this project?

> This app demonstrates not only knowledge of **machine-learning algorithms** but also skills in **Python coding, project structuring, version control, and cloud deployment** – everything an interviewer likes to see.


