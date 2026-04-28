# 🏦 AI Loan Approval System with Explainable AI (XAI)

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![Framework](https://img.shields.io/badge/Framework-Gradio-orange.svg)
![ML](https://img.shields.io/badge/ML-XGBoost-green.svg)
![XAI](https://img.shields.io/badge/XAI-SHAP-red.svg)

## 🎯 Project Context & Objectives

### The Problem: The "Black Box" of AI in Finance
In the banking and financial sector, credit risk assessment is a critical operation. While advanced Machine Learning models like XGBoost provide high accuracy in predicting loan defaults, they often act as "Black Boxes." This lack of transparency makes it difficult for loan officers to:
* Explain to a customer **why** their application was rejected.
* Ensure the model is making decisions based on fair and logical financial factors.
* Comply with financial regulations that require "the right to an explanation."

### The Solution: Interpretable Machine Learning
This project bridges the gap between high-performance prediction and human interpretability. By combining **XGBoost** for robust classification and **SHAP (SHapley Additive exPlanations)** for feature attribution, the system provides a transparent "scorecard" for every individual decision.

### Core Objectives
* **Precision Risk Assessment:** Build a model that accurately distinguishes between low-risk and high-risk loan applicants using historical credit data.
* **Explainable AI (XAI) Integration:** Transform abstract model weights into intuitive visual explanations (Waterfall plots) showing the positive/negative impact of each feature.
* **Real-time Deployment:** Develop an interactive web interface for stakeholders to perform "what-if" analysis in real-time.
* **Operational Transparency:** Provide a tool that helps financial institutions make data-driven decisions while maintaining accountability.

## 🚀 Key Features
- **Predictive Modeling:** Optimized XGBoost classifier for default probability.
- **XAI Integration:** Real-time SHAP Waterfall plots for local interpretability.
- **Interactive UI:** User-friendly Gradio interface with pre-set examples for testing.
- **Synchronized Pipeline:** Fully integrated preprocessing (StandardScaler & OneHotEncoder) ensuring 100% consistency between training and production.

## 🛠️ Tech Stack
- **Machine Learning:** XGBoost, Scikit-learn
- **Interpretability:** SHAP
- **Interface:** Gradio
- **DevOps/Tools:** Python, Pandas, Joblib, Git LFS

## 📖 How to Use
1. **Clone the repo:** `git clone https://github.com/YourUsername/RepoName.git`
2. **Install dependencies:** `pip install -r requirements.txt`
3. **Run the application:** `python app.py`

## 📊 Interpretation Guide
* 🔴 **Red bars (Positive SHAP):** Factors that **increase** the risk score, pushing the decision towards **Rejection**.
* 🔵 **Blue bars (Negative SHAP):** Factors that **decrease** the risk score, helping the application get **Approved**.

## 👤 Author
- **Ngoc Ly Tran (Henry)**
- Master's Student in Data Science at **Chonnam National University**.
