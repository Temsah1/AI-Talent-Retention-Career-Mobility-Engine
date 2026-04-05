<h1 align="center">🧠 AI Talent Retention & Career Mobility Engine v2.0</h1>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.9+-blue.svg" alt="Python Version">
  <img src="https://img.shields.io/badge/Streamlit-Framework-FF4B4B.svg" alt="Streamlit">
  <img src="https://img.shields.io/badge/Machine_Learning-Random_Forest-F7931E.svg" alt="Machine Learning">
  <img src="https://img.shields.io/badge/NLP-TF--IDF-4CAF50.svg" alt="NLP">
  <img src="https://img.shields.io/badge/Database-SQLite3-003B57.svg" alt="SQLite">
</p>

<p align="center">
  <strong>An Enterprise-grade ML & NLP application built to predict employee churn risk and recommend intelligent internal career paths, helping HR teams make data-driven decisions.</strong>
</p>

---

## 📌 Project Overview
Employee turnover costs tech giants millions of dollars annually. This **B2B SaaS application** provides HR departments with a proactive, data-driven approach to talent management. By leveraging **Machine Learning** and **Natural Language Processing (NLP)**, the engine predicts the flight risk of current employees and intelligently suggests internal mobility paths to retain top talent.

### 🌟 What's New in v2.0?
* Fully responsive UI with an **Animated Navbar**.
* Robust **SQLite3** database integration for secure user and admin management.
* Enhanced Dark-Mode styling for a premium Enterprise feel.

## ✨ Core Features

* **🔐 Secure Authentication & RBAC:**
  * Custom-built login/signup system using `SQLite3` and `hashlib` password encryption.
  * Role-Based Access Control (RBAC) to separate standard users from Admins.
* **📊 HR Executive Dashboard:**
  * Real-time metrics overview (Total Employees, Average Satisfaction, Retention Rate).
  * Interactive Plotly visualizations including a **Correlation Heatmap** to identify factors driving employee churn.
* **⚠️ Flight Risk Predictor (ML):**
  * Uses a trained `RandomForestClassifier` to predict the probability of an employee resigning.
  * Features an interactive Gauge Chart and **Explainable AI (Feature Importance)** to show HR *why* an employee is at risk (e.g., Low Salary + High Working Hours).
* **🧠 Smart Career Mobility (NLP):**
  * An AI recommendation engine utilizing `TF-IDF Vectorization` and `Cosine Similarity`.
  * Matches an employee's current skills against internal open roles.
  * Highlights the **Skill Gap** to help L&D (Learning & Development) teams build personalized training tracks.
* **⚙️ Synthetic Data Generator:**
  * Automatically generates a highly realistic, statistically correlated dataset of 2,000 employees for demonstration purposes.

## 🛠️ Technology Stack
* **Frontend/UI:** Streamlit, Custom CSS (Modern Dark-Mode, Glassmorphism, Animations)
* **Data Manipulation:** Pandas, NumPy
* **Machine Learning & NLP:** Scikit-Learn
* **Data Visualization:** Plotly Express, Plotly Graph Objects
* **Database Management:** SQLite3

## 🚀 Installation & Local Setup

**1. Clone the repository:**
```bash
git clone [https://github.com/yourusername/ai-talent-engine.git](https://github.com/yourusername/ai-talent-engine.git)
cd ai-talent-engine
