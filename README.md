# 🌾 Agri-Smart AI: Advanced Crop Recommendation System

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-Framework-FF4B4B.svg)
![Machine Learning](https://img.shields.io/badge/Machine%20Learning-Random%20Forest-brightgreen.svg)

## 📌 Project Overview
Agri-Smart AI is a predictive agriculture dashboard that acts as a digital agronomist. By analyzing chemical soil makeup (Nitrogen, Phosphorus, Potassium, pH) and real-time environmental data, the system uses a **Random Forest Ensemble Model** to recommend the optimal crop for maximizing yield.

This project moves beyond static historical datasets by integrating live satellite telemetry to provide hyper-accurate, real-time predictions.

## ✨ Advanced Features
* **🌍 Live Weather API Integration:** Connects to the OpenWeatherMap REST API to fetch real-time temperature and humidity data based on the user's city.
* **🧠 Machine Learning Engine:** Powered by a Random Forest Classifier trained on 2,200 agricultural profiles, featuring a dynamic confidence breakdown (probability scoring) for the top 3 crop candidates.
* **📊 Interactive Data Analytics:** Built-in Exploratory Data Analysis (EDA) dashboard utilizing `Plotly` for interactive scatter plots and dataset statistical summaries.
* **🧪 Model Diagnostics:** Academic-grade evaluation tab featuring a live Confusion Matrix heatmap and precision/recall metrics to prove mathematical reliability.
* **🎨 Modern UI/UX:** Built with Streamlit, featuring custom CSS animations, Lottie graphics, and a clean, tabbed interface.

## 🛠️ Tech Stack
* **Frontend/UI:** Streamlit, CSS, Lottie
* **Data Manipulation:** Pandas, NumPy
* **Machine Learning:** Scikit-Learn (Random Forest Classifier)
* **Data Visualization:** Plotly Express
* **APIs:** Requests, OpenWeatherMap

## 🚀 How to Run Locally

**1. Clone the repository**
```bash
git clone [https://github.com/YOUR_GITHUB_USERNAME/agri-smart-ai.git](https://github.com/YOUR_GITHUB_USERNAME/agri-smart-ai.git)
cd agri-smart-ai
