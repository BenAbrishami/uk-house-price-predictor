# 🏡 UK House Price Predictor

A machine learning web app that predicts house prices in the UK based on postcode, location, property type, and other features. Built using Python, Scikit-learn, and Streamlit.

---

## 🎯 Problem Statement

Real estate buyers and investors need a simple tool to estimate fair property prices based on local trends and house features. This project delivers a fast and interactive prediction engine.

---

## 🔧 Features

- Predict property sale price from user input
- Input: postcode, property type, number of bedrooms, etc.
- Built with Streamlit for quick interactivity
- Machine Learning regression model using `XGBoost`
- Trained on UK property sale data
- Exportable CSV results (optional)

---

## 📊 Tech Stack

- Python, Pandas, NumPy
- Scikit-learn, XGBoost
- Streamlit
- Git, GitHub

---

## 📁 Project Structure


---

## 🖥️ How to Run Locally

```bash
git clone https://github.com/BenAbrishami/uk-house-price-predictor.git
cd uk-house-price-predictor
pip install -r requirements.txt
streamlit run app/main.py
