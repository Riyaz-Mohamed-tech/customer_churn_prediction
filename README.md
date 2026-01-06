
# Customer Churn Prediction System 🚀

An end-to-end machine learning project to predict customer churn, including data analysis, model training, explainability, deployment, and monitoring.

## 🔹 Features
- Data cleaning, EDA, and feature engineering
- ML models: Logistic Regression, Random Forest, XGBoost, LightGBM
- Model explainability using SHAP & LIME
- FastAPI-based REST API for real-time predictions
- Prometheus-based monitoring

## 🔹 Tech Stack
Python, Pandas, NumPy, Scikit-learn, XGBoost, LightGBM, SHAP, LIME, FastAPI

## 🔹 Project Structure

customer_churn/
│── app.py
│── model/
│── notebooks/
│── requirements.txt
│── README.md


## 🔹 Run Locally

```bash
pip install -r requirements.txt
uvicorn app:app --reload


Open:

http://127.0.0.1:8000/docs
Sample Prediction Input
{
  "credit_score": 650,
  "country": "France",
  "gender": "Male",
  "age": 40,
  "tenure": 3,
  "balance": 50000,
  "products_number": 2,
  "credit_card": 1,
  "active_member": 1,
  "estimated_salary": 60000
}

🔹 Author

Mohamed Riyas
