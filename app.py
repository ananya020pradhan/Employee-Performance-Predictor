import os
import joblib
import pandas as pd
import streamlit as st

from src.utils import ensure_directories
from src.data_generator import generate_employee_data
from src.preprocess import preprocess_data, prepare_features
from src.train import train_model
from src.predict import predict_new_employee


st.set_page_config(page_title="Employee Performance Predictor", layout="wide")


@st.cache_resource
def load_or_train_model():
    ensure_directories()

    model_path = "models/performance_model.pkl"
    clean_path = "data/processed/cleaned_employee_data.csv"

    if os.path.exists(model_path) and os.path.exists(clean_path):
        model = joblib.load(model_path)
        df_clean = pd.read_csv(clean_path)
        X, y, _ = prepare_features(df_clean)
        return model, X.columns, df_clean

    generate_employee_data()
    df_clean = preprocess_data()
    X, y, _ = prepare_features(df_clean)
    model, _, _, _, _ = train_model(X, y)

    return model, X.columns, df_clean


model, expected_columns, df_clean = load_or_train_model()

st.title("Employee Performance Predictor using Data Analytics")
st.write("Predict whether an employee is likely to be a high performer or low performer.")

st.subheader("Dataset Preview")
st.dataframe(df_clean.head())

st.subheader("Enter Employee Details")

col1, col2 = st.columns(2)

with col1:
    age = st.slider("Age", 22, 60, 30)
    experience = st.slider("Experience", 1, 20, 5)
    salary = st.number_input("Salary", min_value=25000, max_value=120000, value=55000, step=1000)
    training_hours = st.slider("Training Hours", 5, 100, 40)

with col2:
    projects_completed = st.slider("Projects Completed", 1, 15, 6)
    attendance_rate = st.slider("Attendance Rate", 60, 100, 90)
    department = st.selectbox("Department", ["HR", "IT", "Sales", "Finance", "Marketing"])

if st.button("Predict Performance"):
    new_employee = {
        "Age": age,
        "Experience": experience,
        "Salary": salary,
        "TrainingHours": training_hours,
        "ProjectsCompleted": projects_completed,
        "AttendanceRate": attendance_rate,
        "Department": department
    }

    prediction = predict_new_employee(model, new_employee, expected_columns)

    if prediction == 1:
        st.success("Prediction: High Performer")
    else:
        st.error("Prediction: Low Performer")