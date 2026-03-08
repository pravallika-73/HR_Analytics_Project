# -----------------------------
# 1️⃣ Imports (ALWAYS FIRST)
# -----------------------------

import streamlit as st
import pandas as pd
import os

from src.preprocessing import load_data, preprocess_data
from src.model import train_model
from src.utils import get_kpi_metrics


# -----------------------------
# 2️⃣ Page Configuration
# -----------------------------

st.set_page_config(
    page_title="HR Analytics Dashboard",
    layout="wide"
)

st.title("HR Analytics - Employee Attrition & Performance")


# -----------------------------
# 3️⃣ Load Data
# -----------------------------

df = load_data("data/WA_Fn-UseC_-HR-Employee-Attrition.csv")
df_processed = preprocess_data(df)


# -----------------------------
# 4️⃣ Train Model
# -----------------------------

model, accuracy = train_model(df_processed)


# -----------------------------
# 5️⃣ KPI Metrics
# -----------------------------

total_employees, attrition_rate, avg_income = get_kpi_metrics(df_processed)

col1, col2, col3 = st.columns(3)

col1.metric("Total Employees", total_employees)
col2.metric("Attrition Rate (%)", attrition_rate)
col3.metric("Average Monthly Income", avg_income)


# -----------------------------
# 6️⃣ Model Accuracy
# -----------------------------

st.subheader("Model Accuracy")
st.success(f"{round(accuracy * 100, 2)} %")


# -----------------------------
# 7️⃣ Attrition Distribution
# -----------------------------

st.subheader("Attrition Distribution")
st.bar_chart(df_processed["Attrition"].value_counts())