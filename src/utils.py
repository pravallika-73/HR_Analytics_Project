import pandas as pd

def get_kpi_metrics(df):
    total_employees = df.shape[0]
    attrition_rate = round(df['Attrition'].mean() * 100, 2)
    avg_income = round(df['MonthlyIncome'].mean(), 2)

    return total_employees, attrition_rate, avg_income