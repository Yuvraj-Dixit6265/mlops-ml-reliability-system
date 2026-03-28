import pandas as pd
import os

def detect_drift():

    log_file = "logs/data_log.csv"

    if not os.path.exists(log_file):
        return {"error": "No production data available yet"}

    train_df = pd.read_csv("data/train_data.csv")
    prod_df = pd.read_csv(log_file)

    # 🔥 Rename training columns to match production
    column_mapping = {
        "sepal length (cm)": "sepal_length",
        "sepal width (cm)": "sepal_width",
        "petal length (cm)": "petal_length",
        "petal width (cm)": "petal_width"
    }

    train_df = train_df.rename(columns=column_mapping)

    # Remove prediction column from prod data
    if "prediction" in prod_df.columns:
        prod_df = prod_df.drop(columns=["prediction"])

    drift_report = {}
    alert = False

    for col in train_df.columns:
        train_mean = train_df[col].mean()
        prod_mean = prod_df[col].mean()

        diff = abs(train_mean - prod_mean)

        drift_report[col] = diff

        # 🚨 Threshold check
        if diff > 1.0:
            alert = True

    return {
        "drift": drift_report,
        "alert": alert
    }