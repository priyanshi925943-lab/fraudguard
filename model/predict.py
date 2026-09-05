import pandas as pd
import numpy as np
import joblib
import os

def predict_risk(csv_path):
    # Load saved model bundle
    bundle_path = os.path.join(os.path.dirname(__file__), "fraud_model.pkl")
    bundle   = joblib.load(bundle_path)
    model    = bundle["model"]
    scaler   = bundle["scaler"]
    columns  = bundle["columns"]  
    # columns = ['V1'...'V28', 'Amount']

    # Load uploaded CSV
    df          = pd.read_csv(csv_path)
    original_df = df.copy()

    # Drop Class column if exists
    if "Class" in df.columns:
        df = df.drop(columns=["Class"])

    # Drop Time column if exists
    if "Time" in df.columns:
        df = df.drop(columns=["Time"])

    # Scale Amount column
    df["Amount"] = scaler.transform(
        df["Amount"].values.reshape(-1, 1)
    )

    # Keep exactly the columns model was trained on
    df = df[columns]

    # Get fraud probabilities
    probabilities = model.predict_proba(df)[:, 1]

    # Assign risk labels
    def get_risk_label(prob):
        if prob >= 0.6:   return "High"
        elif prob >= 0.3: return "Medium"
        else:             return "Low"

    # Build results DataFrame
    result = pd.DataFrame()
    result["Amount"]     = original_df["Amount"]
    result["Risk_Score"] = (probabilities * 100).round(2)
    result["Risk_Label"] = [get_risk_label(p) for p in probabilities]

    return result


if __name__ == "__main__":
    result = predict_risk("data/raw/creditcard.csv")
    print(f"Total: {len(result)}")
    print(f"High  : {(result['Risk_Label'] == 'High').sum()}")
    print(f"Medium: {(result['Risk_Label'] == 'Medium').sum()}")
    print(f"Low   : {(result['Risk_Label'] == 'Low').sum()}")
    print(result.head())