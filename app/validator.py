import pandas as pd

def validate_input(data: dict):
    df = pd.DataFrame([data])

    # Simple validation rules
    if df.isnull().any().any():
        return False, "Missing values detected"

    if (df < 0).any().any():
        return False, "Negative values not allowed"

    if (df > 10).any().any():
        return False, "Values out of expected range"

    return True, "Valid input"