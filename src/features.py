import pandas as pd

"""
This function takes in the df of data and adds the engineered features as new columns into a new df
"""
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df_engineered = df.copy()

    return df_engineered