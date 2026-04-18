import pandas as pd
import os
from pandas.api.types import is_numeric_dtype


def preprocess_data(input_path="data/raw/employee_data.csv",
                    output_path="data/processed/cleaned_employee_data.csv"):
    df = pd.read_csv(input_path)

    df = df.drop_duplicates()

    for col in df.columns:
        if is_numeric_dtype(df[col]):
            df[col] = df[col].fillna(df[col].median())
        else:
            df[col] = df[col].fillna(df[col].mode()[0])

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)

    print(f"Cleaned data saved to: {output_path}")
    return df


def prepare_features(df):
    df_encoded = pd.get_dummies(df, columns=["Department"], drop_first=True)

    X = df_encoded.drop("Performance", axis=1)
    y = df_encoded["Performance"]

    return X, y, df_encoded
