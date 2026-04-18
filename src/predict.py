import pandas as pd


def predict_new_employee(model, sample_input, expected_columns):
    sample_df = pd.DataFrame([sample_input])

    sample_df = pd.get_dummies(sample_df)

    sample_df = sample_df.reindex(columns=expected_columns, fill_value=0)

    prediction = model.predict(sample_df)[0]

    return prediction