print("main.py started")

from src.utils import ensure_directories
from src.data_generator import generate_employee_data
from src.preprocess import preprocess_data, prepare_features


def main():
    print("Inside main function")

    ensure_directories()
    print("Directories created")

    df_raw = generate_employee_data()
    print("Raw dataset generated")
    print(df_raw.head())

    df_clean = preprocess_data()
    print("Cleaned dataset generated")
    print(df_clean.head())

    X, y, df_encoded = prepare_features(df_clean)
    print("Features prepared")
    print("X shape:", X.shape)
    print("y shape:", y.shape)


if __name__ == "__main__":
    main()