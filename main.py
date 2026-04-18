from src.utils import ensure_directories
from src.data_generator import generate_employee_data
from src.preprocess import preprocess_data, prepare_features
from src.eda import perform_eda
from src.train import train_model
from src.evaluate import evaluate_model
from src.predict import predict_new_employee


def main():
    print("Starting Employee Performance Predictor project...")

    # Step 1: Create required folders
    ensure_directories()
    print("Directories created")

    # Step 2: Generate synthetic employee dataset
    df_raw = generate_employee_data()
    print("Raw dataset generated")
    print(df_raw.head())

    # Step 3: Preprocess dataset
    df_clean = preprocess_data()
    print("Cleaned dataset generated")
    print(df_clean.head())

    # Step 4: Perform EDA
    perform_eda(df_clean)
    print("EDA completed")

    # Step 5: Prepare features and target
    X, y, df_encoded = prepare_features(df_clean)
    print("Features prepared successfully")
    print("Feature shape:", X.shape)
    print("Target shape:", y.shape)

    # Step 6: Train model
    model, X_train, X_test, y_train, y_test = train_model(X, y)
    print("Model training completed")

    # Step 7: Evaluate model
    y_pred, accuracy, report, importance_df = evaluate_model(
        model=model,
        X_test=X_test,
        y_test=y_test,
        feature_names=X.columns
    )
    print("Model evaluation completed")
    print(f"Final Accuracy: {accuracy:.4f}")

    # Step 8: Predict a new employee
    new_employee = {
        "Age": 30,
        "Experience": 5,
        "Salary": 55000,
        "TrainingHours": 40,
        "ProjectsCompleted": 6,
        "AttendanceRate": 90,
        "Department": "IT"
    }

    prediction = predict_new_employee(model, new_employee, X.columns)

    print("\nNew Employee Input:")
    print(new_employee)

    if prediction == 1:
        print("Predicted Performance: High Performer")
    else:
        print("Predicted Performance: Low Performer")

    print("\nProject completed successfully.")


if __name__ == "__main__":
    main()