from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import os


def evaluate_model(model, X_test, y_test, feature_names,
                   metrics_path="outputs/metrics.txt",
                   image_dir="images",
                   feature_path="outputs/feature_importance.csv"):
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred)

    os.makedirs(os.path.dirname(metrics_path), exist_ok=True)
    os.makedirs(image_dir, exist_ok=True)

    with open(metrics_path, "w") as f:
        f.write(f"Accuracy: {accuracy:.4f}\n\n")
        f.write("Classification Report:\n")
        f.write(report)

    # Confusion matrix plot
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(f"{image_dir}/confusion_matrix.png")
    plt.close()

    # Feature importance
    importance_df = pd.DataFrame({
        "Feature": feature_names,
        "Importance": model.feature_importances_
    }).sort_values(by="Importance", ascending=False)

    importance_df.to_csv(feature_path, index=False)

    plt.figure(figsize=(10, 6))
    sns.barplot(x="Importance", y="Feature", data=importance_df)
    plt.title("Feature Importance")
    plt.tight_layout()
    plt.savefig(f"{image_dir}/feature_importance.png")
    plt.close()

    print(f"Accuracy: {accuracy:.4f}")
    print("Evaluation complete. Metrics and images saved.")

    return y_pred, accuracy, report, importance_df