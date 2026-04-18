import matplotlib.pyplot as plt
import seaborn as sns
import os


def perform_eda(df, image_dir="images"):
    os.makedirs(image_dir, exist_ok=True)

    # Performance distribution
    plt.figure(figsize=(6, 4))
    sns.countplot(x="Performance", data=df)
    plt.title("Performance Distribution")
    plt.xlabel("Performance (0 = Low, 1 = High)")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(f"{image_dir}/performance_distribution.png")
    plt.close()

    # Department-wise performance
    plt.figure(figsize=(8, 5))
    sns.countplot(x="Department", hue="Performance", data=df)
    plt.title("Department-wise Performance")
    plt.xticks(rotation=30)
    plt.tight_layout()
    plt.savefig(f"{image_dir}/department_performance.png")
    plt.close()

    print("EDA graphs saved in images folder.")