import pandas as pd
import numpy as np
import os


def generate_employee_data(num_records=500, save_path="data/raw/employee_data.csv"):
    np.random.seed(42)

    departments = ["HR", "IT", "Sales", "Finance", "Marketing"]

    data = {
        "Age": np.random.randint(22, 60, num_records),
        "Experience": np.random.randint(1, 21, num_records),
        "Salary": np.random.randint(25000, 120000, num_records),
        "TrainingHours": np.random.randint(5, 100, num_records),
        "ProjectsCompleted": np.random.randint(1, 15, num_records),
        "AttendanceRate": np.random.randint(60, 101, num_records),
        "Department": np.random.choice(departments, num_records)
    }

    df = pd.DataFrame(data)

    noise = np.random.normal(0, 2.5, num_records)

    performance_score = (
        0.25 * df["Experience"] +
        0.20 * (df["Salary"] / 10000) +
        0.20 * (df["TrainingHours"] / 10) +
        0.20 * df["ProjectsCompleted"] +
        0.15 * (df["AttendanceRate"] / 10) +
        noise
    )

    df["Performance"] = np.where(performance_score >= performance_score.median(), 1, 0)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    df.to_csv(save_path, index=False)

    print(f"Dataset generated and saved to: {save_path}")
    return df