import os
from pathlib import Path

import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split


MODEL_PATH = "model"
TARGET_COLUMN = "Churn"

# Log params/metrics otomatis, tapi model disimpan manual agar path artifact pasti "model".
mlflow.sklearn.autolog(log_models=False)


def find_dataset() -> Path:
    """Cari data_clean.csv dari beberapa lokasi umum di repo/MLProject."""
    env_path = os.getenv("DATA_PATH")
    candidates = []

    if env_path:
        candidates.append(Path(env_path))

    candidates.extend(
        [
            Path("TelcoCustomerChurn_raw_preprocessing/data_clean.csv"),
            Path("preprocessing/TelcoCustomerChurn_raw_preprocessing/data_clean.csv"),
            Path("../preprocessing/TelcoCustomerChurn_raw_preprocessing/data_clean.csv"),
        ]
    )

    for path in candidates:
        if path.exists():
            return path

    checked = "\n".join(f"- {path}" for path in candidates)
    raise FileNotFoundError(
        "data_clean.csv tidak ditemukan. Lokasi yang sudah dicek:\n" + checked
    )


def train() -> None:
    data_path = find_dataset()
    print(f"Menggunakan dataset: {data_path.resolve()}")
    print(f"MLFLOW_TRACKING_URI: {mlflow.get_tracking_uri()}")
    print(f"MLFLOW_RUN_ID dari environment: {os.getenv('MLFLOW_RUN_ID')}")

    df = pd.read_csv(data_path)

    if TARGET_COLUMN not in df.columns:
        raise ValueError(f"Kolom target '{TARGET_COLUMN}' tidak ditemukan di dataset.")

    X = df.drop(TARGET_COLUMN, axis=1)
    y = df[TARGET_COLUMN]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y if y.nunique() > 1 else None,
    )

    with mlflow.start_run(run_name="CI_Retraining_Model") as run:
        model = RandomForestClassifier(
            n_estimators=100,
            random_state=42,
        )
        model.fit(X_train, y_train)

        test_accuracy = model.score(X_test, y_test)
        mlflow.log_metric("test_accuracy", test_accuracy)

        # Penting: path ini harus sama dengan MODEL_PATH di workflow Docker.
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path=MODEL_PATH,
        )

        run_id = run.info.run_id
        model_uri = f"runs:/{run_id}/{MODEL_PATH}"

        print("Model CI berhasil dilatih!")
        print(f"RUN_ID={run_id}")
        print(f"MODEL_PATH={MODEL_PATH}")
        print(f"MODEL_URI={model_uri}")
        print(f"TEST_ACCURACY={test_accuracy}")

        # Supaya step berikutnya di GitHub Actions langsung memakai run/model yang benar.
        github_env = os.getenv("GITHUB_ENV")
        if github_env:
            with open(github_env, "a", encoding="utf-8") as env_file:
                env_file.write(f"RUN_ID={run_id}\n")
                env_file.write(f"MODEL_PATH={MODEL_PATH}\n")
                env_file.write(f"MODEL_URI={model_uri}\n")


if __name__ == "__main__":
    train()
