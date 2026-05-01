import os
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# Metrics/params tetap autolog, tapi model kita log manual agar path-nya pasti "model"
mlflow.sklearn.autolog(log_models=False)

def train():
    df = pd.read_csv("TelcoCustomerChurn_raw_preprocessing/data_clean.csv")

    X = df.drop("Churn", axis=1)
    y = df["Churn"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    with mlflow.start_run(run_name="CI_Retraining_Model") as run:
        model = RandomForestClassifier(
            n_estimators=100,
            random_state=42
        )

        model.fit(X_train, y_train)

        # Ini yang penting untuk build-docker runs:/$RUN_ID/model
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model"
        )

        print("Model CI berhasil dilatih!")
        print(f"RUN_ID={run.info.run_id}")
        print("MODEL_PATH=model")

        # Supaya GitHub Actions langsung punya RUN_ID dan MODEL_PATH yang benar
        github_env = os.getenv("GITHUB_ENV")
        if github_env:
            with open(github_env, "a") as f:
                f.write(f"RUN_ID={run.info.run_id}\n")
                f.write("MODEL_PATH=model\n")

if __name__ == "__main__":
    train()
