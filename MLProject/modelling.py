import os
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

def train():
    # Set tracking URI dari environment variable (di-set oleh GitHub Actions)
    tracking_uri = os.environ.get('MLFLOW_TRACKING_URI')
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
        print(f"Tracking URI: {tracking_uri}")

    df = pd.read_csv('TelcoCustomerChurn_raw_preprocessing/data_clean.csv')

    X = df.drop('Churn', axis=1)
    y = df['Churn']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    with mlflow.start_run(run_name="CI_Retraining_Model"):
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)

        # Log model secara eksplisit dengan artifact_path="model"
        # Ini WAJIB ada agar CI bisa download dan build Docker
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",  # <-- harus "model" sesuai MODEL_PATH di CI
            registered_model_name="telco-churn-model"
        )

        print("Model CI berhasil dilatih dan di-log ke MLflow!")

if __name__ == "__main__":
    train()
