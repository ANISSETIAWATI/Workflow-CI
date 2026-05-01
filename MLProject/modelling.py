import os
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

def train():
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

        # Log model tanpa registered_model_name
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model"
        )

        print("Model berhasil di-log!")
        print(f"Artifact URI: {mlflow.get_artifact_uri('model')}")

if __name__ == "__main__":
    train()
