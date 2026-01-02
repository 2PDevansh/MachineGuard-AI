# src/models/train_xgboost.py

import pandas as pd
from pathlib import Path
from xgboost import XGBClassifier
from sklearn.metrics import (
    roc_auc_score,
    precision_score,
    recall_score,
    classification_report
)
import joblib


def load_data():
    """
    Load engineered features and targets.
    """
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    DATA_DIR = PROJECT_ROOT / "src" / "data" / "data" / "processed"

    X_train = pd.read_csv(DATA_DIR / "X_train_fe.csv")
    X_test = pd.read_csv(DATA_DIR / "X_test_fe.csv")
    y_train = pd.read_csv(DATA_DIR / "y_train.csv").values.ravel()
    y_test = pd.read_csv(DATA_DIR / "y_test.csv").values.ravel()

    return X_train, X_test, y_train, y_test


def train_model(X_train, y_train):
    """
    Train XGBoost classifier.
    """
    model = XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=42,
        n_jobs=-1
    )

    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X_test, y_test):
    """
    Evaluate model performance.
    """
    y_proba = model.predict_proba(X_test)[:, 1]
    y_pred = (y_proba >= 0.5).astype(int)

    roc_auc = roc_auc_score(y_test, y_proba)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)

    print("\n📊 Model Evaluation Metrics")
    print(f"ROC-AUC   : {roc_auc:.4f}")
    print(f"Precision : {precision:.4f}")
    print(f"Recall    : {recall:.4f}")

    print("\n📄 Classification Report")
    print(classification_report(y_test, y_pred))


def save_model(model):
    """
    Save trained model artifact.
    """
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    MODEL_DIR = PROJECT_ROOT / "artifacts" / "models"
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    model_path = MODEL_DIR / "xgboost_model.pkl"
    joblib.dump(model, model_path)

    print(f"\n✅ Model saved at: {model_path}")


def main():
    X_train, X_test, y_train, y_test = load_data()

    print("✅ Data loaded successfully")
    print(f"Train shape: {X_train.shape}")
    print(f"Test shape : {X_test.shape}")

    model = train_model(X_train, y_train)
    print("\n🚀 Model training completed")

    evaluate_model(model, X_test, y_test)
    save_model(model)


if __name__ == "__main__":
    main()
