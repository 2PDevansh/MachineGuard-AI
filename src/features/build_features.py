import pandas as pd
import numpy as np
from pathlib import Path


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Domain-driven feature engineering for AI4I predictive maintenance dataset.
    Column names aligned with preprocessed data.
    """
    df = df.copy()

    # -----------------------------
    # Temperature-based features
    # -----------------------------
    df["temp_diff"] = df["Process temperature"] - df["Air temperature"]

    df["thermal_load"] = (
        df["Process temperature"] * df["Rotational speed"]
    )

    # -----------------------------
    # Power-based features
    # -----------------------------
    df["power"] = (
        df["Torque"] *
        (2 * np.pi * df["Rotational speed"] / 60)
    )

    df["speed_torque_ratio"] = (
        df["Rotational speed"] / (df["Torque"] + 1e-6)
    )

    # -----------------------------
    # Wear-based features
    # -----------------------------
    df["wear_torque_interaction"] = (
        df["Tool wear"] * df["Torque"]
    )

    # Product-type-aware normalized wear
    # Type_H is implicit when Type_L == 0 and Type_M == 0
    df["normalized_wear"] = df["Tool wear"] * (
        1
        + 0.2 * (1 - df["Type_L"] - df["Type_M"])  # Type_H
        + 0.1 * df["Type_M"]
    )

    # -----------------------------
    # Log-scaled features
    # -----------------------------
    df["log_power"] = np.log1p(df["power"])
    df["log_tool_wear"] = np.log1p(df["Tool wear"])

    return df


def main():
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    DATA_DIR = PROJECT_ROOT / "src" / "data" / "data" / "processed"

    X_train = pd.read_csv(DATA_DIR / "X_train.csv")
    X_test = pd.read_csv(DATA_DIR / "X_test.csv")

    X_train_fe = engineer_features(X_train)
    X_test_fe = engineer_features(X_test)

    X_train_fe.to_csv(DATA_DIR / "X_train_fe.csv", index=False)
    X_test_fe.to_csv(DATA_DIR / "X_test_fe.csv", index=False)

    print("✅ Feature engineering completed successfully")
    print(f"X_train_fe shape: {X_train_fe.shape}")
    print(f"X_test_fe shape: {X_test_fe.shape}")

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Applies full feature engineering pipeline.
    Used by BOTH training and inference.
    """

    df = df.copy()

    # Temperature features
    df["temp_diff"] = df["Process temperature"] - df["Air temperature"]
    df["thermal_load"] = df["temp_diff"] * df["Tool wear"]

    # Power features
    df["rotational_speed_rad"] = df["Rotational speed"] * (2 * 3.1416 / 60)
    df["power"] = df["Torque"] * df["rotational_speed_rad"]
    df["speed_torque_ratio"] = df["Torque"] / (df["Rotational speed"] + 1)

    # Wear interactions
    df["wear_torque_interaction"] = df["Tool wear"] * df["Torque"]
    df["normalized_wear"] = df["Tool wear"] / (df["Tool wear"].max() + 1)

    # Log transforms (stability)
    df["log_power"] = np.log1p(df["power"])
    df["log_tool_wear"] = np.log1p(df["Tool wear"])

    # One-hot encoding
    if "Type" in df.columns:
        df = pd.get_dummies(df, columns=["Type"], drop_first=True)

    # Ensure dummy columns exist
    for col in ["Type_L", "Type_M"]:
        if col not in df.columns:
            df[col] = 0

    # Drop temp columns
    df.drop(columns=["rotational_speed_rad"], inplace=True, errors="ignore")

    return df

if __name__ == "__main__":
    main()
