# src/data/preprocess.py

import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split


RAW_DATA_PATH = Path("data/raw/ai4i_2020.csv")
PROCESSED_DATA_PATH = Path("data/processed")


def preprocess_data(
    test_size: float = 0.2,
    random_state: int = 42
):
    """
    Preprocess AI4I predictive maintenance dataset.
    
    Steps:
    - Load raw data
    - Select features and target
    - Encode categorical variables
    - Stratified train-test split
    - Save processed datasets
    """

    # Load raw data
    df = pd.read_csv(RAW_DATA_PATH)

    # ---------------------------
    # Target selection
    # ---------------------------
    target = "Machine failure"

    # Drop individual failure modes (not known at inference time)
    drop_cols = ["TWF", "HDF", "PWF", "OSF", "RNF"]
    df = df.drop(columns=drop_cols)

    X = df.drop(columns=[target])
    y = df[target]

    # ---------------------------
    # Encode categorical features
    # ---------------------------
    X = pd.get_dummies(X, columns=["Type"], drop_first=True)

    # ---------------------------
    # Train-test split
    # ---------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )

    # ---------------------------
    # Save processed data
    # ---------------------------
    PROCESSED_DATA_PATH.mkdir(parents=True, exist_ok=True)

    X_train.to_csv(PROCESSED_DATA_PATH / "X_train.csv", index=False)
    X_test.to_csv(PROCESSED_DATA_PATH / "X_test.csv", index=False)
    y_train.to_csv(PROCESSED_DATA_PATH / "y_train.csv", index=False)
    y_test.to_csv(PROCESSED_DATA_PATH / "y_test.csv", index=False)

    print("Preprocessing completed successfully!")
    print(f"Train size: {X_train.shape}")
    print(f"Test size: {X_test.shape}")


if __name__ == "__main__":
    preprocess_data()
