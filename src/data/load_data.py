# src/data/load_data.py

from ucimlrepo import fetch_ucirepo
import pandas as pd
from pathlib import Path


def load_ai4i_dataset(save_raw: bool = True) -> pd.DataFrame:
    """
    Fetch AI4I 2020 Predictive Maintenance Dataset from UCI repository.
    
    Args:
        save_raw (bool): Whether to save raw dataset to data/raw
    
    Returns:
        pd.DataFrame: Combined features + targets dataframe
    """
    # Fetch dataset
    dataset = fetch_ucirepo(id=601)

    # Extract features and targets
    X = dataset.data.features
    y = dataset.data.targets

    # Combine into single dataframe
    df = pd.concat([X, y], axis=1)

    if save_raw:
        raw_data_path = Path("data/raw")
        raw_data_path.mkdir(parents=True, exist_ok=True)

        file_path = raw_data_path / "ai4i_2020.csv"
        df.to_csv(file_path, index=False)

    return df


if __name__ == "__main__":
    df = load_ai4i_dataset()
    print("Dataset loaded successfully!")
    print(df.head())
