import pandas as pd
from pathlib import Path


COLUMNS = (
    ["engine_id", "cycle"]
    + [f"op{i}" for i in range(1, 4)]
    + [f"s{i}" for i in range(1, 22)]
)

KNOWN_LOW_VARIANCE_SENSORS = [
    "s1",
    "s5",
    "s6",
    "s10",
    "s16",
    "s18",
    "s19",
]


def load_cmapss(file_path: Path) -> pd.DataFrame:
    """
    Load NASA CMAPSS train/test file.
    Handles trailing whitespace columns automatically.
    """

    df = pd.read_csv(
        file_path,
        sep=r"\s+",
        header=None,
    )

    df = df.iloc[:, : len(COLUMNS)]
    df.columns = COLUMNS

    df = df.apply(pd.to_numeric, errors="coerce")

    return df


def load_rul(file_path: Path) -> pd.DataFrame:
    """
    Load test-set RUL labels and attach engine IDs.
    """

    rul = pd.read_csv(
        file_path,
        header=None,
        names=["RUL"]
    )

    rul["engine_id"] = range(1, len(rul) + 1)

    return rul[["engine_id", "RUL"]]


def dataset_summary(df: pd.DataFrame) -> None:
    """
    Print useful metadata for EDA.
    """

    print(f"Shape: {df.shape}")
    print(f"Engines: {df['engine_id'].nunique()}")
    print(f"Max Cycle: {df['cycle'].max()}")
    print(f"Missing Values: {df.isnull().sum().sum()}")


def main() -> None:

    DATA_DIR = Path("data/raw")

    train_df = load_cmapss(DATA_DIR / "train_FD001.txt")
    test_df = load_cmapss(DATA_DIR / "test_FD001.txt")
    rul_df = load_rul(DATA_DIR / "RUL_FD001.txt")

    print("\nTRAIN DATA")
    dataset_summary(train_df)

    print("\nTEST DATA")
    dataset_summary(test_df)

    print("\nRUL DATA")
    print(rul_df.shape)

    assert train_df.shape[1] == len(COLUMNS)
    assert test_df.shape[1] == len(COLUMNS)

    print("\nLoader validation passed.")


if __name__ == "__main__":
    main()