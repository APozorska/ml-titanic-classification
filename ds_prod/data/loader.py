import pandas as pd
from pathlib import Path


# TODO: Implement different functions and mechaniscm for selecting correct function
def load_csv(path: str | Path) -> pd.DataFrame:
    """
    Load CSV file...
    Args:
        ...
    Returns:
        ...
    """
    data = pd.read_csv(path)
    return data
