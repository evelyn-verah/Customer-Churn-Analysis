import pandas as pd

def load_logs(path: str) -> pd.DataFrame:
    return pd.read_csv(path)
