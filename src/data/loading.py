from pathlib import Path

import pandas as pd

URL = (
    "https://www.philadelphiafed.org/-/media/FRBP/Assets/"
    "Surveys-And-Data/survey-of-professional-forecasters/"
    "data-files/files/Individual_CPI.xlsx"
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
DEFAULT_CSV_PATH = DATA_DIR / "SPF_Individual_CPI.csv"


def download_spf_data(url: str = URL) -> pd.DataFrame:
    """Download the SPF individual CPI workbook and return it as a DataFrame."""
    return pd.read_excel(url)


def save_csv(df: pd.DataFrame, csv_path: str | Path = DEFAULT_CSV_PATH) -> Path:
    """Save the DataFrame to CSV and return the resolved path."""
    path = Path(csv_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return path


def load_or_download_csv(
    csv_path: str | Path = DEFAULT_CSV_PATH,
    url: str = URL,
) -> pd.DataFrame:
    """Load local CSV if present, otherwise download the source workbook and cache it."""
    path = Path(csv_path)
    if path.exists():
        return pd.read_csv(path)

    df = pd.read_excel(url)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return df
