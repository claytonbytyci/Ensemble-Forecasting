from pathlib import Path

import pandas as pd

URL = (
    "https://www.philadelphiafed.org/-/media/FRBP/Assets/"
    "Surveys-And-Data/survey-of-professional-forecasters/"
    "data-files/files/Individual_CPI.xlsx"
)
SPF_CPI_REALTIME_URL = (
    "https://www.philadelphiafed.org/-/media/FRBP/Assets/Surveys-And-Data/"
    "real-time-data/data-files/xlsx/cpiQvMd.xlsx?sc_lang=en&hash=9659566E92130D1A394C7FEDB6EC4115"
)
SPF_CPI_REALIZ5_URL = (
    "https://www.philadelphiafed.org/-/media/FRBP/Assets/Surveys-And-Data/"
    "survey-of-professional-forecasters/data-files/CPI/"
    "Data_SPF_Error_Statistics_CPI_1_AIC.xls?sc_lang=en&hash=E1779D894F61DDC4C9D6C1BECC5ED860"
)
VIX_CSV_URL = (
    "https://fred.stlouisfed.org/graph/fredgraph.csv?"
    "id=VIXCLS&cosd=1990-01-02&coed=2026-02-19"
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
DEFAULT_CSV_PATH = DATA_DIR / "SPF_Individual_CPI.csv"
DEFAULT_SPF_CPI_REALTIME_CSV_PATH = DATA_DIR / "SPF_CPI_realtime_diagonal_source.csv"
DEFAULT_SPF_CPI_REALIZ5_CSV_PATH = DATA_DIR / "SPF_CPI_Realiz5.csv"
DEFAULT_VIX_CSV_PATH = DATA_DIR / "VIXCLS.csv"
M3_QUARTERLY_ACTUALS_URL = "https://forvis.github.io/data/M3_monthly_TSTS.csv"
M3_QUARTERLY_FORECASTS_URL = "https://forvis.github.io/data/M3_monthly_FTS.csv"
DEFAULT_M3_QUARTERLY_ACTUALS_CSV_PATH = DATA_DIR / "M3_monthly_TSTS.csv"
DEFAULT_M3_QUARTERLY_FORECASTS_CSV_PATH = DATA_DIR / "M3_monthly_FTS.csv"


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


def _load_or_download_csv_from_url(csv_path: str | Path, url: str) -> pd.DataFrame:
    """Load local CSV if present, otherwise download from URL and cache to csv_path."""
    path = Path(csv_path)
    if path.exists():
        return pd.read_csv(path)
    df = pd.read_csv(url)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return df


def download_spf_realtime_cpi_actuals(url: str = SPF_CPI_REALTIME_URL) -> pd.DataFrame:
    """
    Download Philadelphia Fed real-time CPI vintage matrix workbook and return first sheet as DataFrame.
    """
    return pd.read_excel(url)


def download_spf_realiz5_actuals(url: str = SPF_CPI_REALIZ5_URL) -> pd.DataFrame:
    """Download SPF CPI error-statistics table containing Realiz5 actuals."""
    return pd.read_excel(url)


def load_or_download_spf_realtime_cpi_actuals(
    csv_path: str | Path = DEFAULT_SPF_CPI_REALTIME_CSV_PATH,
    url: str = SPF_CPI_REALTIME_URL,
) -> pd.DataFrame:
    """Load cached real-time CPI source CSV, or download workbook and cache as CSV."""
    path = Path(csv_path)
    if path.exists():
        return pd.read_csv(path)
    df = pd.read_excel(url)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return df


def load_or_download_spf_realiz5_actuals(
    csv_path: str | Path = DEFAULT_SPF_CPI_REALIZ5_CSV_PATH,
    url: str = SPF_CPI_REALIZ5_URL,
) -> pd.DataFrame:
    """Load cached SPF Realiz5 actuals source CSV, or download workbook and cache as CSV."""
    path = Path(csv_path)
    if path.exists():
        return pd.read_csv(path)
    df = pd.read_excel(url)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return df


def download_vix_data(url: str = VIX_CSV_URL) -> pd.DataFrame:
    """Download VIX daily close CSV from FRED-style endpoint."""
    return pd.read_csv(url)


def load_or_download_vix_data(
    csv_path: str | Path = DEFAULT_VIX_CSV_PATH,
    url: str = VIX_CSV_URL,
) -> pd.DataFrame:
    """Load cached VIX CSV, or download and cache if missing."""
    return _load_or_download_csv_from_url(csv_path=csv_path, url=url)


def download_m3_quarterly_actuals(url: str = M3_QUARTERLY_ACTUALS_URL) -> pd.DataFrame:
    """Download M3 quarterly actual values (TSTS) and return as a DataFrame."""
    return pd.read_csv(url)


def download_m3_quarterly_forecasts(url: str = M3_QUARTERLY_FORECASTS_URL) -> pd.DataFrame:
    """Download M3 quarterly forecaster values (FTS) and return as a DataFrame."""
    return pd.read_csv(url)


def load_or_download_m3_quarterly_actuals(
    csv_path: str | Path = DEFAULT_M3_QUARTERLY_ACTUALS_CSV_PATH,
    url: str = M3_QUARTERLY_ACTUALS_URL,
) -> pd.DataFrame:
    """Load cached M3 quarterly actuals CSV, or download and cache if missing."""
    return _load_or_download_csv_from_url(csv_path=csv_path, url=url)


def load_or_download_m3_quarterly_forecasts(
    csv_path: str | Path = DEFAULT_M3_QUARTERLY_FORECASTS_CSV_PATH,
    url: str = M3_QUARTERLY_FORECASTS_URL,
) -> pd.DataFrame:
    """Load cached M3 quarterly forecasts CSV, or download and cache if missing."""
    return _load_or_download_csv_from_url(csv_path=csv_path, url=url)
