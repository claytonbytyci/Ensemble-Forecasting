import numpy as np
import pandas as pd


META_COLS = {"YEAR", "QUARTER", "ID", "INDUSTRY"}


def prepare_spf_data(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Apply core cleaning steps and return cleaned data with forecast column names."""
    cleaned = df.copy()

    cleaned = cleaned.replace(r"^\\s*$", np.nan, regex=True)

    cleaned["YEAR"] = pd.to_numeric(cleaned["YEAR"], errors="coerce")
    cleaned["QUARTER"] = pd.to_numeric(cleaned["QUARTER"], errors="coerce")
    cleaned["ID"] = pd.to_numeric(cleaned["ID"], errors="coerce")

    forecast_cols = [c for c in cleaned.columns if c not in META_COLS]
    for col in forecast_cols:
        cleaned[col] = pd.to_numeric(cleaned[col], errors="coerce")

    cleaned = cleaned.dropna(subset=["CPI1"])
    cleaned = cleaned.sort_values(["YEAR", "QUARTER", "ID"]).reset_index(drop=True)

    year_str = cleaned["YEAR"].astype("Int64").astype(str)
    quarter_str = cleaned["QUARTER"].astype("Int64").astype(str)
    period_str = year_str + "Q" + quarter_str
    cleaned["period"] = pd.PeriodIndex(period_str, freq="Q").to_timestamp("Q")

    return cleaned, forecast_cols


def missingness_by_horizon(df: pd.DataFrame, forecast_cols: list[str]) -> pd.Series:
    """Return missing-share by forecast horizon sorted descending."""
    return df[forecast_cols].isna().mean().sort_values(ascending=False)


def average_forecasts_by_period(df: pd.DataFrame, forecast_cols: list[str]) -> pd.DataFrame:
    """Return horizon averages by period."""
    return df.groupby("period")[forecast_cols].mean().sort_index()


def long_forecast_frame(df: pd.DataFrame, forecast_cols: list[str]) -> pd.DataFrame:
    """Return long-format (ID, period, horizon, forecast) view for scatter plots."""
    long = df.melt(
        id_vars=["ID", "period"],
        value_vars=forecast_cols,
        var_name="horizon",
        value_name="forecast",
    )
    return long.dropna(subset=["forecast"])
