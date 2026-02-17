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


def prepare_m3_monthly_data(
    actuals_df: pd.DataFrame,
    forecasts_df: pd.DataFrame,
    category: str = "MACRO",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Index]:
    """
    Clean M3 monthly actuals/forecasts and keep only the requested actuals category.

    Expected actuals columns:
      - series_id, category, value, timestamp
    Expected forecasts columns:
      - series_id, method_id, forecast, horizon, timestamp, origin_timestamp

    Returns:
      - actuals_clean: cleaned actuals for selected category
      - forecasts_clean: cleaned forecasts filtered to selected category series
      - series_ids: selected series IDs (Index)
    """
    actuals = actuals_df.copy()
    forecasts = forecasts_df.copy()

    actuals.columns = [str(c).strip().lower() for c in actuals.columns]
    forecasts.columns = [str(c).strip().lower() for c in forecasts.columns]

    required_actuals = {"series_id", "category", "value", "timestamp"}
    required_forecasts = {"series_id", "method_id", "forecast", "horizon", "timestamp", "origin_timestamp"}
    missing_a = required_actuals.difference(actuals.columns)
    missing_f = required_forecasts.difference(forecasts.columns)
    if missing_a:
        raise ValueError(f"actuals_df missing required columns: {sorted(missing_a)}")
    if missing_f:
        raise ValueError(f"forecasts_df missing required columns: {sorted(missing_f)}")

    actuals = actuals.replace(r"^\s*$", np.nan, regex=True)
    forecasts = forecasts.replace(r"^\s*$", np.nan, regex=True)

    actuals["series_id"] = actuals["series_id"].astype(str).str.strip()
    actuals["category"] = actuals["category"].astype(str).str.strip().str.upper()
    actuals["actual"] = pd.to_numeric(actuals["value"], errors="coerce")
    actuals["timestamp"] = actuals["timestamp"].astype(str).str.strip()
    actuals["period"] = pd.PeriodIndex(actuals["timestamp"], freq="M")

    cat = str(category).strip().upper()
    actuals = actuals.loc[actuals["category"] == cat].copy()
    actuals = actuals.dropna(subset=["series_id", "actual", "period"]).sort_values(
        ["series_id", "period"]
    )
    actuals = actuals.drop_duplicates(subset=["series_id", "period"], keep="last")

    series_ids = pd.Index(actuals["series_id"].unique(), name="series_id")

    forecasts["series_id"] = forecasts["series_id"].astype(str).str.strip()
    forecasts["method_id"] = forecasts["method_id"].astype(str).str.strip()
    forecasts["forecast"] = pd.to_numeric(forecasts["forecast"], errors="coerce")
    forecasts["horizon"] = pd.to_numeric(forecasts["horizon"], errors="coerce").astype("Int64")
    forecasts["timestamp"] = forecasts["timestamp"].astype(str).str.strip()
    forecasts["origin_timestamp"] = forecasts["origin_timestamp"].astype(str).str.strip()
    forecasts["target_period"] = pd.PeriodIndex(forecasts["timestamp"], freq="M")
    forecasts["origin_period"] = pd.PeriodIndex(forecasts["origin_timestamp"], freq="M")

    forecasts = forecasts.loc[forecasts["series_id"].isin(series_ids)].copy()
    forecasts = forecasts.dropna(
        subset=["series_id", "method_id", "forecast", "horizon", "target_period", "origin_period"]
    )
    forecasts = forecasts.sort_values(["series_id", "method_id", "origin_period", "horizon"])
    forecasts = forecasts.drop_duplicates(
        subset=["series_id", "method_id", "origin_period", "horizon", "target_period"],
        keep="last",
    )

    return actuals, forecasts, series_ids


def align_m3_monthly_actuals_and_forecasts(
    actuals_clean: pd.DataFrame,
    forecasts_clean: pd.DataFrame,
) -> pd.DataFrame:
    """
    Create long-form aligned panel:
      series_id, method_id, horizon, origin_period, target_period, forecast, actual

    Also includes:
      expected_target_period, horizon_consistent
    """
    actuals = actuals_clean.copy()
    forecasts = forecasts_clean.copy()

    actual_lookup = actuals[["series_id", "period", "actual"]].rename(columns={"period": "target_period"})
    aligned = forecasts.merge(actual_lookup, how="left", on=["series_id", "target_period"])
    aligned["expected_target_period"] = aligned["origin_period"] + aligned["horizon"].astype(int)
    aligned["horizon_consistent"] = aligned["expected_target_period"] == aligned["target_period"]

    cols = [
        "series_id",
        "method_id",
        "horizon",
        "origin_period",
        "target_period",
        "expected_target_period",
        "horizon_consistent",
        "forecast",
        "actual",
    ]
    aligned = aligned[cols].sort_values(["series_id", "method_id", "origin_period", "horizon"])
    return aligned.reset_index(drop=True)


def build_m3_series_horizon_matrix(
    aligned_df: pd.DataFrame,
    series_id: str,
    horizon: int,
    require_actual: bool = True,
) -> pd.DataFrame:
    """
    Build an ensemble-ready matrix for one (series_id, horizon):
      index: origin_period
      columns: method_id forecasts + actual + target_period
    """
    d = aligned_df.copy()
    d = d.loc[(d["series_id"] == str(series_id)) & (d["horizon"].astype(int) == int(horizon))].copy()
    if require_actual:
        d = d.loc[d["actual"].notna()].copy()

    pivot = d.pivot_table(
        index="origin_period",
        columns="method_id",
        values="forecast",
        aggfunc="last",
    )
    meta = d.groupby("origin_period", as_index=True).agg(
        actual=("actual", "last"),
        target_period=("target_period", "last"),
    )
    out = meta.join(pivot, how="left").sort_index()
    out.columns.name = None
    return out.reset_index()
