import numpy as np
import pandas as pd
import re


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


def _parse_quarter_period(value: object) -> pd.Period | None:
    """Parse broad quarter labels (e.g., 1992Q3, 1992-Q3, 1992-09-30) to pandas Period('Q')."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None

    s = str(value).strip()
    if not s:
        return None

    m = re.match(r"^(\d{4})\s*[-/]?\s*[Qq]\s*([1-4])$", s)
    if m:
        return pd.Period(f"{m.group(1)}Q{m.group(2)}", freq="Q")

    m = re.match(r"^[A-Za-z]+(\d{2})\s*[Qq]\s*([1-4])$", s)
    if m:
        yy = int(m.group(1))
        # RTDSM files use 2-digit years in headers (e.g., CPI65Q4, CPI00Q1).
        year = 1900 + yy if yy >= 50 else 2000 + yy
        return pd.Period(f"{year}Q{m.group(2)}", freq="Q")

    m = re.match(r"^(\d{2})\s*[Qq]\s*([1-4])$", s)
    if m:
        yy = int(m.group(1))
        year = 1900 + yy if yy >= 50 else 2000 + yy
        return pd.Period(f"{year}Q{m.group(2)}", freq="Q")

    m = re.match(r"^(\d{4})\s*([1-4])$", s)
    if m:
        return pd.Period(f"{m.group(1)}Q{m.group(2)}", freq="Q")

    # Permissive fallback for labels containing an embedded y/q token
    # e.g., "CPI65Q4", "vintage_CPI2008Q3", "CPI_00Q1_rev"
    m = re.search(r"(\d{4})\s*[Qq]\s*([1-4])", s)
    if m:
        return pd.Period(f"{m.group(1)}Q{m.group(2)}", freq="Q")
    m = re.search(r"(\d{2})\s*[Qq]\s*([1-4])", s)
    if m:
        yy = int(m.group(1))
        year = 1900 + yy if yy >= 50 else 2000 + yy
        return pd.Period(f"{year}Q{m.group(2)}", freq="Q")

    m = re.match(r"^(\d{4})[:/-](\d{1,2})$", s)
    if m:
        year = int(m.group(1))
        token = int(m.group(2))
        # SPF error-stats files often encode quarter as YYYY:01..04.
        # Treat 1..4 as explicit quarter code; only map >4 as month.
        if 1 <= token <= 4:
            return pd.Period(f"{year}Q{token}", freq="Q")
        if 1 <= token <= 12:
            q = ((token - 1) // 3) + 1
            return pd.Period(f"{year}Q{q}", freq="Q")

    dt = pd.to_datetime(s, errors="coerce")
    if pd.notna(dt):
        return dt.to_period("Q")
    return None


def clean_spf_realiz5_actuals(
    realiz_df: pd.DataFrame,
    realiz_col: str = "Realiz5",
) -> pd.DataFrame:
    """
    Clean SPF Realiz5 table into a 2-column quarterly series:
      - time
      - cpi_actual
    """
    d = realiz_df.copy()
    colmap = {str(c).strip(): c for c in d.columns}
    if realiz_col in colmap:
        real_col = colmap[realiz_col]
    else:
        # Case-insensitive / punctuation-insensitive fallback.
        target = re.sub(r"[^a-z0-9]", "", realiz_col.lower())
        normalized = {
            re.sub(r"[^a-z0-9]", "", str(k).lower()): v
            for k, v in colmap.items()
        }
        if target in normalized:
            real_col = normalized[target]
        else:
            candidates = [v for k, v in normalized.items() if "realiz5" in k or "realization" in k]
            if not candidates:
                raise ValueError(
                    f"Column '{realiz_col}' not found in Realiz5 dataset. "
                    f"Available columns sample: {list(d.columns)[:12]}"
                )
            real_col = candidates[0]

    # Infer quarter time from YEAR/QUARTER or DATE/PERIOD-style column.
    if "YEAR" in colmap and "QUARTER" in colmap:
        yy = pd.to_numeric(d[colmap["YEAR"]], errors="coerce")
        qq = pd.to_numeric(d[colmap["QUARTER"]], errors="coerce")
        mask = yy.notna() & qq.notna()
        periods = pd.Series(pd.PeriodIndex([f"{int(y)}Q{int(q)}" for y, q in zip(yy[mask], qq[mask])], freq="Q"), index=d.index[mask])
        d = d.loc[mask].copy()
        d["period"] = periods.values
    else:
        date_candidates = [
            c for c in d.columns
            if str(c).strip().upper() in {"DATE", "PERIOD", "TIME", "YEARQ", "YQ"}
        ]
        obs_col = date_candidates[0] if date_candidates else d.columns[0]
        d["period"] = d[obs_col].apply(_parse_quarter_period)
        d = d.loc[d["period"].notna()].copy()

    d["cpi_actual"] = pd.to_numeric(d[real_col], errors="coerce")
    d = d.dropna(subset=["period", "cpi_actual"]).copy()
    d = d.sort_values("period").drop_duplicates(subset=["period"], keep="last")
    d["time"] = d["period"].dt.to_timestamp("Q")
    out = d[["time", "cpi_actual"]].reset_index(drop=True)
    return out


def realiz5_to_actual_inflation(
    cleaned_realiz5: pd.DataFrame,
    assume_already_annualized: bool = True,
) -> pd.DataFrame:
    """
    Convert cleaned Realiz5 2-column series to comparison-ready actual series.

    If assume_already_annualized=True, Realiz5 is treated as the actual annualized CPI.
    Otherwise, cpi_actual is treated as an index level and converted to annualized change.
    """
    d = cleaned_realiz5.copy()
    if "time" not in d.columns or "cpi_actual" not in d.columns:
        raise ValueError("cleaned_realiz5 must include columns ['time', 'cpi_actual'].")

    # Accept quarter-like labels (including Period objects) in addition to timestamps.
    q = d["time"].apply(_parse_quarter_period)
    d["time"] = pd.PeriodIndex(q, freq="Q").to_timestamp("Q")
    d["cpi_actual"] = pd.to_numeric(d["cpi_actual"], errors="coerce")
    d = d.dropna(subset=["time", "cpi_actual"]).sort_values("time").reset_index(drop=True)

    if assume_already_annualized:
        out = d.rename(columns={"cpi_actual": "actual_inflation_annualized"})[["time", "actual_inflation_annualized"]]
        return out

    ratio = d["cpi_actual"] / d["cpi_actual"].shift(1)
    out = pd.DataFrame(
        {
            "time": d["time"],
            "actual_inflation_annualized": 100.0 * (ratio.pow(4) - 1.0),
        }
    ).dropna()
    return out.reset_index(drop=True)


def build_spf_consensus_vs_realizations(
    spf_df: pd.DataFrame,
    realiz_df: pd.DataFrame,
    forecast_col: str = "CPI1",
    realiz_initial_col: str = "Realiz1",
    realiz_latest_col: str = "Realiz5",
) -> pd.DataFrame:
    """
    Build a period-matched comparison table with no horizon realignment.

    This is intended for diagnostic plotting of SPF average forecasts against
    initial and latest realizations in the same survey quarter.

    Output columns:
      - period (Period[Q])
      - time (quarter-end Timestamp)
      - avg_forecast
      - realiz_initial
      - realiz_latest
      - revision (latest - initial)
    """
    s = spf_df.copy()
    if forecast_col not in s.columns:
        raise ValueError(f"SPF dataframe missing forecast column: {forecast_col}")
    if "YEAR" not in s.columns or "QUARTER" not in s.columns:
        raise ValueError("SPF dataframe must include YEAR and QUARTER columns.")

    s["YEAR"] = pd.to_numeric(s["YEAR"], errors="coerce")
    s["QUARTER"] = pd.to_numeric(s["QUARTER"], errors="coerce")
    s["forecast"] = pd.to_numeric(s[forecast_col], errors="coerce")
    s = s.dropna(subset=["YEAR", "QUARTER", "forecast"]).copy()
    s["period"] = pd.PeriodIndex(
        [f"{int(y)}Q{int(q)}" for y, q in zip(s["YEAR"], s["QUARTER"])], freq="Q"
    )
    fc = (
        s.groupby("period", as_index=False)
        .agg(avg_forecast=("forecast", "mean"))
        .sort_values("period")
    )

    r = realiz_df.copy()
    colmap = {str(c).strip().lower(): c for c in r.columns}
    if "year" in colmap and "quarter" in colmap:
        yy = pd.to_numeric(r[colmap["year"]], errors="coerce")
        qq = pd.to_numeric(r[colmap["quarter"]], errors="coerce")
        mask = yy.notna() & qq.notna()
        r = r.loc[mask].copy()
        r["period"] = pd.PeriodIndex(
            [f"{int(y)}Q{int(q)}" for y, q in zip(yy.loc[mask], qq.loc[mask])], freq="Q"
        )
    else:
        obs_col = r.columns[0]
        r["period"] = r[obs_col].apply(_parse_quarter_period)
        r = r.loc[r["period"].notna()].copy()

    normalized = {re.sub(r"[^a-z0-9]", "", str(c).lower()): c for c in r.columns}
    i_key = re.sub(r"[^a-z0-9]", "", realiz_initial_col.lower())
    l_key = re.sub(r"[^a-z0-9]", "", realiz_latest_col.lower())
    if i_key not in normalized or l_key not in normalized:
        raise ValueError(
            "Realization columns not found in realiz_df. "
            f"Requested: {realiz_initial_col}, {realiz_latest_col}. "
            f"Columns sample: {list(r.columns)[:12]}"
        )
    i_col = normalized[i_key]
    l_col = normalized[l_key]
    r["realiz_initial"] = pd.to_numeric(r[i_col], errors="coerce")
    r["realiz_latest"] = pd.to_numeric(r[l_col], errors="coerce")
    rz = (
        r[["period", "realiz_initial", "realiz_latest"]]
        .dropna(subset=["period", "realiz_initial", "realiz_latest"])
        .drop_duplicates(subset=["period"], keep="last")
        .sort_values("period")
    )

    out = fc.merge(rz, on="period", how="inner")
    if out.empty:
        raise ValueError("No overlapping periods between SPF forecasts and realizations.")
    out["revision"] = out["realiz_latest"] - out["realiz_initial"]
    out["time"] = out["period"].dt.to_timestamp("Q")
    return out[["period", "time", "avg_forecast", "realiz_initial", "realiz_latest", "revision"]]


def clean_spf_error_stats_revision_table(
    realiz_df: pd.DataFrame,
    date_col: str = "Date",
    forecast_col: str = "SPFfor_Step1",
    realiz_initial_col: str = "Realiz1",
    realiz_latest_col: str = "Realiz5",
) -> pd.DataFrame:
    """
    Clean the SPF error-statistics table into a directly aligned comparison frame.

    This uses one source table where SPF one-step forecasts and realizations already
    share the same date key, so no additional horizon realignment is performed.

    Output columns:
      - Date
      - DateTime
      - SPFfor_Step1
      - Realiz1
      - Realiz5
      - Revision
    """
    d = realiz_df.copy()
    colmap = {str(c).strip().lower(): c for c in d.columns}

    def _pick(name: str) -> str:
        key = re.sub(r"[^a-z0-9]", "", name.lower())
        normalized = {re.sub(r"[^a-z0-9]", "", str(k).lower()): v for k, v in colmap.items()}
        if key not in normalized:
            raise ValueError(
                f"Column '{name}' not found in SPF error-stats table. "
                f"Columns sample: {list(d.columns)[:12]}"
            )
        return normalized[key]

    date_src = _pick(date_col)
    fc_src = _pick(forecast_col)
    r1_src = _pick(realiz_initial_col)
    r5_src = _pick(realiz_latest_col)

    out = d[[date_src, fc_src, r1_src, r5_src]].copy()
    out = out.rename(
        columns={
            date_src: "Date",
            fc_src: "SPFfor_Step1",
            r1_src: "Realiz1",
            r5_src: "Realiz5",
        }
    )
    out["SPFfor_Step1"] = pd.to_numeric(out["SPFfor_Step1"], errors="coerce")
    out["Realiz1"] = pd.to_numeric(out["Realiz1"], errors="coerce")
    out["Realiz5"] = pd.to_numeric(out["Realiz5"], errors="coerce")
    out = out.dropna(subset=["Date", "SPFfor_Step1", "Realiz1", "Realiz5"]).copy()

    # Date keys are quarterly labels like "1968:04" -> 1968Q4.
    date_parts = out["Date"].astype(str).str.strip().str.extract(r"^(?P<y>\d{4})[:/-]?(?P<q>\d{1,2})$")
    if date_parts.isna().all(axis=None):
        out["DateTime"] = pd.to_datetime(out["Date"], errors="coerce")
        if out["DateTime"].isna().all():
            raise ValueError("Could not parse Date column in SPF error-stats table.")
    else:
        y = pd.to_numeric(date_parts["y"], errors="coerce")
        q = pd.to_numeric(date_parts["q"], errors="coerce")
        valid = y.notna() & q.notna() & q.between(1, 4)
        out = out.loc[valid].copy()
        out["DateTime"] = pd.PeriodIndex(
            [f"{int(yy)}Q{int(qq)}" for yy, qq in zip(y.loc[valid], q.loc[valid])],
            freq="Q",
        ).to_timestamp("Q")

    out = out.sort_values("DateTime").drop_duplicates(subset=["DateTime"], keep="last")
    out["Revision"] = out["Realiz5"] - out["Realiz1"]
    return out.reset_index(drop=True)


def prepare_vix_data(vix_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Clean VIX daily data and aggregate to quarterly mean.

    Expected FRED-like columns: DATE, VIXCLS
    Returns:
      - daily: DATE, vix
      - quarterly: period (Period[Q]), vix_q_mean
    """
    d = vix_df.copy()
    d.columns = [str(c).strip().upper() for c in d.columns]
    if "DATE" not in d.columns:
        if "OBSERVATION_DATE" in d.columns:
            d = d.rename(columns={"OBSERVATION_DATE": "DATE"})
        else:
            raise ValueError("VIX data must include DATE or observation_date column.")
    value_col = "VIXCLS" if "VIXCLS" in d.columns else next((c for c in d.columns if c != "DATE"), None)
    if value_col is None:
        raise ValueError("VIX data must include a value column (expected VIXCLS).")

    d["DATE"] = pd.to_datetime(d["DATE"], errors="coerce")
    d["vix"] = pd.to_numeric(d[value_col], errors="coerce")
    d = d.dropna(subset=["DATE", "vix"]).sort_values("DATE").reset_index(drop=True)

    q = d.copy()
    q["period"] = q["DATE"].dt.to_period("Q")
    q = q.groupby("period", as_index=False).agg(vix_q_mean=("vix", "mean"))
    return d[["DATE", "vix"]], q


def clean_cpi_actuals_index_series(
    realtime_df: pd.DataFrame,
    observation_col: str | None = None,
    as_index: bool = True,
) -> pd.DataFrame:
    """
    Build a clean CPI actuals level series from the real-time vintage matrix.

    Output:
      - if as_index=True: index=time (quarter end timestamp), column=cpi_actual
      - if as_index=False: columns=[time, cpi_actual]
    """
    d = realtime_df.copy()
    d.columns = [str(c).strip() for c in d.columns]

    obs_col = observation_col
    if obs_col is None:
        candidates = [c for c in d.columns if str(c).strip().upper() in {"DATE", "OBS", "OBS_DATE", "PERIOD"}]
        obs_col = candidates[0] if candidates else d.columns[0]
    if obs_col not in d.columns:
        raise ValueError(f"observation_col '{obs_col}' not found in real-time DataFrame.")

    d["obs_period"] = d[obs_col].apply(_parse_quarter_period)
    d = d.loc[d["obs_period"].notna()].copy()
    if d.empty:
        raise ValueError("No observation dates parsed from real-time actuals matrix.")

    vintage_cols: list[str] = []
    vintage_map: dict[str, pd.Period] = {}
    for c in d.columns:
        if c in {obs_col, "obs_period"}:
            continue
        p = _parse_quarter_period(c)
        if p is not None:
            vintage_cols.append(c)
            vintage_map[c] = p
            d[c] = pd.to_numeric(d[c], errors="coerce")

    if not vintage_cols:
        sample_cols = [str(c) for c in d.columns[:12]]
        raise ValueError(
            "No vintage columns parsed from real-time matrix. "
            f"obs_col={obs_col!r}; sample columns={sample_cols}"
        )

    v_sorted = sorted(vintage_cols, key=lambda c: vintage_map[c])
    rows: list[dict[str, object]] = []
    for _, row in d.iterrows():
        q = row["obs_period"]
        # Leading diagonal: choose earliest vintage for quarter q (v >= q).
        candidates_ge = [c for c in v_sorted if vintage_map[c] >= q]
        chosen = candidates_ge[0] if candidates_ge else v_sorted[-1]
        value = row.get(chosen, np.nan)
        if pd.isna(value):
            valid = [c for c in v_sorted if pd.notna(row.get(c, np.nan))]
            if not valid:
                continue
            valid_ge = [c for c in valid if vintage_map[c] >= q]
            chosen = valid_ge[0] if valid_ge else valid[-1]
            value = row.get(chosen, np.nan)
        if pd.isna(value):
            continue
        rows.append({"period": q, "cpi_actual": float(value)})

    out = pd.DataFrame(rows)
    if out.empty:
        raise ValueError("No diagonal CPI actual values extracted from matrix.")

    # If source rows are monthly, aggregate to quarterly average index level.
    out = out.groupby("period", as_index=False).agg(cpi_actual=("cpi_actual", "mean"))
    out = out.sort_values("period").reset_index(drop=True)
    out["time"] = out["period"].dt.to_timestamp("Q")
    out = out[["time", "cpi_actual"]]
    if as_index:
        out = out.set_index("time").sort_index()
    return out


def cpi_index_to_annualized_change(
    cpi_level_df: pd.DataFrame,
    method: str = "compound",
    as_index: bool = True,
) -> pd.DataFrame:
    """
    Convert CPI index level to annualized quarterly inflation for forecast comparison.

    method='compound': 100 * ((CPI_t / CPI_{t-1})^4 - 1)
    method='log':      400 * log(CPI_t / CPI_{t-1})
    """
    d = cpi_level_df.copy()
    if "time" in d.columns and "cpi_actual" in d.columns:
        d = d[["time", "cpi_actual"]].copy()
    elif "cpi_actual" in d.columns and d.index.name is not None:
        d = d.reset_index().rename(columns={d.index.name: "time"})[["time", "cpi_actual"]]
    elif "cpi_level_rt" in d.columns and "period" in d.columns:
        d = d[["period", "cpi_level_rt"]].rename(columns={"period": "time", "cpi_level_rt": "cpi_actual"})
    else:
        raise ValueError("Input must include 'cpi_actual' with a time column/index.")

    # Accept quarter labels (including Period objects like 2020Q1) as well as timestamps.
    q = d["time"].apply(_parse_quarter_period)
    d["time"] = pd.PeriodIndex(q, freq="Q").to_timestamp("Q")
    d["cpi_actual"] = pd.to_numeric(d["cpi_actual"], errors="coerce")
    d = d.dropna(subset=["time", "cpi_actual"]).sort_values("time").reset_index(drop=True)

    ratio = d["cpi_actual"] / d["cpi_actual"].shift(1)
    if method == "compound":
        d["actual_inflation_annualized"] = 100.0 * (ratio.pow(4) - 1.0)
    elif method == "log":
        d["actual_inflation_annualized"] = 400.0 * np.log(ratio)
    else:
        raise ValueError("method must be 'compound' or 'log'")

    out = d[["time", "actual_inflation_annualized"]].dropna().reset_index(drop=True)
    if as_index:
        out = out.set_index("time").sort_index()
    return out


def extract_realtime_diagonal_actuals(
    realtime_df: pd.DataFrame,
    observation_col: str | None = None,
) -> pd.DataFrame:
    """
    Backward-compatible wrapper around the new CPI actuals cleaner.
    Returns columns: period, cpi_level_rt.
    """
    cleaned = clean_cpi_actuals_index_series(
        realtime_df=realtime_df,
        observation_col=observation_col,
        as_index=False,
    ).copy()
    out = cleaned.rename(columns={"time": "period", "cpi_actual": "cpi_level_rt"})
    out["period"] = pd.PeriodIndex(pd.to_datetime(out["period"]), freq="Q")
    return out


def compute_annualized_cpi_inflation_from_level(
    diagonal_actuals: pd.DataFrame,
    method: str = "compound",
) -> pd.DataFrame:
    """
    Backward-compatible wrapper for annualized change conversion.
    Returns columns: period, actual_inflation_annualized.
    """
    d = diagonal_actuals.copy()
    if "period" in d.columns and "cpi_level_rt" in d.columns:
        base = d.rename(columns={"period": "time", "cpi_level_rt": "cpi_actual"})
    else:
        base = d
    out = cpi_index_to_annualized_change(
        cpi_level_df=base,
        method=method,
        as_index=False,
    )
    out = out.rename(columns={"time": "period"})
    out["period"] = pd.PeriodIndex(pd.to_datetime(out["period"]), freq="Q")
    return out


def build_spf_cpi_panel_with_actuals(
    spf_clean: pd.DataFrame,
    actual_inflation: pd.DataFrame,
    horizons: list[str] | None = None,
) -> pd.DataFrame:
    """
    Build long SPF panel with same-quarter realized CPI actuals (Realiz5 path).

    Output columns:
      ID, origin_period, horizon_name, horizon, forecast, actual
    """
    d = spf_clean.copy()
    if "period" not in d.columns:
        raise ValueError("spf_clean must include 'period'.")

    all_h = [c for c in ["CPI1", "CPI2", "CPI3", "CPI4"] if c in d.columns]
    use_h = all_h if horizons is None else [h for h in horizons if h in d.columns]
    if not use_h:
        raise ValueError("No requested SPF horizon columns found.")

    d["origin_period"] = pd.PeriodIndex(pd.to_datetime(d["period"]), freq="Q")

    a = actual_inflation.copy()
    if "period" not in a.columns or "actual_inflation_annualized" not in a.columns:
        raise ValueError("actual_inflation must include 'period' and 'actual_inflation_annualized'.")
    a["origin_period"] = a["period"].apply(_parse_quarter_period)
    a = a.dropna(subset=["origin_period", "actual_inflation_annualized"]).copy()
    a_lookup = a[["origin_period", "actual_inflation_annualized"]].drop_duplicates(subset=["origin_period"], keep="last")

    rows: list[pd.DataFrame] = []
    for h_name in use_h:
        h_num = int(re.sub(r"[^0-9]", "", h_name))
        sub = d[["ID", "origin_period", h_name]].copy()
        sub = sub.rename(columns={h_name: "forecast"})
        sub["horizon_name"] = h_name
        sub["horizon"] = h_num
        rows.append(sub)

    panel = pd.concat(rows, axis=0, ignore_index=True)
    panel["forecast"] = pd.to_numeric(panel["forecast"], errors="coerce")
    panel = panel.merge(a_lookup, how="left", on="origin_period")
    panel = panel.rename(columns={"actual_inflation_annualized": "actual"})
    panel = panel.dropna(subset=["ID", "origin_period", "forecast", "actual"]).copy()
    panel = panel.sort_values(["horizon_name", "origin_period", "ID"]).reset_index(drop=True)
    return panel


def alignment_diagnostics_spf_offsets(
    spf_clean: pd.DataFrame,
    actual_inflation: pd.DataFrame,
    horizons: list[str] | None = None,
    offsets: list[int] | None = None,
) -> pd.DataFrame:
    """Alignment offset diagnostics are deprecated; SPF uses same-quarter matching only."""
    raise ValueError(
        "alignment_diagnostics_spf_offsets is deprecated. "
        "Use build_spf_cpi_panel_with_actuals() same-quarter output directly."
    )


def find_optimal_reporting_window(
    spf_clean: pd.DataFrame,
    forecast_col: str,
    min_years: int = 8,
    required_date: str | None = None,
) -> dict[str, object]:
    """Return best contiguous window maximizing fully reporting forecasters for one horizon."""
    if forecast_col not in spf_clean.columns:
        raise ValueError(f"{forecast_col} not found in SPF dataframe.")
    d = spf_clean.copy()
    d["period"] = pd.PeriodIndex(pd.to_datetime(d["period"]), freq="Q")
    min_window = int(min_years) * 4
    required = pd.Period(pd.Timestamp(required_date), freq="Q") if required_date else None

    report = (
        d.pivot_table(index="ID", columns="period", values=forecast_col, aggfunc="last")
        .notna()
        .astype(int)
        .sort_index(axis=1)
    )
    periods = report.columns.to_list()
    vals = report.to_numpy(dtype=int)
    n_p = vals.shape[1]
    best: tuple[int, int, int] | None = None

    for w in range(min_window, n_p + 1):
        for start in range(0, n_p - w + 1):
            end = start + w
            if required is not None and not (periods[start] <= required <= periods[end - 1]):
                continue
            full = (vals[:, start:end] == 1).all(axis=1)
            count = int(full.sum())
            if best is None or count > best[0] or (count == best[0] and w > (best[2] - best[1])):
                best = (count, start, end)

    if best is None:
        raise ValueError("No feasible window satisfies constraints.")

    _, s, e = best
    full = (vals[:, s:e] == 1).all(axis=1)
    ids = report.index[full].tolist()
    return {
        "forecast_col": forecast_col,
        "start_period": periods[s],
        "end_period": periods[e - 1],
        "n_periods": int(e - s),
        "n_ids": int(len(ids)),
        "ids": ids,
    }


def build_spf_horizon_matrix(
    panel_long: pd.DataFrame,
    horizon_name: str,
    start_period: pd.Period | None = None,
    end_period: pd.Period | None = None,
    forecaster_ids: list[int] | None = None,
    min_obs: int = 24,
    min_forecasters: int = 3,
) -> dict[str, object] | None:
    """
    Build ensemble-ready matrix for one SPF horizon over a chosen origin-period window.
    """
    d = panel_long.loc[panel_long["horizon_name"] == horizon_name].copy()
    if d.empty:
        return None
    if start_period is not None:
        d = d.loc[d["origin_period"] >= start_period].copy()
    if end_period is not None:
        d = d.loc[d["origin_period"] <= end_period].copy()
    if forecaster_ids is not None:
        d = d.loc[d["ID"].isin(forecaster_ids)].copy()
    if d.empty:
        return None

    pivot = d.pivot_table(index="origin_period", columns="ID", values="forecast", aggfunc="last")
    target = d.groupby("origin_period", as_index=True).agg(actual=("actual", "last"))
    mat = target.join(pivot, how="left").sort_index().reset_index()
    mat = mat.rename(columns={"origin_period": "date"})
    method_cols = [c for c in mat.columns if c not in {"date", "actual"}]
    if len(method_cols) < int(min_forecasters):
        return None
    mat2 = mat[["date", *method_cols, "actual"]].dropna().copy()
    if len(mat2) < int(min_obs):
        return None

    y = mat2["actual"].to_numpy(dtype=float)
    F = mat2[method_cols].to_numpy(dtype=float)
    disagreement = np.std(F, axis=1)
    mat_for_eval = mat2.rename(columns={"date": "origin_period"})
    return {
        "mat": mat_for_eval,
        "mat_clean": mat2,
        "y": y,
        "F": F,
        "forecaster_ids": method_cols,
        "disagreement": disagreement,
    }


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
    # Keep only forecasts whose target period matches origin+horizon exactly.
    if "horizon_consistent" in d.columns:
        d = d.loc[d["horizon_consistent"]].copy()
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
