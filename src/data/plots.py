import pandas as pd
import matplotlib.pyplot as plt


def plot_missingness_bar(missing_rate: pd.Series) -> None:
    plt.figure(figsize=(8, 4))
    missing_rate.plot(kind="bar", color="#4C78A8")
    plt.ylabel("Share Missing")
    plt.title("Missingness by Horizon")
    plt.tight_layout()
    plt.show()


def plot_reporting_forecasters_over_time(df: pd.DataFrame, forecast_cols: list[str]) -> None:
    for horizon in forecast_cols:
        forecasters_per_period = (
            df.loc[df[horizon].notna()].groupby("period")["ID"].nunique().sort_index()
        )

        plt.figure(figsize=(10, 4))
        forecasters_per_period.plot(color="#F58518")
        plt.ylabel("# Forecasters Reporting")
        plt.title(f"Reporting Forecasters Over Time ({horizon})")
        plt.tight_layout()
        plt.show()


def summarize_and_plot_reporting_consistency(
    df: pd.DataFrame,
    forecast_cols: list[str],
    display_fn=None,
) -> None:
    total_periods = df["period"].nunique()
    show = display_fn if display_fn is not None else print

    for horizon in forecast_cols:
        report_counts = (
            df.loc[df[horizon].notna()]
            .groupby("ID")["period"]
            .nunique()
            .sort_values(ascending=False)
        )

        report_fraction = report_counts / total_periods

        print(f"Horizon: {horizon}")
        print(f"  Total periods: {total_periods}")
        print(f"  Forecasters with 100% reporting: {(report_fraction == 1).sum()}")
        print(f"  Forecasters with >=90% reporting: {(report_fraction >= 0.9).sum()}")
        print(f"  Forecasters with >=75% reporting: {(report_fraction >= 0.75).sum()}")

        plt.figure(figsize=(8, 4))
        plt.hist(report_fraction, bins=20, color="#54A24B", edgecolor="white")
        plt.xlabel("Fraction of Periods Reported")
        plt.ylabel("# Forecasters")
        plt.title(f"Forecaster Reporting Consistency ({horizon})")
        plt.tight_layout()
        plt.show()

        top_consistent = (
            pd.DataFrame({"report_fraction": report_fraction, "report_count": report_counts})
            .sort_values(["report_fraction", "report_count"], ascending=False)
            .head(10)
        )
        show(top_consistent)
        print()


def analyze_optimal_reporting_windows(
    df: pd.DataFrame,
    forecast_cols: list[str],
    min_years: int = 15,
    required_date: str | None = None,
) -> None:
    min_window = min_years * 4
    required_ts = pd.Timestamp(required_date) if required_date else None

    for horizon in forecast_cols:
        report_pivot = (
            df.pivot_table(index="ID", columns="period", values=horizon, aggfunc="max")
            .notna()
            .astype(int)
        )
        report_pivot = report_pivot.sort_index(axis=1)

        periods = report_pivot.columns.to_list()
        values = report_pivot.to_numpy(dtype=int)

        n_periods = values.shape[1]
        best = None
        max_by_window = []

        for window_len in range(min_window, n_periods + 1):
            max_count = -1
            best_window = None
            for start in range(0, n_periods - window_len + 1):
                end = start + window_len

                if required_ts is not None and not (
                    periods[start] <= required_ts <= periods[end - 1]
                ):
                    continue

                fully_reporting = (values[:, start:end] == 1).all(axis=1)
                count = int(fully_reporting.sum())
                if count > max_count:
                    max_count = count
                    best_window = (start, end)

            if best_window is None:
                continue

            max_by_window.append((window_len, max_count))
            if best is None or max_count > best[0] or (
                max_count == best[0] and window_len > (best[2] - best[1])
            ):
                best = (max_count, best_window[0], best_window[1])

        if best is None:
            print(f"Horizon: {horizon}")
            print("  No window satisfies the constraints.")
            print()
            continue

        best_count, best_start, best_end = best
        best_start_period = periods[best_start]
        best_end_period = periods[best_end - 1]

        print(f"Horizon: {horizon}")
        print("  Optimal window (max fully reporting forecasters):")
        print(
            f"    Window: {best_start_period} to {best_end_period} "
            f"(len={best_end - best_start} quarters)"
        )
        print(f"    Fully reporting forecasters: {best_count}")

        optimal_fully_reporting = (values[:, best_start:best_end] == 1).all(axis=1)
        optimal_ids = report_pivot.index[optimal_fully_reporting].to_list()
        print(f"    IDs (n={len(optimal_ids)}): {optimal_ids}")

        window_lens = [window_len for window_len, _ in max_by_window]
        max_counts = [count for _, count in max_by_window]
        plt.figure(figsize=(8, 4))
        plt.plot(window_lens, max_counts, color="#E45756")
        plt.xlabel("Window Length (quarters)")
        plt.ylabel("Max Fully Reporting Forecasters")
        plt.title(
            f"Best Possible Fully-Reporting Count by Window Length ({horizon})"
        )
        plt.tight_layout()
        plt.show()
        print()


def plot_average_forecasts_over_time(
    avg_by_period: pd.DataFrame,
    forecast_cols: list[str],
) -> None:
    for horizon in forecast_cols:
        plt.figure(figsize=(10, 5))
        plt.plot(avg_by_period.index, avg_by_period[horizon], label=horizon, color="#4C78A8")
        plt.title(f"Average Forecast Over Time ({horizon})")
        plt.ylabel("Forecast Value")
        plt.xlabel("Period")
        plt.tight_layout()
        plt.show()


def plot_dispersion_scatter(long_df: pd.DataFrame) -> None:
    for horizon, subset in long_df.groupby("horizon"):
        plt.figure(figsize=(10, 5))
        plt.scatter(
            subset["period"],
            subset["forecast"],
            s=6,
            alpha=0.15,
            color="#4C78A8",
        )
        plt.title(f"{horizon} Forecasts Over Time (All IDs)")
        plt.xlabel("Period")
        plt.ylabel("Forecast Value")
        plt.tight_layout()
        plt.show()


def plot_reporting_waterfall(df: pd.DataFrame, forecast_cols: list[str]) -> None:
    for horizon in forecast_cols:
        report_matrix = (
            df.assign(reported=df[horizon].notna())
            .pivot_table(index="ID", columns="period", values="reported", aggfunc="max")
            .fillna(0)
        )

        report_matrix = report_matrix.loc[
            report_matrix.mean(axis=1).sort_values(ascending=False).index
        ]

        period_labels = report_matrix.columns.to_period("Q").astype(str)

        plt.figure(figsize=(12, 8))
        plt.imshow(
            report_matrix.to_numpy(dtype=float),
            aspect="auto",
            interpolation="nearest",
            cmap="Greys",
        )
        plt.colorbar(label=f"Reported {horizon} (1=yes)")
        plt.title(f"Forecaster Reporting Over Time (Waterfall, {horizon})")
        plt.xlabel("Period")
        plt.ylabel("Forecaster ID (sorted by reporting rate)")

        if report_matrix.shape[1] > 12:
            step = max(1, report_matrix.shape[1] // 12)
            xticks = list(range(0, report_matrix.shape[1], step))
            plt.xticks(xticks, [period_labels[i] for i in xticks], rotation=45)
        else:
            plt.xticks(range(report_matrix.shape[1]), period_labels, rotation=45)

        plt.yticks([])
        plt.tight_layout()
        plt.show()
