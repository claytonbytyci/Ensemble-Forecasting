"""EDA helper modules for SPF data loading, cleaning, and plotting."""

from .cleaning import (
    average_forecasts_by_period,
    long_forecast_frame,
    missingness_by_horizon,
    prepare_spf_data,
)
from .loading import (
    DATA_DIR,
    DEFAULT_CSV_PATH,
    URL,
    download_spf_data,
    load_or_download_csv,
    save_csv,
)
from .plots import (
    analyze_optimal_reporting_windows,
    plot_average_forecasts_over_time,
    plot_dispersion_scatter,
    plot_missingness_bar,
    plot_reporting_forecasters_over_time,
    plot_reporting_waterfall,
    summarize_and_plot_reporting_consistency,
)

__all__ = [
    "DATA_DIR",
    "DEFAULT_CSV_PATH",
    "URL",
    "analyze_optimal_reporting_windows",
    "average_forecasts_by_period",
    "download_spf_data",
    "load_or_download_csv",
    "long_forecast_frame",
    "missingness_by_horizon",
    "plot_average_forecasts_over_time",
    "plot_dispersion_scatter",
    "plot_missingness_bar",
    "plot_reporting_forecasters_over_time",
    "plot_reporting_waterfall",
    "prepare_spf_data",
    "save_csv",
    "summarize_and_plot_reporting_consistency",
]
