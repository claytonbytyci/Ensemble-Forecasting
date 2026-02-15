"""Data utilities package (EDA helpers + simulation tools)."""

# Optional EDA imports (require pandas/matplotlib in the runtime env).
try:
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
except ModuleNotFoundError:
    # Keep simulator imports available even when notebook deps are absent.
    pass
from .simulator import (
    MacroSimConfig,
    build_forecaster_panel,
    make_environment_and_forecasts,
    simulate_macro_environment,
)

__all__ = [
    "MacroSimConfig",
    "build_forecaster_panel",
    "make_environment_and_forecasts",
    "simulate_macro_environment",
]

_optional_exports = [
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
for _name in _optional_exports:
    if _name in globals():
        __all__.append(_name)
