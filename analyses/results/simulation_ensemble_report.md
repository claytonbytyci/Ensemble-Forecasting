# Simulation Ensembling Analysis

## Module Helper Inventory

### `src.data.simulator`
- `MacroSimConfig`
- `build_forecaster_panel`
- `make_environment_and_forecasts`
- `simulate_macro_environment`

### `src.ensemblers.ensemblers`
- `BaseEnsembler`
- `EnsembleResult`
- `MWUMBothKL`
- `MWUMConcentrationOnlyKL`
- `MWUMVanilla`
- `MeanEnsembler`
- `MedianEnsembler`
- `OGDConcentrationBoth`
- `OGDConcentrationOnly`
- `OGDVanilla`

### `src.evaluation.evaluation_helpers`
- `best_forecaster_yhat`
- `cumulative_loss`
- `evaluate_and_plot`
- `hhi_from_weights`
- `linex_loss`
- `loss_series`
- `loss_table`
- `mae`
- `mse`
- `plot_actions_over_time`
- `plot_hhi_over_time`
- `plot_kappa_over_time`
- `plot_loss_over_time`
- `plot_policy_diagnostics`
- `rolling_mean`

## Aggregated Results

### Horizon h=1
- MWUMVanilla: MSE=0.0970 (std 0.0142), MAE=0.2314, LINEX=0.0512, avg HHI=0.0843
- MWUMBothKL: MSE=0.0971 (std 0.0144), MAE=0.2310, LINEX=0.0512, avg HHI=0.0559
- MWUMConcOnlyKL: MSE=0.0971 (std 0.0143), MAE=0.2311, LINEX=0.0512, avg HHI=0.0564
- OGDBoth: MSE=0.0974 (std 0.0147), MAE=0.2312, LINEX=0.0513, avg HHI=0.0574
- Median: MSE=0.0974 (std 0.0145), MAE=0.2317, LINEX=0.0514, avg HHI=nan
- Mean: MSE=0.0977 (std 0.0147), MAE=0.2315, LINEX=0.0516, avg HHI=0.0556
- OGDVanilla: MSE=0.0985 (std 0.0151), MAE=0.2326, LINEX=0.0519, avg HHI=0.0792
- OGDConcOnly: MSE=0.1077 (std 0.0215), MAE=0.2408, LINEX=0.0561, avg HHI=0.0880

### Horizon h=4
- OGDConcOnly: MSE=0.2690 (std 0.0717), MAE=0.3736, LINEX=0.1571, avg HHI=0.1312
- MWUMConcOnlyKL: MSE=0.2772 (std 0.0643), MAE=0.3835, LINEX=0.1638, avg HHI=0.0619
- OGDBoth: MSE=0.2798 (std 0.0686), MAE=0.3823, LINEX=0.1660, avg HHI=0.0752
- OGDVanilla: MSE=0.2811 (std 0.0704), MAE=0.3825, LINEX=0.1671, avg HHI=0.1274
- MWUMVanilla: MSE=0.2980 (std 0.0732), MAE=0.3937, LINEX=0.1791, avg HHI=0.3112
- MWUMBothKL: MSE=0.3016 (std 0.0725), MAE=0.3989, LINEX=0.1803, avg HHI=0.0577
- Median: MSE=0.3081 (std 0.0766), MAE=0.4005, LINEX=0.1852, avg HHI=nan
- Mean: MSE=0.3163 (std 0.0773), MAE=0.4091, LINEX=0.1906, avg HHI=0.0556

### Horizon h=8
- OGDVanilla: MSE=0.3292 (std 0.0972), MAE=0.4146, LINEX=0.1970, avg HHI=0.1447
- OGDBoth: MSE=0.3312 (std 0.0980), MAE=0.4180, LINEX=0.1986, avg HHI=0.0857
- MWUMConcOnlyKL: MSE=0.3411 (std 0.0969), MAE=0.4294, LINEX=0.2038, avg HHI=0.0661
- OGDConcOnly: MSE=0.3561 (std 0.1191), MAE=0.4339, LINEX=0.2202, avg HHI=0.1563
- MWUMVanilla: MSE=0.3604 (std 0.1027), MAE=0.4352, LINEX=0.2193, avg HHI=0.4751
- MWUMBothKL: MSE=0.3772 (std 0.1090), MAE=0.4514, LINEX=0.2273, avg HHI=0.0610
- Median: MSE=0.4069 (std 0.1279), MAE=0.4642, LINEX=0.2512, avg HHI=nan
- Mean: MSE=0.4316 (std 0.1365), MAE=0.4834, LINEX=0.2652, avg HHI=0.0556
