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

### `src.evaluation.optuna_tuning`
- `TuneResult`
- `tune_all_methods_optuna`
- `tune_method_optuna`

## Tuning Setup
- Scenario: discriminating
- Tuning seeds: [0, 1, 2, 3]
- Test seeds: [4, 5, 6, 7, 8, 9]
- Optuna trials per method: 35
- Tuning mode: optuna_by_horizon

## Tuned Hyperparameters (by horizon)

### Horizon h=1
- MWUMVanilla: params={'eta': 0.01580148447850615}, objective_mse=0.9275967740804423
- MWUMBothKL: params={'eta': 0.015919552043565138, 'kappa': 0.005714357133536371}, objective_mse=0.9276072196891759
- Median: params={}, objective_mse=0.9351405990174786
- OGDVanilla: params={'eta': 0.001004380875007612}, objective_mse=0.9363174265651122
- OGDBoth: params={'eta': 0.001000231236412198, 'kappa': 0.05203263616499159}, objective_mse=0.9363951771480309
- MWUMConcOnlyKL: params={'kappa': 7.849943290518327}, objective_mse=0.9436778985286702
- Mean: params={}, objective_mse=0.951494710698005
- OGDConcOnly: params={'kappa': 7.989067040220606}, objective_mse=1.0591123854559412

### Horizon h=4
- MWUMConcOnlyKL: params={'kappa': 0.28056807110434306}, objective_mse=2.3110477195341685
- MWUMBothKL: params={'eta': 2.9821867212394793, 'kappa': 0.5913228643145931}, objective_mse=2.436336792645921
- OGDBoth: params={'eta': 0.03643044147708752, 'kappa': 0.8840210944332434}, objective_mse=2.5986572663695844
- OGDVanilla: params={'eta': 0.04449333821791166}, objective_mse=2.617438665876078
- OGDConcOnly: params={'kappa': 7.973929436896223}, objective_mse=2.741631494002637
- MWUMVanilla: params={'eta': 0.014568470700968195}, objective_mse=3.208999573430413
- Median: params={}, objective_mse=3.257653711387143
- Mean: params={}, objective_mse=3.300878809265483

### Horizon h=8
- MWUMConcOnlyKL: params={'kappa': 0.39742005940341457}, objective_mse=2.934682642435865
- MWUMBothKL: params={'eta': 2.900336707567024, 'kappa': 0.6678824415504225}, objective_mse=3.0253191922396407
- OGDBoth: params={'eta': 0.015110282798066584, 'kappa': 0.7407595779704955}, objective_mse=3.470104023728017
- OGDVanilla: params={'eta': 0.01529830399119274}, objective_mse=3.4878812727270203
- OGDConcOnly: params={'kappa': 7.9869664267965925}, objective_mse=4.083511276008283
- MWUMVanilla: params={'eta': 2.8200403424806297}, objective_mse=4.265079734439067
- Median: params={}, objective_mse=4.704949890767167
- Mean: params={}, objective_mse=4.886158056388866

## Aggregated Results

### Horizon h=1
- MWUMBothKL: MSE=0.8260 (std 0.1223), MAE=0.6170, LINEX=0.7475, avg HHI=0.0676
- MWUMVanilla: MSE=0.8260 (std 0.1223), MAE=0.6171, LINEX=0.7472, avg HHI=0.0678
- Median: MSE=0.8316 (std 0.1238), MAE=0.6188, LINEX=0.7407, avg HHI=nan
- OGDVanilla: MSE=0.8321 (std 0.1238), MAE=0.6218, LINEX=0.7545, avg HHI=0.0603
- OGDBoth: MSE=0.8321 (std 0.1238), MAE=0.6218, LINEX=0.7544, avg HHI=0.0601
- MWUMConcOnlyKL: MSE=0.8403 (std 0.1266), MAE=0.6252, LINEX=0.7522, avg HHI=0.0566
- Mean: MSE=0.8442 (std 0.1273), MAE=0.6329, LINEX=0.7219, avg HHI=0.0556
- OGDConcOnly: MSE=0.9249 (std 0.1478), MAE=0.6535, LINEX=0.8911, avg HHI=0.0860

### Horizon h=4
- MWUMConcOnlyKL: MSE=2.0969 (std 0.3979), MAE=0.9977, LINEX=3.4653, avg HHI=0.2957
- MWUMBothKL: MSE=2.2235 (std 0.4099), MAE=1.0367, LINEX=3.6229, avg HHI=0.1916
- OGDBoth: MSE=2.2650 (std 0.4327), MAE=1.0521, LINEX=3.5191, avg HHI=0.1672
- OGDVanilla: MSE=2.2678 (std 0.4420), MAE=1.0529, LINEX=3.5129, avg HHI=0.2298
- OGDConcOnly: MSE=2.3789 (std 0.5370), MAE=1.0869, LINEX=4.1541, avg HHI=0.1321
- MWUMVanilla: MSE=2.9936 (std 0.5752), MAE=1.2111, LINEX=7.4537, avg HHI=0.2440
- Median: MSE=3.0434 (std 0.6486), MAE=1.2203, LINEX=8.6394, avg HHI=nan
- Mean: MSE=3.0623 (std 0.6386), MAE=1.2447, LINEX=9.1668, avg HHI=0.0556

### Horizon h=8
- MWUMConcOnlyKL: MSE=2.6249 (std 0.6435), MAE=1.1190, LINEX=5.1589, avg HHI=0.3003
- MWUMBothKL: MSE=2.6814 (std 0.6293), MAE=1.1399, LINEX=5.2487, avg HHI=0.2327
- OGDBoth: MSE=2.9933 (std 0.6935), MAE=1.2221, LINEX=6.5905, avg HHI=0.1635
- OGDVanilla: MSE=3.0040 (std 0.6954), MAE=1.2250, LINEX=6.6408, avg HHI=0.1969
- OGDConcOnly: MSE=3.2954 (std 0.8867), MAE=1.3060, LINEX=9.4258, avg HHI=0.1664
- MWUMVanilla: MSE=3.8891 (std 0.8552), MAE=1.3976, LINEX=14.9285, avg HHI=0.9768
- Median: MSE=4.2037 (std 1.0563), MAE=1.4574, LINEX=19.0906, avg HHI=nan
- Mean: MSE=4.4315 (std 1.1429), MAE=1.5255, LINEX=22.2751, avg HHI=0.0556
