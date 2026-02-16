# Dual-Loss Ensemble + RL Analysis

- LINEX parameter `a`: 1.0

## MSE Results

### Tuned Hyperparameters (Regular Methods)
- Horizon 1:
  - MWUMVanilla: params={'eta': 0.01580148447850615} objective=0.9275967740804423
  - MWUMBothKL: params={'eta': 0.015919552043565138, 'kappa': 0.005714357133536371} objective=0.9276072196891759
  - Median: params={} objective=0.9351405990174786
  - OGDVanilla: params={'eta': 0.001004380875007612} objective=0.9363174265651122
  - OGDBoth: params={'eta': 0.001000231236412198, 'kappa': 0.05203263616499159} objective=0.9363951771480309
  - MWUMConcOnlyKL: params={'kappa': 7.849943290518327} objective=0.9436778985286702
  - Mean: params={} objective=0.951494710698005
  - OGDConcOnly: params={'kappa': 7.989067040220606} objective=1.0591123854559412
- Horizon 4:
  - MWUMConcOnlyKL: params={'kappa': 0.28056807110434306} objective=2.3110477195341685
  - MWUMBothKL: params={'eta': 2.9821867212394793, 'kappa': 0.5913228643145931} objective=2.436336792645921
  - OGDBoth: params={'eta': 0.03643044147708752, 'kappa': 0.8840210944332434} objective=2.5986572663695844
  - OGDVanilla: params={'eta': 0.04449333821791166} objective=2.617438665876078
  - OGDConcOnly: params={'kappa': 7.973929436896223} objective=2.741631494002637
  - MWUMVanilla: params={'eta': 0.014568470700968195} objective=3.208999573430413
  - Median: params={} objective=3.257653711387143
  - Mean: params={} objective=3.300878809265483
- Horizon 8:
  - MWUMConcOnlyKL: params={'kappa': 0.39742005940341457} objective=2.934682642435865
  - MWUMBothKL: params={'eta': 2.900336707567024, 'kappa': 0.6678824415504225} objective=3.0253191922396407
  - OGDBoth: params={'eta': 0.015110282798066584, 'kappa': 0.7407595779704955} objective=3.470104023728017
  - OGDVanilla: params={'eta': 0.01529830399119274} objective=3.4878812727270203
  - OGDConcOnly: params={'kappa': 7.9869664267965925} objective=4.083511276008283
  - MWUMVanilla: params={'eta': 2.8200403424806297} objective=4.265079734439067
  - Median: params={} objective=4.704949890767167
  - Mean: params={} objective=4.886158056388866

### Summary (Lower Objective Is Better)
- Horizon 1:
  - MWUMBothKL: objective=0.8260 (std 0.1223), MSE=0.8260, MAE=0.6170, LINEX=0.7475, avg HHI=0.0676, excess vs best indiv=-0.0305
  - MWUMVanilla: objective=0.8260 (std 0.1223), MSE=0.8260, MAE=0.6171, LINEX=0.7472, avg HHI=0.0678, excess vs best indiv=-0.0304
  - Median: objective=0.8316 (std 0.1238), MSE=0.8316, MAE=0.6188, LINEX=0.7407, avg HHI=nan, excess vs best indiv=-0.0249
  - OGDVanilla: objective=0.8321 (std 0.1238), MSE=0.8321, MAE=0.6218, LINEX=0.7545, avg HHI=0.0603, excess vs best indiv=-0.0244
  - OGDBoth: objective=0.8321 (std 0.1238), MSE=0.8321, MAE=0.6218, LINEX=0.7544, avg HHI=0.0601, excess vs best indiv=-0.0243
  - MWUMConcOnlyKL: objective=0.8403 (std 0.1266), MSE=0.8403, MAE=0.6252, LINEX=0.7522, avg HHI=0.0566, excess vs best indiv=-0.0162
  - Mean: objective=0.8442 (std 0.1273), MSE=0.8442, MAE=0.6329, LINEX=0.7219, avg HHI=0.0556, excess vs best indiv=-0.0123
  - RLRuleBandit: objective=0.8444 (std 0.1193), MSE=0.8444, MAE=0.6272, LINEX=0.7514, avg HHI=0.0618, excess vs best indiv=-0.0121
  - RLKappaBandit: objective=0.9097 (std 0.1327), MSE=0.9097, MAE=0.6422, LINEX=0.8987, avg HHI=0.2298, excess vs best indiv=0.0532
  - OGDConcOnly: objective=0.9249 (std 0.1478), MSE=0.9249, MAE=0.6535, LINEX=0.8911, avg HHI=0.0860, excess vs best indiv=0.0685
  - RLSimplexBandit: objective=1.0623 (std 0.4960), MSE=1.0623, MAE=0.6968, LINEX=0.9601, avg HHI=0.4257, excess vs best indiv=0.2058
- Horizon 4:
  - MWUMConcOnlyKL: objective=2.0969 (std 0.3979), MSE=2.0969, MAE=0.9977, LINEX=3.4653, avg HHI=0.2957, excess vs best indiv=-0.8883
  - MWUMBothKL: objective=2.2235 (std 0.4099), MSE=2.2235, MAE=1.0367, LINEX=3.6229, avg HHI=0.1916, excess vs best indiv=-0.7617
  - RLKappaBandit: objective=2.2241 (std 0.4278), MSE=2.2241, MAE=1.0353, LINEX=3.6903, avg HHI=0.3843, excess vs best indiv=-0.7611
  - OGDBoth: objective=2.2650 (std 0.4327), MSE=2.2650, MAE=1.0521, LINEX=3.5191, avg HHI=0.1672, excess vs best indiv=-0.7202
  - OGDVanilla: objective=2.2678 (std 0.4420), MSE=2.2678, MAE=1.0529, LINEX=3.5129, avg HHI=0.2298, excess vs best indiv=-0.7174
  - OGDConcOnly: objective=2.3789 (std 0.5370), MSE=2.3789, MAE=1.0869, LINEX=4.1541, avg HHI=0.1321, excess vs best indiv=-0.6063
  - RLRuleBandit: objective=2.5774 (std 0.6077), MSE=2.5774, MAE=1.1164, LINEX=5.3632, avg HHI=0.1773, excess vs best indiv=-0.4078
  - MWUMVanilla: objective=2.9936 (std 0.5752), MSE=2.9936, MAE=1.2111, LINEX=7.4537, avg HHI=0.2440, excess vs best indiv=0.0084
  - Median: objective=3.0434 (std 0.6486), MSE=3.0434, MAE=1.2203, LINEX=8.6394, avg HHI=nan, excess vs best indiv=0.0582
  - Mean: objective=3.0623 (std 0.6386), MSE=3.0623, MAE=1.2447, LINEX=9.1668, avg HHI=0.0556, excess vs best indiv=0.0771
  - RLSimplexBandit: objective=3.2636 (std 0.6767), MSE=3.2636, MAE=1.2814, LINEX=8.3857, avg HHI=0.7558, excess vs best indiv=0.2784
- Horizon 8:
  - MWUMConcOnlyKL: objective=2.6249 (std 0.6435), MSE=2.6249, MAE=1.1190, LINEX=5.1589, avg HHI=0.3003, excess vs best indiv=-1.2072
  - MWUMBothKL: objective=2.6814 (std 0.6293), MSE=2.6814, MAE=1.1399, LINEX=5.2487, avg HHI=0.2327, excess vs best indiv=-1.1507
  - RLKappaBandit: objective=2.7940 (std 0.7234), MSE=2.7940, MAE=1.1640, LINEX=5.8040, avg HHI=0.3782, excess vs best indiv=-1.0381
  - OGDBoth: objective=2.9933 (std 0.6935), MSE=2.9933, MAE=1.2221, LINEX=6.5905, avg HHI=0.1635, excess vs best indiv=-0.8388
  - OGDVanilla: objective=3.0040 (std 0.6954), MSE=3.0040, MAE=1.2250, LINEX=6.6408, avg HHI=0.1969, excess vs best indiv=-0.8281
  - OGDConcOnly: objective=3.2954 (std 0.8867), MSE=3.2954, MAE=1.3060, LINEX=9.4258, avg HHI=0.1664, excess vs best indiv=-0.5367
  - RLRuleBandit: objective=3.6694 (std 1.1482), MSE=3.6694, MAE=1.3501, LINEX=83.8387, avg HHI=0.2204, excess vs best indiv=-0.1628
  - MWUMVanilla: objective=3.8891 (std 0.8552), MSE=3.8891, MAE=1.3976, LINEX=14.9285, avg HHI=0.9768, excess vs best indiv=0.0570
  - Median: objective=4.2037 (std 1.0563), MSE=4.2037, MAE=1.4574, LINEX=19.0906, avg HHI=nan, excess vs best indiv=0.3716
  - Mean: objective=4.4315 (std 1.1429), MSE=4.4315, MAE=1.5255, LINEX=22.2751, avg HHI=0.0556, excess vs best indiv=0.5994
  - RLSimplexBandit: objective=5.6000 (std 2.0781), MSE=5.6000, MAE=1.7140, LINEX=53.6239, avg HHI=0.8362, excess vs best indiv=1.7679

## LINEX Results

### Tuned Hyperparameters (Regular Methods)
- Horizon 1:
  - Mean: params={} objective=1.5716887880116488
  - MWUMConcOnlyKL: params={'kappa': 7.849943290518327} objective=1.8737130724116813
  - MWUMBothKL: params={'eta': 0.05059635030485148, 'kappa': 7.982919948391065} objective=1.8868371070546055
  - MWUMVanilla: params={'eta': 0.001071380345127959} objective=1.9679959458159566
  - Median: params={} objective=2.1397734634077263
  - OGDVanilla: params={'eta': 0.0031709028216475085} objective=2.29131275968619
  - OGDBoth: params={'eta': 0.00302272642675273, 'kappa': 0.003670244761572792} objective=2.291582585620061
  - OGDConcOnly: params={'kappa': 7.989067040220606} objective=2.579140724483963
- Horizon 4:
  - OGDBoth: params={'eta': 0.031220593023844168, 'kappa': 3.624385325514904} objective=5.3657827366635145
  - MWUMConcOnlyKL: params={'kappa': 1.8557563808482023} objective=5.381439649967893
  - MWUMBothKL: params={'eta': 2.9230972720137793, 'kappa': 2.439287239235953} objective=5.47413495328397
  - OGDVanilla: params={'eta': 0.0495541420385862} objective=5.481725480929249
  - MWUMVanilla: params={'eta': 1.2322634388058582} objective=9.440538812744247
  - Mean: params={} objective=14.812864759185958
  - Median: params={} objective=14.977728192844564
  - OGDConcOnly: params={'kappa': 7.973929436896223} objective=18.846284429662138
- Horizon 8:
  - OGDBoth: params={'eta': 0.033144088276003625, 'kappa': 1.3919459890346317} objective=10.561033791949372
  - OGDVanilla: params={'eta': 0.028792972760974003} objective=10.594387449501564
  - MWUMConcOnlyKL: params={'kappa': 0.96553628534458} objective=10.701371566188174
  - MWUMBothKL: params={'eta': 2.7054467838824405, 'kappa': 1.4723907206505458} objective=10.813543879969608
  - MWUMVanilla: params={'eta': 2.6977035594911487} objective=19.60451370140126
  - Median: params={} objective=28.772192927198226
  - Mean: params={} objective=28.899487357446475
  - OGDConcOnly: params={'kappa': 7.9869664267965925} objective=44.4058864109314

### Summary (Lower Objective Is Better)
- Horizon 1:
  - Mean: objective=0.7219 (std 0.1549), MSE=0.8442, MAE=0.6329, LINEX=0.7219, avg HHI=0.0556, excess vs best indiv=0.0873
  - MWUMVanilla: objective=0.7236 (std 0.1430), MSE=0.8325, MAE=0.6242, LINEX=0.7236, avg HHI=0.0580, excess vs best indiv=0.0890
  - MWUMBothKL: objective=0.7286 (std 0.1520), MSE=0.8470, MAE=0.6297, LINEX=0.7286, avg HHI=0.0565, excess vs best indiv=0.0940
  - MWUMConcOnlyKL: objective=0.7327 (std 0.1540), MSE=0.8500, MAE=0.6303, LINEX=0.7327, avg HHI=0.0574, excess vs best indiv=0.0981
  - OGDBoth: objective=0.7334 (std 0.1667), MSE=0.9085, MAE=0.6645, LINEX=0.7334, avg HHI=0.1301, excess vs best indiv=0.0988
  - OGDVanilla: objective=0.7347 (std 0.1674), MSE=0.9092, MAE=0.6648, LINEX=0.7347, avg HHI=0.1321, excess vs best indiv=0.1002
  - Median: objective=0.7407 (std 0.1428), MSE=0.8316, MAE=0.6188, LINEX=0.7407, avg HHI=nan, excess vs best indiv=0.1062
  - RLRuleBandit: objective=0.7466 (std 0.1611), MSE=0.8583, MAE=0.6355, LINEX=0.7466, avg HHI=0.0684, excess vs best indiv=0.1120
  - OGDConcOnly: objective=0.7969 (std 0.1583), MSE=0.8786, MAE=0.6372, LINEX=0.7969, avg HHI=0.0756, excess vs best indiv=0.1623
  - RLKappaBandit: objective=0.8623 (std 0.1805), MSE=0.8933, MAE=0.6423, LINEX=0.8623, avg HHI=0.1815, excess vs best indiv=0.2277
  - RLSimplexBandit: objective=3.5880 (std 3.4115), MSE=1.8214, MAE=0.9136, LINEX=3.5880, avg HHI=0.5689, excess vs best indiv=2.9535
- Horizon 4:
  - OGDVanilla: objective=3.2881 (std 1.9914), MSE=2.5936, MAE=1.1176, LINEX=3.2881, avg HHI=0.2508, excess vs best indiv=-2.3138
  - OGDBoth: objective=3.3022 (std 1.9098), MSE=2.5319, MAE=1.1118, LINEX=3.3022, avg HHI=0.1113, excess vs best indiv=-2.2997
  - MWUMConcOnlyKL: objective=3.3218 (std 1.8684), MSE=2.4805, MAE=1.1088, LINEX=3.3218, avg HHI=0.0973, excess vs best indiv=-2.2801
  - RLKappaBandit: objective=3.4106 (std 1.9372), MSE=2.2450, MAE=1.0427, LINEX=3.4106, avg HHI=0.2765, excess vs best indiv=-2.1914
  - MWUMBothKL: objective=3.4357 (std 1.8712), MSE=2.5876, MAE=1.1337, LINEX=3.4357, avg HHI=0.0914, excess vs best indiv=-2.1663
  - OGDConcOnly: objective=3.5005 (std 2.0148), MSE=2.4262, MAE=1.0950, LINEX=3.5005, avg HHI=0.1125, excess vs best indiv=-2.1014
  - RLRuleBandit: objective=4.6859 (std 2.5221), MSE=2.8176, MAE=1.1745, LINEX=4.6859, avg HHI=0.2155, excess vs best indiv=-0.9160
  - MWUMVanilla: objective=7.8044 (std 5.3439), MSE=3.3490, MAE=1.3097, LINEX=7.8044, avg HHI=0.9623, excess vs best indiv=2.2025
  - Median: objective=8.6394 (std 5.6749), MSE=3.0434, MAE=1.2203, LINEX=8.6394, avg HHI=nan, excess vs best indiv=3.0375
  - RLSimplexBandit: objective=8.9653 (std 4.1319), MSE=3.3958, MAE=1.3013, LINEX=8.9653, avg HHI=0.7817, excess vs best indiv=3.3634
  - Mean: objective=9.1668 (std 6.8162), MSE=3.0623, MAE=1.2447, LINEX=9.1668, avg HHI=0.0556, excess vs best indiv=3.5649
- Horizon 8:
  - MWUMConcOnlyKL: objective=4.9678 (std 2.7692), MSE=2.9629, MAE=1.2151, LINEX=4.9678, avg HHI=0.1482, excess vs best indiv=-2.2456
  - MWUMBothKL: objective=5.0742 (std 2.9579), MSE=3.1663, MAE=1.2627, LINEX=5.0742, avg HHI=0.1344, excess vs best indiv=-2.1392
  - OGDBoth: objective=5.1449 (std 2.8465), MSE=3.1401, MAE=1.2431, LINEX=5.1449, avg HHI=0.1691, excess vs best indiv=-2.0685
  - OGDVanilla: objective=5.3811 (std 2.8306), MSE=3.3870, MAE=1.2870, LINEX=5.3811, avg HHI=0.2689, excess vs best indiv=-1.8323
  - RLKappaBandit: objective=5.4491 (std 2.9321), MSE=2.9618, MAE=1.2030, LINEX=5.4491, avg HHI=0.3143, excess vs best indiv=-1.7643
  - OGDConcOnly: objective=5.9454 (std 3.2663), MSE=3.2298, MAE=1.2903, LINEX=5.9454, avg HHI=0.1387, excess vs best indiv=-1.2679
  - RLRuleBandit: objective=8.0707 (std 4.3518), MSE=3.6037, MAE=1.3383, LINEX=8.0707, avg HHI=0.1689, excess vs best indiv=0.8573
  - MWUMVanilla: objective=9.0167 (std 5.2777), MSE=4.3507, MAE=1.4923, LINEX=9.0167, avg HHI=0.9728, excess vs best indiv=1.8033
  - Median: objective=19.0906 (std 17.3029), MSE=4.2037, MAE=1.4574, LINEX=19.0906, avg HHI=nan, excess vs best indiv=11.8773
  - Mean: objective=22.2751 (std 19.7149), MSE=4.4315, MAE=1.5255, LINEX=22.2751, avg HHI=0.0556, excess vs best indiv=15.0617
  - RLSimplexBandit: objective=110.1577 (std 109.6555), MSE=6.3816, MAE=1.8707, LINEX=110.1577, avg HHI=0.8643, excess vs best indiv=102.9443

