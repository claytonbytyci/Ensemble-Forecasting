# Dual-Loss Ensemble + RL Analysis

- LINEX parameter `a`: 1.0

## MSE Results

### Tuned Hyperparameters (Regular Methods)
- Horizon 1:
  - MWUMVanilla: params={'eta': 0.01580148447850615} objective=0.9275967740804423
  - MWUMBothKL: params={'eta': 0.015919552043565138, 'kappa': 0.005714357133536371} objective=0.9276027672026388
  - Median: params={} objective=0.9351405990174786
  - OGDVanilla: params={'eta': 0.001004380875007612} objective=0.9363174265651122
  - OGDBoth: params={'eta': 0.0010089037878103623, 'kappa': 0.14340181023302656} objective=0.9365165142555085
  - Mean: params={} objective=0.951494710698005
  - MWUMConcOnlyKL: params={'kappa': 7.849943290518327} objective=0.9519221512310448
  - OGDConcOnly: params={'kappa': 7.3665560605325995} objective=1.1041852360579676
- Horizon 4:
  - MWUMConcOnlyKL: params={'kappa': 0.45564289138541536} objective=2.307407447875409
  - MWUMBothKL: params={'eta': 2.99138715807153, 'kappa': 1.102821750702496} objective=2.4236141833202196
  - OGDBoth: params={'eta': 0.035882137858637725, 'kappa': 3.4595834023016296} objective=2.592423240902611
  - OGDVanilla: params={'eta': 0.04449333821791166} objective=2.617438665876078
  - OGDConcOnly: params={'kappa': 7.973929436896223} objective=2.7568134316670343
  - MWUMVanilla: params={'eta': 0.014568470700968195} objective=3.208999573430413
  - Median: params={} objective=3.257653711387143
  - Mean: params={} objective=3.300878809265483
- Horizon 8:
  - MWUMConcOnlyKL: params={'kappa': 0.581453887083425} objective=2.9312089748609425
  - MWUMBothKL: params={'eta': 2.958728127774368, 'kappa': 1.0504226437089312} objective=3.015381101626353
  - OGDBoth: params={'eta': 0.015110282798066584, 'kappa': 0.7407595779704955} objective=3.471352264769482
  - OGDVanilla: params={'eta': 0.01529830399119274} objective=3.4878812727270203
  - OGDConcOnly: params={'kappa': 7.9869664267965925} objective=4.120663328383975
  - MWUMVanilla: params={'eta': 2.8200403424806297} objective=4.265079734439067
  - Median: params={} objective=4.704949890767167
  - Mean: params={} objective=4.886158056388866

### Summary (Lower Objective Is Better)
- Horizon 1:
  - MWUMBothKL: objective=0.8906 (std 0.1187), MSE=0.8906, MAE=0.6371, LINEX=1.1601, avg HHI=0.0684, excess vs best indiv=-0.0331, improvement vs best indiv=3.64%
  - MWUMVanilla: objective=0.8907 (std 0.1187), MSE=0.8907, MAE=0.6371, LINEX=1.1599, avg HHI=0.0685, excess vs best indiv=-0.0331, improvement vs best indiv=3.63%
  - OGDVanilla: objective=0.8957 (std 0.1194), MSE=0.8957, MAE=0.6424, LINEX=1.1251, avg HHI=0.0609, excess vs best indiv=-0.0280, improvement vs best indiv=3.09%
  - OGDBoth: objective=0.8959 (std 0.1194), MSE=0.8959, MAE=0.6425, LINEX=1.1250, avg HHI=0.0605, excess vs best indiv=-0.0278, improvement vs best indiv=3.07%
  - Median: objective=0.8966 (std 0.1184), MSE=0.8966, MAE=0.6378, LINEX=1.1410, avg HHI=nan, excess vs best indiv=-0.0272, improvement vs best indiv=2.98%
  - MWUMConcOnlyKL: objective=0.9089 (std 0.1231), MSE=0.9089, MAE=0.6432, LINEX=1.1863, avg HHI=0.0577, excess vs best indiv=-0.0148, improvement vs best indiv=1.69%
  - Mean: objective=0.9089 (std 0.1210), MSE=0.9089, MAE=0.6522, LINEX=1.0611, avg HHI=0.0556, excess vs best indiv=-0.0148, improvement vs best indiv=1.66%
  - RLRuleBandit: objective=0.9145 (std 0.1235), MSE=0.9145, MAE=0.6470, LINEX=1.1426, avg HHI=0.0625, excess vs best indiv=-0.0092, improvement vs best indiv=1.06%
  - RLKappaBandit: objective=0.9966 (std 0.1412), MSE=0.9966, MAE=0.6650, LINEX=2.2384, avg HHI=0.2403, excess vs best indiv=0.0729, improvement vs best indiv=-7.76%
  - OGDConcOnly: objective=1.0491 (std 0.1786), MSE=1.0491, MAE=0.6830, LINEX=2.4060, avg HHI=0.1061, excess vs best indiv=0.1254, improvement vs best indiv=-13.13%
- Horizon 4:
  - MWUMConcOnlyKL: objective=2.1975 (std 0.3285), MSE=2.1975, MAE=1.0192, LINEX=6.6345, avg HHI=0.2833, excess vs best indiv=-0.9171, improvement vs best indiv=29.26%
  - RLKappaBandit: objective=2.2897 (std 0.3494), MSE=2.2897, MAE=1.0478, LINEX=9.4774, avg HHI=0.3743, excess vs best indiv=-0.8249, improvement vs best indiv=26.33%
  - MWUMBothKL: objective=2.3082 (std 0.3440), MSE=2.3082, MAE=1.0558, LINEX=6.6207, avg HHI=0.1703, excess vs best indiv=-0.8064, improvement vs best indiv=25.70%
  - OGDBoth: objective=2.3841 (std 0.3681), MSE=2.3841, MAE=1.0742, LINEX=16.6080, avg HHI=0.1349, excess vs best indiv=-0.7305, improvement vs best indiv=23.30%
  - OGDVanilla: objective=2.3946 (std 0.3817), MSE=2.3946, MAE=1.0741, LINEX=16.9284, avg HHI=0.2401, excess vs best indiv=-0.7200, improvement vs best indiv=23.01%
  - OGDConcOnly: objective=2.5442 (std 0.4329), MSE=2.5442, MAE=1.1217, LINEX=29.5749, avg HHI=0.1599, excess vs best indiv=-0.5704, improvement vs best indiv=18.34%
  - RLRuleBandit: objective=2.7553 (std 0.5166), MSE=2.7553, MAE=1.1467, LINEX=154.8744, avg HHI=0.1723, excess vs best indiv=-0.3593, improvement vs best indiv=11.68%
  - MWUMVanilla: objective=3.1196 (std 0.5069), MSE=3.1196, MAE=1.2328, LINEX=32.6878, avg HHI=0.2213, excess vs best indiv=0.0050, improvement vs best indiv=-0.15%
  - Median: objective=3.1577 (std 0.5325), MSE=3.1577, MAE=1.2443, LINEX=40.5831, avg HHI=nan, excess vs best indiv=0.0431, improvement vs best indiv=-1.28%
  - Mean: objective=3.1776 (std 0.5182), MSE=3.1776, MAE=1.2712, LINEX=34.8956, avg HHI=0.0556, excess vs best indiv=0.0630, improvement vs best indiv=-2.01%
- Horizon 8:
  - MWUMConcOnlyKL: objective=2.8510 (std 0.5504), MSE=2.8510, MAE=1.1501, LINEX=25.2025, avg HHI=0.2947, excess vs best indiv=-1.1761, improvement vs best indiv=28.88%
  - MWUMBothKL: objective=2.8867 (std 0.5372), MSE=2.8867, MAE=1.1679, LINEX=25.3095, avg HHI=0.2166, excess vs best indiv=-1.1404, improvement vs best indiv=27.92%
  - RLKappaBandit: objective=2.9957 (std 0.5704), MSE=2.9957, MAE=1.1910, LINEX=27.2552, avg HHI=0.3964, excess vs best indiv=-1.0314, improvement vs best indiv=25.22%
  - OGDBoth: objective=3.1949 (std 0.5872), MSE=3.1949, MAE=1.2393, LINEX=29.8697, avg HHI=0.1669, excess vs best indiv=-0.8322, improvement vs best indiv=20.18%
  - OGDVanilla: objective=3.2015 (std 0.5872), MSE=3.2015, MAE=1.2410, LINEX=29.8549, avg HHI=0.1920, excess vs best indiv=-0.8256, improvement vs best indiv=20.01%
  - RLRuleBandit: objective=3.7170 (std 0.7785), MSE=3.7170, MAE=1.3263, LINEX=33617.4164, avg HHI=0.2725, excess vs best indiv=-0.3101, improvement vs best indiv=7.14%
  - OGDConcOnly: objective=3.7187 (std 0.8033), MSE=3.7187, MAE=1.3593, LINEX=37.0362, avg HHI=0.1978, excess vs best indiv=-0.3084, improvement vs best indiv=7.64%
  - MWUMVanilla: objective=4.0720 (std 0.8152), MSE=4.0720, MAE=1.4049, LINEX=43.3371, avg HHI=0.9848, excess vs best indiv=0.0449, improvement vs best indiv=-1.23%
  - Median: objective=4.3920 (std 0.9453), MSE=4.3920, MAE=1.4699, LINEX=716.8541, avg HHI=nan, excess vs best indiv=0.3649, improvement vs best indiv=-8.88%
  - Mean: objective=4.5913 (std 0.9575), MSE=4.5913, MAE=1.5391, LINEX=591.6474, avg HHI=0.0556, excess vs best indiv=0.5642, improvement vs best indiv=-13.99%

## LINEX Results

### Tuned Hyperparameters (Regular Methods)
- Horizon 1:
  - Mean: params={} objective=1.5716887880116488
  - MWUMVanilla: params={'eta': 0.001071380345127959} objective=1.9679959458159566
  - MWUMConcOnlyKL: params={'kappa': 7.849943290518327} objective=2.0671452640488184
  - Median: params={} objective=2.1397734634077263
  - MWUMBothKL: params={'eta': 2.659494788779418, 'kappa': 0.0011509254400450255} objective=2.1510027870386694
  - OGDVanilla: params={'eta': 0.0031709028216475085} objective=2.29131275968619
  - OGDBoth: params={'eta': 0.00302272642675273, 'kappa': 0.003670244761572792} objective=2.2914138080633633
  - OGDConcOnly: params={'kappa': 7.989067040220606} objective=2.80896714188421
- Horizon 4:
  - OGDBoth: params={'eta': 0.03764200803282203, 'kappa': 2.841565616132992} objective=5.329783839166686
  - MWUMConcOnlyKL: params={'kappa': 3.0807731507848457} objective=5.388545194986812
  - MWUMBothKL: params={'eta': 1.2395799335272268, 'kappa': 2.46387160716886} objective=5.417556773169373
  - OGDVanilla: params={'eta': 0.0495541420385862} objective=5.481725480929249
  - MWUMVanilla: params={'eta': 1.2322634388058582} objective=9.440538812744247
  - Mean: params={} objective=14.812864759185958
  - Median: params={} objective=14.977728192844564
  - OGDConcOnly: params={'kappa': 7.973929436896223} objective=25.859249264175265
- Horizon 8:
  - OGDBoth: params={'eta': 0.03201348093933163, 'kappa': 2.321990267574276} objective=10.567067119087252
  - OGDVanilla: params={'eta': 0.028792972760974003} objective=10.594387449501564
  - MWUMConcOnlyKL: params={'kappa': 1.6291463329239688} objective=10.705561198031877
  - MWUMBothKL: params={'eta': 2.7054467838824405, 'kappa': 1.4723907206505458} objective=10.789711787139817
  - MWUMVanilla: params={'eta': 2.6977035594911487} objective=19.60451370140126
  - Median: params={} objective=28.772192927198226
  - Mean: params={} objective=28.899487357446475
  - OGDConcOnly: params={'kappa': 7.9869664267965925} objective=93.90752011802049

### Summary (Lower Objective Is Better)
- Horizon 1:
  - Mean: objective=1.0611 (std 1.1035), MSE=0.9089, MAE=0.6522, LINEX=1.0611, avg HHI=0.0556, excess vs best indiv=0.2160, improvement vs best indiv=-18.20%
  - MWUMVanilla: objective=1.1357 (std 1.5164), MSE=0.9013, MAE=0.6452, LINEX=1.1357, avg HHI=0.0625, excess vs best indiv=0.2905, improvement vs best indiv=-19.67%
  - Median: objective=1.1410 (std 1.3979), MSE=0.8966, MAE=0.6378, LINEX=1.1410, avg HHI=nan, excess vs best indiv=0.2959, improvement vs best indiv=-21.90%
  - OGDVanilla: objective=1.8581 (std 5.5451), MSE=0.9669, MAE=0.6782, LINEX=1.8581, avg HHI=0.1446, excess vs best indiv=1.0130, improvement vs best indiv=-73.42%
  - OGDBoth: objective=1.8604 (std 5.5457), MSE=0.9663, MAE=0.6779, LINEX=1.8604, avg HHI=0.1429, excess vs best indiv=1.0153, improvement vs best indiv=-73.50%
  - MWUMBothKL: objective=1.8990 (std 5.5216), MSE=0.9925, MAE=0.6968, LINEX=1.8990, avg HHI=0.9406, excess vs best indiv=1.0539, improvement vs best indiv=-68.35%
  - RLRuleBandit: objective=1.9076 (std 5.5619), MSE=0.9375, MAE=0.6584, LINEX=1.9076, avg HHI=0.1913, excess vs best indiv=1.0625, improvement vs best indiv=-76.74%
  - MWUMConcOnlyKL: objective=1.9111 (std 5.5410), MSE=0.9168, MAE=0.6474, LINEX=1.9111, avg HHI=0.0587, excess vs best indiv=1.0659, improvement vs best indiv=-76.20%
  - RLKappaBandit: objective=2.1174 (std 5.5683), MSE=0.9888, MAE=0.6623, LINEX=2.1174, avg HHI=0.1973, excess vs best indiv=1.2722, improvement vs best indiv=-100.81%
  - OGDConcOnly: objective=2.1442 (std 6.7686), MSE=0.9671, MAE=0.6584, LINEX=2.1442, avg HHI=0.0845, excess vs best indiv=1.2990, improvement vs best indiv=-103.19%
- Horizon 4:
  - MWUMConcOnlyKL: objective=5.7328 (std 8.4066), MSE=2.5791, MAE=1.1339, LINEX=5.7328, avg HHI=0.0969, excess vs best indiv=-1.3547, improvement vs best indiv=28.26%
  - OGDBoth: objective=5.8424 (std 9.0609), MSE=2.5894, MAE=1.1183, LINEX=5.8424, avg HHI=0.1387, excess vs best indiv=-1.2450, improvement vs best indiv=28.58%
  - MWUMBothKL: objective=5.8902 (std 8.4386), MSE=2.6964, MAE=1.1515, LINEX=5.8902, avg HHI=0.1096, excess vs best indiv=-1.1972, improvement vs best indiv=25.77%
  - RLKappaBandit: objective=5.9263 (std 8.9860), MSE=2.3497, MAE=1.0642, LINEX=5.9263, avg HHI=0.3013, excess vs best indiv=-1.1611, improvement vs best indiv=26.68%
  - OGDVanilla: objective=6.2282 (std 11.0463), MSE=2.6650, MAE=1.1307, LINEX=6.2282, avg HHI=0.2691, excess vs best indiv=-0.8592, improvement vs best indiv=27.69%
  - MWUMVanilla: objective=9.5267 (std 10.2008), MSE=3.4547, MAE=1.3276, LINEX=9.5267, avg HHI=0.9694, excess vs best indiv=2.4393, improvement vs best indiv=-27.31%
  - OGDConcOnly: objective=9.5303 (std 28.5165), MSE=2.4974, MAE=1.1083, LINEX=9.5303, avg HHI=0.1321, excess vs best indiv=2.4429, improvement vs best indiv=12.95%
  - Mean: objective=34.8956 (std 135.3049), MSE=3.1776, MAE=1.2712, LINEX=34.8956, avg HHI=0.0556, excess vs best indiv=27.8081, improvement vs best indiv=-270.34%
  - RLRuleBandit: objective=37.3307 (std 141.3807), MSE=2.9846, MAE=1.2119, LINEX=37.3307, avg HHI=0.2402, excess vs best indiv=30.2433, improvement vs best indiv=-283.90%
  - Median: objective=40.5831 (std 170.1704), MSE=3.1577, MAE=1.2443, LINEX=40.5831, avg HHI=nan, excess vs best indiv=33.4957, improvement vs best indiv=-330.44%
- Horizon 8:
  - MWUMBothKL: objective=23.3948 (std 65.6375), MSE=3.3528, MAE=1.2697, LINEX=23.3948, avg HHI=0.1542, excess vs best indiv=-16.8695, improvement vs best indiv=33.57%
  - OGDVanilla: objective=23.4533 (std 65.9603), MSE=3.5953, MAE=1.3085, LINEX=23.4533, avg HHI=0.2713, excess vs best indiv=-16.8110, improvement vs best indiv=33.18%
  - OGDBoth: objective=23.4945 (std 66.4516), MSE=3.3645, MAE=1.2690, LINEX=23.4945, avg HHI=0.1645, excess vs best indiv=-16.7699, improvement vs best indiv=34.02%
  - RLKappaBandit: objective=23.9619 (std 65.9761), MSE=3.1731, MAE=1.2332, LINEX=23.9619, avg HHI=0.3048, excess vs best indiv=-16.3025, improvement vs best indiv=28.75%
  - MWUMConcOnlyKL: objective=23.9752 (std 66.8828), MSE=3.2068, MAE=1.2509, LINEX=23.9752, avg HHI=0.1389, excess vs best indiv=-16.2892, improvement vs best indiv=31.88%
  - OGDConcOnly: objective=26.6590 (std 65.4984), MSE=3.5366, MAE=1.3218, LINEX=26.6590, avg HHI=0.1623, excess vs best indiv=-13.6054, improvement vs best indiv=4.90%
  - MWUMVanilla: objective=51.2666 (std 142.3725), MSE=4.7327, MAE=1.5106, LINEX=51.2666, avg HHI=0.9775, excess vs best indiv=11.0022, improvement vs best indiv=-211.93%
  - RLRuleBandit: objective=573.6548 (std 3320.4832), MSE=3.9712, MAE=1.3898, LINEX=573.6548, avg HHI=0.2389, excess vs best indiv=533.3905, improvement vs best indiv=-281.21%
  - Mean: objective=591.6474 (std 2324.5758), MSE=4.5913, MAE=1.5391, LINEX=591.6474, avg HHI=0.0556, excess vs best indiv=551.3830, improvement vs best indiv=-1902.61%
  - Median: objective=716.8541 (std 3836.2729), MSE=4.3920, MAE=1.4699, LINEX=716.8541, avg HHI=nan, excess vs best indiv=676.5898, improvement vs best indiv=-606.51%

