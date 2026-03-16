# Dual-Loss Ensemble + RL Analysis

- LINEX parameter `a`: 1.0

## MSE Results

### Tuned Hyperparameters (Regular Methods)
- Horizon 1:
  - MWUMBothKL: params={'eta': 0.014187626945931071, 'kappa': 0.03600282826691213} objective=0.9271714870229324
  - MWUMVanilla: params={'eta': 0.013783816558470288} objective=0.9272279839775619
  - Median: params={} objective=0.9340880694295579
  - OGDVanilla: params={'eta': 0.001004380875007612} objective=0.93532546699805
  - OGDBoth: params={'eta': 0.0010089037878103623, 'kappa': 0.14340181023302656} objective=0.9355025963929027
  - MWUMConcOnlyKL: params={'eta': 0.04000975920197938, 'kappa': 7.788171024743676} objective=0.9443016286381749
  - OGDConcOnly: params={'eta': 0.01889879259941431, 'kappa': 7.908550293895121} objective=0.9469512598749436
  - Mean: params={} objective=0.9508282742933337
- Horizon 4:
  - MWUMConcOnlyKL: params={'eta': 2.5332277751016674, 'kappa': 0.25608619358030765} objective=2.311522784364965
  - MWUMBothKL: params={'eta': 2.99138715807153, 'kappa': 1.102821750702496} objective=2.431739014294856
  - OGDBoth: params={'eta': 0.03288929935310459, 'kappa': 2.6338496245322305} objective=2.5811956326715024
  - OGDVanilla: params={'eta': 0.033110894270227446} objective=2.614810380977775
  - OGDConcOnly: params={'eta': 0.0017461524219968072, 'kappa': 2.672302940624942} objective=2.6683625145544543
  - MWUMVanilla: params={'eta': 0.01975905899018788} objective=3.2175799227502355
  - Median: params={} objective=3.263253206160238
  - Mean: params={} objective=3.3076637738264623
- Horizon 8:
  - MWUMConcOnlyKL: params={'eta': 0.2788500263129488, 'kappa': 0.2566672788130432} objective=2.9392786590445903
  - MWUMBothKL: params={'eta': 2.958728127774368, 'kappa': 1.0504226437089312} objective=3.0407156314546966
  - OGDBoth: params={'eta': 0.014286069428104169, 'kappa': 0.16207531409816334} objective=3.5638050643010213
  - OGDVanilla: params={'eta': 0.013709098506762857} objective=3.5662427730061297
  - OGDConcOnly: params={'eta': 0.003111324209734068, 'kappa': 2.8861943489415576} objective=3.7932315911265784
  - MWUMVanilla: params={'eta': 2.6029134173667132} objective=4.388585424780673
  - Median: params={} objective=4.740030592316158
  - Mean: params={} objective=4.908320973470071

### Summary (Lower Objective Is Better)
- Horizon 1:
  - MWUMBothKL: objective=0.8904 (std 0.1185), MSE=0.8904, MAE=0.6370, LINEX=1.1612, avg HHI=0.0663, excess vs best indiv=-0.0333, improvement vs best indiv=3.66%
  - MWUMVanilla: objective=0.8905 (std 0.1186), MSE=0.8905, MAE=0.6371, LINEX=1.1600, avg HHI=0.0671, excess vs best indiv=-0.0332, improvement vs best indiv=3.65%
  - OGDVanilla: objective=0.8957 (std 0.1194), MSE=0.8957, MAE=0.6424, LINEX=1.1251, avg HHI=0.0609, excess vs best indiv=-0.0280, improvement vs best indiv=3.09%
  - OGDBoth: objective=0.8959 (std 0.1194), MSE=0.8959, MAE=0.6425, LINEX=1.1250, avg HHI=0.0605, excess vs best indiv=-0.0278, improvement vs best indiv=3.07%
  - Median: objective=0.8966 (std 0.1184), MSE=0.8966, MAE=0.6378, LINEX=1.1410, avg HHI=nan, excess vs best indiv=-0.0272, improvement vs best indiv=2.98%
  - RLRuleBandit: objective=0.9001 (std 0.1213), MSE=0.9001, MAE=0.6443, LINEX=1.1428, avg HHI=0.0579, excess vs best indiv=-0.0237, improvement vs best indiv=2.63%
  - MWUMConcOnlyKL: objective=0.9039 (std 0.1216), MSE=0.9039, MAE=0.6445, LINEX=1.1446, avg HHI=0.0562, excess vs best indiv=-0.0198, improvement vs best indiv=2.21%
  - OGDConcOnly: objective=0.9063 (std 0.1234), MSE=0.9063, MAE=0.6451, LINEX=1.0276, avg HHI=0.0573, excess vs best indiv=-0.0174, improvement vs best indiv=1.98%
  - Mean: objective=0.9089 (std 0.1210), MSE=0.9089, MAE=0.6522, LINEX=1.0611, avg HHI=0.0556, excess vs best indiv=-0.0148, improvement vs best indiv=1.66%
  - RLKappaBandit: objective=0.9966 (std 0.1412), MSE=0.9966, MAE=0.6650, LINEX=2.2384, avg HHI=0.2403, excess vs best indiv=0.0729, improvement vs best indiv=-7.76%
- Horizon 4:
  - MWUMConcOnlyKL: objective=2.1972 (std 0.3280), MSE=2.1972, MAE=1.0200, LINEX=6.4800, avg HHI=0.2688, excess vs best indiv=-0.9174, improvement vs best indiv=29.27%
  - RLKappaBandit: objective=2.2897 (std 0.3494), MSE=2.2897, MAE=1.0478, LINEX=9.4774, avg HHI=0.3743, excess vs best indiv=-0.8249, improvement vs best indiv=26.33%
  - MWUMBothKL: objective=2.3082 (std 0.3440), MSE=2.3082, MAE=1.0558, LINEX=6.6207, avg HHI=0.1703, excess vs best indiv=-0.8064, improvement vs best indiv=25.70%
  - OGDBoth: objective=2.3926 (std 0.3665), MSE=2.3926, MAE=1.0762, LINEX=16.7063, avg HHI=0.1417, excess vs best indiv=-0.7220, improvement vs best indiv=23.01%
  - OGDVanilla: objective=2.4173 (std 0.3711), MSE=2.4173, MAE=1.0824, LINEX=17.0255, avg HHI=0.2219, excess vs best indiv=-0.6973, improvement vs best indiv=22.22%
  - OGDConcOnly: objective=2.4679 (std 0.3828), MSE=2.4679, MAE=1.1079, LINEX=14.1221, avg HHI=0.0862, excess vs best indiv=-0.6467, improvement vs best indiv=20.63%
  - RLRuleBandit: objective=2.7731 (std 0.5051), MSE=2.7731, MAE=1.1558, LINEX=320.4956, avg HHI=0.1254, excess vs best indiv=-0.3415, improvement vs best indiv=11.03%
  - MWUMVanilla: objective=3.1176 (std 0.5071), MSE=3.1176, MAE=1.2326, LINEX=33.5192, avg HHI=0.2812, excess vs best indiv=0.0030, improvement vs best indiv=-0.08%
  - Median: objective=3.1577 (std 0.5325), MSE=3.1577, MAE=1.2443, LINEX=40.5831, avg HHI=nan, excess vs best indiv=0.0431, improvement vs best indiv=-1.28%
  - Mean: objective=3.1776 (std 0.5182), MSE=3.1776, MAE=1.2712, LINEX=34.8956, avg HHI=0.0556, excess vs best indiv=0.0630, improvement vs best indiv=-2.01%
- Horizon 8:
  - MWUMConcOnlyKL: objective=2.8529 (std 0.5523), MSE=2.8529, MAE=1.1490, LINEX=25.1822, avg HHI=0.3105, excess vs best indiv=-1.1742, improvement vs best indiv=28.84%
  - MWUMBothKL: objective=2.8867 (std 0.5372), MSE=2.8867, MAE=1.1679, LINEX=25.3095, avg HHI=0.2166, excess vs best indiv=-1.1404, improvement vs best indiv=27.92%
  - RLKappaBandit: objective=2.9957 (std 0.5704), MSE=2.9957, MAE=1.1910, LINEX=27.2552, avg HHI=0.3964, excess vs best indiv=-1.0314, improvement vs best indiv=25.22%
  - OGDBoth: objective=3.2117 (std 0.5876), MSE=3.2117, MAE=1.2438, LINEX=29.3126, avg HHI=0.1823, excess vs best indiv=-0.8154, improvement vs best indiv=19.75%
  - OGDVanilla: objective=3.2226 (std 0.5889), MSE=3.2226, MAE=1.2468, LINEX=29.0960, avg HHI=0.1873, excess vs best indiv=-0.8045, improvement vs best indiv=19.48%
  - OGDConcOnly: objective=3.3898 (std 0.6905), MSE=3.3898, MAE=1.3002, LINEX=34.4118, avg HHI=0.0970, excess vs best indiv=-0.6374, improvement vs best indiv=15.65%
  - RLRuleBandit: objective=3.6764 (std 0.7625), MSE=3.6764, MAE=1.3268, LINEX=1659.1073, avg HHI=0.2215, excess vs best indiv=-0.3508, improvement vs best indiv=8.22%
  - MWUMVanilla: objective=4.0747 (std 0.8231), MSE=4.0747, MAE=1.4049, LINEX=43.3890, avg HHI=0.9836, excess vs best indiv=0.0476, improvement vs best indiv=-1.27%
  - Median: objective=4.3920 (std 0.9453), MSE=4.3920, MAE=1.4699, LINEX=716.8541, avg HHI=nan, excess vs best indiv=0.3649, improvement vs best indiv=-8.88%
  - Mean: objective=4.5913 (std 0.9575), MSE=4.5913, MAE=1.5391, LINEX=591.6474, avg HHI=0.0556, excess vs best indiv=0.5642, improvement vs best indiv=-13.99%

## LINEX Results

### Tuned Hyperparameters (Regular Methods)
- Horizon 1:
  - Mean: params={} objective=1.5353887522221132
  - MWUMBothKL: params={'eta': 0.0010869466553146418, 'kappa': 7.51011975594033} objective=1.7568333150058975
  - MWUMConcOnlyKL: params={'eta': 0.04000975920197938, 'kappa': 7.788171024743676} objective=1.8111150145722488
  - MWUMVanilla: params={'eta': 0.001071380345127959} objective=1.942031992972344
  - Median: params={} objective=2.132740367416537
  - OGDConcOnly: params={'eta': 0.01889879259941431, 'kappa': 7.908550293895121} objective=2.1391590985568434
  - OGDVanilla: params={'eta': 0.0031709028216475085} objective=2.2390864857021597
  - OGDBoth: params={'eta': 0.0031576228955195983, 'kappa': 0.0024886637247900295} objective=2.239273620757578
- Horizon 4:
  - OGDBoth: params={'eta': 0.029469702954236822, 'kappa': 5.032305705987864} objective=5.468778949750622
  - MWUMConcOnlyKL: params={'eta': 0.17973232378405318, 'kappa': 1.5823556074993874} objective=5.5528534813989054
  - MWUMBothKL: params={'eta': 2.9499438147542123, 'kappa': 3.4245102609633378} objective=5.563164894537939
  - OGDVanilla: params={'eta': 0.05184093746478533} objective=5.796234982125387
  - MWUMVanilla: params={'eta': 1.8147265964847274} objective=8.79815432411173
  - OGDConcOnly: params={'eta': 0.07598643993749689, 'kappa': 7.805506547777622} objective=8.91469542329136
  - Median: params={} objective=14.424355637046158
  - Mean: params={} objective=14.613262449811607
- Horizon 8:
  - MWUMBothKL: params={'eta': 1.4779367164159243, 'kappa': 3.500325229967218} objective=10.707712104572659
  - OGDBoth: params={'eta': 0.005477724519907216, 'kappa': 1.4861073477971558} objective=10.983407977312808
  - OGDVanilla: params={'eta': 0.005083257982033165} objective=11.002999171221804
  - MWUMConcOnlyKL: params={'eta': 0.781849322599175, 'kappa': 0.9537720082048889} objective=11.105118435645517
  - OGDConcOnly: params={'eta': 0.033268215320327636, 'kappa': 4.84571470836092} objective=11.389613792718214
  - MWUMVanilla: params={'eta': 0.23450380966828216} objective=19.330832966091933
  - Median: params={} objective=28.902900481992447
  - Mean: params={} objective=28.984654140212907

### Summary (Lower Objective Is Better)
- Horizon 1:
  - Mean: objective=1.0611 (std 1.1035), MSE=0.9089, MAE=0.6522, LINEX=1.0611, avg HHI=0.0556, excess vs best indiv=0.2160, improvement vs best indiv=-18.20%
  - MWUMBothKL: objective=1.1118 (std 1.3606), MSE=0.9049, MAE=0.6471, LINEX=1.1118, avg HHI=0.0579, excess vs best indiv=0.2666, improvement vs best indiv=-19.17%
  - MWUMVanilla: objective=1.1357 (std 1.5164), MSE=0.9013, MAE=0.6452, LINEX=1.1357, avg HHI=0.0625, excess vs best indiv=0.2905, improvement vs best indiv=-19.67%
  - Median: objective=1.1410 (std 1.3979), MSE=0.8966, MAE=0.6378, LINEX=1.1410, avg HHI=nan, excess vs best indiv=0.2959, improvement vs best indiv=-21.90%
  - OGDConcOnly: objective=1.8193 (std 5.5105), MSE=0.9160, MAE=0.6494, LINEX=1.8193, avg HHI=0.0586, excess vs best indiv=0.9742, improvement vs best indiv=-73.87%
  - RLRuleBandit: objective=1.8321 (std 5.1086), MSE=0.9158, MAE=0.6526, LINEX=1.8321, avg HHI=0.0690, excess vs best indiv=0.9870, improvement vs best indiv=-69.39%
  - OGDVanilla: objective=1.8581 (std 5.5451), MSE=0.9669, MAE=0.6782, LINEX=1.8581, avg HHI=0.1446, excess vs best indiv=1.0130, improvement vs best indiv=-73.42%
  - OGDBoth: objective=1.8583 (std 5.5452), MSE=0.9668, MAE=0.6781, LINEX=1.8583, avg HHI=0.1442, excess vs best indiv=1.0132, improvement vs best indiv=-73.43%
  - MWUMConcOnlyKL: objective=1.8916 (std 5.5394), MSE=0.9135, MAE=0.6482, LINEX=1.8916, avg HHI=0.0570, excess vs best indiv=1.0465, improvement vs best indiv=-73.61%
  - RLKappaBandit: objective=2.1174 (std 5.5683), MSE=0.9888, MAE=0.6623, LINEX=2.1174, avg HHI=0.1973, excess vs best indiv=1.2722, improvement vs best indiv=-100.81%
- Horizon 4:
  - MWUMConcOnlyKL: objective=5.7364 (std 8.4050), MSE=2.5839, MAE=1.1352, LINEX=5.7364, avg HHI=0.0961, excess vs best indiv=-1.3511, improvement vs best indiv=28.19%
  - MWUMBothKL: objective=5.7969 (std 8.3994), MSE=2.6534, MAE=1.1490, LINEX=5.7969, avg HHI=0.0957, excess vs best indiv=-1.2905, improvement vs best indiv=27.10%
  - OGDBoth: objective=5.8188 (std 8.8225), MSE=2.6214, MAE=1.1297, LINEX=5.8188, avg HHI=0.1172, excess vs best indiv=-1.2687, improvement vs best indiv=28.24%
  - RLKappaBandit: objective=5.9263 (std 8.9860), MSE=2.3497, MAE=1.0642, LINEX=5.9263, avg HHI=0.3013, excess vs best indiv=-1.1611, improvement vs best indiv=26.68%
  - OGDConcOnly: objective=6.1225 (std 8.5646), MSE=2.7555, MAE=1.1843, LINEX=6.1225, avg HHI=0.0691, excess vs best indiv=-0.9650, improvement vs best indiv=21.82%
  - OGDVanilla: objective=6.3043 (std 11.4774), MSE=2.6543, MAE=1.1282, LINEX=6.3043, avg HHI=0.2704, excess vs best indiv=-0.7831, improvement vs best indiv=27.51%
  - MWUMVanilla: objective=9.7707 (std 12.0283), MSE=3.4915, MAE=1.3338, LINEX=9.7707, avg HHI=0.9739, excess vs best indiv=2.6832, improvement vs best indiv=-26.31%
  - RLRuleBandit: objective=32.2545 (std 131.1770), MSE=3.0243, MAE=1.2245, LINEX=32.2545, avg HHI=0.2101, excess vs best indiv=25.1671, improvement vs best indiv=-235.76%
  - Mean: objective=34.8956 (std 135.3049), MSE=3.1776, MAE=1.2712, LINEX=34.8956, avg HHI=0.0556, excess vs best indiv=27.8081, improvement vs best indiv=-270.34%
  - Median: objective=40.5831 (std 170.1704), MSE=3.1577, MAE=1.2443, LINEX=40.5831, avg HHI=nan, excess vs best indiv=33.4957, improvement vs best indiv=-330.44%
- Horizon 8:
  - MWUMBothKL: objective=23.3404 (std 65.5197), MSE=3.5647, MAE=1.3268, LINEX=23.3404, avg HHI=0.1143, excess vs best indiv=-16.9239, improvement vs best indiv=33.49%
  - MWUMConcOnlyKL: objective=23.9079 (std 66.7330), MSE=3.2566, MAE=1.2634, LINEX=23.9079, avg HHI=0.1316, excess vs best indiv=-16.3565, improvement vs best indiv=32.02%
  - RLKappaBandit: objective=23.9619 (std 65.9761), MSE=3.1731, MAE=1.2332, LINEX=23.9619, avg HHI=0.3048, excess vs best indiv=-16.3025, improvement vs best indiv=28.75%
  - OGDBoth: objective=24.5889 (std 66.2485), MSE=4.0957, MAE=1.4036, LINEX=24.5889, avg HHI=0.1862, excess vs best indiv=-15.6754, improvement vs best indiv=28.54%
  - OGDVanilla: objective=24.7296 (std 66.2373), MSE=4.1912, MAE=1.4201, LINEX=24.7296, avg HHI=0.2516, excess vs best indiv=-15.5348, improvement vs best indiv=27.44%
  - OGDConcOnly: objective=24.7771 (std 70.0978), MSE=3.6445, MAE=1.3574, LINEX=24.7771, avg HHI=0.0862, excess vs best indiv=-15.4873, improvement vs best indiv=29.18%
  - MWUMVanilla: objective=34.8754 (std 79.1054), MSE=4.7205, MAE=1.5342, LINEX=34.8754, avg HHI=0.9353, excess vs best indiv=-5.3890, improvement vs best indiv=-15.42%
  - Mean: objective=591.6474 (std 2324.5758), MSE=4.5913, MAE=1.5391, LINEX=591.6474, avg HHI=0.0556, excess vs best indiv=551.3830, improvement vs best indiv=-1902.61%
  - Median: objective=716.8541 (std 3836.2729), MSE=4.3920, MAE=1.4699, LINEX=716.8541, avg HHI=nan, excess vs best indiv=676.5898, improvement vs best indiv=-606.51%
  - RLRuleBandit: objective=589403028854827.1250 (std 3680820735443012.5000), MSE=4.1828, MAE=1.4325, LINEX=589403028854827.1250, avg HHI=0.1993, excess vs best indiv=589403028854786.8750, improvement vs best indiv=-4321152503477085.50%

