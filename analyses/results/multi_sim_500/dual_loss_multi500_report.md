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
  - MWUMBothKL: objective=0.9052 (std 0.1176), MSE=0.9052, MAE=0.6379, LINEX=3.4303, avg HHI=0.0666, excess vs best indiv=-0.0293, improvement vs best indiv=3.13%
  - MWUMVanilla: objective=0.9052 (std 0.1176), MSE=0.9052, MAE=0.6379, LINEX=3.4224, avg HHI=0.0675, excess vs best indiv=-0.0293, improvement vs best indiv=3.13%
  - OGDVanilla: objective=0.9114 (std 0.1192), MSE=0.9114, MAE=0.6437, LINEX=3.4177, avg HHI=0.0609, excess vs best indiv=-0.0231, improvement vs best indiv=2.48%
  - OGDBoth: objective=0.9116 (std 0.1192), MSE=0.9116, MAE=0.6437, LINEX=3.4211, avg HHI=0.0606, excess vs best indiv=-0.0230, improvement vs best indiv=2.46%
  - Median: objective=0.9117 (std 0.1188), MSE=0.9117, MAE=0.6389, LINEX=3.2944, avg HHI=nan, excess vs best indiv=-0.0228, improvement vs best indiv=2.44%
  - RLRuleBandit: objective=0.9166 (std 0.1203), MSE=0.9166, MAE=0.6464, LINEX=2.5833, avg HHI=0.0579, excess vs best indiv=-0.0179, improvement vs best indiv=1.92%
  - MWUMConcOnlyKL: objective=0.9217 (std 0.1232), MSE=0.9217, MAE=0.6463, LINEX=3.3913, avg HHI=0.0563, excess vs best indiv=-0.0129, improvement vs best indiv=1.41%
  - Mean: objective=0.9240 (std 0.1208), MSE=0.9240, MAE=0.6537, LINEX=2.3010, avg HHI=0.0556, excess vs best indiv=-0.0105, improvement vs best indiv=1.12%
  - OGDConcOnly: objective=0.9244 (std 0.1244), MSE=0.9244, MAE=0.6471, LINEX=3.5361, avg HHI=0.0573, excess vs best indiv=-0.0101, improvement vs best indiv=1.12%
  - RLKappaBandit: objective=1.0163 (std 0.1429), MSE=1.0163, MAE=0.6659, LINEX=5.1581, avg HHI=0.2377, excess vs best indiv=0.0818, improvement vs best indiv=-8.67%
- Horizon 4:
  - MWUMConcOnlyKL: objective=2.2424 (std 0.3654), MSE=2.2424, MAE=1.0230, LINEX=16.8736, avg HHI=0.2719, excess vs best indiv=-0.9316, improvement vs best indiv=29.09%
  - RLKappaBandit: objective=2.3449 (std 0.3782), MSE=2.3449, MAE=1.0551, LINEX=11.6818, avg HHI=0.3600, excess vs best indiv=-0.8291, improvement vs best indiv=25.81%
  - MWUMBothKL: objective=2.3543 (std 0.3722), MSE=2.3543, MAE=1.0594, LINEX=9.5910, avg HHI=0.1726, excess vs best indiv=-0.8197, improvement vs best indiv=25.58%
  - OGDBoth: objective=2.4411 (std 0.3906), MSE=2.4411, MAE=1.0809, LINEX=12.7369, avg HHI=0.1428, excess vs best indiv=-0.7330, improvement vs best indiv=22.86%
  - OGDVanilla: objective=2.4657 (std 0.3937), MSE=2.4657, MAE=1.0866, LINEX=12.6522, avg HHI=0.2214, excess vs best indiv=-0.7083, improvement vs best indiv=22.08%
  - OGDConcOnly: objective=2.5185 (std 0.4060), MSE=2.5185, MAE=1.1157, LINEX=11.9164, avg HHI=0.0868, excess vs best indiv=-0.6556, improvement vs best indiv=20.45%
  - RLRuleBandit: objective=2.8550 (std 0.4948), MSE=2.8550, MAE=1.1666, LINEX=75.0359, avg HHI=0.1309, excess vs best indiv=-0.3190, improvement vs best indiv=9.85%
  - MWUMVanilla: objective=3.1779 (std 0.5449), MSE=3.1779, MAE=1.2380, LINEX=24.8001, avg HHI=0.3166, excess vs best indiv=0.0038, improvement vs best indiv=-0.13%
  - Median: objective=3.2299 (std 0.5554), MSE=3.2299, MAE=1.2514, LINEX=42.3381, avg HHI=nan, excess vs best indiv=0.0559, improvement vs best indiv=-1.77%
  - Mean: objective=3.2547 (std 0.5504), MSE=3.2547, MAE=1.2802, LINEX=36.5615, avg HHI=0.0556, excess vs best indiv=0.0806, improvement vs best indiv=-2.61%
- Horizon 8:
  - MWUMConcOnlyKL: objective=2.9402 (std 0.6641), MSE=2.9402, MAE=1.1571, LINEX=1924887.0475, avg HHI=0.3120, excess vs best indiv=-1.2158, improvement vs best indiv=28.66%
  - MWUMBothKL: objective=2.9760 (std 0.5975), MSE=2.9760, MAE=1.1769, LINEX=28.7802, avg HHI=0.2190, excess vs best indiv=-1.1800, improvement vs best indiv=27.83%
  - OGDBoth: objective=3.3143 (std 0.6543), MSE=3.3143, MAE=1.2549, LINEX=9026497161178884.0000, avg HHI=0.1833, excess vs best indiv=-0.8417, improvement vs best indiv=19.66%
  - OGDVanilla: objective=3.3244 (std 0.6526), MSE=3.3244, MAE=1.2578, LINEX=7283189208974045.0000, avg HHI=0.1883, excess vs best indiv=-0.8316, improvement vs best indiv=19.42%
  - RLKappaBandit: objective=3.3680 (std 6.7470), MSE=3.3680, MAE=1.2013, LINEX=276797.2073, avg HHI=0.3692, excess vs best indiv=-0.7879, improvement vs best indiv=16.59%
  - OGDConcOnly: objective=3.5061 (std 0.7811), MSE=3.5061, MAE=1.3114, LINEX=82665168158063232.0000, avg HHI=0.0983, excess vs best indiv=-0.6499, improvement vs best indiv=15.29%
  - RLRuleBandit: objective=3.7914 (std 0.8762), MSE=3.7914, MAE=1.3371, LINEX=201194415.4509, avg HHI=0.2274, excess vs best indiv=-0.3646, improvement vs best indiv=8.47%
  - MWUMVanilla: objective=4.2145 (std 0.9088), MSE=4.2145, MAE=1.4189, LINEX=82.2674, avg HHI=0.9849, excess vs best indiv=0.0586, improvement vs best indiv=-1.45%
  - Median: objective=4.5270 (std 0.9910), MSE=4.5270, MAE=1.4855, LINEX=141.0028, avg HHI=nan, excess vs best indiv=0.3710, improvement vs best indiv=-8.94%
  - Mean: objective=4.7350 (std 1.0090), MSE=4.7350, MAE=1.5550, LINEX=211.2749, avg HHI=0.0556, excess vs best indiv=0.5790, improvement vs best indiv=-14.11%

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
  - Mean: objective=2.3010 (std 13.2588), MSE=0.9240, MAE=0.6537, LINEX=2.3010, avg HHI=0.0556, excess vs best indiv=1.1814, improvement vs best indiv=-36.06%
  - RLRuleBandit: objective=2.5917 (std 13.8567), MSE=0.9376, MAE=0.6567, LINEX=2.5917, avg HHI=0.0732, excess vs best indiv=1.4721, improvement vs best indiv=-47.58%
  - MWUMBothKL: objective=2.5945 (std 15.6466), MSE=0.9302, MAE=0.6515, LINEX=2.5945, avg HHI=0.0625, excess vs best indiv=1.4749, improvement vs best indiv=-41.57%
  - MWUMConcOnlyKL: objective=2.7022 (std 16.1338), MSE=0.9336, MAE=0.6503, LINEX=2.7022, avg HHI=0.0574, excess vs best indiv=1.5826, improvement vs best indiv=-50.71%
  - MWUMVanilla: objective=2.7966 (std 18.6805), MSE=0.9333, MAE=0.6515, LINEX=2.7966, avg HHI=0.0733, excess vs best indiv=1.6770, improvement vs best indiv=-45.96%
  - OGDConcOnly: objective=2.8777 (std 19.0573), MSE=0.9362, MAE=0.6518, LINEX=2.8777, avg HHI=0.0589, excess vs best indiv=1.7581, improvement vs best indiv=-55.72%
  - Median: objective=3.2944 (std 28.2636), MSE=0.9117, MAE=0.6389, LINEX=3.2944, avg HHI=nan, excess vs best indiv=2.1748, improvement vs best indiv=-59.04%
  - OGDVanilla: objective=3.3274 (std 27.1569), MSE=0.9875, MAE=0.6803, LINEX=3.3274, avg HHI=0.1567, excess vs best indiv=2.2078, improvement vs best indiv=-60.30%
  - OGDBoth: objective=3.3280 (std 27.1740), MSE=0.9874, MAE=0.6802, LINEX=3.3280, avg HHI=0.1563, excess vs best indiv=2.2084, improvement vs best indiv=-60.31%
  - RLKappaBandit: objective=4.9440 (std 51.2427), MSE=1.0060, MAE=0.6633, LINEX=4.9440, avg HHI=0.1966, excess vs best indiv=3.8244, improvement vs best indiv=-120.66%
- Horizon 4:
  - MWUMConcOnlyKL: objective=7.3707 (std 16.7299), MSE=2.6396, MAE=1.1422, LINEX=7.3707, avg HHI=0.0979, excess vs best indiv=-1.9248, improvement vs best indiv=29.51%
  - MWUMBothKL: objective=7.4108 (std 16.7048), MSE=2.7135, MAE=1.1559, LINEX=7.4108, avg HHI=0.0976, excess vs best indiv=-1.8847, improvement vs best indiv=28.84%
  - OGDConcOnly: objective=7.7086 (std 16.7350), MSE=2.8170, MAE=1.1926, LINEX=7.7086, avg HHI=0.0701, excess vs best indiv=-1.5870, improvement vs best indiv=23.88%
  - RLKappaBandit: objective=8.0862 (std 17.6280), MSE=2.4262, MAE=1.0760, LINEX=8.0862, avg HHI=0.3030, excess vs best indiv=-1.2093, improvement vs best indiv=21.83%
  - OGDBoth: objective=8.8970 (std 35.5093), MSE=2.6847, MAE=1.1368, LINEX=8.8970, avg HHI=0.1197, excess vs best indiv=-0.3985, improvement vs best indiv=24.15%
  - OGDVanilla: objective=10.8324 (std 59.7851), MSE=2.7245, MAE=1.1355, LINEX=10.8324, avg HHI=0.2743, excess vs best indiv=1.5369, improvement vs best indiv=8.93%
  - MWUMVanilla: objective=11.9801 (std 27.5449), MSE=3.5597, MAE=1.3384, LINEX=11.9801, avg HHI=0.9759, excess vs best indiv=2.6845, improvement vs best indiv=-33.23%
  - Mean: objective=36.5615 (std 225.8232), MSE=3.2547, MAE=1.2802, LINEX=36.5615, avg HHI=0.0556, excess vs best indiv=27.2659, improvement vs best indiv=-114.32%
  - Median: objective=42.3381 (std 282.6717), MSE=3.2299, MAE=1.2514, LINEX=42.3381, avg HHI=nan, excess vs best indiv=33.0425, improvement vs best indiv=-139.21%
  - RLRuleBandit: objective=54.2151 (std 303.7108), MSE=3.1148, MAE=1.2355, LINEX=54.2151, avg HHI=0.2214, excess vs best indiv=44.9195, improvement vs best indiv=-229.14%
- Horizon 8:
  - OGDVanilla: objective=21.0478 (std 58.5505), MSE=4.3420, MAE=1.4432, LINEX=21.0478, avg HHI=0.2550, excess vs best indiv=-3.1277, improvement vs best indiv=0.82%
  - MWUMBothKL: objective=21.0674 (std 56.0419), MSE=3.6341, MAE=1.3353, LINEX=21.0674, avg HHI=0.1174, excess vs best indiv=-3.1081, improvement vs best indiv=14.13%
  - OGDBoth: objective=21.4869 (std 65.9542), MSE=4.2349, MAE=1.4239, LINEX=21.4869, avg HHI=0.1920, excess vs best indiv=-2.6887, improvement vs best indiv=16.39%
  - MWUMConcOnlyKL: objective=24.0113 (std 77.9237), MSE=3.3345, MAE=1.2714, LINEX=24.0113, avg HHI=0.1341, excess vs best indiv=-0.1642, improvement vs best indiv=6.25%
  - MWUMVanilla: objective=28.3211 (std 58.3657), MSE=4.7648, MAE=1.5234, LINEX=28.3211, avg HHI=0.9420, excess vs best indiv=4.1456, improvement vs best indiv=-24.34%
  - Median: objective=141.0028 (std 1186.1340), MSE=4.5270, MAE=1.4855, LINEX=141.0028, avg HHI=nan, excess vs best indiv=116.8273, improvement vs best indiv=-227.68%
  - Mean: objective=211.2749 (std 1529.6108), MSE=4.7350, MAE=1.5550, LINEX=211.2749, avg HHI=0.0556, excess vs best indiv=187.0994, improvement vs best indiv=-594.98%
  - RLKappaBandit: objective=43968.8870 (std 743201.4291), MSE=3.2310, MAE=1.2294, LINEX=43968.8870, avg HHI=0.3257, excess vs best indiv=43944.7115, improvement vs best indiv=-399786.76%
  - OGDConcOnly: objective=99584.0492 (std 2221239.2170), MSE=3.7285, MAE=1.3670, LINEX=99584.0492, avg HHI=0.0877, excess vs best indiv=99559.8737, improvement vs best indiv=-332252.40%
  - RLRuleBandit: objective=47152242308640.3281 (std 1053301307032563.5000), MSE=4.3745, MAE=1.4611, LINEX=47152242308640.3281, avg HHI=0.2241, excess vs best indiv=47152242308616.1484, improvement vs best indiv=-345692200279341.75%

