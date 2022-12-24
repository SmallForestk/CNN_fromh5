# CNN_fromh5

## regresion
### Datasetの作成
make_3Dimage_h5.shとmake_3Dhitmap_h5.pyが
```
path=/home/kobayashik/geant4/pi_k_experiment/AHCAL/regression/energy1_30GeV/CNN_3cm/
```
においてあるので、make_3Dimage_h5.shを実行してDatasetを作成する。

### train
ConvNN_3D_muh5.pyとConv3D_mutrain.shが
```
path=/home/kobayashik/geant4/pi_k_experiment/AHCAL/regression/energy1_30GeV/CNN_MAPE
```
においてあるのでConv3D_mutrain.shを実行するとtrainが始まる

### retest
ConvNN_3D_h5retest.pyとConv3D_retest.shがtrainのときと同じpathの中においてあるので、Conv3D_retest.shを実行してEnergyがランダムなDatasetでのtestを行う。
読み込むmodelはtrainの最中のLossの値を見ながらConv3D_retest.shのなかで
