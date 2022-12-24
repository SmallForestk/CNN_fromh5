# CNN_fromh5

## regresion
### Datasetの作成
make_3Dimage_h5.shとmake_3Dhitmap_h5.pyが
```
/home/kobayashik/geant4/pi_k_experiment/AHCAL/regression/energy1_30GeV/CNN_3cm/
```
においてあるので、make_3Dimage_h5.shを実行してDatasetを作成する。

### train
ConvNN_3D_muh5.pyとConv3D_mutrain.shが
```
/home/kobayashik/geant4/pi_k_experiment/AHCAL/regression/energy1_30GeV/CNN_MAPE
```
においてあるのでConv3D_mutrain.shを実行するとtrainが始まる

### 解析
解析はtrainのときと同じpathの中にあるanalysis_3D_3lrmu.ipynbとanalysis_3Dmu.pyを実行して行う。
解析はtrainを行ったあととtestとretestを行ったあとの2回に分けて行う。
### 1.train後の解析
trainをしたあとはtrainの最中のLossを確認する。
Lossが最小のときのepochが"In[5]:"の中のvloss.argmin()で表示されるのでその値を参照してtestとretestで読み込むmodelを決める。
### 2.testとretest後の解析
testとretestのあとに
```
/home/kobayashik/geant4/pi_k_experiment/AHCAL/regression/energy1_30GeV/CNN_MAPE
```
においてあるanalysis_3D.shを実行すると
```
/home/kobayashik/geant4/pi_k_experiment/AHCAL/regression/energy1_30GeV/
```
の中にあるanalysis_3Dmu.pyを実行して解析結果をまとめたrootファイルがanalysis_3D.shと同じディレクトリの中に作成されるので、analysis_3D_3lrmu.ipynbの中の残りを実行するとrootファイルの中のヒストグラムやグラフを見ることができる。

### retest
ConvNN_3D_h5retest.pyとConv3D_retest.shがtrainのときと同じpathの中においてあるので、Conv3D_retest.shを実行してEnergyがランダムなDatasetでのtestを行う。
読み込むmodelはtrainの最中のLossの値を見ながらConv3D_retest.shのなかで指定する。

### test
ConvNN_3D_h5testmu.pyとConv3D_mutest.shがtrainのときと同じpathの中においてあるので、Conv3D_mutest.shを実行してEnergyがランダムなDatasetでのtestを行う。
読み込むmodelはtrainの最中のLossの値を見ながらConv3D_mutest.shのなかで指定する。

##　classification
classificationで使用するファイルはすべて
```
/home/kobayashik/geant4/pi_k_experiment/AHCAL/classification/stable_energy/CNN
```
において使用する

### Datasetの作成
make_3Dimage_h5.shを時以降するとmake_3Dhitmap_h5.pyが実行され、Datasetが作成できる。

###train
trainはConvNN_3Dh5.shを実行するとConvNN_3D_h5.pyが実行され、trainが実行される。

###解析
解析はtrainのあとにanalysis_3D.ipynbで行う。
