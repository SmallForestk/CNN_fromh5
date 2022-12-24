# CNN_fromh5

## make_3Dhitmap.py
rootファイルは
```
path=/home/kobayashik/geant4/pi_k_experiment/AHCAL/regression/energy1_30GeV/CNN_3cm/

```
の中にtrain_datasetとtest_datasetというディレクトリがあり、train用のrootファイルはtrain_datasetの中にあり、test用のrootファイルはtest_datasetの中にenergyごとのディレクトリがあり、その中においてある。
基本的には"/CNN_3cm"にmake_3Dimage_h5.shとmake_3Dhitmap_h5.pyをおいておき、make_3Dimage_h5.shを実行してDatasetを作成する。
