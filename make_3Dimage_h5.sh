python make_3Dhitmap_h5.py train_dataset kaon 90 0 90 0 48 0 30 30
python make_3Dhitmap_h5.py train_dataset pi 90 0 90 0 48 0 30 30
for i in `seq 1 5`
do
    energy=`expr $i \* 2`
    energy=`expr ${energy} + 10`
    echo "execute in Cnn_${energy}GeV"
    python make_3Dhitmap_h5.py "test_dataset/data_${energy}GeV" kaon 90 0 90 0 48 0 30 30
    python make_3Dhitmap_h5.py "test_dataset/data_${energy}GeV" pi 90 0 90 0 48 0 30 30
done
