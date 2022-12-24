for i in `seq 1 5`
do
    energy=`expr $i \* 2`
    energy=`expr ${energy} + 10`
    echo "execute in Cnn_${energy}GeV"
    python make_3Dhitmap_h5_ver2.py "data_${energy}GeV" kaon 30 0 30 0 48 0 
    python make_3Dhitmap_h5_ver2.py "data_${energy}GeV" pi 30 0 30 0 48 0 
done
