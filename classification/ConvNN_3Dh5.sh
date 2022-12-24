for i in `seq 1 5`
do
    energy=`expr $i \* 2`
    energy=`expr ${energy} + 10`
    echo "execute in Cnn_${energy}GeV"
    cp "./data_${energy}GeV/hitmap3D/hitmap.h5" /mnt/scratch/kobayashik/
    python ConvNN_3D_h5ver2.py "data_${energy}GeV" 3 10 pi kaon
    rm /mnt/scratch/kobayashik/hitmap.h5
    python ../messege.py "finish_ConvNN3D_in_data_${energy}GeV"
done
