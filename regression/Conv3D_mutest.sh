python ../messege.py "start_test"
# for i in `seq 1 15`
# do
#     energy=`expr $i \* 2`
#     echo ${energy}
#     cp "../CNN_3cm/test_dataset/data_${energy}GeV/hitmap3D/hitmap.h5" /mnt/scratch/kobayashik
# #     python ../messege.py "start test Energy=${energy}"
#     echo "execute test Energy=${energy}"
#     python ConvNN_3D_h5testmu.py "test_dataset/data_${energy}GeV" 3 65 pi "${energy}" filter_mu
#     python ConvNN_3D_h5testmu.py "test_dataset/data_${energy}GeV" 3 65 kaon "${energy}" filter_mu
#     rm /mnt/scratch/kobayashik/hitmap.h5
# done
for i in `seq 1 15`
do
    energy=`expr $i \* 2`
    echo ${energy}
    cp "../CNN_3cm/test_dataset/data_${energy}GeV/hitmap3D/hitmap.h5" /mnt/scratch/kobayashik
#     python ../messege.py "start test Energy=${energy}"
    echo "execute test Energy=${energy}"
    python ConvNN_3D_h5testmu.py "test_dataset/data_${energy}GeV" 4 399 pi "${energy}" filter_mu
    python ConvNN_3D_h5testmu.py "test_dataset/data_${energy}GeV" 4 399 kaon "${energy}" filter_mu
    rm /mnt/scratch/kobayashik/hitmap.h5
done
python ../messege.py "finish_test"
