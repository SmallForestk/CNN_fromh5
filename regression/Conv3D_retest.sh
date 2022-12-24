python ../messege.py "start_retest"
cp ../CNN_3cm/train_dataset/hitmap3D/hitmap.h5 /mnt/scratch/kobayashik
# python ConvNN_3D_h5retest.py 3 65 pi
python ConvNN_3D_h5retest.py 4 399 pi
# python ConvNN_3D_h5retest.py 5 167 pi
rm /mnt/scratch/kobayashik/hitmap.h5
python ../messege.py "finish_retest"
