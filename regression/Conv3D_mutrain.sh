python ../messege.py "start_training"
cp ../CNN_3cm/train_dataset/hitmap3D/hitmap.h5 /mnt/scratch/kobayashik

python ../messege.py start Learning Rate=10^-3
echo "execute Learning Rate=10^-3"
python ConvNN_3D_muh5.py train_dataset 3 200 pi 94720
python ../messege.py "end Learning Rate=10^-3"

python ../messege.py start Learning Rate=10^-4
echo "execute Learning Rate=10^-4"
python ConvNN_3D_muh5.py train_dataset 4 400 pi 94720
python ../messege.py "end Learning Rate=10^-4"

rm /mnt/scratch/kobayashik/hitmap.h5
