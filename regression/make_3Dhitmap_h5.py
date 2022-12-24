import ROOT as r
import numpy as np
import h5py
from tqdm import tqdm
import sys
sys.path.append("../")

import Cnn_tool
import Line_module

exe_file = sys.argv[1]
particle = sys.argv[2]
# decide hitmap size
xmax = int(sys.argv[3])
xmin = int(sys.argv[4])
ymax = int(sys.argv[5])
ymin = int(sys.argv[6])
zmax = int(sys.argv[7])
zmin = int(sys.argv[8])
nofx = int(sys.argv[9])
nofy = int(sys.argv[10])


Line_module.notify_to_line("make hitmap in " + exe_file + " particle:" + particle)
print("make hitmap in " + exe_file + " particle:" + particle)

# read root file and get ttree
rf = r.TFile("./" + exe_file + "/" + particle + ".root")
tree = rf.Get("Edep")

# read root file for energy label
nd_energy = r.RDataFrame("Event_Condition", "./" + exe_file + "/" + particle +".root").Filter("PNumber==-1 || PNumber==1").AsNumpy()
energy = nd_energy["InEnergy"]
print("Energy label shape", energy.shape)

# get ttree profile
nEntry = tree.GetEntries()
nofEvent = int(tree.GetMaximum("Enumber")+1)
print("nofEvent=", nofEvent, ", Entry=", nEntry)

# list for keep 1Event info
layer = []
xnumber = []
ynumber = []
edep = []

# before event number(default=0)
before_event = 0

# save create status
create_list = np.ones(nofEvent)

with h5py.File("./" + exe_file + "/hitmap3D/hitmap.h5", "a") as f:
    for group in f:
        print(group)
        if group==particle:
            del f[particle]
        if group==(particle+"_energy"):
            del f[particle+"_energy"]
    f.create_group(particle)
    f.create_group(particle+"_energy")
    f[particle].create_dataset("nofEvent", dtype=np.float32, data=nofEvent)
    for i in tqdm(range(nEntry)):
        tree.GetEntry(i)
        if before_event!=tree.Enumber:
            arr = np.stack([
                np.ones(len(layer)),
                np.array(layer),
                np.array(xnumber),
                np.array(ynumber),
                np.array(edep)
            ])
            hitmap = Cnn_tool.make_image.hitmap3DbyEvent(arr, xmax, xmin, ymax, ymin, zmax, zmin, nofx, nofy)
            f[particle].create_dataset(f"{before_event}", dtype=np.float32, data=hitmap)
            f[particle + "_energy"].create_dataset(f"{before_event}", dtype=np.float32, data=energy[int(before_event)])
            create_list[before_event]=0
            layer = []
            xnumber = []
            ynumber = []
            edep = []
        layer.append(tree.Lnumber)
        xnumber.append(tree.TXnumber)
        ynumber.append(tree.TYnumber)
        edep.append(tree.Edep)
        before_event = int(tree.Enumber)
    arr = np.stack([
        np.ones(len(layer)),
        np.array(layer),
        np.array(xnumber),
        np.array(ynumber),
        np.array(edep)
    ])
    hitmap = Cnn_tool.make_image.hitmap3DbyEvent(arr, xmax, xmin, ymax, ymin, zmax, zmin, nofx, nofy)
    f[particle].create_dataset(f"{before_event}", dtype=np.float32, data=hitmap)
    f[particle + "_energy"].create_dataset(f"{before_event}", dtype=np.float32, data=energy[int(before_event)])
    create_list[before_event]=0
    if create_list.sum()!=0:
        print(f"create empty hitmap")
        empty_list = np.arange(nofEvent)
        empty_list = empty_list[create_list==1]
        print("create list", empty_list)
        hitmap = np.zeros([nofx, nofy, int(zmax-zmin)])
        for i in range(len(empty_list)):
            print(f"{empty_list[i]} create")
            f[particle].create_dataset(f"{empty_list[i]}", dtype=np.float32, data=hitmap)
            f[particle + "_energy"].create_dataset(f"{empty_list[i]}", dtype=np.float32, data=-energy[int(empty_list[i])])

Line_module.notify_to_line("finish in " + exe_file + " particle:" + particle)
print("finish in " + exe_file + " particle:" + particle)
