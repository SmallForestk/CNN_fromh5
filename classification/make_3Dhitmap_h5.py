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

Line_module.notify_to_line("make hitmap in " + exe_file + " particle:" + particle)
print("make hitmap in " + exe_file + " particle:" + particle)

# read root file and get ttree
rf = r.TFile("./" + exe_file + "/" + particle + ".root")
tree = rf.Get("Edep")

# get ttree profile
nEntry = tree.GetEntries()
nofEvent = tree.GetMaximum("Enumber")+1
print("nofEvent=", nofEvent, ", Entry=", nEntry)

# list for keep 1Event info
layer = []
xnumber = []
ynumber = []
edep = []

# before event number(default=0)
before_event = 0

with h5py.File("./" + exe_file + "/hitmap3D/hitmap.h5", "a") as f:
    for group in f:
        if group==particle:
            del f[particle]
    f.create_group(particle)
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
            hitmap = Cnn_tool.make_image.hitmap3DbyEvent(arr, xmax, xmin, ymax, ymin, zmax, zmin)
            f[particle].create_dataset(f"{before_event}", dtype=np.float32, data=hitmap)
            layer = []
            xnumber = []
            ynumber = []
            edep = []
        layer.append(tree.Lnumber)
        xnumber.append(tree.TXnumber)
        ynumber.append(tree.TYnumber)
        edep.append(tree.Edep)
        before_event = tree.Enumber
    arr = np.stack([
        np.ones(len(layer)),
        np.array(layer),
        np.array(xnumber),
        np.array(ynumber),
        np.array(edep)
    ])
    hitmap = Cnn_tool.make_image.hitmap3DbyEvent(arr, xmax, xmin, ymax, ymin, zmax, zmin)
    f[particle].create_dataset(f"{before_event}", dtype=np.float32, data=hitmap)

Line_module.notify_to_line("finish in " + exe_file + " particle:" + particle)
print("finish in " + exe_file + " particle:" + particle)
