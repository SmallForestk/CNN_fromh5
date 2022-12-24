import ROOT as r
import numpy as np
import sys

sys.path.append("../")
import Reg_tool
import Cnn_toolr

read_lr = int(sys.argv[1])
read_epoch = int(sys.argv[2])
save_file = sys.argv[3]
dimension = int(sys.argv[4])

y_label = np.load(f"./train_dataset/Conv{dimension}D_result/y_label{read_lr}_filmu.npy")
y_output = np.load(f"./train_dataset/Conv{dimension}D_result/y_reoutput{read_epoch}epoch_{read_lr}lr_pi_filmu.npy")

# regression-label matrix
h2_labelreg = r.TH2D("h2_labelreg", "Regression Energy to Truth Energy", 60, 0, 32000, 60, 0, 32000)
h2_labelreg.FillN(len(y_label), y_label.astype(np.double), y_output.astype(np.double), np.ones(len(y_label)).astype(np.double))
h2_labelreg.GetXaxis().SetTitle("True Energy(MeV)")
h2_labelreg.GetYaxis().SetTitle("Regression Energy(MeV)")

h2_labelreg_ratio = Reg_tool.label_output_per(y_label, y_output, 10, 1000, 30000, 10, 1000, 30000, "h2_labelreg_ratio", "Regression Energy to Truth Energy")
h2_labelreg_ratio.SetAxisRange(0, 1, "Z")
h2_labelreg_ratio.GetZaxis().SetMaxDigits(2)
h2_labelreg_ratio.SetMarkerSize(1.8)

h2_shift = Reg_tool.energy_error(y_label, y_output, 29, 1000, 30000, 20, 300, h2label="h2_shift", h2name="Energy Error2")

# stable energy dataset regression
RegE_list = Reg_tool.hist_fit_list(0, 35000, "gaus", 2)
for i in range(15):
    energy = (i+1)*2
    # regression_energy = np.load(f"./test_dataset/data_{energy}GeV/Conv{dimension}D_result/y_output{read_epoch}epoch_{read_lr}lr_filmupi.npy")
    regression_energy =np.concatenate([
        np.load(f"./test_dataset/data_{energy}GeV/Conv{dimension}D_result/y_output{read_epoch}epoch_{read_lr}lr_filmukaon.npy"),
        np.load(f"./test_dataset/data_{energy}GeV/Conv{dimension}D_result/y_output{read_epoch}epoch_{read_lr}lr_filmupi.npy")
    ])
    RegE_list.add_ndarray(
        regression_energy,
        "regression energy",
        f"Regression Energy in {energy}GeV",
        f"f1_energy{i}")
h1_list = RegE_list.get_h1()
f1_list = RegE_list.get_f1()
parameter = RegE_list.get_fit_par()

# regression energy resolution
resolution = parameter[2]/parameter[0]
resolutionerror = np.sqrt(
    np.power(resolution*parameter[1]/parameter[0], 2) + np.power(resolution*parameter[3]/parameter[2], 2))
g1_resolution = r.TGraphErrors(
    len(resolution),
    ((np.arange(15)+1)*2).astype(np.float32),
    resolution.astype(np.float32),
    np.zeros(len(resolution)).astype(np.float32),
    resolutionerror.astype(np.float32))
g1_resolution.SetMarkerStyle(20)
g1_resolution.SetMarkerSize(1)
g1_resolution.SetTitle("Energy Resolution(Regression Energy)")
g1_resolution.GetYaxis().SetRangeUser(0, 0.5)
# g1_resolution.GetXaxis().SetRangeUser(0, 30)
g1_resolution.GetXaxis().SetTitle("Energy(GeV)")
g1_resolution.GetYaxis().SetTitle("#sigma Energy/Energy Mean")

# plot label-gausmean(regression)
g1_mean = r.TGraphErrors(
    len(parameter[0]),
    ((np.arange(15)+1)*2).astype(np.float32),
    (parameter[0]/1000).astype(np.float32),
    np.zeros(15).astype(np.float32),
    (parameter[1]/1000).astype(np.float32))
g1_mean.SetMarkerStyle(20)
g1_mean.SetMarkerSize(1)
g1_mean.SetTitle("Regression Energy Mean")
g1_mean.GetXaxis().SetTitle("Dataset Energy(GeV)")
g1_mean.GetYaxis().SetTitle("Mean Energy(GeV)")
g1_mean.GetXaxis().SetRangeUser(0, 32)
g1_mean.GetYaxis().SetRangeUser(0, 32)

# plot regression energy shift
energy_list = (np.arange(15)+1)*2
error = energy_list-parameter[0]/1000
g1_meanshift = r.TGraphErrors(
    len(error),
    energy_list.astype(np.float32),
    error.astype(np.float32),
    np.zeros(len(error)).astype(np.float32),
    (parameter[1]/1000).astype(np.float32))
g1_meanshift.SetMarkerStyle(20)
g1_meanshift.SetMarkerSize(1)
g1_meanshift.SetTitle("Energy Shift(=Label - Regression)")
g1_meanshift.GetXaxis().SetTitle("Energy(GeV)")
g1_meanshift.GetYaxis().SetTitle("Energy Shift(GeV)")

# save to root file
file = r.TFile(save_file, "recreate")
for i in range(15):
    h1_list[i].Write(f"h1_regenergy{i}")
    f1_list[i].Write(f"f1_regenergy{i}")
g1_resolution.Write("g1_resolution")
g1_meanshift.Write("g1_meanshift")
g1_mean.Write("g1_mean")
h2_shift.Write("h2_shift")
h2_labelreg.Write("h2_labelreg")
h2_labelreg_ratio.Write("h2_labelreg_ratio")
file.Close()