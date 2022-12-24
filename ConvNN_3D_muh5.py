#!/usr/bin/env python
# coding: utf-8

import sys
if len(sys.argv)==1:
    print("Error:file name not found")
    sys.exit()
exe_file = sys.argv[1]
if len(sys.argv)==2:
    LR = 3
else:
    LR = int(sys.argv[2])
if len(sys.argv)==3:
    num_epoch = 15
else:
    num_epoch = int(sys.argv[3])

particle = sys.argv[4]
use_nofevent = int(sys.argv[5])

import numpy as np
from sklearn.model_selection import train_test_split
import h5py
import torch
sys.path.append("../")
import Line_module

Line_module.notify_to_line("start in " + exe_file + f"learning rate={LR}, {num_epoch}epoch filter mu")

# Get number of Event
h5py_path = "/mnt/scratch/kobayashik/hitmap.h5"
with h5py.File(h5py_path) as f:
    nofevent = f[particle]["nofEvent"][()]
    filter_list = f[particle]["filter_mu"][::]

# create event number list
event_number = np.arange(nofevent)
event_number = event_number[filter_list==0]
np.random.shuffle(event_number)
event_number = event_number[:use_nofevent]
tv_t_seed = np.random.randint(1, 10000)
en_train_valid, en_test = train_test_split(event_number, train_size=0.8, random_state=tv_t_seed)

t_v_seed = np.random.randint(1, 10000)
en_train, en_valid = train_test_split(en_train_valid, train_size=0.75, random_state=t_v_seed)
print("tv_t split seed=", tv_t_seed, " t_v split seed=", t_v_seed)
print(en_train.shape, en_valid.shape, en_test.shape)

# import
from torch.utils.data import DataLoader, Dataset
from torch import tensor, float32


# definition dataset
class HDF5dataset(Dataset):
    def __init__(self, path, enlist, particle):
        self.path = path
        self.fh5 = h5py.File(self.path, "r")
        self.enlist = enlist
        
        self.hitmap = self.fh5[particle]
        self.label = self.fh5[particle + "_energy"]

    def __getitem__(self, idx):
        event_number = int(self.enlist[int(idx)])
        x = self.hitmap[str(event_number)][::]
        y = self.label[str(event_number)][()]
        if y<0:
            y = 0
        return tensor(x[np.newaxis, :, :, :], dtype=float32), tensor(y, dtype=float32)
    
    def __len__(self):
        return len(self.enlist)
            
train_c1 = HDF5dataset(h5py_path, en_train, particle)
valid_c1 = HDF5dataset(h5py_path, en_valid, particle)
test_c1 = HDF5dataset(h5py_path, en_test, particle)


#DataLoader
batch_size = 512
train_c1_dataloader = DataLoader(train_c1, batch_size=batch_size, shuffle=True)
valid_c1_dataloader = DataLoader(valid_c1, batch_size=batch_size, shuffle=True)
test_c1_dataloader = DataLoader(test_c1, batch_size=batch_size, shuffle=False)


# modelの定義

import torch.nn as nn
from torch.nn import Sequential, Flatten, Conv3d, MaxPool3d, Linear, ReLU

class ConvNet(nn.Module):
    def __init__(self, input_shape):
        super().__init__()
        self.conv_relu_stack = Sequential(
            Conv3d(input_shape, 64, 3), # output_channels=64, karnel_size=3
            Conv3d(64, 64, 3), # input_channels=64, output_channels=64, karnel_size=3
            MaxPool3d(2, 2), # karnel_size=2, stride=2
            ReLU(),
            Conv3d(64, 32, 3), # input_channels=64, output_channels=32, karnel_size=3
            Conv3d(32, 32, 3), # input_channels=32, output_channels=32, karnel_size=3
            MaxPool3d(2, 2), # karnel_size=2, stride=2
            ReLU()
        ) # input:W=H=30 D=48, C output:W=H=4 D=9, C=32
        self.linear_relu_stack = Sequential(
            Flatten(),
            Linear(32*4*4*9, 128), #input=32*5*5*9, output=128
            ReLU(),
            Linear(128, 1), #input=128, output=1
        )

    def forward(self, x):
        x = self.conv_relu_stack(x)
        x = self.linear_relu_stack(x)
        return x

# definition loss function
class mylossMAPE(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, y_pred, target):
        loss = torch.sum(torch.abs((y_pred-target)/target))*(100/target.numel())
        return loss

#trainingの時とtestの時にmodelを動かすための関数

def train(dataloader, model, loss_fn, optimizer):
    # modelをtrain modelに切り替える
    model.train()
    # 1 epochでのlossの合計を入力する変数
    train_loss_total = 0.
    for x, y in dataloader:
        x = x.to("cuda")
        y = y.detach().numpy().copy()
        y = tensor(y[:, np.newaxis], dtype=float32).to("cuda")
        # 順伝播
        y_pred = model(x)
        # loss function
        loss = loss_fn(y_pred, y)
        train_loss_total += loss/y.numel()
        # back propagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    # loss average
    train_loss = train_loss_total/len(dataloader)
    return train_loss, loss

def valid(dataloader, model, loss_fn, threshold=0.5):
    # modelをevaluation modeに切り替える
    model.eval()
    # 1 epochでのlossの合計を入力する変数
    valid_loss_total = 0.
    with torch.no_grad():
        for x, y in dataloader:
            x = x.to("cuda")
            y = y.detach().numpy().copy()
            y = tensor(y[:, np.newaxis], dtype=float32).to("cuda")
            # 順伝播
            y_pred = model(x)
            # loss function
            loss = loss_fn(y_pred, y)
            valid_loss_total += loss/y.numel()
    # loss average
    valid_loss = valid_loss_total/len(dataloader)
    return valid_loss


def test(dataloader, model):
    # modelをevaluation modeに切り替える
    model.eval()
    with torch.no_grad():
        for i, (x, y) in enumerate(dataloader):
            x, y = x.to("cuda"), y.to("cuda")
            # 順伝播
            y_pred = model(x)
            # print(y_pred.size(), y.size())
            np_y_pred = y_pred.to("cpu").detach().numpy()
            np_y = y.to("cpu").detach().numpy()
            if i==0:
                output = np_y_pred
                true = np_y
            else:
                output = np.concatenate([output, np_y_pred], axis=0)
                true = np.concatenate([true, np_y], axis=0)
    # output = np.array(output)
    # true = np.array(true)
    return output, true

# Learning

# loss_function = MAPEloss, optimizer = Adam
from torch.optim import Adam

# model instance
# input_shape:batch_size=64, channels=1, W=H=100
model = ConvNet(1).to("cuda")

loss_fn = mylossMAPE()
optimizer = Adam(model.parameters(), lr=pow(10, -LR))
# epochごとのlossを入力するリスト
tloss = []
vloss = []

# training
for i_epoch in range(num_epoch):
    #train
    train_loss, output_loss = train(train_c1_dataloader, model, loss_fn, optimizer)
    tloss.append(train_loss.to("cpu").detach().numpy())
    #validation
    valid_loss = valid(valid_c1_dataloader, model, loss_fn)
    vloss.append(valid_loss.to("cpu").detach().numpy())

    print(f"Train loss: {train_loss:.5f}, Validation loss: {valid_loss:.5f}")
    if i_epoch%10==0:
        Line_module.notify_to_line(f"epoch{i_epoch}_filmu TL:{train_loss:.3f}, VL:{valid_loss:.3f}")
    #　パラメータの保存
    torch.save(model.state_dict(), "./" + exe_file + f"/CNNparameter/Conv3D{i_epoch}epoch_{LR}lr_filmu")

tloss = np.array(tloss)
vloss = np.array(vloss)


np.save("./" + exe_file + f"/Conv3D_result/vloss{LR}_filmu", vloss)
np.save("./" + exe_file + f"/Conv3D_result/tloss{LR}_filmu", tloss)
torch.save(
    {'epoch': i_epoch,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'loss': output_loss},
    "./" + exe_file + f"/CNNparameter/Conv3D{LR}lr_filmu")

output_y, true_y = test(test_c1_dataloader, model)
output_y = output_y.reshape(len(en_test))
true_y = true_y.reshape(len(en_test))
print(output_y.shape, type(output_y))
print(true_y.shape, type(true_y))

np.save("./" + exe_file + f"/Conv3D_result/y_output{LR}_filmu", output_y)
np.save("./" + exe_file + f"/Conv3D_result/y_label{LR}_filmu", true_y)
np.save("./" + exe_file + f"/Conv3D_result/event_number{LR}_filmu", en_test)

Line_module.notify_to_line("finish_training")




