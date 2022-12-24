#!/usr/bin/env python
# coding: utf-8
# In[5]:

import sys
import re
LR = int(sys.argv[1])
num_epoch = int(sys.argv[2])

particle = sys.argv[3]

import numpy as np
from sklearn.model_selection import train_test_split
import h5py
import torch
sys.path.append("../")
import Line_module

# Get Event number list
event_number = np.load(f"./train_dataset/Conv3D_result/event_number{LR}_filmu.npy")

# Get number of Event
h5py_path = "/mnt/scratch/kobayashik/hitmap.h5"
with h5py.File(h5py_path) as f:
    nofsignal = int(f[particle]["nofEvent"][()])

# import
from torch.utils.data import DataLoader, Dataset
from torch import tensor, float32

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

test_c1 = HDF5dataset(h5py_path, event_number, particle)
# In[16]:


#DataLoader
batch_size = 512
test_c1_dataloader = DataLoader(test_c1, batch_size=batch_size, shuffle=False)


# ### modelの定義

# In[12]:


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

# In[14]:


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


# ### 学習

# ### input channel = 1

# In[15]:


# loss_function = BCEloss, optimizer = Adam
from torch.nn import MSELoss
from torch.optim import Adam

# model instance
# input_shape:batch_size=64, channels=1, W=H=100
model = ConvNet(1).to("cuda")
model.load_state_dict(torch.load(f"./train_dataset/CNNparameter/Conv3D{num_epoch}epoch_{LR}lr_filmu"))


# In[19]:
output_y, true_y = test(test_c1_dataloader, model)
output_y = output_y.reshape(len(event_number))
true_y = true_y.reshape(len(event_number))
print(output_y.shape, type(output_y))
print(true_y.shape, type(true_y))


# In[21]:


np.save(f"./train_dataset/Conv3D_result/y_reoutput{num_epoch}epoch_{LR}lr_" + particle + "_filmu", output_y)

# Line_module.notify_to_line("finish_test")
