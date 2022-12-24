#!/usr/bin/env python
# coding: utf-8
# In[5]:

import sys
if len(sys.argv)==1:
    print("Error:file name not found")
    sys.exit()
exe_file = sys.argv[1]
if len(sys.argv)==2:
    lr = 3
else:
    lr = int(sys.argv[2])
if len(sys.argv)==3:
    num_epoch = 15
else:
    num_epoch = int(sys.argv[3])

sparticle = sys.argv[4]
bparticle = sys.argv[5]

import numpy as np
from sklearn.model_selection import train_test_split
import h5py
import torch
sys.path.append("../")
import Line_module

Line_module.notify_to_line("start in " + exe_file + f"learning rate={lr}, {num_epoch}epoch")

# Get number of Event
h5py_path = "/mnt/scratch/kobayashik/hitmap_100kE.h5"
with h5py.File(h5py_path) as f:
    nofsignal = f[sparticle]["nofEvent"][()]
    nofbackgraund = f[bparticle]["nofEvent"][()]

# create event number list
event_number = np.arange(nofsignal+nofbackgraund)
tv_t_seed = np.random.randint(1, 10000)
en_train_valid, en_test = train_test_split(event_number, train_size=0.8, random_state=tv_t_seed)

t_v_seed = np.random.randint(1, 10000)
en_train, en_valid = train_test_split(en_train_valid, train_size=0.75, random_state=t_v_seed)
print(en_train.shape, en_valid.shape, en_test.shape)

# import
from torch.utils.data import DataLoader, Dataset
from torch import tensor, float32

class HDF5dataset(Dataset):
    def __init__(self, path, enlist, signal, backgraund, nofbackgraund):
        self.path = path
        self.fh5 = h5py.File(self.path, "r")
        self.enlist = enlist
        self.nofbackgraund = nofbackgraund
        
        self.signal = self.fh5[signal]
        self.backgraund = self.fh5[backgraund]

    def __getitem__(self, idx):
        event_number = self.enlist[int(idx)]
        if event_number>=self.nofbackgraund:
            hitmap = self.signal[str(event_number-nofbackgraund)][::]
            y = 1
        elif event_number<self.nofbackgraund:
            hitmap = self.backgraund[str(event_number)][::]
            y = 0
        return tensor(hitmap[np.newaxis, :, :, :], dtype=float32), tensor(y, dtype=float32)
    
    def __len__(self):
        return len(self.enlist)
            
train_c1 = HDF5dataset(h5py_path, en_train, sparticle, bparticle, nofbackgraund)
valid_c1 = HDF5dataset(h5py_path, en_valid, sparticle, bparticle, nofbackgraund)
test_c1 = HDF5dataset(h5py_path, en_test, sparticle, bparticle, nofbackgraund)
# In[16]:


#DataLoader
batch_size = 512
train_c1_dataloader = DataLoader(train_c1, batch_size=batch_size, shuffle=True)
valid_c1_dataloader = DataLoader(valid_c1, batch_size=batch_size, shuffle=True)
test_c1_dataloader = DataLoader(test_c1, batch_size=batch_size, shuffle=False)


# ### modelの定義

# In[17]:


import torch.nn as nn
from torch.nn import Sequential, Flatten, Conv3d, MaxPool3d, Linear, ReLU, Sigmoid

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
            Linear(32*4*4*9, 128), #input=32*4*4*9, output=128
            ReLU(),
            Linear(128, 1), #input=128, output=1
            Sigmoid()
        )
    
    def forward(self, x):
        x = self.conv_relu_stack(x)
        x = self.linear_relu_stack(x)
        return x


# In[18]:


#trainingの時とtestの時にmodelを動かすための関数

def train(dataloader, model, loss_fn, optimizer):
    # modelをtrain modelに切り替える
    model.train()
    # 1 epochでのlossの合計を入力する変数
    train_loss_total = 0.
    for i, (x, y) in enumerate(dataloader):
        x = x.to("cuda")
        y = y.detach().numpy().copy()
        y = tensor(y[:, np.newaxis], dtype=float32).to("cuda")
        # 順伝播
        y_pred = model(x)
        # loss function
        loss = loss_fn(y_pred, y)
        train_loss_total += loss
        # back propagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    # loss average
    train_loss = train_loss_total/len(dataloader)
    return train_loss

def valid(dataloader, model, loss_fn, threshold=0.5):
    # modelをevaluation modeに切り替える
    model.eval()
    # 1 epochでのlossの合計を入力する変数
    valid_loss_total = 0.
    # anserとyが一致した回数を入力する変数
    correct = 0
    with torch.no_grad():
        for x, y in dataloader:
            x = x.to("cuda")
            y = y.detach().numpy().copy()
            y = tensor(y[:, np.newaxis], dtype=float32).to("cuda")
            # 順伝播
            y_pred = model(x)
            # loss function
            loss = loss_fn(y_pred, y)
            valid_loss_total += loss
            # accuracyを計算するためのthreshold
            np_threshold = np.full(y.shape[0],threshold)
            tensor_threshold = tensor(np_threshold[:, np.newaxis], dtype=float32).to("cuda")
            # anserとyが一致した回数
            anser = (y_pred>=tensor_threshold).type(float32)
            correct += (anser==y).type(float32).sum().item()
    # loss average
    valid_loss = valid_loss_total/len(dataloader)
    # accuracy
    correct /= len(dataloader.dataset)
    return valid_loss, correct


# In[19]:


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

# In[20]:


# loss_function = BCEloss, optimizer = Adam
from torch.nn import BCELoss
from torch.optim import Adam

# model instance
# input_shape:batch_size=64, channels=1, W=H=100
model = ConvNet(1).to("cuda")

loss_fn = BCELoss()
optimizer = Adam(model.parameters(), lr=pow(10, -lr))
# epochごとのlossを入力するリスト
tloss = []
vloss = []
# epochごとのaccuracyを入力するリスト
taccuracy = []
vaccuracy = []

# training
for i_epoch in range(num_epoch):
    #train
    train_loss = train(train_c1_dataloader, model, loss_fn, optimizer)
    _, train_accuracy = valid(train_c1_dataloader, model, loss_fn) 
    tloss.append(train_loss.to("cpu").detach().numpy())
    taccuracy.append(train_accuracy)
    #validation
    valid_loss, valid_accuracy = valid(valid_c1_dataloader, model, loss_fn)
    vloss.append(valid_loss.to("cpu").detach().numpy())
    vaccuracy.append(valid_accuracy)

    print(f"Train loss: {train_loss:.5f}, Train accuraxy: {train_accuracy:.5f}, Validation loss: {valid_loss:.5f}, Validation accuracy: {valid_accuracy:.5f}")
    Line_module.notify_to_line(f"epoch{i_epoch} TL: {train_loss:.2f}, TA: {train_accuracy:.2f}, VL: {valid_loss:.2f}, VA: {valid_accuracy:.2f}")
    if i_epoch%10==0:
        torch.save(model.state_dict(), "./" + exe_file + f"/CNNparameter/Conv3Ds{i_epoch}epoch_{lr}lr")

tloss = np.array(tloss)
taccuracy = np.array(taccuracy)
vloss = np.array(vloss)
vaccuracy = np.array(vaccuracy)


# 評価

# In[21]:


#　パラメータの保存
torch.save(model.state_dict(), "./" + exe_file + f"/CNNparameter/Conv3Ds{lr}lr")


# In[23]:


np.save("./" + exe_file + f"/Conv3Ds_result/tloss{lr}lr", tloss)
np.save("./" + exe_file + f"/Conv3Ds_result/vloss{lr}lr", vloss)
np.save("./" + exe_file + f"/Conv3Ds_result/taccuracy{lr}lr", taccuracy)
np.save("./" + exe_file + f"/Conv3Ds_result/vaccuracy{lr}lr", vaccuracy)


# In[24]:


output_y, true_y = test(test_c1_dataloader, model)
output_y = output_y.reshape(en_test.shape)
true_y = true_y.reshape(en_test.shape)
print(output_y.shape, type(output_y))
print(true_y.shape, type(true_y))


# # In[26]:


np.save("./" + exe_file + f"/Conv3Ds_result/y_output{lr}lr", output_y)
np.save("./" + exe_file + f"/Conv3Ds_result/y_label{lr}lr", true_y)


# In[ ]:




