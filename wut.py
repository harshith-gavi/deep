# !pip install tonic
# !pip install snntorch
# !pip install torchplot
print('IMPORTING LIBRARIES...', end = '\t')
import os
import h5py
import numpy as np
import matplotlib.pyplot as plt
import itertools
from tqdm import tqdm

import tonic
import tonic.transforms as ttr

import torch
import torch.nn as nn
import torchplot as tp
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

import snntorch as snn
from snntorch import spikeplot as splt
from snntorch import spikegen

print('DONE')

datapath = '../data/'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def data_mod(X, y, batch_size, step_size, input_size, max_time, shuffle=True):
    '''
    This function modifies SHD dataset into batches of data
    '''
    labels = np.array(y, int)
    nb_batches = len(labels)//batch_size
    sample_index = np.arange(len(labels))

    # compute discrete firing times
    firing_times = X['times']
    units_fired = X['units']

    time_bins = np.linspace(0, max_time, num=step_size)

    if shuffle:
        np.random.shuffle(sample_index)

    total_batch_count = 0
    counter = 0
    mod_data = []
    while counter<nb_batches:
        batch_index = sample_index[batch_size*counter:batch_size*(counter+1)]

        coo = [ [] for i in range(3) ]
        for bc,idx in enumerate(batch_index):
            times = np.digitize(firing_times[idx], time_bins)
            units = units_fired[idx]
            batch = [bc for _ in range(len(times))]

            coo[0].extend(batch)
            coo[1].extend(times)
            coo[2].extend(units)

        i = torch.LongTensor(coo).to(device)
        v = torch.FloatTensor(np.ones(len(coo[0]))).to(device)

        X_batch = torch.sparse.FloatTensor(i, v, torch.Size([batch_size,step_size,input_size])).to(device)
        y_batch = torch.tensor(labels[batch_index], device = device)

        mod_data.append((X_batch.to(device), y_batch.to(device)))

        counter += 1

    return mod_data

print('PREPROCESSING DATA...', end = '\t')
# shd_train = tonic.datasets.SHD(save_to = datapath + 'train_data')
# shd_test = tonic.datasets.SHD(save_to = datapath + 'test_data', train = False)

shd_train = h5py.File(datapath + 'train_data/SHD/shd_train.h5', 'r')
shd_test = h5py.File(datapath + 'test_data/SHD/shd_test.h5', 'r')

shd_train = data_mod(shd_train['spikes'], shd_train['labels'], batch_size = 256, step_size = 100, input_size = tonic.datasets.SHD.sensor_size[0], max_time = 1.4)
shd_test = data_mod(shd_test['spikes'], shd_test['labels'], batch_size = 1, step_size = 100, input_size = tonic.datasets.SHD.sensor_size[0], max_time = 1.4)
print('DONE')

#Straight from the github

gamma = .5  # gradient scale
lens = 0.3

def gaussian(x, mu=0., sigma=.5):
    return torch.exp(-((x - mu) ** 2) / (2 * sigma ** 2)) / torch.sqrt(2 * torch.tensor(math.pi)) / sigma


class ActFun_adp(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):  # input = membrane potential- threshold
        ctx.save_for_backward(input)
        return input.gt(0).float()  # is firing ???

    @staticmethod
    def backward(ctx, grad_output):  # approximate the gradients
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        # temp = abs(input) < lens
        scale = 6.0
        hight = .15
        # temp = torch.exp(-(input**2)/(2*lens**2))/torch.sqrt(2*torch.tensor(math.pi))/lens
        temp = gaussian(input, mu=0., sigma=lens) * (1. + hight) \
               - gaussian(input, mu=lens, sigma=scale * lens) * hight \
               - gaussian(input, mu=-lens, sigma=scale * lens) * hight
        # temp =  gaussian(input, mu=0., sigma=lens)
        return grad_input * temp.float() * gamma
        # return grad_input


act_fun_adp = ActFun_adp.apply

class LSNN(nn.Module):
    def __init__(self, i_size, h_size, o_size, b_size):
        super(LSNN, self).__init__()

        self.u_r = 0                                            # Resting Potential
        self.thr = 0.5                                          # Threshold
        self.thr_min = 0.01                                     # Threshold Baseline

        self.u1 = torch.zeros(b_size, h_size[0]).to(device)                # Membrane Potentials
        self.u2 = torch.zeros(b_size, h_size[1]).to(device)
        self.u3 = torch.zeros(b_size, o_size).to(device)

        self.b1 = torch.zeros(b_size, h_size[0]).to(device)
        self.b2 = torch.zeros(b_size, h_size[1]).to(device)
        self.b3 = torch.zeros(b_size, o_size).to(device)

        self.spk1 = torch.zeros(b_size, h_size[0]).to(device)              # Spikes
        self.spk2 = torch.zeros(b_size, h_size[1]).to(device)
        self.spk_out = torch.zeros(b_size, o_size).to(device)

        self.syn1 = nn.Linear(i_size, h_size[0])                # Synapses/Connections
        self.syn2 = nn.Linear(h_size[0], h_size[1])
        self.syn3 = nn.Linear(h_size[1], o_size)

        self.l1_T_adp = nn.Linear(h_size[0], h_size[0])         # Adaptation Time Constant
        self.l1_T_m = nn.Linear(h_size[0], h_size[0])           # Membrane Time Constant

        self.l2_T_adp = nn.Linear(h_size[1], h_size[1])
        self.l2_T_m = nn.Linear(h_size[1], h_size[1])

        self.o_T_adp = nn.Linear(o_size, o_size)
        self.o_T_m = nn.Linear(o_size, o_size)

        self.act = nn.Sigmoid()

        nn.init.ones_(self.syn1.weight)                         # Parameter Initialisation
        nn.init.zeros_(self.syn1.bias)
        nn.init.ones_(self.l1_T_adp.weight)
        nn.init.ones_(self.l1_T_m.weight)
        nn.init.zeros_(self.l1_T_adp.bias)
        nn.init.zeros_(self.l1_T_m.bias)

        nn.init.ones_(self.syn2.weight)
        nn.init.zeros_(self.syn2.bias)
        nn.init.ones_(self.l2_T_adp.weight)
        nn.init.ones_(self.l2_T_m.weight)
        nn.init.zeros_(self.l2_T_adp.bias)
        nn.init.zeros_(self.l2_T_m.bias)

        nn.init.ones_(self.syn3.weight)
        nn.init.zeros_(self.syn3.bias)
        nn.init.ones_(self.o_T_adp.weight)
        nn.init.ones_(self.o_T_m.weight)
        nn.init.zeros_(self.o_T_adp.bias)
        nn.init.zeros_(self.o_T_m.bias)

    def update_params(self, op, u_t, spk, t_m, t_adp, b_t):
        """
        Used to update the parameters
        INPUT: Layer output, Membrane Potential, Spikes, T_adp, T_m and Intermediate State Variable (b_t)
        OUTPUT: Membrane Potential, Spikes and Intermediate State Variable (b_t)
        """
        alpha = t_m
        rho = t_adp

        b_t = (rho * b_t) + ((1 - rho) * spk)
        thr = self.thr_min + (1.8 * b_t)

        du = (-u_t + op) / alpha
        u_t = u_t + du

        spk = act_fun_adp(u_t - thr)
        u_t = u_t * (1 - spk) + (self.u_r * spk)

        return u_t, spk, b_t

    def FPTT(self, x_t):
        """
        Used to train using Forward Pass Through Time Algorithm
        INPUT: Spikes
        OUTPUT: Spikes
        """
        L1 = self.syn1(x_t)
        T_m = self.act(self.l1_T_m(L1 + self.u1))
        T_adp = self.act(self.l1_T_adp(L1 + self.b1))
        self.u1, self.spk1, self.b1 = self.update_params(L1, self.u1, self.spk1, T_m, T_adp, self.b1)

        L2 = self.syn2(self.spk1)
        T_m = self.act(self.l2_T_m(L2 + self.u2))
        T_adp = self.act(self.l2_T_adp(L2 + self.b2))
        self.u2, self.spk2, self.b2  = self.update_params(L2, self.u2, self.spk2, T_m, T_adp, self.b2)

        L3 = self.syn3(self.spk2)
        T_m = self.act(self.o_T_m(L3 + self.u3))
        T_adp = self.act(self.o_T_adp(L3 + self.b3))
        self.u3, self.spk_out, self.b3 =  self.update_params(L3, self.u3, self.spk_out, T_m, T_adp, self.b3)

def save_checkpoint(model, epoch, batch):
    checkpoint = {'model_state_dict': model.state_dict(), 'epoch': epoch, 'batch': batch}
    checkpoint_path = f'checkpoint_epoch_{epoch}_batch_{batch}.pt'
    torch.save(checkpoint, checkpoint_path)

def load_checkpoint(model, checkpoint_path):
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    epoch = checkpoint['epoch']
    batch = checkpoint['batch']
    print(f'Checkpoint loaded: {checkpoint_path}')
    return epoch, batch

model = LSNN(700, [256, 64], 20, 256).to(device)

print('TRAINING THE MODEL...')
model_u = []
model_spk = []
start_epoch = 1
start_batch = 0
resume_training = False
shd_train = shd_train[:1]

checkpoint_files = [file for file in os.listdir() if file.startswith('checkpoint')]
if checkpoint_files:
    resume_training = True
    latest_checkpoint = max(checkpoint_files)
    start_epoch, start_batch = load_checkpoint(model, latest_checkpoint)

for _ in range(start_epoch, 10):
    progress_bar = tqdm(total = len(shd_train), desc = 'Epoch {}'.format(_))
    for batch_idx, batch in enumerate(shd_train[start_batch:], start=start_batch):
        inputs, labels = batch
        b_size, seq_num, i_size = inputs.shape
        # xx = inputs.to_dense()
        for i in range(seq_num):
            xx = inputs.to_dense()[:, i, :]
            model.FPTT(xx)
            model_spk.append(model.spk_out)

            save_checkpoint(model, _, batch_idx)
        
        progress_bar.update(1)
    progress_bar.close()
    start_batch = 0
    
print('DONE')
