import h5py
import numpy as np
import matplotlib.pyplot as plt
import itertools

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

datapath = './thesis/'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#----------------------------------------------------------------------------------

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
        y_batch = torch.tensor(labels[batch_index],device=device)

        mod_data.append((X_batch.to(device=device), y_batch.to(device=device)))

        counter += 1

    return mod_data

#----------------------------------------------------------------------------------

shd_train = tonic.datasets.SHD(save_to = datapath + 'train_data')
shd_test = tonic.datasets.SHD(save_to = datapath + 'test_data', train = False)

shd_train = h5py.File(datapath + 'train_data/SHD/shd_train.h5', 'r')
shd_test = h5py.File(datapath + 'test_data/SHD/shd_test.h5', 'r')

shd_train = data_mod(shd_train['spikes'], shd_train['labels'], batch_size = 32, step_size = 100, input_size = tonic.datasets.SHD.sensor_size[0], max_time = 1.4)
shd_test = data_mod(shd_test['spikes'], shd_test['labels'], batch_size = 1, step_size = 100, input_size = tonic.datasets.SHD.sensor_size[0], max_time = 1.4)

#----------------------------------------------------------------------------------
# # Liquid Spiking Neural Network module

#----------------------------------------------------------------------------------
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

#----------------------------------------------------------------------------------

class LSNN(nn.Module):
    def __init__(self):
        super(LSNN, self).__init__()
        self.tc = 1                                 # Time Constant
        self.u_t = 0.3                              # Membrane Potential at time t = 0
        self.u_r = 0                                # Resting Potential
        self.thr = 0.5                              # Threshold

        self.b_t = 0.95                             # Intermediate State Variable   
        self.alpha = 0.95                           # Used for Dynamic Regularisation Penalty 
        self.beta = 0.95                            # Used for the instantaneous loss
        self.l_r = 0.01                             # Learning Rate
        self.loss = 0                               # Loss at each time step
        self.spk = 0

        self.l1 = nn.Linear(700, 100)               # Layers
        self.l2 = nn.Linear(100, 20)

        self.l1_T_adp = nn.Linear(100, 100)         # T_adp
        self.l1_T_m = nn.Linear(100, 100)           # T_m

        self.act = nn.Sigmoid()

        nn.init.ones_(self.l1.weight)
        nn.init.ones_(self.l1_T_adp.weight)
        nn.init.ones_(self.l1_T_m.weight)

        nn.init.zeros_(self.l1.bias)
        nn.init.constant_(self.l1_T_adp.bias,-5.)
        nn.init.constant_(self.l1_T_m.bias,-5.)

    def forward(self, x_t):
        
        L1 = self.l1(x_t)
        T_m = self.act(self.l1_T_m(L1 + self.u_t))
        T_adp = self.act(self.l1_T_adp(L1 + self.b_t))
        # L2 = self.l2(A1)
        # A2 = self.act(L2)

        # # rho = 
        # self.tc = 1 / A3
        
        # # self.b_t = (rho * self.b_t) + ((1 - rho) * self.spk)
        # self.thr = 0.1 + (1.8 * self.b_t)

        return L1, T_m, T_adp

    def update_params(self, l1, t_m, t_adp):

        self.alpha = t_m
        self.rho = t_adp

        self.b_t = (self.rho * self.b_t) + ((1 - self.rho) * self.spk)
        self.thr = 0.1 + (1.8 * self.b_t)

        #1 - du = -self.u_t + l1
        #1 - self.u_t = self.u_t + (du * self.alpha)
        du = (-self.u_t + l1) / self.alpha
        self.u_t = self.u_t + du
        

        self.spk = act_fun_adp(self.u_t - self.thr)
        self.u_t = self.u_t * (1 - self.spk) + (self.u_r * self.spk)


    # def predict(self, x_t):

    #     L1 = self.l1(x_t)
    #     A1 = self.act(L1)
    #     L2 = self.l2(A1)
    #     A2 = self.act(L2)
    #     L3 = self.l3(A2)
    #     A3 = self.act(L3)

    #     return A3
      
    # def dynamic_loss_t(self, y_true, y_pred):
    #     # dl = 1                                                                                    # Derivative of Loss
    #     l_t_div = - sum(y_pred[i] * np.log2(y_true) for i in range(len(y_true)))                  # Divergence Term
    #     l_t = (self.beta * nn.CrossEntropyLoss(y_true, y_pred)) + ((1 - self.beta) * l_t_div)     # Loss
    #     r_t = 0.5 * self.alpha * (self.w3 - self.W_avg - (1 / (2 * self.alpha) * dl))**2          # Dynamic Regularisation Penalty
    #     self.loss = l_t + r_t                                                                     # Total Loss
                                      
    # def update_weights(self):
    #     dl = 1
    #     self.w3 = self.w3 - (self.l_r * dl)
    #     self.W_avg = (0.5 * (self.W_avg + self.W)) - ((1/(2*self.alpha)) * dl)

#----------------------------------------------------------------------------------

# #Training the model

#----------------------------------------------------------------------------------
# time_step = 1e-3
model = LSNN().to(device)

model_u = []
model_spk = []

for _ in range(10):
    for batch in shd_train:
        inputs, labels = batch
        xx = inputs.to_dense()
        out, t_m, t_adp = model.forward(xx)
        model.update_params(out, t_m, t_adp)
        model_u.append(model.u_t)
        model_spk.append(model.spk) 

#----------------------------------------------------------------------------------
# def plot_spk(spk_list,shape='*',label='spk',baseline=0.):
#     spk_t = np.where(spk_list==1)[0]
#     plt.plot(spk_t,np.ones(len(spk_t))+baseline,shape,label=label)

# mem_ltc_np = np.array(mem_ltc_list)
# spk_ltc_np = np.array(spk_ltc_list)




