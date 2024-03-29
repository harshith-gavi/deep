print('IMPORTING LIBRARIES...')
import argparse
import h5py
import numpy as np

from tqdm import tqdm

import tonic

import torch
import torch.nn as nn

datapath = '../data/'
device_0 = torch.device('cpu')
device_1 = torch.device('cuda:0')  # First CUDA device
device_2 = torch.device('cuda:1')  # Second CUDA device

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

        i = torch.LongTensor(coo).to(device_0)
        v = torch.FloatTensor(np.ones(len(coo[0]))).to(device_0)

        X_batch = torch.sparse.FloatTensor(i, v, torch.Size([batch_size,step_size,input_size])).to(device_0)
        y_batch = torch.tensor(labels[batch_index], device = device_0)

        mod_data.append((X_batch.to(device_0), y_batch.to(device_0)))

        counter += 1

    return mod_data

class LSNN(nn.Module):
    def __init__(self, i_size, h_size, o_size):
        super(LSNN, self).__init__()

        b_size = 64
        self.u_r = 0                                                # Resting Potential
        self.thr = 0.5                                              # Threshold
        self.thr_min = 0.01                                         # Threshold Baseline

        self.u1 = torch.zeros(b_size, h_size[0]).to(device_1)         # Membrane Potentials
        self.u2 = torch.zeros(b_size, h_size[1]).to(device_2)
        self.u3 = torch.zeros(b_size, o_size).to(device_2)

        self.spk1 = torch.zeros(b_size, h_size[0]).to(device_1)       # Spikes
        self.spk2 = torch.zeros(b_size, h_size[1]).to(device_2)
        self.spk_out = torch.zeros(b_size, o_size).to(device_2)

        self.syn1 = nn.Linear(i_size, h_size[0]).to(device_1)                    # Synapses/Connections
        self.syn2 = nn.Linear(h_size[0], h_size[1]).to(device_2)
        self.syn3 = nn.Linear(h_size[1], o_size).to(device_2)

        self.l1_T_adp = nn.Linear(h_size[0], h_size[0]).to(device_1)             # Adaptation Time Constant
        self.l1_T_m = nn.Linear(h_size[0], h_size[0]).to(device_1)               # Membrane Time Constant

        self.l2_T_adp = nn.Linear(h_size[1], h_size[1]).to(device_2)
        self.l2_T_m = nn.Linear(h_size[1], h_size[1]).to(device_2)

        self.o_T_adp = nn.Linear(o_size, o_size).to(device_2)
        self.o_T_m = nn.Linear(o_size, o_size).to(device_2)

        self.act = nn.Sigmoid()

        nn.init.ones_(self.syn1.weight)                             # Parameter Initialisation
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

    def update_params(self, op, u_t, spk, t_m, t_adp, b_t_):
        """
        Used to update the parameters
        INPUT: Layer output, Membrane Potential, Spikes, T_adp, T_m and Intermediate State Variable (b_t)
        OUTPUT: Membrane Potential, Spikes and Intermediate State Variable (b_t)
        """
        alpha = t_m
        rho = t_adp

        b_t_ = (rho * b_t_) + ((1 - rho) * spk)
        thr = self.thr_min + (1.8 * b_t_)

        du = (-u_t + op) / alpha
        u_t = u_t + du

        spk = u_t - thr
        spk = spk.gt(0).float()
        u_t = u_t * (1 - spk) + (self.u_r * spk)

        return u_t, spk, b_t_

    def FPTT(self, x_t, b_t):
        """
        Used to train using Forward Pass Through Time Algorithm
        INPUT: Spikes
        OUTPUT: Spikes
        """
        x_t = x_t.to(device_1)
        L1 = self.syn1(x_t)
        T_m = self.act(self.l1_T_m(L1 + self.u1))
        T_adp = self.act(self.l1_T_adp(L1 + b_t[0]))
        self.u1, self.spk1, b_t[0] = self.update_params(L1, self.u1, self.spk1, T_m, T_adp, b_t[0])
        temp = self.spk1
        temp = temp.to(device_2)
        L2 = self.syn2(temp)
        T_m = self.act(self.l2_T_m(L2 + self.u2))
        T_adp = self.act(self.l2_T_adp(L2 + b_t[1]))
        self.u2, self.spk2, b_t[1]  = self.update_params(L2, self.u2, self.spk2, T_m, T_adp, b_t[1])

        L3 = self.syn3(self.spk2)
        T_m = self.act(self.o_T_m(L3 + self.u3))
        T_adp = self.act(self.o_T_adp(L3 + b_t[2]))
        self.u3, self.spk_out, b_t[2] =  self.update_params(L3, self.u3, self.spk_out, T_m, T_adp, b_t[2])

        del x_t, T_m, T_adp, L1, L2, L3, temp

        return b_t


def es_geht():
    print('PARSING ARGUMENTS...')
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=32)
    args = parser.parse_args()

    print('PREPROCESSING DATA...')
    shd_train = h5py.File(datapath + 'train_data/SHD/shd_train.h5', 'r')
    shd_test = h5py.File(datapath + 'test_data/SHD/shd_test.h5', 'r')

    shd_train = data_mod(shd_train['spikes'], shd_train['labels'], batch_size = args.batch_size, step_size = 100, input_size = tonic.datasets.SHD.sensor_size[0], max_time = 1.4)
    shd_test = data_mod(shd_test['spikes'], shd_test['labels'], batch_size = 1, step_size = 100, input_size = tonic.datasets.SHD.sensor_size[0], max_time = 1.4)

    shd_train = shd_train[:int(0.8 * len(shd_train))]

    print('Available CUDA memory: ', torch.cuda.mem_get_info()[0] / (1024 * 1024))
    print('CREATING A MODEL...')    
    b_size = args.batch_size
    i_size = 700
    h_size = [128, 64]
    o_size = 20

    model_spk = []                                                    # Output Spikes

    # # Membrane Potentials
    # self.u1 = torch.zeros(b_size, h_size[0]).to(device_1)
    # self.u2 = torch.zeros(b_size, h_size[1]).to(device_2)
    # self.u3 = torch.zeros(b_size, o_size).to(device_2)

    b = [torch.zeros(b_size, h_size[0]).to(device_1),
          torch.zeros(b_size, h_size[1]).to(device_2),
          torch.zeros(b_size, o_size).to(device_2)]

    model = LSNN(i_size, h_size, o_size)
    print('Available CUDA memory: ', torch.cuda.mem_get_info()[0] / (1024 * 1024))
    print('TRAINING THE MODEL...')
    for _ in range(1, 5):
        progress_bar = tqdm(total = len(shd_train), desc = 'Epoch {}'.format(_))
        for batch in shd_train:
            inputs, labels = batch
            b_size, seq_num, i_size = inputs.shape

            for i in range(seq_num):
                xx = inputs.to_dense()[:, i, :]
                b = model.FPTT(xx, b)
                model_spk.append(model.spk_out)
                del xx

            progress_bar.update(1)
        progress_bar.close()

    torch.cuda.empty_cache()
    print('Available CUDA memory: ', torch.cuda.mem_get_info()[0] / (1024 * 1024))

    for j in range(20):
        print(model_spk[0][j])

es_geht()
