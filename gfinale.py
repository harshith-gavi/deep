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

def data_mod(X, y, batch_size, step_size, input_size, max_time, shuffle=False):
    '''
    This function generates batches of sparse data from the SHD dataset
    '''
    labels = np.array(y, int)
    nb_batches = len(labels)//batch_size
    sample_index = np.arange(len(labels))

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

        i = torch.LongTensor(coo).to(device_2)
        v = torch.FloatTensor(np.ones(len(coo[0]))).to(device_2)

        X_batch = torch.sparse.FloatTensor(i, v, torch.Size([batch_size,step_size,input_size])).to(device_2)
        y_batch = torch.tensor(labels[batch_index], device = device_2)

        mod_data.append((X_batch.to(device_2), y_batch.to(device_2)))

        counter += 1

    return mod_data

class Processing_layer(nn.Module):
    """
    LSNN layer module
    INPUT: Input Size, Output Size
    """
    def __init__(self, i_size, o_size):
        super(Processing_layer, self).__init__()

        self.u_r = 0.0                                                                  # Resting Potential
        self.thr = torch.full((1, o_size), 2.0, dtype=torch.float32).to(device_1)       # Threshold
        self.thr_min = 0.1                                                              # Threshold Baseline
        self.u_t = torch.zeros((1, o_size), dtype=torch.float32).to(device_1)           # Membrane Potential
        self.b_t = torch.zeros((1, o_size), dtype=torch.float32).to(device_1)           # Intermediate State Variable
        self.spk = torch.zeros((1, o_size), dtype=torch.float32).to(device_1)           # Output Spikes

        self.syn = nn.Linear(i_size, o_size).to(device_1)                               # Synapses/Connections
        self.T_adp = nn.Linear(o_size, o_size).to(device_1)                             # Adaptation Time Constant
        self.T_m = nn.Linear(o_size, o_size).to(device_1)                               # Membrane Time Constant

        self.act = nn.Sigmoid()

        # Parameter Initialisation
        nn.init.ones_(self.syn.weight)
        nn.init.zeros_(self.syn.bias)
        nn.init.ones_(self.T_adp.weight)
        nn.init.ones_(self.T_m.weight)
        nn.init.zeros_(self.T_adp.bias)
        nn.init.zeros_(self.T_m.bias)

    def forward(self, x_t):
        """
            Used to train the layer using Forward Pass Through Time Algorithm and update its parameters
            INPUT: Input Spikes
            OUTPUT: Membrane Potential, Spikes and Intermediate State Variable (b_t)
        """
        with torch.no_grad():

            L1 = self.syn(x_t.to(device_1))

            T_m = self.act(self.T_m(L1 + self.u_t))
            T_adp = self.act(self.T_adp(L1 + self.b_t))

            alpha = T_m
            rho = T_adp

            self.b_t = (rho * self.b_t) + ((1 - rho) * self.spk)
            self.thr = self.thr_min + (1.8 * self.b_t)

            du = (-self.u_t + L1) / alpha
            self.u_t = self.u_t + du

            self.spk = self.u_t - self.thr
            self.spk = self.spk.gt(0).float()
            self.u_t = self.u_t * (1 - self.spk) + (self.u_r * self.spk)
            self.spk = self.spk

            return self.spk

class Output_layer(nn.Module):
    """
    LSNN Hidden layer module
    INPUT: Input Size, Output Size
    """
    def __init__(self, i_size, o_size):
        super(Output_layer, self).__init__()

        self.u_r = 0.0                                                                  # Resting Potential
        self.thr = torch.full((1, o_size), 2.0, dtype=torch.float32).to(device_1)       # Threshold
        self.thr_min = 0.1                                                              # Threshold Baseline
        self.u_t = torch.zeros((1, o_size), dtype=torch.float32).to(device_1)           # Membrane Potential
        self.b_t = torch.zeros((1, o_size), dtype=torch.float32).to(device_1)           # Intermediate State Variable
        self.spk = torch.zeros((1, o_size), dtype=torch.float32).to(device_1)           # Output Spikes

        self.syn = nn.Linear(i_size, o_size).to(device_1)                               # Synapses/Connections
        self.T_adp = nn.Linear(o_size, o_size).to(device_1)                             # Adaptation Time Constant
        self.T_m = nn.Linear(o_size, o_size).to(device_1)                               # Membrane Time Constant

        self.act = nn.Sigmoid()

        # Parameter Initialisation
        nn.init.ones_(self.syn.weight)
        nn.init.zeros_(self.syn.bias)
        nn.init.ones_(self.T_adp.weight)
        nn.init.ones_(self.T_m.weight)
        nn.init.zeros_(self.T_adp.bias)
        nn.init.zeros_(self.T_m.bias)

    def forward(self, x_t):
        """
        Used to train the layer using Forward Pass Through Time Algorithm and update its parameters
        INPUT: Input Spikes
        OUTPUT: Membrane Potential, Spikes and Intermediate State Variable (b_t)
        """
        with torch.no_grad():

            L1 = self.syn(x_t.to(device_1))

            T_m = self.act(self.T_m(L1 + self.u_t))
            T_adp = self.act(self.T_adp(L1 + self.b_t))

            alpha = T_m
            rho = T_adp

            self.b_t = (rho * self.b_t) + ((1 - rho) * self.spk)
            self.thr = self.thr_min + (1.8 * self.b_t)

            du = (-self.u_t + L1) / alpha
            self.u_t = self.u_t + du

            self.spk = self.u_t - self.thr
            self.spk = self.spk.gt(0).float()
            self.u_t = self.u_t * (1 - self.spk) + (self.u_r * self.spk)
            self.spk = self.spk

            return self.spk

class LSNN(nn.Module):
    def __init__(self):
        super(LSNN, self).__init__()

        i_size = 700
        h_size = [256, 64]
        o_size = 20

        layers = [Processing_layer(i_size, h_size[0]), Processing_layer(h_size[0], h_size[1]), Output_layer(h_size[1], o_size)]
        self.network = nn.Sequential(*layers)

    def forward(self, x_t):
        with torch.no_grad():
            return self.network(x_t)
    
def es_geht():
    print('PARSING ARGUMENTS...')
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type = int, default = 128)
    parser.add_argument('--epochs', type = int, default = 100)
    parser.add_argument('--evaluation_metric', type = str, default = 'spike_count')
    args = parser.parse_args()
    b_size = args.batch_size
    epochs = args.epochs
    metric = args.evaluation_metric
    
    print('PREPROCESSING DATA...')
    shd_train = h5py.File(datapath + 'train_data/SHD/shd_train.h5', 'r')
    shd_test = h5py.File(datapath + 'test_data/SHD/shd_test.h5', 'r')
    
    shd_train = data_mod(shd_train['spikes'], shd_train['labels'], batch_size = b_size, step_size = 100, input_size = tonic.datasets.SHD.sensor_size[0], max_time = 1.4)
    shd_test = data_mod(shd_test['spikes'], shd_test['labels'], batch_size = 1, step_size = 100, input_size = tonic.datasets.SHD.sensor_size[0], max_time = 1.4)

    shd_train = shd_train[:int(0.8 * len(shd_train))]
    shd_val = shd_train[int(0.8 * len(shd_train)):]
    
    print('Available CUDA memory: ', torch.cuda.mem_get_info()[0] / (1024 * 1024))
    print('CREATING A MODEL...')    
    model = LSNN()

    print('Available CUDA memory: ', torch.cuda.mem_get_info()[0] / (1024 * 1024))
    print('TRAINING THE MODEL...')
    model.train()
    
    for _ in range(1, epochs+1):
        progress_bar = tqdm(total = len(shd_train), desc = 'Epoch {}'.format(_))
        preds = []
        acc = 0
        
        for batch in shd_train:
            inputs, labels = batch
            b_size, seq_num, i_size = inputs.shape
            b_spk = []
            
            for i in range(seq_num):
                xx = inputs.to_dense()[:, i, :]
                out_spk = model(xx)
                b_spk.append(out_spk)

            b_spk = torch.sum(torch.stack(b_spk), dim=0)
            val, idx = torch.max(b_spk, dim=1)
            print(idx.shape)
            preds.append(idx)
            progress_bar.update(1)

        progress_bar.close()
        print(preds)
        # Calculate and print('Accuracy: ', 1)

es_geht()
