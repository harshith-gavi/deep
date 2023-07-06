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

class LSNN_layer(nn.Module):
    """
    LSNN layer module
    INPUT: Input Size, Output Size
    """
    def __init__(self, i_size, o_size, b_size):
        super(LSNN_layer, self).__init__()

        self.u_r = 0.01                                                     # Resting Potential
        self.thr = 0.5                                                      # Threshold
        self.thr_min = 0.01                                                 # Threshold Baseline
        self.u_t = torch.zeros(b_size, o_size).to(device_1)                 # Membrane Potential
        self.b_t = torch.zeros(b_size, o_size).to(device_1)                 # Intermediate State Variable
        self.spk = torch.zeros(b_size, o_size).to(device_1)                 # Output Spikea

        self.syn = nn.Linear(i_size, o_size).to(device_1)                   # Synapses/Connections
        self.T_adp = nn.Linear(o_size, o_size).to(device_1)                 # Adaptation Time Constant
        self.T_m = nn.Linear(o_size, o_size).to(device_1)                   # Membrane Time Constant

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
        L1 = self.syn(x_t)
        
        # T_m = self.act(self.T_m(L1 + self.u_t))
        # T_adp = self.act(self.T_adp(L1 + self.b_t))
        
        alpha = self.act(self.T_m(L1 + self.u_t))
        rho = self.act(self.T_adp(L1 + self.b_t))
      
        self.b_t = (rho * self.b_t) + ((1 - rho) * self.spk)
        self.thr = self.thr_min + (1.8 * self.b_t)

        du = (-self.u_t + L1) / alpha
        self.u_t = self.u_t + du

        self.spk = self.u_t - self.thr
        self.spk = self.spk.gt(0).float()
        self.u_t = self.u_t * (1 - self.spk) + (self.u_r * self.spk)

        o_spk = self.spk
        
        del x_t, L1, alpha, rho, du #T_m, T_adp,
        return o_spk

class LSNN_network(nn.Module):
    def __init__(self, b_size):
        super(LSNN_network, self).__init__()

        i_size = 700
        h_size = [128, 64]
        o_size = 20

        layers = [LSNN_layer(i_size, h_size[0], b_size), LSNN_layer(h_size[0], h_size[1], b_size), LSNN_layer(h_size[1], o_size, b_size)]
        self.network = nn.Sequential(*layers)

    def forward(self, x_t):
        return self.network(x_t)
    
def es_geht():
    print('PARSING ARGUMENTS...')
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type = int, default = 128)
    parser.add_argument('--epochs', type = int, default = 10)
    args = parser.parse_args()
    b_size = args.batch_size
    epochs = args.epochs
    
    print('PREPROCESSING DATA...')
    shd_train = h5py.File(datapath + 'train_data/SHD/shd_train.h5', 'r')
    shd_test = h5py.File(datapath + 'test_data/SHD/shd_test.h5', 'r')
    
    shd_train = data_mod(shd_train['spikes'], shd_train['labels'], batch_size = b_size, step_size = 140, input_size = tonic.datasets.SHD.sensor_size[0], max_time = 1.4)
    shd_test = data_mod(shd_test['spikes'], shd_test['labels'], batch_size = 1, step_size = 140, input_size = tonic.datasets.SHD.sensor_size[0], max_time = 1.4)

    shd_train = shd_train[:int(0.8 * len(shd_train))]
    
    print('Available CUDA memory: ', torch.cuda.mem_get_info()[0] / (1024 * 1024))
    print('CREATING A MODEL...')    
    model = LSNN_network(b_size)

    print('Available CUDA memory: ', torch.cuda.mem_get_info()[0] / (1024 * 1024))
    print('TRAINING THE MODEL...')
    
    for _ in range(1, epochs+1):
        progress_bar = tqdm(total = len(shd_train), desc = 'Epoch {}'.format(_))
        model_spk = []

        for batch in shd_train:
            inputs, labels = batch
            b_size, seq_num, i_size = inputs.shape
            b_spk = 0
            
            for i in range(seq_num):
                xx = inputs.to_dense()[:, i, :].to(device_1)
                b_spk = model(xx)
                del xx
                
            b_spk.to(device_2)
            model_spk.append(b_spk)
            progress_bar.update(1)   
        progress_bar.close()
        print(len(model_spk))
        # Calculate and print('Accuracy: ', 1)
        
        torch.cuda.empty_cache()
        print('Available CUDA memory: ', torch.cuda.mem_get_info()[0] / (1024 * 1024))

    for j in range(20):
        print(model_spk[0][j])

es_geht()
