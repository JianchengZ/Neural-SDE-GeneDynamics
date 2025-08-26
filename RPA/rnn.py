import os

import ot
from matplotlib import pyplot as plt
from torch import nn, relu
import numpy as np
import torch
import pandas as pd
import time
import csv
from tqdm import tqdm
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import csv
import numpy as np
import ot
import pandas as pd
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
import torch
from tqdm import tqdm
import time
import random
import ot
import pandas as pd
import torch
import torchsde

torch.set_num_threads(1)
from torch import nn, relu
import numpy as np
from tqdm import tqdm
import argparse
import csv

# determine batch_hyper using argparse
parser = argparse.ArgumentParser()
parser.add_argument('--repeat', type=int, default=1, help='repeat')
args = parser.parse_args()
repeat = args.repeat
starttime=time.time()
def load_data(file_path,csv_data):
    """
    :param file_path: the path of data saved
    :param csv_data: the empty list
    :return: csv_data
    """
    with open(file_path, 'r') as csvfile:
        csvreader = csv.reader(csvfile)
        i0 = 0
        for row in csvreader:
            if i0 == 0:
                i0 += 1
                continue
            temp = []
            j0 = 0
            for i in row:
                if j0 == 0:
                    j0 += 1
                    continue
                temp.append(float(i))
            csv_data.append(temp)
            temp2=list(np.zeros([361,]))
            csv_data.append(temp2)
    csv_data=(np.array(csv_data))
    return csv_data


batch_size, state_size, brownian_size = 50, 2, 1
noise_size=1
H = 32
t_size = 361


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")




class MyRNN(nn.Module):
    """
    This class creates rnn model that consists of four linear layers with activation functions and two layers of RNN.
    The input size and hidden size are the same as paper's.

    """
    def __init__(self):
        super(MyRNN, self).__init__()
        self.rnn = nn.RNN(input_size=state_size+noise_size, hidden_size=256, batch_first=True, num_layers=2,dropout=0.3)
        self.fc1 = nn.Linear(256, 128)
        self.fcfc1=nn.Linear(128,64)
        self.fcfc=nn.Linear(64,32)
        self.fc2 = nn.Linear(32, state_size+noise_size)

    def forward(self, x):
        x = np.array(x.cpu().detach().numpy())
        x = torch.tensor(x)
        x = x.to(device)
        (h_n, c_n) = self.rnn(torch.reshape(x, [ batch_size,(state_size+noise_size)]))
        output =  h_n

        out = relu(self.fc1(output))
        out = relu(self.fcfc1(out))
        out = relu(self.fcfc(out))

        out = relu(self.fc2(out))
        return out


rnn = MyRNN()
rnn.to(device)
ts = torch.linspace(0, 2, t_size)
ts = ts.to(device)



all_traj=[]
TRAINPAS=[]
all_train_data=[]
Ts=[]
TRAINDATA=[]
TRAIN0=[]
TESTDATA=[]
TEST0=[]
TESTPAS=[]
for i in range(31):
    """
    Load data under all parameter situation 
    """
    csv_data=[]
    file_path = '../data/ndata_tmp_e%d.csv'%i
    csv_data2 = []
    file_path2 = '../data/ndata_tmp_e_%d.csv' % i
    csv_data=load_data(file_path=file_path,csv_data=csv_data)
    csv_data2=load_data(file_path=file_path2,csv_data=csv_data2)
    for j in range(202):
        if j % 2 == 0:
            csv_data[j] = csv_data[j]
        else:
            csv_data[j] = csv_data2[j - 1]
    csv_data = np.reshape(csv_data, [101, -1, 361])
    cdata = np.zeros([101, 361, 2])
    for ke in range(101):
        cdata[ke] = csv_data[ke].T
    csv_data = cdata
    all_traj.append(csv_data)

    paras = np.load("../data/nparas%d.npy"%i)

    paras = torch.tensor(paras)
    paras = paras.float()
    paras = paras.to(device)
    TRAINPAS.append(paras)
    all_train_data.append( np.transpose(csv_data[:50],(1,0,2)))
    y0 = torch.zeros(batch_size, state_size)
    y0 = y0.to(device)
    ts = torch.linspace(0, 2, t_size)
    ts = ts.to(device)
    Ts.append(ts)
    ys_truth = torch.tensor(all_train_data[i])
    TRAINDATA.append(ys_truth)
    TRAIN0.append(y0)
    test_data = csv_data[50:100]
    y0_test = torch.zeros(batch_size, state_size)
    y0_test = y0_test.to(device)
    ys_truth_test = torch.zeros(t_size, batch_size, state_size)
    test_data = np.transpose(test_data, (1, 0, 2))
    ys_truth_test = torch.tensor(test_data).to(device)
    TESTDATA.append(ys_truth_test)
    TEST0.append(y0_test)
    TESTPAS.append(paras)

def relative_error(p, PARAS):
    error = (abs(p - PARAS)) / (abs(PARAS) + abs(PARAS))
    return error
def distance(P, Q):
    cost_matrix = ot.dist(P, Q, metric='sqeuclidean')
    return cost_matrix
def w2_decoupled(y, y_pred):
    """
    :param y: the real trajectories(ground truth)
    :param y_pred: the predicted trajectories using model
    :return: W_{2} as loss
    """
    batch_size = y.shape[1]
    state_size = y.shape[2]
    t_size = y.shape[0]
    loss = 0
    for i in range(1, t_size):
        weights = torch.tensor([1 / batch_size for _ in range(batch_size)])
        loss += (ot.emd2(weights.to(device), weights.to(device),
                             distance((y[i, :, :]).float(), (y_pred[i, :, :]).float())))


    return loss




criterion = torch.nn.MSELoss(reduction='sum')

optimizer = torch.optim.Adam(rnn.parameters(), lr=0.0005)
epoch = 2001
loss_list = []
ttime = time.time()




def train(Ys_truth,Y0,Paras,model):
    """
    :param Ys_truth: ground truth of training datasets
    :param Y0: y0 of training datasets
    :param Paras: parameters of trajectories
    :param model: model that needs to train
    :return: trained model
    """
    LOSS=[]
    for k in range(31):
        y0=Y0[k]
        paras=Paras[k]
        ys_truth=Ys_truth[k]
        tmpy0 = torch.zeros([y0.shape[0], state_size]).to(device)
        y0 = tmpy0
        ys = torch.zeros([361, batch_size, state_size+noise_size])

        for tm in range(361):
            if tm==0:
                ys[tm, :, :2] = y0
                ys[tm, :, 2] = paras[3]

            else:

                ys[tm,:,:2]+=model(torch.concatenate((ys_truth[tm-1].float(),torch.reshape(ys[tm-1,:,2].float(),[batch_size,1])),dim=1))[:,:2]

                ys[tm, :, 2] = paras[3]

        loss = w2_decoupled(ys[:, :, :2], ys_truth[:, :, :2])
        LOSS.append(loss)
    loss=LOSS[0]
    for l in range(30):
        loss += LOSS[l + 1]
    loss /= 31





    if i0 % 20 == 0:
        print(i0, loss)
    optimizer.zero_grad()
    loss.backward()
    nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()
    return model
def _mytest_(model,y0_test,ys_truth,l1,l2,l3,l4):
    tmpy0 = torch.zeros([y0_test.shape[0], state_size]).to(device)
    y0_test = tmpy0
    ys = torch.zeros([361, batch_size, state_size + noise_size])
    for tm in range(361):
        if tm == 0:
            ys[tm,:,:2] = (y0_test)
        else:

            ys[tm, :, :2] += model(
                torch.concatenate((ys_truth[tm - 1].float(), torch.reshape(ys[tm - 1, :, 2].float(), [batch_size, 1])),
                                  dim=1))[:, :2]



    loss = w2_decoupled(ys[:, :, :2], ys_truth[:, :, :2])
    print("test_W2 loss: ", loss)
for i0 in tqdm(range(epoch)):
    rnn=train(TRAINDATA,TRAIN0,TRAINPAS,rnn)

    if i0 % 50==0:
        torch.save(rnn, r"../results/repeat%d/rnn/rnn%d.pth"%(repeat,i0))
        for A in tqdm(range(31)):
            rnn = _mytest_(rnn, TEST0[A], TESTDATA[A],
                         '../results/repeat%d/rnn/ground_truth_verify_traj%d.csv' % (repeat, A),
                         '../results/repeat%d/rnn/predict_verify_traj%d.csv' % (repeat, A),
                         '../results/repeat%d/rnn/ground_truth_verify2_traj%d.csv' % (repeat, A),
                         '../results/repeat%d/rnn/predict_verify2_traj%d.csv' % (repeat, A))

def mytest(model,y0_test,ys_truth,l1,l2,l3,l4):
    """
    :param model: trained model
    :param y0_test: y0 of testing datasets
    :param ys_truth: ground truth of testing datasets
    :param l1: path for saving results
    :param l2: path for saving results
    :param l3: path for saving results
    :param l4: path for saving results
    :return: testing results
    """
    tmpy0 = torch.zeros([y0_test.shape[0], state_size]).to(device)
    y0_test = tmpy0
    ys = torch.zeros([361, batch_size, state_size + noise_size])
    for tm in range(361):
        if tm == 0:
            ys[tm,:,:2] = (y0_test)
        else:

            ys[tm, :, :2] += model(
                torch.concatenate((ys_truth[tm - 1].float(), torch.reshape(ys[tm - 1, :, 2].float(), [batch_size, 1])),
                                  dim=1))[:, :2]



    loss = w2_decoupled(ys[:, :, :2], ys_truth[:, :, :2])
    print("test_W2 loss: ", loss)
    truth_data = pd.DataFrame(data=[[float(ys_truth[i, j, 0]) for i in range(t_size)] for j in range(batch_size)])
    truth_data.to_csv(l1, header=False, index=False)
    predict_data = pd.DataFrame(data=[[float(ys[i, j, 0]) for i in range(t_size)] for j in range(batch_size)])
    predict_data.to_csv(l2, header=False, index=False)
    truth_data2 = pd.DataFrame(data=[[float(ys_truth[i, j, 1]) for i in range(t_size)] for j in range(batch_size)])
    truth_data2.to_csv(l3, header=False, index=False)
    predict_data2 = pd.DataFrame(data=[[float(ys[i, j, 1]) for i in range(t_size)] for j in range(batch_size)])
    predict_data2.to_csv(l4, header=False, index=False)
    return model

for A in tqdm(range(31)):
    rnn = mytest(rnn, TEST0[A], TESTDATA[A],
                   '../results/repeat%d/rnn/ground_truth_verify_traj%d.csv' % (repeat,A),
                   '../results/repeat%d/rnn/predict_verify_traj%d.csv' % (repeat,A),
                   '../results/repeat%d/rnn/ground_truth_verify2_traj%d.csv' % (repeat,A),
                   '../results/repeat%d/rnn/predict_verify2_traj%d.csv' % (repeat,A))

endtime=time.time()
print('Running Time is: ',(endtime-starttime))





