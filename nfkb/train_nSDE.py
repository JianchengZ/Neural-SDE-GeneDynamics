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
parser.add_argument('--batch_hyper', type=int, default=32, help='batch_hyper')
parser.add_argument('--repeat', type=int, default=1, help='repeat')
args = parser.parse_args()
batch_hyper = args.batch_hyper
repeat=args.repeat
starttime=time.time()

class TwoLayerNet(torch.nn.Module):
    def __init__(self, D_in, H, D_out):
        """
        In the constructor we instantiate two nn.Linear modules and assign them as
        member variables.
        """
        super(TwoLayerNet, self).__init__()
        self.linear1 = torch.nn.Linear(D_in, H)
        self.linearmid = torch.nn.Linear(H, H)
        self.linear2 = torch.nn.Linear(H, D_out)

    def forward(self, x):
        """
        In the forward function we accept a Tensor of input data and we must return
        a Tensor of output data. We can use Modules defined in the constructor as
        well as arbitrary operators on Tensors.
        """
        h_relu1 = self.linear1(x)
        h_relu1 = torch.relu(h_relu1)
        h_relu1 = torch.relu(self.linearmid(h_relu1))
        y_pred = self.linear2(h_relu1)

        return y_pred


class SDE(torch.nn.Module):
    """
    The class is used to constructing the SDE model to fit the data.
    For the drift part and diffusion part, they are same as the paper desciption.
    """
    noise_type = 'general'
    sde_type = 'ito'

    def __init__(self):
        super().__init__()
        self.mu = TwoLayerNet(state_size + noise_size, H,
                              state_size + noise_size)
        self.sigma = TwoLayerNet(state_size + noise_size, H,
                                 (state_size + noise_size) * brownian_size)
        self.ptr = 0
        self.epochctr = 0
        self.ptr2 = 0
        self.epochctr2 = 0
        self.t_step = 0
        self.t_step2 = 0

    # Drift
    def f(self, t, y):
        newy = y.clone().to(device)
        tmpf = self.mu(newy)
        spacev = torch.zeros([tmpf.shape[0], tmpf.shape[1]]).to(device)

        spacev[:, :state_size] = tmpf[:, :state_size]

        return spacev

    # Diffusion
    def g(self, t, y):
        newy = y.clone().to(device)

        tmpg = torch.abs(self.sigma(newy).view(batch_size,
                                               state_size + noise_size,
                                               brownian_size))
        spacev = torch.zeros([tmpg.shape[0], tmpg.shape[1], brownian_size]).to(device)
        spacev[:, [5, 8, 9], :] = tmpg[:, [5, 8, 9], :]
        return spacev


class encoder(nn.Module):
    """
    The class is used to solve the sde model.
    """
    def __init__(self, ts):
        super(encoder, self).__init__()
        self.sde = SDE()
        self.ts = ts

    def forward(self, y0):
        ys = torchsde.sdeint(self.sde, y0, self.ts)

        return ys


class en_de(nn.Module):
    """
    The class is the model for training.
    """
    def __init__(self, ts):
        super(en_de, self).__init__()
        self.encoder = encoder(ts)


    def forward(self, y0, paras):
        tmpy0 = torch.zeros([y0.shape[0], state_size + noise_size]).to(device)
        tmpy0[:, state_size:] = paras
        tmpy0[:, :state_size] = y0
        y0 = tmpy0
        ys = self.encoder(y0)
        return ys


def distance(P, Q):
    cost_matrix = ot.dist(P.float(), Q.float(), metric='sqeuclidean')
    return cost_matrix


def distance(P, Q):
    cost_matrix = ot.dist(P, Q, metric='sqeuclidean')
    return cost_matrix


def w2_decoupled(y, y_pred, Train):
    batch_size = y.shape[1]
    state_size = y.shape[2]
    t_size = y.shape[0]
    loss = 0

    for i in range(1, t_size):
        weights = torch.tensor([1 / batch_size for _ in range(batch_size)])

        if Train == True:
            loss += (ot.emd2(weights.to(device), weights.to(device),
                             distance((y[i, :, :]).float(), (y_pred[i, :, :]).float())))
        else:
            loss += (ot.emd2(weights.to(device), weights.to(device),
                             distance((y[i, :, :]).float(), (y_pred[i, :, :]).float())))

    return loss


def relative_error(p, PARAS):
    error = (abs(p[0] - PARAS[0]) + abs(p[1] - PARAS[1])) / (abs(PARAS[0]) + abs(PARAS[1]))

    return error


batch_size, state_size, brownian_size = 50, 52, 2
noise_size = 2
H = 200
t_size = 31
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = "cpu"
ts = torch.linspace(0, 2.5, t_size)
ts = ts.to(device)

model = en_de(ts)
model = model.to(device)
criterion = torch.nn.MSELoss(reduction='sum')
optimizer = torch.optim.Adam(model.encoder.parameters(), lr=0.002)
epoch = 2001

ttime = time.time()


def train( Ys_truth, Y0, Paras, model, N):
    """
    :param Ys_truth: The ground truth of trajectories
    :param Y0: The initial values of trajectories.
    :param Paras: The noise used to compute the SDE.
    :param model: The model we have to fit data.
    :param N: The number of noise combinations.
    :return: trained well model.
    """
    LOSS = []
    for k in (range(N)):
        y0 = Y0[k]
        paras = Paras[k]
        ys_truth = Ys_truth[k]
        ys = model(y0, paras)
        ys = torch.reshape(ys, (ys.shape[0], ys.shape[1], 52 + noise_size))


        loss = w2_decoupled(ys[:, :, [4, 9]], ys_truth[:, :, [4, 9]],
                            Train=True)
        LOSS.append(loss)

    loss = LOSS[0]
    for l in range(N - 1):
        loss += LOSS[l + 1]

    optimizer.zero_grad()
    loss.backward()
    nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()

    return model


noise1 = []
noise2 = []
L0 = []
L00 = []
noiseidx = []


def mytest(A, L0, L00,model, y0_test, ys_truth, paras, l1, l2, l3, l4, noise1, noise2):
    """
    :param model: trained model
    :param y0_test: The initial values of trajectories.
    :param ys_truth: The ground truth of trajectories.
    :param paras: The noise for calculating the SDE.
    :param l1-l4: The save path for results.
    :param of rest: They exist for saving results
    :return:
    """
    ys = model(y0_test, paras)

    ys = torch.reshape(ys, (ys.shape[0], ys.shape[1], 52 + noise_size))


    loss = w2_decoupled(ys[:, :, [4, 9]], ys_truth[:, :, [4, 9]],
                        Train=False)

    print("test_W2 loss: ", loss)
    L00.append(loss.detach().numpy())
    noise1.append(paras[0].detach().numpy())
    noise2.append(paras[1].detach().numpy())
    noiseidx.append(A)
    truth_data = pd.DataFrame(data=[[float(ys_truth[i, j, 4]) for i in range(t_size)] for j in range(batch_size)])
    truth_data.to_csv(l1, header=False, index=False)
    predict_data = pd.DataFrame(data=[[float(ys[i, j, 4]) for i in range(t_size)] for j in range(batch_size)])
    predict_data.to_csv(l2, header=False, index=False)
    truth_data2 = pd.DataFrame(data=[[float(ys_truth[i, j, 9]) for i in range(t_size)] for j in range(batch_size)])
    truth_data2.to_csv(l3, header=False, index=False)
    predict_data2 = pd.DataFrame(data=[[float(ys[i, j, 9]) for i in range(t_size)] for j in range(batch_size)])
    predict_data2.to_csv(l4, header=False, index=False)

    return model


TRAINDATA = []
TRAIN0 = []
TRAINPAS = []
TESTDATA = []
TEST0 = []
TESTPAS = []
K1 = []
K2 = []
nolist = [i for i in range(121)]
randomindex = random.sample(nolist, 121)
defaultindex = nolist
randomindex = [53, 27, 79, 55, 42, 44, 12, 60, 26, 8, 89, 52, 38, 88, 119, 51, 13, 116, 78, 82, 77, 114, 93, 1, 83, 3,
               112, 21, 17, 70, 30, 115, 107, 35, 24, 6, 15, 68, 40, 39, 11, 104, 5, 47, 50, 57, 92, 95, 101, 117, 105,
               10, 76, 20, 46, 74, 4, 32, 94, 103, 98, 34, 45, 65, 62, 0, 75, 25, 66, 87, 14, 72, 100, 113, 96, 54, 111,
               64, 48, 19, 23, 7, 58, 36, 43, 85, 67, 86, 118, 61, 41, 108, 99, 110, 9, 91, 120, 80, 84, 33, 49, 109,
               18, 28, 71, 90, 106, 69, 22, 37, 2, 81, 31, 56, 102, 73, 63, 97, 16, 59, 29]



for D in tqdm(range(121)):

    file_path = 'data/ndata_tmp_e%d.csv' % D

    csv_data = []
    csvreader = csv.reader(file_path)
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
    csv_data = (np.array(csv_data))
    csv_data = np.reshape(csv_data, [100, -1, 52])
    csv_data = csv_data[:, :151, :]

    paras = np.load("data/nparas%d.npy" % D)
    paras = torch.tensor(paras)
    paras = paras.float()
    paras = paras.to(device)

    train_data = csv_data[0::2]
    train_mean, train_std = np.mean(train_data), np.std(train_data)

    y0 = torch.tensor(train_data[:, 0, :]).float().to(device)
    ys_truth = torch.zeros(t_size, batch_size, state_size)
    ys_truth.to(device)
    train_data = np.transpose(train_data, (1, 0, 2))
    ys_truth = torch.tensor(train_data).float().to(device)
    TRAIN0.append(y0)
    TRAINPAS.append(paras)

    TRAINDATA.append(ys_truth[::5])

    test_data = csv_data[1::2]
    K1.append(train_mean)
    K2.append(train_std)
    y0_test = torch.tensor(test_data[:, 0, :]).float().to(device)
    TEST0.append(y0_test)
    ys_truth_test = torch.zeros(t_size, batch_size, state_size)
    test_data = np.transpose(test_data, (1, 0, 2))
    ys_truth_test = torch.tensor(test_data).float().to(device)
    TESTDATA.append(ys_truth_test[::5])
    TESTPAS.append(paras)

for i0 in tqdm(range(epoch), colour='green'):

    model = train(TRAINDATA, TRAIN0, TRAINPAS, model, N=96)
    if i0 % 50 == 0:
        torch.save(model.encoder, "results/repeat%d/en_de_temp%d/big_loss_encoder%d.pth" % (repeat,batch_hyper, (i0)))

IDX = []

for t in range(1):
    for aA in tqdm(range(121)):
        A = randomindex[aA]
        IDX.append(A)
        model = mytest(A, L0, L00,  model, TEST0[A], TESTDATA[A], TESTPAS[A],
                       'results/repeat%d/en_de_temp%d/ground_truth_verify_traj%d.csv' % (repeat,batch_hyper, A),
                       'results/repeat%d/en_de_temp%d/predict_verify_traj%d.csv' % (repeat,batch_hyper, A),
                       'results/repeat%d/en_de_temp%d/ground_truth_verify2_traj%d.csv' % (repeat,batch_hyper, A),
                       'results/repeat%d/en_de_temp%d/predict_verify2_traj%d.csv' % (repeat,batch_hyper, A), noise1, noise2)

np.save(r'results/repeat%d/en_de_temp%d/noise1%d.npy' % (repeat,batch_hyper, batch_hyper), np.array(noise1))
np.save(r'results/repeat%d/en_de_temp%d/noise2%d.npy' % (repeat,batch_hyper, batch_hyper), np.array(noise2))
np.save(r'results/repeat%d/en_de_temp%d/noise_error%d.npy' % (repeat,batch_hyper, batch_hyper), np.array(L0))
np.save(r'results/repeat%d/en_de_temp%d/w2_error%d.npy' % (repeat,batch_hyper, batch_hyper), np.array(L00))
np.save(r'results/repeat%d/en_de_temp%d/noiseidx%d.npy' % (repeat,batch_hyper, batch_hyper), np.array(noiseidx))
np.save(r'results/repeat%d/en_de_temp%d/IDX%d.npy' % (repeat,batch_hyper, batch_hyper), np.array(IDX))
endtime=time.time()
print('Running Time is: ',(endtime-starttime))