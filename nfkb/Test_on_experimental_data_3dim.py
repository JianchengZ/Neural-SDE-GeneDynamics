import csv
import time
import random
import ot
import pandas as pd
import torch
import torchsde
# restrict to 1 thread on cpu
torch.set_num_threads(1)
from torch import nn, relu

import numpy as np
from tqdm import tqdm


import argparse

# determine batch_hyper using argparse
parser = argparse.ArgumentParser()
parser.add_argument('--batch_hyper', type=int, default=16, help='batch_hyper')
parser.add_argument('--repeat', type=int, default=1, help='repeat')
batch_hyper = 16
repeat=1
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
    noise_type = 'general'
    sde_type = 'ito'

    def __init__(self):
        super().__init__()
        self.mu = TwoLayerNet(3, H,
                              2)
        self.sigma = TwoLayerNet(3, H,
                                 (2) * brownian_size)
        self.ptr = 0
        self.epochctr = 0

        self.ptr2 = 0
        self.epochctr2 = 0
        self.t_step = 0
        self.t_step2 = 0

    # Drift
    def f(self, t, y):

        lt = t.expand(y.size(0), 1)
        y = torch.cat([lt, y], dim=1)
        newy = y.clone().to(device)

        tmpf = self.mu(newy)
        spacev = torch.zeros([tmpf.shape[0], tmpf.shape[1]]).to(device)
        spacev[:, :2] = tmpf[:, :2]

        return spacev

    # Diffusion
    def g(self, t, y):
        lt = t.expand(y.size(0), 1)
        y = torch.cat([lt, y], dim=1)
        newy = y.clone().to(device)
        tmpg = torch.abs(self.sigma(newy).view(batch_size,
                                               2,
                                               brownian_size))
        spacev = torch.zeros([tmpg.shape[0], tmpg.shape[1], brownian_size]).to(device)
        spacev[:, :2, :] = tmpg[:, :2, :]

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

    def forward(self, y0):

        tmpy0 = torch.zeros([y0.shape[0], 2]).to(device)
        tmpy0[:, :] = y0
        y0 = tmpy0
        ys = self.encoder(y0)

        return ys


def distance(P, Q):
    cost_matrix = ot.dist(P.float(), Q.float(), metric='sqeuclidean')
    return cost_matrix


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


def train(Ys_truth, Y0, Paras, model, N):
    """
    :param K1,K2: list saving mean of data
    :param Ys_truth: ground truth of training datasets
    :param Y0: y0 of training datasets
    :param Paras: parameters of trajectories
    :param model: model that needs to train
    :param N: the amount of data group used for training
    :return: trained model
    """
    LOSS = []
    for k in (range(N)):
        y0 = Y0[k]
        paras = Paras[k]
        ys_truth = Ys_truth[k]
        ys, resy0, respas = model(y0)
        ys = torch.reshape(ys, (ys.shape[0], ys.shape[1], 2))

        loss = w2_decoupled(ys, ys_truth)
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
noiseidx=[]

def mytest(A,L0, L00,  model, y0_test, ys_truth, paras, l1, l2, l3, l4):
    """
    :param L0,L00: list for saving loss on testing datasets.
    :param A: index that is selected to determine we use which data.
    :param model: trained model
    :param y0_test: y0 of testing datasets
    :param ys_truth: ground truth of testing datasets
    :param paras: parameters of testing datasets
    :param l1: path for saving results
    :param l2: path for saving results
    :param l3: path for saving results
    :param l4: path for saving results
    :param noise1,noise2: list for saving paras
    :return: testing results
    """
    ys, resy0_test, respas_test = model(y0_test)

    ys = torch.reshape(ys, (ys.shape[0], ys.shape[1], 2))

    loss = w2_decoupled(ys, ys_truth)

    print("test_W2 loss: ", loss)
    L00.append(loss.detach().numpy())
    noise1.append(paras[0].detach().numpy())
    noise2.append(paras[1].detach().numpy())
    noiseidx.append(A)


    truth_data = pd.DataFrame(data=[[float(ys_truth[i, j, 0]) for i in range(t_size)] for j in range(batch_size)])
    truth_data.to_csv(l1, header=False, index=False)
    predict_data = pd.DataFrame(data=[[float(ys[i, j, 0]) for i in range(t_size)] for j in range(batch_size)])
    predict_data.to_csv(l2, header=False, index=False)
    truth_data2 = pd.DataFrame(data=[[float(ys_truth[i, j, 1]) for i in range(t_size)] for j in range(batch_size)])
    truth_data2.to_csv(l3, header=False, index=False)
    predict_data2 = pd.DataFrame(data=[[float(ys[i, j, 1]) for i in range(t_size)] for j in range(batch_size)])
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
randomindex =[53, 27, 79, 55, 42, 44, 12, 60, 26, 8, 89, 52, 38, 88, 119, 51, 13, 116, 78, 82, 77, 114, 93, 1, 83, 3, 112, 21, 17, 70, 30, 115, 107, 35, 24, 6, 15, 68, 40, 39, 11, 104, 5, 47, 50, 57, 92, 95, 101, 117, 105, 10, 76, 20, 46, 74, 4, 32, 94, 103, 98, 34, 45, 65, 62, 0, 75, 25, 66, 87, 14, 72, 100, 113, 96, 54, 111, 64, 48, 19, 23, 7, 58, 36, 43, 85, 67, 86, 118, 61, 41, 108, 99, 110, 9, 91, 120, 80, 84, 33, 49, 109, 18, 28, 71, 90, 106, 69, 22, 37, 2, 81, 31, 56, 102, 73, 63, 97, 16, 59, 29]




"""
testing on the experimental data
"""
tmpy0 = torch.zeros([50, state_size + noise_size]).to(device)


ys = torch.load(r'my_tensor.pt')


"""
do sort
"""
testing_data = pd.read_csv('TNF_NFKB_non_smooth_exp_rescale.csv').to_numpy()

tI = []
testing_data = testing_data[:, :31]

testing_data = torch.tensor(testing_data)

def cosine_similarity2(vec1, vec2):
    dot_product = torch.sum(vec1 * vec2, dim=-1)
    norm_vec1 = torch.norm(vec1, p=2, dim=-1)
    norm_vec2 = torch.norm(vec2, p=2, dim=-1)
    eps = 1e-8
    cosine_sim = dot_product / (norm_vec1 * norm_vec2 + eps)
    return cosine_sim
dis = [-cosine_similarity2(torch.reshape(ys, [t_size]), torch.reshape(tensor, [t_size])) for tensor in
       testing_data]
sorted_indices = sorted(range(len(dis)), key=lambda k: dis[k])




for i in range(32*(32//batch_hyper)):#32
    ti = testing_data[sorted_indices[batch_hyper * i:batch_hyper * (i + 1)], :]
    tI.append(ti)



for i in range(32*(32//batch_hyper)):
    tmp = tI[i]
    ys_truth_test = torch.tensor(tI[i]).float().to(device)
    TESTDATA.append(ys_truth_test)



"""
predict trajectories based on predicted noise.
"""

TRAINPAS = torch.load("results/repeat%d/en_de_temp%d/res%d.pt" % (repeat,batch_hyper,batch_hyper))

TRAIN0 = []

pl=[-2.231496916358205e-24,  # 1 stim
                    0.004208863742123,  # 2 IkBa
                    7.243145820414400e-05,  # 3 IkBan
                    0.073041047041980,  # 4 IkBaNFkB
                    3.797846073277533e-04,  # 5 IkBaNFkBn
                    1.904011434510926e-06,  # 6 IkBat
                    2.775518764090452e-07,  # 7 IKKIkBaNFkB
                    1.599344582923025e-08,  # 8 IKKIkBa
                    5.880889989729376e-04,  # 9 NFkB
                    0.021917267689900,  # 10 NFkBn
                    0.079998817578927,  # 11 IKK_off
                    7.999881757892686e-07,  # 12 IKK
                    8.888757508769650e-08,  # 13 IKK_i
                    -4.870769118528821e-43,  # 14 LPS
                    1.275626422798158,  # 15 CD14
                    -4.115469906928813e-41,  # 16 CD14LPS
                    -4.128676637845660e-41,  # 17 CD14LPSen
                    0.016215599102276,  # 18 TLR4
                    8.039816232771825e-04,  # 19 TLR4en
                    -7.153247040091529e-41,  # 20 TLR4LPS
                    -6.540006656455026e-41,  # 21 TLR4LPSen
                    0.100000000000000,  # 22 MyD88_off
                    -3.409493618824104e-62,  # 23 MyD88
                    0.100000000000000,  # 24 TRIF_off
                    2.553712654161919e-31,  # 25 TRIF
                    0.100000000000000,  # 26 TRAF6_off
                    8.171577283926864e-32,  # 27 TRAF6
                    0.019230769230769,  # 28 TNF
                    3.449664429530201e-04,  # 29 TNFR
                    -9.096526145120623e-48,  # 30 TNFR_TNF
                    0.035000000000000,  # 31 TTR
                    -7.518390367944331e-47,  # 32 C1_off
                    -4.399730501920834e-49,  # 33 C1
                    0.100000000000000,  # 34 TAK1_off
                    1.653829281361072e-33,  # 35 TAK1
                    0,  # 36 Pam3CSK
                    0.002499829692340,  # 37 TLR2
                    0,  # 38 CD14_P3CSK
                    0,  # 39 TLR2_P3CSK
                    9.193695496617080e-33,  # 40 polyIC
                    3.060193542961056e-31,  # 41 polyIC_en
                    0.004285714068304,  # 42 TLR3
                    2.553730109144204e-32,  # 43 TLR3_polyIC
                    1.583239309648607e-29,  # 44 CpG
                    1.667662468655748e-29,  # 45 CpG_en
                    0.004999659384679,  # 46 TLR9
                    4.842461745709273e-31,  # 47 TLR9_CpG
                    0.004999659384679,  # 48 TLR9_N
                    1.904011434510926e-06,  # 49 IkBat_cas1
                    1.087774852089695e-05,  # 50 IkBat_cas2
                    0.021917267689900,  # 51 NFkBn_cas1
                    0.021917267689900  # 52 NFkBn_cas2
                    ]
for i in range(1):
    tmp = []
    for j in range(1):
        tmp.append([x for x in pl])
    TRAIN0.append(tmp)
TRAIN0 = torch.tensor(TRAIN0)
encoder = torch.load(r'three_dim/repeat%d/en_de_temp32/big_loss_encoder2000.pth'%repeat )
tmpy0 = torch.zeros([32*(32//batch_hyper), 50, 54])
for k in (range(32*(32//batch_hyper))):
    for i in range(50):
        tmpy0[k][i][:52] = TRAIN0[0][0]
        tmpy0[k][i][52:] = TRAINPAS[k]
for k in tqdm(range(32*(32//batch_hyper))):
    y0 = tmpy0[k]

    ys_pred = encoder(y0[:,[4,9]])

    ys_pred = (torch.sum(ys_pred[:, :batch_hyper, :], dim=2)).T

    tmpd = TESTDATA[k]

    truth_data = pd.DataFrame(data=[[float(tmpd[i, j]) for j in range(t_size)] for i in range(batch_hyper)])
    truth_data.to_csv('three_dim/repeat%d/en_de_temp%d/real_truth_traj%d.csv' % (repeat,batch_hyper, k), header=True, index=False)
    predict_data = pd.DataFrame(data=[[float(ys_pred[i, j]) for j in range(t_size)] for i in range(batch_hyper)])
    predict_data.to_csv('three_dim/repeat%d/en_de_temp%d/real_pred_traj%d.csv' % (repeat,batch_hyper, k), header=True, index=False)

endtime=time.time()
print('Running Time is: ',(endtime-starttime))
