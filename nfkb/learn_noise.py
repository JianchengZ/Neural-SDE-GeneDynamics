
import csv
import time
import random
from idlelib import testing
from statistics import mean

import ot
import numpy as np
# from sklearn.preprocessing import StandardScaler, MinMaxScaler
import pandas as pd
import torch
import torchsde

# restrict to 1 thread on cpu
torch.set_num_threads(1)
from torch import nn, relu, cosine_similarity, optim
import math
import numpy as np
from tqdm import tqdm
import torch.nn.functional as F

import argparse

# determine batch_hyper using argparse
parser = argparse.ArgumentParser()
parser.add_argument('--batch_hyper', type=int, default=1, help='batch_hyper')
parser.add_argument('--repeat', type=int, default=1, help='repeat')
parser.add_argument('--t_size',type=int,default=2)
args = parser.parse_args()
batch_hyper = args.batch_hyper
repeat=args.repeat
starttime=time.time()
import random
global_dict={'16':10,'11':15,'8':20,'6':30,'31':5,'3':75,'2':150}

indices = random.sample(range(0, 121), 96)
indices = [53, 27, 79, 55, 42, 44, 12, 60, 26, 8, 89, 52, 38, 88, 119, 51, 13, 116, 78, 82, 77, 114, 93, 1, 83, 3, 112,
           21, 17, 70, 30, 115, 107, 35, 24, 6, 15, 68, 40, 39, 11, 104, 5, 47, 50, 57, 92, 95, 101, 117, 105, 10, 76,
           20, 46, 74, 4, 32, 94, 103, 98, 34, 45, 65, 62, 0, 75, 25, 66, 87, 14, 72, 100, 113, 96, 54, 111, 64, 48, 19,
           23, 7, 58, 36, 43, 85, 67, 86, 118, 61, 41, 108, 99, 110, 9, 91, 120, 80, 84, 33, 49, 109, 18, 28, 71, 90,
           106, 69, 22, 37, 2, 81, 31, 56, 102, 73, 63, 97, 16, 59, 29]


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

t_size = args.t_size
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = "cpu"
ts = torch.linspace(0, 2.5, t_size)
ts = ts.to(device)


class MLP(nn.Module):
    """
    The class can learn noise from trajectories. And we will sort data and add attention mechanism into the model.
    The concrete process can be found in the paper.
    """
    def __init__(self):
        super(MLP, self).__init__()

        self.fc1 = nn.Linear(t_size * batch_hyper, 64)

        self.fc2 = nn.Linear(64, 128)

        self.fc3 = nn.Linear(128, 2)
        self.query_layer = nn.Linear(t_size, t_size)
        self.key_layer = nn.Linear(batch_hyper, batch_hyper)
        self.value_layer = nn.Linear(batch_hyper, batch_hyper)

    def forward(self, tensor_2d, ys, k, s):
        num_vectors = tensor_2d.shape[1]
        similarities = torch.zeros(num_vectors)
        """
        分组
        """
        for i in range(num_vectors):
            # cos_sim = F.cosine_similarity(tensor_2d[:, i].unsqueeze(0), ys.unsqueeze(0), dim=1)
            similarities[i] = -i
        top_k_indices = torch.topk(similarities, k=k * s).indices[k * (s - 1):]
        top_k_vectors = tensor_2d[:, top_k_indices]


        query = self.query_layer(ys).unsqueeze(1)  # [input_dim] -> [input_dim, 1]
        keys = self.key_layer(top_k_vectors)  # [input_dim, num_keys]
        values = self.value_layer(top_k_vectors)  # [input_dim, num_keys]


        attention_scores = torch.matmul(query.T, keys) / (keys.shape[0] ** 0.5)  # [1, num_keys]
        attention_weights = F.softmax(attention_scores, dim=-1)  # [1, num_keys]
        values = values * attention_weights

        x = F.relu(self.fc1(values.T.reshape([batch_hyper * t_size])))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


model = MLP().to(device)

criterion = torch.nn.MSELoss(reduction='sum')
optimizer = torch.optim.Adam(model.parameters(), lr=0.002)
epoch =4001 #4001

ttime = time.time()
P_ = []

TRAINDATA = []
TRAIN0 = []
TRAINPAS = []
TESTDATA = []
TEST0 = []
TESTPAS = []
K1 = []
K2 = []
nolist = [i for i in range(121)]

"""
Load simulation data
"""
for D in tqdm(range(121)):

    file_path = 'data/ndata_tmp_e%d.csv' % indices[D]

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

    csv_data = csv_data[:, ::int((global_dict[f'{args.t_size}']) / 5)]
    paras = np.load("data/nparas%d.npy" % indices[D])

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
    ys_truth = torch.sum(ys_truth, dim=2)
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
    ys_truth_test = torch.sum(ys_truth_test, dim=2)
    TESTDATA.append(ys_truth_test[::5])
    TESTPAS.append(paras)
"""
Test the simulation.
"""

"""
load and test on experimental data.
"""
tmpy0 = torch.zeros([50, state_size + noise_size]).to(device)
ys = torch.load(r'my_tensor.pt')
ys=ys[::int((global_dict[f'{args.t_size}']) / 5)]
"""
do sort using the cos-similarity
"""

testing_data = pd.read_csv('TNF_NFKB_non_smooth_exp_rescale.csv').to_numpy()

tI = []
testing_data = testing_data[:, :31]
testing_data = testing_data[:, ::int((global_dict[f'{args.t_size}'])/5)]

knnidx = [i for i in range(1054)]
testing_data = torch.tensor(testing_data)
def cosine_similarity2(vec1, vec2):
    # 计算点积
    dot_product = torch.sum(vec1 * vec2, dim=-1)

    # 计算向量的范数 (欧几里得范数)
    norm_vec1 = torch.norm(vec1, p=2, dim=-1)
    norm_vec2 = torch.norm(vec2, p=2, dim=-1)

    # 防止除以0
    eps = 1e-8
    cosine_sim = dot_product / (norm_vec1 * norm_vec2 + eps)

    return cosine_sim


dis = [-cosine_similarity2(torch.reshape(ys, [t_size]), torch.reshape(tensor, [t_size])) for tensor in
       testing_data]

sorted_indices = sorted(range(len(dis)), key=lambda k: dis[k])
testing_data=testing_data[sorted_indices]
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.003, weight_decay=1e-4)


for i in range(epoch):
    loss = 0
    for j_ in range(96):
        j=j_
        optimizer.zero_grad()
        output = model(TRAINDATA[j],ys,batch_hyper,1)
        loss += criterion(output, TRAINPAS[j]) / 96
    loss.backward()
    # print("epoch%d loss:%.4f" % (i, loss.item()))
    optimizer.step()
    if i % 50 == 0:
        relative_errors = []
        with torch.no_grad():
            for j in range(len(TESTDATA)):
                test_data = TESTDATA[j]
                test_output = model(test_data, ys, batch_hyper, 1)
                relative_loss0 = relative_error(torch.tensor(test_output), torch.tensor(TESTPAS[j]))
                relative_errors.append(relative_loss0.item())
        # print mean and std
        print("relative errors:", np.mean(relative_errors), np.std(relative_errors))

torch.save(model, "results/repeat%d/en_de_temp%d/model%d_%d.pth" % (repeat,batch_hyper, batch_hyper,args.t_size))
res = []
noise_error = []
with torch.no_grad():
    for j in range(len(TESTDATA)):
        test_outputs = model(TESTDATA[j], ys, batch_hyper, 1)
        relative_loss0 = relative_error(torch.tensor(test_outputs), torch.tensor(TESTPAS[j]))
        noise_error.append(relative_loss0)
        print("Test outputs:", test_outputs[0].item(), test_outputs[1].item(), "truth:", TESTPAS[j][0].item(), TESTPAS[j][1].item(), "test_loss:", relative_loss0)

    np.save(r'results/repeat%d/en_de_temp%d/noise_err%d_%d.npy' % (repeat,batch_hyper, batch_hyper,args.t_size), np.array(noise_error))
    for j in tqdm(range(32*(32//batch_hyper))):
        test_outputs = model(torch.tensor(testing_data).float().T, ys, batch_hyper, j + 1)
        res.append(test_outputs)
torch.save(res, 'results/repeat%d/en_de_temp%d/res%d_%d.pt'%(repeat,batch_hyper, batch_hyper,args.t_size))



endtime=time.time()
print('Running Time is: ',(endtime-starttime))




