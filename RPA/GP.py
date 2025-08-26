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


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
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
t_size = 360
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

for i in tqdm(range(31)):
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
    paras = np.load("../data/nparas%d.npy"%i)#Load parameters


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
def _mytest_(model,ys_truth,l1,l2,l3,l4):
    test_data = ys_truth
    ml = torch.rand([361, 50, 2])

    for li in range(361):
        for h in range(50):
            for l in range(2):
                ml[li][h][0] = test_data[li][h][0]
                ml[li][h][1] = test_data[li][h][1]

    test_x = ml[0:180, :, :].reshape([-1, 2])
    test_y = ml[1:, :, :].reshape([-1, 2])
    pred = gpr.predict(test_x)

    pred = pred.reshape([180, 50, 2])

    ys_truth_test = torch.tensor(ml[180:360])
    ys_test = torch.tensor(pred)
    loss = w2_decoupled(ys_test[:, :, :2], ys_truth_test[:, :, :2])
    print("test_W2 loss: ", loss)
def mytest(model,ys_truth,l1,l2,l3,l4):
    """
    :param model: trained model
    :param ys_truth: ground truth of testing datasets
    :param l1: path for saving results
    :param l2: path for saving results
    :param l3: path for saving results
    :param l4: path for saving results
    :return: testing results
    """
    test_data=ys_truth
    ml = torch.rand([361, 50, 2])

    for li in range(361):
        for h in range(50):
            for l in range(2):
                ml[li][h][0] = test_data[li][h][0]
                ml[li][h][1] = test_data[li][h][1]

    test_x = ml[0:180, :, :].reshape([-1, 2])
    test_y = ml[1:, :, :].reshape([-1, 2])
    pred = gpr.predict(test_x)

    pred=pred.reshape([180,50,2])


    ys_truth_test = torch.tensor(ml[180:360])
    ys_test = torch.tensor(pred)
    loss = w2_decoupled(ys_test[:, :, :2], ys_truth_test[:, :, :2])
    print("test_W2 loss: ", loss)
    truth_data = pd.DataFrame(data=[[float(ys_truth_test[i, j, 0]) for i in range(180)] for j in range(batch_size)])
    truth_data.to_csv(l1, header=False, index=False)
    predict_data = pd.DataFrame(data=[[float(ys_test[i, j, 0]) for i in range(180)] for j in range(batch_size)])
    predict_data.to_csv(l2, header=False, index=False)
    truth_data2 = pd.DataFrame(data=[[float(ys_truth_test[i, j, 1]) for i in range(180)] for j in range(batch_size)])
    truth_data2.to_csv(l3, header=False, index=False)
    predict_data2 = pd.DataFrame(data=[[float(ys_test[i, j, 1]) for i in range(180)] for j in range(batch_size)])
    predict_data2.to_csv(l4, header=False, index=False)
    return model

epoch=1
kernel = ( RBF(length_scale_bounds=(1e-5, 1e5)))
from joblib import dump
from joblib import load

for k in tqdm(range(31)):
    gpr = GaussianProcessRegressor(kernel=kernel, alpha=0.3, n_restarts_optimizer=5)#create GP model
    # s=load('gp%d.joblib' % k)
    # exit()
    for i0 in tqdm(range(epoch)):
        ys_truth = TRAINDATA[k]
        paras = TRAINPAS[k]
        ml = torch.rand([361,50, 2])

        for li in range(361):
            for h in range(50):
                for l in range(2):
                    ml[li][h][0] = ys_truth[li][h][0]
                    ml[li][h][1] = ys_truth[li][h][1]

        x=ml[0:180,:,:].reshape([-1,2])
        y=ml[180:360,:,:].reshape([-1,2])#construct training data as discussed in paper.

        gpr.fit(x,y)#model training process
        dump(gpr, '../results/repeat%d/gp/gp%d.joblib'%(repeat,k))
        if i0%50==0:
            gpr = _mytest_(gpr,  TESTDATA[k],
                         '../results/repeat%d/gp/ground_truth_verify_traj%d.csv' % (repeat,k),
                         '../results/repeat%d/gp/predict_verify_traj%d.csv' % (repeat,k),
                         '../results/repeat%d/gp/ground_truth_verify2_traj%d.csv' % (repeat,k),
                         '../results/repeat%d/gp/predict_verify2_traj%d.csv' % (repeat,k))
        if i0==(epoch-1):
            gpr = mytest(gpr,  TESTDATA[k],
                         '../results/repeat%d/gp/ground_truth_verify_traj%d.csv' % (repeat,k),
                         '../results/repeat%d/gp/predict_verify_traj%d.csv' % (repeat,k),
                         '../results/repeat%d/gp/ground_truth_verify2_traj%d.csv' % (repeat,k),
                         '../results/repeat%d/gp/predict_verify2_traj%d.csv' % (repeat,k))
endtime=time.time()
print('Running Time is: ',(endtime-starttime))