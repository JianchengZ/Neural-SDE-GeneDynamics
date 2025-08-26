# -*- coding: utf-8 -*-
"""
Created on Fri Dec  1 23:27:26 2023

@author: 12147
"""

# -*- coding: utf-8 -*-
import numpy

"""
Created on Sat Nov  4 22:32:02 2023

@author: 12147
"""

# -*- coding: utf-8 -*-
"""
Created on Sat Nov  4 21:15:22 2023

@author: 12147
"""

# -*- coding: utf-8 -*-
"""
Created on Sat Nov  4 12:31:48 2023

@author: 12147
"""

# -*- coding: utf-8 -*-
"""
Created on Sat Nov  4 00:35:03 2023

@author: 12147
"""

# -*- coding: utf-8 -*-
import sys

# 设置新的递归深度限制
sys.setrecursionlimit(1500)

import torch
import torchsde
import pandas as pd
import random
import time
import csv
from tqdm import tqdm
from math import sqrt
import pdb
from tqdm import tqdm
device = torch.device("cpu" if torch.cuda.is_available() else "cpu")

batch_size, state_size, brownian_size = 50, 2, 2
noise_size = 2
H = 100
t_size = 11
print(device)


def Circadian(sigma_para,sigma1, noise_type,k1,k2):
    #sigma_para:sigma_0
    #sigma1:sigma_1
    #k1,k2:k1,k2
    class TwoLayerNet(torch.nn.Module):
        def __init__(self, D_in, H, D_out):
            """
            In the constructor we instantiate two nn.Linear modules and assign them as
            member variables.
            """
            super(TwoLayerNet, self).__init__()
            self.Relu = torch.nn.ELU(inplace=True)
            self.linear1 = torch.nn.Linear(D_in, H)
            self.linear2 = torch.nn.Linear(H, H)
            self.linear3 = torch.nn.Linear(H, H)
            self.linear5 = torch.nn.Linear(H, D_out)

            # return data

        def forward(self, x):
            """
            In the forward function we accept a Tensor of input data and we must return
            a Tensor of output data. We can use Modules defined in the constructor as
            well as arbitrary operators on Tensors.
            """
            h_relu1 = self.linear1(x)
            h_relu1 = self.Relu(h_relu1)
            h_relu2 = self.linear2(h_relu1)
            h_relu2 = self.Relu(h_relu2)
            h_relu3 = self.linear3(h_relu2)
            h_relu3 = self.Relu(h_relu3)
            y_pred = self.linear5(h_relu3)
            return y_pred

    params = [0.19, 0.21]
    state_size=2
    # params = [0, 1]
    class SDE(torch.nn.Module):
        noise_type = 'general'
        sde_type = 'ito'

        def __init__(self):
            super().__init__()
            self.mu = TwoLayerNet(state_size + noise_size, H,  # torch.nn.Linear(state_size,
                                  state_size + noise_size)
            self.sigma = TwoLayerNet(state_size + noise_size, H,  # torch.nn.Linear(state_size,
                                     (state_size + noise_size) * brownian_size)  # torch.nn.Linear(state_size,
            # state_size * brownian_size)

        # Drift
        def f(self, t, y):
            t = t.expand(y.size(0), 1)
            ty = torch.cat([t, y], dim=1).to(device)

            newy = y.clone().to(device)
            tmpf = self.mu(newy)  # shape (batch_size, state_size)
            spacev = torch.zeros([tmpf.shape[0], tmpf.shape[1]]).to(device)
            # print(spacev.shape,tmpf.shape)
            spacev[:, :state_size] = tmpf[:, :state_size]

            return spacev  # shape (batch_size, state_size)

        # Diffusion
        def g(self, t, y):
            t = t.expand(y.size(0), 1)
            ty = torch.cat([t, y], dim=1).to(device)
            newy = y.clone().to(device)
            tmpg = torch.abs(self.sigma(newy).view(batch_size,
                                                   state_size + noise_size,
                                                   brownian_size))
            spacev = torch.zeros([tmpg.shape[0], tmpg.shape[1], brownian_size]).to(device)
            # spacev[:, :state_size, :] = tmpg[:, :state_size, :]
            spacev[:, :state_size, :] = tmpg[:, :state_size, :]
            # sigma_matrix = self.sigma(y).view(batch_size,
            #                           state_size,
            #                           brownian_size)

            # sigma_result = torch.zeros(batch_size, state_size, brownian_size)
            # sigma_result[:, 0, 0] = sigma_matrix[:, 0, 0]
            # sigma_result[:, 0, 1] = sigma_matrix[:, 1, 1] * sqrt(abs(params[1] / params[0]))
            # sigma_result[:, 1, 0] = -sigma_matrix[:, 0, 0] * sqrt(abs(params[1] / params[0]))
            # sigma_result[:, 1, 1] = sigma_matrix[:, 1, 1]
            # print(sigma_result[1, :, :])
            # pdb.set_trace()
            # for i in range(y.shape[0]):
            #    sigma_matrix[i, 0, 1] = 0
            #    sigma_matrix[i, 1, 0] = 0
            return spacev  # self.sigma(y).view(batch_size,
            #      state_size,
            #     brownian_size)

    a = 0.1
    b = 5
    sigma = 0.4  # 0.5
    v = 0.1
    from math import sqrt
    print('start')
    #
    # for each sigma = 0.02:0.02:0.18, each cor=-1:-.25:1
    # for each noise type const, linear, langevin

    # repeat 10 experiments, a total of 810 models
    sigma_para = sigma_para
    # cor = cor



    Sigma = [[0.1, -0.05], [-0.05, 0.05]]
    noise_type1 = noise_type

    class SDE_truth(torch.nn.Module):
        noise_type = 'general'
        sde_type = 'ito'

        def __init__(self):
            super().__init__()

        # Drift
        def f(self, t, y):
            # print(y.shape)
            f_truth = torch.zeros(y.shape[0], y.shape[1]).to(device)
            for i in range(y.shape[0]):
                f_truth[i, 0] = -params[0] * y[i, 0] - params[1] * y[i, 1]  # - y[i, 1]
                f_truth[i, 1] = params[1] * y[i, 0] - params[0] * y[i, 1]  # y[i, 0]

            return f_truth  # shape (batch_size, state_size)

        # Diffusion
        def g(self, t, y):
            nonlocal noise_type1
            g_truth = torch.zeros(y.shape[0], y.shape[1], brownian_size).to(device)
            for i in range(y.shape[0]):
                if noise_type1 == 'const':

                    g_truth[i, 0, 0] = y[i, 2]  # sqrt(abs(params[0]*y[i, 0])) #* sqrt(t+1)
                    g_truth[i, 0, 1] = y[i, 2] * y[i, 3]  # sqrt(abs(params[1]*y[i, 1]))#Sigma[0][1] * y[i, 1]
                    g_truth[i, 1, 0] = y[i, 2] * y[i, 3]
                    # g_truth[i, 1, 0] = -sqrt(abs(params[1]*y[i, 0]))#Sigma[1][0] * y[i, 0] #* sqrt(t+1)
                    g_truth[i, 1, 1] = y[i, 2]  # sqrt(abs(params[0]*y[i, 1]))#Sigma[1][1] * y[i, 1]
                elif noise_type1 == 'linear':
                    g_truth[i, 0, 0] = y[i, 2] * y[i, 0]  # * sqrt(t+1)
                    g_truth[i, 0, 1] = y[i, 2] * y[i, 3] * y[i, 1]  # sqrt(abs(params[1]*y[i, 1]))#Sigma[0][1] * y[i, 1]
                    g_truth[i, 1, 0] = y[i, 2] * y[i, 3] * y[i, 0]
                    # g_truth[i, 1, 0] = -sqrt(abs(params[1]*y[i, 0]))#Sigma[1][0] * y[i, 0] #* sqrt(t+1)
                    g_truth[i, 1, 1] = y[i, 2] * y[i, 0]
                elif noise_type1 == 'langevin':

                    # g_truth[i, 0, 0] = y[i, 2] * sqrt(abs(y[i, 0]))  # * sqrt(t+1)
                    g_truth[i, 0, 0] = (k1 + k2 * y[i, 2]) * sqrt(abs(y[i, 0]))

                    # g_truth[i, 0, 1] = y[i, 2] * y[i, 3] * sqrt(
                    #     abs(y[i, 1]))  # sqrt(abs(params[1]*y[i, 1]))#Sigma[0][1] * y[i, 1]
                    g_truth[i, 0, 1] = 0#y[i, 2]

                    # g_truth[i, 1, 0] = y[i, 2] * y[i, 3] * sqrt(abs(y[i, 0]))
                    g_truth[i, 1, 0] = 0#y[i, 3]*k1

                    # g_truth[i, 1, 1] = y[i, 2] * sqrt(abs(y[i, 0]))
                    g_truth[i, 1, 1] = (k1 + k2 * y[i, 3]) * sqrt(abs(y[i, 0]))
            
            #import pdb
            #print(g_truth, noise_type)
            #pdb.set_trace()
            return g_truth

    for repeat in range(1):
        sde = SDE().to(device)
        sde_truth = SDE_truth().to(device)
        noise_now = [sigma_para, sigma1]
        noise_now=torch.tensor(numpy.array(noise_now))
        # print(noise_now.shape)/
        # exit()
        state_size = 2

        t_end = 1#2  # 10
        ts = torch.linspace(0, t_end, t_size).to(device)
        ys_truth = [[0 for j in range(len(noise_now[1]))] for i in range(len(noise_now[0]))]
        # print(torch.tensor(ys_truth).shape)
        # exit()
        for i1 in range(len(noise_now[0])):
            for i2 in range(len(noise_now[1])):
                y0 = torch.zeros(batch_size, state_size + noise_size).to(device)
                for i in range(batch_size):
                    y0[i, 0] = 0  # 0.5 #+ float(torch.randn(1)[0] * 0.5) #csv_data[i][0]
                    y0[i, 1] = 1

                    # y0[i, 2] = noise_now[0][i1]
                    # y0[i, 3] = noise_now[1][i2]
                    y0[i, 2]=noise_now[0][i1]
                    y0[i,3]=noise_now[1][i2]


                ys_truth[i1][i2] = torchsde.sdeint(sde_truth, y0, ts).to(device)

        # print('start')
        for i1 in range(len(noise_now[0])):
            for i2 in range(len(noise_now[1])):
                truth_data = pd.DataFrame(data=[[float(ys_truth[i1][i2][i, j, 0]) for i in range(t_size)] for j in range(batch_size)])
                truth_data.to_csv(
                    r'../results/ground_truth_cir1_damped2cuda' + '_' + noise_type + '_' + str(
                        i1) + '_' + str(i2) + '_' + str(repeat) + '_' +str(k1) +'_'+str(k2)+'_'+'.csv', header=False, index=False)

                truth_data = pd.DataFrame(data=[[float(ys_truth[i1][i2][i, j, 1]) for i in range(t_size)] for j in range(batch_size)])
                truth_data.to_csv(
                    r'../results/ground_truth_cir2_damped2cuda' + '_' + noise_type + '_' + str(
                        i1) + '_' + str(i2) + '_' + str(repeat) + '_'+str(k1) +'_'+str(k2)+'_' + '.csv', header=False, index=False)

        criterion = torch.nn.MSELoss(reduction='sum')
        optimizer = torch.optim.Adam(sde.parameters(), lr=0.002, betas=(0.9, 0.999), weight_decay=0.005)
        epoch = 500
        loss_list = []
        ttime = time.time()
        g_error_list = []
        sigma_error_list = []

        from math import pi

        import ot
        def distance(P, Q):
            cost_matrix = ot.dist(P, Q, metric='sqeuclidean')
            return cost_matrix

        def w2_decoupled(y, y_pred):

            batch_size = y.shape[1]
            state_size = y.shape[2]
            t_size = y.shape[0]
            loss = 0
            for i in range(1, t_size):
                weights = torch.tensor([1 / batch_size for _ in range(batch_size)]).to(device)

                loss += ot.emd2(weights, weights, distance(y[i, :, :state_size], y_pred[i, :, :state_size]))

            return loss

        import pdb

        error_f = []
        error_s = []

        def total_error1(sde_truth, sde, i0):
            nonlocal noise_now, state_size

            f_error = [[0 for i in range(len(noise_now[1]))] for j in range(len(noise_now[0]))]
            s_error = [[0 for i in range(len(noise_now[1]))] for j in range(len(noise_now[0]))]
            for i1 in range(len(noise_now[0])):
                for i2 in range(len(noise_now[1])):
                    for i in range(t_size):
                        summ = 0
                        summ_ref = 0
                        summ += torch.sum(torch.abs(sde.f(ts[i], ys_truth[i1][i2][i,:, :]) - sde_truth.f(ts[i], ys_truth[
                                                                                                                     i1][
                                                                                                                     i2][
                                                                                                                 i,:,
                                                                                                                 :])))
                        summ_ref += torch.sum(torch.abs(sde_truth.f(ts[i], ys_truth[i1][i2][i,:,
                                                                 :])))  # + torch.sum(torch.abs(sde.f(ts[i], ys_truth[i])))

                    f_error[i1][i2] = summ / summ_ref

            for i1 in range(len(noise_now[0])):
                for i2 in range(len(noise_now[1])):
                    summ1 = 0
                    summ1_ref = 0
                    for i in range(t_size):
                        # print(ys_truth[i1][i2].shape)
                        s1 = sde_truth.g(ts[i], ys_truth[i1][i2][i,:, :])[:, :state_size, :]
                        #print(s1)
                        #pdb.set_trace()
                        s2 = sde.g(ts[i], ys_truth[i1][i2][i,:, :])[:, :state_size, :]
                        for j in range(ys.shape[1]):
                            summ1_ref += torch.sum(torch.abs(torch.matmul(s1[j], s1[j].transpose(0,
                                                                                                 1))))  # + torch.sum(torch.abs(torch.matmul(s2[j], s2[j].transpose(0, 1))))
                            summ1 += torch.sum(torch.abs(
                                torch.abs(torch.matmul(s1[j], s1[j].transpose(0, 1))) - torch.abs(torch.matmul(s2[j],
                                                                                          s2[j].transpose(0, 1)))))

                    s_error[i1][i2] =  summ1 / summ1_ref

            print(f_error,  s_error)
            return f_error, s_error

    

        for i0 in tqdm(range(epoch)):
            # print(i)
            # ys_truth = [[0 for j in range(len(noise_now[1]))] for i in range(len(noise_now[0]))]
            i1=len(noise_now[0])
            i2=len(noise_now[1])
            loss_list = torch.zeros(i1, i2)
            for i1 in range(len(noise_now[0])):
                for i2 in range(len(noise_now[1])):
            # for i1 in range(1):
            #     for i2 in range(1):
                    y0 = torch.zeros(batch_size, state_size + noise_size).to(device)
                    for i in range(batch_size):
                        y0[i, 0] = 0  # 0.5 #+ float(torch.randn(1)[0] * 0.5) #csv_data[i][0]
                        y0[i, 1] = 1
                        y0[i, 2] = noise_now[0][i1]
                        y0[i, 3] = noise_now[1][i2]



                    ys = torchsde.sdeint(sde, y0, ts).to(device)
                    # print(ys[:, :, :2].shape,ys_truth[i1][i2][:, :, :2].shape)
                    loss_list[i1, i2] += w2_decoupled(ys[:, :, :2], ys_truth[i1][i2][:, :, :2])

            loss = torch.sum(loss_list)

            if i0 % 10 == 0:
                print(i0, loss.item(), time.time() - ttime)

                f_error, s_error = total_error1(sde_truth, sde, i0)
                error_f.append(f_error)
                error_s.append(s_error)
                ttime = time.time()

            # loss_list.append(loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        torch.save(sde.state_dict(), r'../results/model' + '_' + noise_type +  '_'+ str(k1) + str(k2) + str(repeat) + '_' + '.pkl')
        error_f = []
        error_s = []
        i1 = len(noise_now[0])
        i2 = len(noise_now[1])
        loss_list = torch.zeros(i1, i2)

        for i1 in range(len(noise_now[0])):
            for i2 in range(len(noise_now[1])):

                y0 = torch.zeros(batch_size, state_size + noise_size).to(device)
                for i in range(batch_size):
                    y0[i, 0] = 0  # 0.5 #+ float(torch.randn(1)[0] * 0.5) #csv_data[i][0]
                    y0[i, 1] = 1
                    y0[i, 2] = noise_now[0][i1]
                    y0[i, 3] = noise_now[1][i2]


                ys = torchsde.sdeint(sde, y0, ts).to(device)
                # print(loss_list[i1,i2])

                loss_list[i1, i2] += w2_decoupled(ys[:, :, :2], ys_truth[i1][i2][:, :, :2])
                predict_data = pd.DataFrame(data=[[float(ys[i, j, 0]) for i in range(t_size)] for j in range(batch_size)])
                predict_data.to_csv(
                    r'../results/predict2d1_cir_damped2cuda' + '_' + noise_type + '_' + str(
                        i1) + '_' + str(i2) + '_'+ str(k1) + str(k2) + str(repeat) + '_' + '.csv', header=False, index=False)

                predict_data = pd.DataFrame(data=[[float(ys[i, j, 1]) for i in range(t_size)] for j in range(batch_size)])
                predict_data.to_csv(
                    r'../results/predict2d2_cir_damped2cuda' + '_' + noise_type + '_' + str(
                        i1) + '_'+ str(k1) + str(k2) + str(i2) + '_' + str(repeat) + '_' + '.csv', header=False, index=False)
        f_error, s_error = total_error1(sde_truth, sde, 0)
        error_f.append(f_error)
        error_s.append(s_error)
        loss_data = pd.DataFrame(data=loss_list)
        loss_data.to_csv(
                    r'../results/loss2d_cir_damped2cuda' + '_' + noise_type +'_' + str(repeat) + str(k1) + str(k2) + '_' + '.csv', header=False, index=False)


        loss_data = pd.DataFrame(data=error_f)
        loss_data.to_csv(
                    r'../results/f_error_cir_damped2cuda' + '_' + noise_type + '_'  + str(repeat)+ str(k1) + str(k2) + '_' + '.csv', header=False, index=False)

        loss_data = pd.DataFrame(data=error_s)
        loss_data.to_csv(
                    r'../results/s_error_cir_damped2cuda' + '_' + noise_type +  '_' + str(repeat)+ str(k1) + str(k2) + '_' + '.csv', header=False, index=False)


if __name__ == '__main__':
    starttime=time.time()
    sigma_para = numpy.arange(0.1, 0.51, 0.05)  # 0.1:0.05:0.5
    cor = numpy.arange(-1, 1.0001, 0.25)  # -1:-.25:1
    x=[]
    for i in range(9):
        c=cor[i]
        if c == 0:
            x.append(1)
        else:
            x.append((2 - sqrt(4 - 4 * c ** 2)) / 2 / c)
    # noise_type_list = ['langevin', 'const', 'linear'  ]
    noise_type_list = ['langevin']
    # print(cor)
    k1=[-1, -0.5, 0., 0.5, 1]
    k2=[-1, -0.5, 0., 0.5, 1]
    for k in range(1):
        for i in range((3)):
            for j in range((3)):
                Circadian(sigma_para=k1, sigma1=k2,
                                  noise_type=noise_type_list[k],k1=0.1+0.1*i,k2=0.05*j)
                # exit()
    endtime=time.time()
    print('Totol time: ',(endtime-starttime))