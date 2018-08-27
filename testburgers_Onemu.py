#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Aug  7 22:55:32 2018

@author: xuping
"""

#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Aug  7 00:12:39 2018

@author: xuping
"""

#import tensorflow as tf
import numpy as np
import scipy.io
from scipy.integrate import odeint
import sys
sys.path.insert(0, '../MultistepNNs')

from Multistep_NN import Multistep_NN
from numpy import linalg as LA
import matplotlib.pyplot as plt


#filename = 'Data/Burgers_cond2_0.005.mat'
#filename = 'Data/Onemu_Training.mat'
filename = 'Data/2Dcylinder.mat'
#filename = 'Data/3Dcylinder.mat'

X = scipy.io.loadmat(filename)

Xtrain = X['a_train']

tspan = np.linspace(0, 5, 2500)       #2D
#tspan = tspan.reshape([201, ])
#tspan = np.linspace(62.5, 362.4250, 4000)   #3D
print("X_train shape is", Xtrain.shape)
d = 10
neural = 128
layers = [d, neural, d]
#layers = [d, neural, neural, d]
#layers = [d, neural, neural, neural, d]
#layers = [d, neural, neural, neural, neural,d]
#layers = [d, neural, neural, neural, neural, neural, d]
L = len(layers) - 1
M = 1
scheme = 'AM'
dt = tspan[1]-tspan[0]
#dt = 0.002
noise = 0

OneXstar = np.transpose(Xtrain[0:d, :])
#OneXstar = OneXstar + noise*OneXstar.std(1, keepdims=True)*np.random.randn(OneXstar.shape[0], OneXstar.shape[1])
OneXtrain = np.reshape(OneXstar, (1, OneXstar.shape[0], OneXstar.shape[1]))
#OneXtrain = OneXtrain + noise*OneXtrain.std(0)*np.random.randn(OneXtrain.shape[0], OneXtrain.shape[1], OneXtrain.shape[2])

# define the NN model
model = Multistep_NN(dt, OneXtrain, layers, M, scheme)

#training the model
N_iter = 10000
model.train(N_iter)

# define the prediction f
def learned_f(x, t):
    f = model.predict_f(x[None, :])
    return f.flatten()

#  system reconstruction from learned F ( predict f)
#St, Nt, Dt = OneXtrain.shape
#recons_X = np.zeros((St, Nt, Dt))
#recons_X0 = np.zeros((St, Dt))
#for k in range(St):
#    recons_X0[k, :] = X_train[k, 1, :]
x0 = OneXstar[0, :]
recons_X = odeint(learned_f, x0, tspan)

a_infer=np.transpose(recons_X)
#error = a_infer - Xtrain[0:d, :]
#OneXer = np.reshape(OneXtrain, (a_infer.shape[0], a_infer.shape[1]))
error = a_infer - Xtrain[0:d, :]
print("error is",  np.linalg.norm(error,np.inf))
#filenamesave = 'Data/DataOne' + str(neural)+ 'r'+str(d)+'M'+str(M)+'.mat'
filenamesave = 'Data/2Dinfer' +str(L)+'layers'+ str(neural)+'units_r'+str(d)+scheme+'step'+str(M)+'noise'+str(noise)+'.mat'

scipy.io.savemat(filenamesave, {'a_infer':a_infer})





