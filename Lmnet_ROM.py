#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Aug  7 22:55:32 2018
THis is the main driver to train the nonlinear function for reduced order model

da/dt = f(a)

a: the ROM coefficients,  state x = \sum_{j=1}^r a_j \phi_j,
\phi_j are the POD basis.

"""

#import tensorflow as tf
import numpy as np
import scipy.io
from scipy.integrate import odeint
import sys
#sys.path.insert(0, '../MultistepNNs')

from Multistep_NN import Multistep_NN
from numpy import linalg as LA
import matplotlib.pyplot as plt
import time


#======== load the data, coefficients (a) at each time step.===========
#filename = 'Data/Burgers_cond2_0.005.mat'
filename = 'Data/2Dcylinder.mat'
#filename = 'Data/2DcylinderT03.mat'
#filename = 'Data/3Dcylinder.mat'
X = scipy.io.loadmat(filename)
Xtrain = X['a_train']
tspan = np.linspace(0, 5, 2500)       #2D T=[0, 5], 2500 snapshots
#tspan = tspan.reshape([201, ])
#tspan = np.linspace(62.5, 362.4250, 4000)   #3D
print("X_train shape is", Xtrain.shape)

#=================Define the Nueral Net.==============================
#=====================================================================
d = 8    # input dimension
neural = 128   # number of units each layer
layers = [d, neural, d]  # number of layers, [input, hidden, output]
#layers = [d, neural, neural, neural, neural, neural, d]
L = len(layers) - 1
M = 1   # set K-step
scheme = 'AM'   # choose discretization method, 
dt = (tspan[1]-tspan[0]) # \delta t in the time integrator discretization
#dt = 0.002
noise = 0

#=================Prepre training data ==============================
#=====================================================================
OneXstar = np.transpose(Xtrain[0:d, :]) 
#OneXstar = OneXstar + noise*OneXstar.std(1, keepdims=True)*np.random.randn(OneXstar.shape[0], OneXstar.shape[1])
# reshape the training data to match input dimension of Net
OneXtrain = np.reshape(OneXstar, (1, OneXstar.shape[0], OneXstar.shape[1]))
#OneXtrain = OneXtrain + noise*OneXtrain.std(0)*np.random.randn(OneXtrain.shape[0], OneXtrain.shape[1], OneXtrain.shape[2])

# define the NN model
N_iter = 50000
model = Multistep_NN(dt, OneXtrain, layers, M, scheme)


#=================Train the model ====================================
#=====================================================================
print("start training...\n")
time_counter_start = time.clock()
model.train(N_iter)

elapsed = (time.clock()-time_counter_start)
print("time cost is", elapsed)

#=================obtain the learned function=========================
#=====================================================================
# define the prediction f, learned from the data
def learned_f(x, t):
    f = model.predict_f(x[None, :])
    return f.flatten()

#  system reconstruction from learned F ( predict f)
#integrate the function for reconstruction and prediction
x0 = OneXstar[0, :]
recons_X = odeint(learned_f, x0, tspan)  
a_infer=np.transpose(recons_X)   # the inferred coefficients, a

#error = a_infer - Xtrain[0:d, :]
#OneXer = np.reshape(OneXtrain, (a_infer.shape[0], a_infer.shape[1]))
error = a_infer - Xtrain[0:d, :]
print("error is",  np.linalg.norm(error))
#filenamesave = 'Data/DataOne' + str(neural)+ 'r'+str(d)+'M'+str(M)+'.mat'
filenamesave = 'Data/2Dinfer' +str(L)+'layers'+ str(neural)+'units_r'+str(d)+scheme+'step'+str(M)+'noise'+str(noise)+'.mat'

scipy.io.savemat(filenamesave, {'a_infer':a_infer})





