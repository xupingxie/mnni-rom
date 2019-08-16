#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Sep  1 00:13:51 2018
This class is designed particularly for predictor-corrector discretization
scheme for LMNet
"""

import numpy as np
import tensorflow as tf
import nodepy.linear_multistep_method as lm
import timeit

np.random.seed(1)
tf.set_random_seed(1)

class PECE_NN:
    def __init__(self, dt, X, layers, M):

        self.dt = dt
        self.X = X # S x N x D
        
        self.S = X.shape[0] # number of trajectories
        self.N = X.shape[1] # number of time snapshots
        self.D = X.shape[2] # number of dimensions
        
        self.M = M # number of Adams-Moulton steps
        
        self.layers = layers
        
        #predictor:
        pe_method = lm.Adams_Bashforth(M)
        #corrector:
        ce_method = lm.Adams_Moulton(M)
        
        self.alpha_pe = np.float32(-pe_method.alpha[::-1])
        self.beta_pe = np.float32(pe_method.beta[::-1])
        self.alpha_ce = np.float32(-ce_method.alpha[::-1])
        self.beta_ce = np.float32(ce_method.beta[::-1])
        
        # tf placeholders and graph
        self.sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
        
        self.X_tf = tf.placeholder(tf.float32, shape=[self.S, None, self.D]) # S x N x D
        self.X_star_tf = tf.placeholder(tf.float32, shape=[None, self.D]) # N_star x D
        
        scope_name = str(np.random.randint(1e6))
        with tf.variable_scope(scope_name) as scope:
            self.f_pred = self.neural_net(self.X_star_tf) # N_star x D
        with tf.variable_scope(scope, reuse=True):
            self.Y_pred = self.net_Y(self.X_tf) # S x N x D
        
        self.loss = self.D*tf.reduce_mean(tf.square(self.Y_pred))
        
        self.optimizer_Adam = tf.train.AdamOptimizer()
        self.train_op_Adam = self.optimizer_Adam.minimize(self.loss)
        
        init = tf.global_variables_initializer()
        self.sess.run(init)
                
    def neural_net(self, H):
        num_layers = len(self.layers)
        for l in range(0,num_layers-2):
            with tf.variable_scope("layer%d" %(l+1)):
                H = tf.layers.dense(inputs=H, units=self.layers[l+1], activation=tf.nn.tanh)
        with tf.variable_scope("layer%d" %(num_layers-1)):
            H = tf.layers.dense(inputs=H, units=self.layers[-1], activation=None)
        return H
    
    def net_F(self, X): # S x (N-M+1) x D
        X_reshaped = tf.reshape(X, [-1,self.D]) # S(N-M+1) x D
        F_reshaped = self.neural_net(X_reshaped) # S(N-M+1) x D
        F = tf.reshape(F_reshaped, [self.S,-1,self.D]) # S x (N-M+1) x D
        return F # S x (N-M+1) x D
    
    def net_Y(self, X): # S x N x D
        M = self.M
        
        Y_pe = self.alpha_pe[0]*X[:,M:,:] + self.dt*self.beta_pe[0]*self.net_F(X[:,M:,:])
        Y_ce = self.alpha_ce[0]*X[:,M:,:]
        for m in range(1, M+1):
            Y_pe = Y_pe + self.alpha_pe[m]*X[:,M-m:-m,:] + self.dt*self.beta_pe[m]*self.net_F(X[:,M-m:-m,:])
            #+ self.beta[0]*self.dt*self.net_F(X[:,M:,:])# S x (N-M+1) x D
            Y_ce = Y_ce + self.alpha_ce[m]*X[:,M-m:-m,:] + self.dt*self.beta_ce[m]*self.net_F(X[:,M-m:-m,:])
        Y = Y_pe+Y_ce
        return Y # S x (N-M+1) x D
    
    def train(self, N_Iter):
        
        tf_dict = {self.X_tf: self.X}
        
        start_time = timeit.default_timer()
        for it in range(N_Iter):
            
            self.sess.run(self.train_op_Adam, tf_dict)
            
            # Print
            if it % 100 == 0:
                elapsed = timeit.default_timer() - start_time
                loss_value = self.sess.run(self.loss, tf_dict)
                print('It: %d, Loss: %.3e, Time: %.2f' % 
                      (it, loss_value, elapsed))
                start_time = timeit.default_timer()

    
    def predict_f(self, X_star):
        
        F_star = self.sess.run(self.f_pred, {self.X_star_tf: X_star})
        
        return F_star
