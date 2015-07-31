#!usr/bin/env python
#coding:utf-8

import numpy as np

def normVector(vector):
    s = sum(vector)
    assert(s != 0)
    return vector/s

def normMatrix(matrix):
    s = matrix.sum(axis=1)
    matrix = matrix/s[..., np.newaxis]
    return matrix

class CHSMM:
    def __init__(self, init_probs=None, trans_probs=None, durt_probs=None, emss_probs=None, ):
        self.init_probs = init_probs
        self.trans_probs = trans_probs
        self.durt_probs = durt_probs
        self.emss_probs = emss_probs
        if self.init_probs is None:
            self.states_num = 0
        else:
            self.states_num = self.init_probs.shape[0]
        if self.emss_probs is None:
            self.observation_num = 0
        else:
            self.observation_num = self.emss_probs.shape[1]
        if self.durt_probs is None:
            self.duration_max = 0
        else:
            self.duration_max = self.durt_probs.shape[1]

    def randInit(self, states_num, duration_max, observation_num):
        self.states_num = states_num    # The number of latent states
        self.duration_max = duration_max    # The maximum length of time duration periods
        self.observation_num = observation_num  # The number of observations

        self.init_probs = np.random.rand(self.states_num)   # The initial probability of each state
        self.init_probs = normVector(self.init_probs)   #normalize the probability

        self.trans_probs = np.random.rand(self.states_num, self.states_num)
        self.trans_probs = normMatrix(self.trans_probs)

        self.emss_probs = np.random.rand(self.states_num, self.observation_num)
        self.emss_probs = normMatrix(self.emss_probs)

        self.durt_probs = np.random.rand(self.states_num, self.duration_max)
        self.durt_probs = normMatrix(self.durt_probs)



    def generate(self):



