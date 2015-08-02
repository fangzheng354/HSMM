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

    def generate(self, T):
        def draw_from(probs):
            return np.where(np.random.multinomial(1, probs) == 1)[0][0]

        states = np.zeros(T)
        observations = np.zeros(T)


    def viterbi(self, observations):
        T = observations.shape[0]

        log_trans_probs =  np.log(self.trans_probs)
        log_emss_probs = np.log(self.emss_probs)
        log_init_probs = np.log(self.init_probs)
        log_durt_probs = np.log(self.durt_probs)

        log_gamma = np.zeros((T,self.states_num))
        log_gamma_star = np.zeros((T, self.states_num))
        back = np.zeros((T, self.states_num))
        back_star  = np.zeros((T, self.states_num))

        log_gamma_star[0] = log_init_probs
        for t in range(T-1):
            dmax = np.min(self.duration_max, t+1)
            a = log_gamma_star[t+1-dmax:t+1] + log_durt_probs[:dmax][::-1] + np.cumsum(log_emss_probs[t+1-dmax:t+1][::-1], axis=0)[::-1]
            a = a[::-1]
            log_gamma[t] = np.max(a, axis=0)
            back[t] = np.argmax(a, axis=0)

            a = log_gamma[t][:, np.newaxis] + log_trans_probs
            log_gamma_star[t+1] = np.max(a, axis=0)
            back_star[t+1] = np.argmax(a, axis=0)

        t = T-1
        dmax = np.min(self.duration_max, t+1)
        a = log_gamma_star[t+1-dmax:t+1] + log_durt_probs[:dmax][::-1] + np.cumsum(log_emss_probs[t+1-dmax:t+1][::-1], axis=0)[::-1]
        a = a[::-1]
        log_gamma[t] = np.max(a, axis=0)
        back[t] = np.argmax(a, axis=0)

        #recover the sequence
        t = T-1
        seq = []
        i = int(np.argmax(log_gamma[t]))
        d = int(back[t, i])
        while t >= 0:
            seq.extend([i] * (d+1))
            i = int(back_star[t-d, i])
            t = t-d-1
            d = int(back[t, i])
        return np.array(list(reversed(seq))), log_gamma
