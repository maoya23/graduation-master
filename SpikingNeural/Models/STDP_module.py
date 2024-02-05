import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
from Models.Neurons import ConductanceBasedLIF,DiehlAndCook2015LIF
from Models.Synapses import SingleExponentialSynapse
from Models.Connections import FullConnection,DelayConnection

np.random.seed(seed=0)

def online_load_and_encoding_dataset(dataset, i, dt, n_time, max_fr=32,
                                    norm=140):
    fr_tmp = max_fr*norm/np.sum(dataset[i][0])
    fr = fr_tmp*np.repeat(np.expand_dims(dataset[i][0],
                                        axis=0), n_time, axis=0)
    input_spikes = np.where(np.random.rand(n_time, 784) < fr*dt, 1, 0)
    input_spikes = input_spikes.astype(np.uint8)

    return input_spikes

# ラベルの割り当て
def assign_labels(spikes, labels, n_labels, rates=None, alpha=1.0):

    n_neurons = spikes.shape[1] 
    
    if rates is None:        
        rates = np.zeros((n_neurons, n_labels)).astype(np.float32)
    
    # 時間の軸でスパイク数の和を取る
    for i in range(n_labels):
        # サンプル内の同じラベルの数を求める
        n_labeled = np.sum(labels == i).astype(np.int16)
    
        if n_labeled > 0:
            # label == iのサンプルのインデックスを取得
            indices = np.where(labels == i)[0]
            
            # label == iに対する各ニューロンごとの平均発火率を計算(前回の発火率との移動平均)
            rates[:, i] = alpha*rates[:, i] + (np.sum(spikes[indices], axis=0)/n_labeled)
    
    sum_rate = np.sum(rates, axis=1)
    sum_rate[sum_rate==0] = 1
    # クラスごとの発火頻度の割合を計算する
    proportions = rates / np.expand_dims(sum_rate, 1) # (n_neurons, n_labels)
    proportions[proportions != proportions] = 0  # Set NaNs to 0
    
    # 最も発火率が高いラベルを各ニューロンに割り当てる
    assignments = np.argmax(proportions, axis=1).astype(np.uint8) # (n_neurons,)

    return assignments, proportions, rates

# assign_labelsで割り当てたラベルからサンプルのラベルの予測をする
def prediction(spikes, assignments, n_labels):

    n_samples = spikes.shape[0]
    
    # 各サンプルについて各ラベルの発火率を見る
    rates = np.zeros((n_samples, n_labels)).astype(np.float32)
    
    for i in range(n_labels):
        # 各ラベルが振り分けられたニューロンの数
        n_assigns = np.sum(assignments == i).astype(np.uint8)
    
        if n_assigns > 0:
            # 各ラベルのニューロンのインデックスを取得
            indices = np.where(assignments == i)[0]
    
            # 各ラベルのニューロンのレイヤー全体における平均発火数を求める
            rates[:, i] = np.sum(spikes[:, indices], axis=1) / n_assigns
    
    # レイヤーの平均発火率が最も高いラベルを出力
    return np.argmax(rates, axis=1).astype(np.uint8) # (n_samples, )



class Network:
    def __init__(self, n_in=784, n_neurons=100, wexc=2.25, winh=0.875,
                dt=1e-3, wmin=0.0, wmax=5e-2, lr=(1e-2, 1e-4),
                update_nt=100):

        
        self.dt = dt
        self.lr_p, self.lr_m = lr
        self.wmax = wmax
        self.wmin = wmin

        # Neurons
        self.exc_neurons = DiehlAndCook2015LIF(n_neurons, dt=dt, tref=5e-3,
                                            tc_m=1e-1,
                                            vrest=-65, vreset=-65, 
                                            init_vthr=-52,
                                            vpeak=20, theta_plus=0.05,
                                            theta_max=35,
                                            tc_theta=1e4,
                                            e_exc=0, e_inh=-100)

        self.inh_neurons = ConductanceBasedLIF(n_neurons, dt=dt, tref=2e-3,
                                            tc_m=1e-2,
                                            vrest=-60, vreset=-45,
                                            vthr=-40, vpeak=20,
                                            e_exc=0, e_inh=-85)
        # Synapses
        self.input_synapse = SingleExponentialSynapse(n_in, dt=dt, td=1e-3)
        self.exc_synapse = SingleExponentialSynapse(n_neurons, dt=dt, td=1e-3)
        self.inh_synapse = SingleExponentialSynapse(n_neurons, dt=dt, td=2e-3)
        
        self.input_synaptictrace = SingleExponentialSynapse(n_in, dt=dt,
                                                            td=2e-2)
        self.exc_synaptictrace = SingleExponentialSynapse(n_neurons, dt=dt,
                                                        td=2e-2)
        
        # Connections
        initW = 1e-3*np.random.rand(n_neurons, n_in)
        self.input_conn = FullConnection(n_in, n_neurons,
                                        initW=initW)
        self.exc2inh_W = wexc*np.eye(n_neurons)
        self.inh2exc_W = (winh/(n_neurons-1))*(np.ones((n_neurons, n_neurons)) - np.eye(n_neurons))
        
        self.delay_input = DelayConnection(N=n_neurons, delay=5e-3, dt=dt)
        self.delay_exc2inh = DelayConnection(N=n_neurons, delay=2e-3, dt=dt)
        
        self.norm = 0.1
        self.g_inh = np.zeros(n_neurons)
        self.tcount = 0
        self.update_nt = update_nt
        self.n_neurons = n_neurons
        self.n_in = n_in
        self.s_in_ = np.zeros((self.update_nt, n_in)) 
        self.s_exc_ = np.zeros((n_neurons, self.update_nt))
        self.x_in_ = np.zeros((self.update_nt, n_in)) 
        self.x_exc_ = np.zeros((n_neurons, self.update_nt))
        
    # スパイクトレースのリセット
    def reset_trace(self):
        self.s_in_ = np.zeros((self.update_nt, self.n_in)) 
        self.s_exc_ = np.zeros((self.n_neurons, self.update_nt))
        self.x_in_ = np.zeros((self.update_nt, self.n_in)) 
        self.x_exc_ = np.zeros((self.n_neurons, self.update_nt))
        self.tcount = 0
    
    # 状態の初期化
    def initialize_states(self):
        self.exc_neurons.initialize_states()
        self.inh_neurons.initialize_states()
        self.delay_input.initialize_states()
        self.delay_exc2inh.initialize_states()
        self.input_synapse.initialize_states()
        self.exc_synapse.initialize_states()
        self.inh_synapse.initialize_states()
        
    def __call__(self, s_in, stdp=True):
        # 入力層
        c_in = self.input_synapse(s_in)
        x_in = self.input_synaptictrace(s_in)
        g_in = self.input_conn(c_in)

        # 興奮性ニューロン層
        s_exc = self.exc_neurons(self.delay_input(g_in), self.g_inh)
        c_exc = self.exc_synapse(s_exc)
        g_exc = np.dot(self.exc2inh_W, c_exc)
        x_exc = self.exc_synaptictrace(s_exc)

        # 抑制性ニューロン層        
        s_inh = self.inh_neurons(self.delay_exc2inh(g_exc), 0)
        c_inh = self.inh_synapse(s_inh)
        self.g_inh = np.dot(self.inh2exc_W, c_inh)

        if stdp:
            # スパイク列とスパイクトレースを記録
            self.s_in_[self.tcount] = s_in
            self.s_exc_[:, self.tcount] = s_exc
            self.x_in_[self.tcount] = x_in 
            self.x_exc_[:, self.tcount] = x_exc
            self.tcount += 1

            # Online STDP
            if self.tcount == self.update_nt:
                W = np.copy(self.input_conn.W)
                
                # postに投射される重みが均一になるようにする
                W_abs_sum = np.expand_dims(np.sum(np.abs(W), axis=1), 1)
                W_abs_sum[W_abs_sum == 0] = 1.0
                W *= self.norm / W_abs_sum
                
                # STDP則
                dW = self.lr_p*(self.wmax - W)*np.dot(self.s_exc_, self.x_in_)
                dW -= self.lr_m*W*np.dot(self.x_exc_, self.s_in_)
                clipped_dW = np.clip(dW / self.update_nt, -1e-3, 1e-3)
                self.input_conn.W = np.clip(W + clipped_dW,
                                            self.wmin, self.wmax)
                self.reset_trace() # スパイク列とスパイクトレースをリセット
        
        return s_exc
        