import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class CurrentBasedLIF:
    def __init__ (self,N,dt=1e-4,tref=5e-3,tc_m=1e-2,vrest=-60,vreset=-60,
                vthr=-50,vpeak=20):
        
        self.N=N
        self.dt=dt#離散化した時間
        self.tref=tref#不応期
        self.tc_m=tc_m#膜時定数
        self.vrest=vrest#静止膜電位
        self.vreset=vreset#リセット電位
        self.vthr=vthr#閾値電位
        self.vpeak=vpeak#ピーク電位

        self.v=self.vrest*np.ones(N)#電位の変化
        self.v_=None
        self.tlast=0#最後に発火したタイミング
        self.tcount=0

    def initialize_states(self,random_state=False):
        if random_state:
            self.v=self.vrest + np.random.rand(self.N)*(self.vthr - self.vrest)
        else:
            self.v=self.vrest*np.ones(self.N)

    def __call__(self,I):
        #callメソッドは関数の様に変数を呼び出せる
        dv = (self.vrest-self.v +I)/self.tc_m
        v=self.v+(self.dt+self.tcount > (self.tlast+self.tref))*(dv * self.dt)
        s=1*(v >= self.vthr)#発火したら1に
        self.tlast=self.tlast*(1-s)+self.dt*self.tcount*s
        v=v*(1-s)+self.vpeak*s
        self.v_=v#v_は発火時の電位を保存する変数
        self.v = v*(1-s) + self.vreset*s
        self.tcount +=1

        return s


class ConductanceBasedLIF:
    def __init__(self, N, dt=1e-4, tref=5e-3, tc_m=1e-2, 
                 vrest=-60, vreset=-60, vthr=-50, vpeak=20,
                 e_exc=0, e_inh=-100):

        self.N = N
        self.dt = dt
        self.tref = tref
        self.tc_m = tc_m 
        self.vrest = vrest
        self.vreset = vreset
        self.vthr = vthr
        self.vpeak = vpeak
        
        self.e_exc = e_exc # 興奮性シナプスの平衡電位
        self.e_inh = e_inh # 抑制性シナプスの平衡電位
        
        self.v = self.vreset*np.ones(N)
        self.v_ = None
        self.tlast = 0
        self.tcount = 0
    
    def initialize_states(self, random_state=False):
        if random_state:
            self.v = self.vreset + np.random.rand(self.N)*(self.vthr-self.vreset) 
        else:
            self.v = self.vreset*np.ones(self.N)
        self.tlast = 0
        self.tcount = 0
        
    def __call__(self, g_exc, g_inh):
        I_synExc = g_exc*(self.e_exc - self.v) 
        I_synInh = g_inh*(self.e_inh - self.v)
        dv = (self.vrest - self.v + I_synExc + I_synInh) / self.tc_m #Voltage equation with refractory period 
        v = self.v + ((self.dt*self.tcount) > (self.tlast + self.tref))*dv*self.dt
        
        s = 1*(v>=self.vthr) #発火時は1, その他は0の出力
        self.tlast = self.tlast*(1-s) + self.dt*self.tcount*s #最後の発火時の更新
        v = v*(1-s) + self.vpeak*s #閾値を超えると膜電位をvpeakにする
        self.v_ = v #発火時の電位も含めて記録するための変数
        self.v = v*(1-s) + self.vreset*s  #発火時に膜電位をリセット
        self.tcount += 1
        
        return s    
    

class LealyRNN(nn.Module):
    def __init__(self,input_dim,mid_dim,alpha,sigma_rec=0.1):
        super(LealyRNN,self).__init__()
        self.Wx=nn.Linear(input_dim,mid_dim)
        self.Wr=nn.Linear(mid_dim,mid_dim,bias=False)
        self.input_dim=input_dim
        self.mid_dim=mid_dim
        self.alpha=alpha
        self.sigma_rec=sigma_rec

    def reset_state(self,r=None):
        self.r=r

    def initialize_states(self,shape):
        self.r=torch.tensors(torch.zeros(shape[0],self.mid_dim))

    def forward(self,x):
        if self.r is None:
            self.r=self.initialize_states(x.shape)
        
        z=self.Wr(self.r)+self.Wx(x)
        if self.sigma_rec is not None:
            z +=torch.randn((x.shape[0], self.mid)) * self.sigma_rec
        r= (1 - self.alpha)*self.r + self.alpha*F.relu(z)
        self.r=r

        return r


class DiehlAndCook2015LIF:
    def __init__(self, N, dt=1e-3, tref=5e-3, tc_m=1e-1,
                 vrest=-65, vreset=-65, init_vthr=-52, vpeak=20,
                 theta_plus=0.05, theta_max=35, tc_theta=1e4, e_exc=0, e_inh=-100):
        self.N = N
        self.dt = dt
        self.tref = tref
        self.tc_m = tc_m 
        self.vreset = vreset
        self.vrest = vrest
        self.init_vthr = init_vthr
        self.theta = np.zeros(N)
        self.theta_plus = theta_plus
        self.theta_max = theta_max
        self.tc_theta = tc_theta
        self.vpeak = vpeak

        self.e_exc = e_exc # 興奮性シナプスの平衡電位
        self.e_inh = e_inh # 抑制性シナプスの平衡電位
        
        self.v = self.vreset*np.ones(N)
        self.vthr = self.init_vthr
        self.v_ = None
        self.tlast = 0
        self.tcount = 0
    
    def initialize_states(self, random_state=False):
        if random_state:
            self.v = self.vreset + np.random.rand(self.N)*(self.vthr-self.vreset) 
        else:
            self.v = self.vreset*np.ones(self.N)
        self.vthr = self.init_vthr
        self.theta = np.zeros(self.N)
        self.tlast = 0
        self.tcount = 0
        
    def __call__(self, g_exc, g_inh):
        I_synExc = g_exc*(self.e_exc - self.v) 
        I_synInh = g_inh*(self.e_inh - self.v)
        dv = (self.vrest - self.v + I_synExc + I_synInh) / self.tc_m #Voltage equation with refractory period 
        v = self.v + ((self.dt*self.tcount) > (self.tlast + self.tref))*dv*self.dt
        
        s = 1*(v>=self.vthr) #発火時は1, その他は0の出力
        theta = (1-self.dt/self.tc_theta)*self.theta + self.theta_plus*s
        self.theta = np.clip(theta, 0, self.theta_max)
        self.vthr = self.theta + self.init_vthr
        self.tlast = self.tlast*(1-s) + self.dt*self.tcount*s #最後の発火時の更新
        v = v*(1-s) + self.vpeak*s #閾値を超えると膜電位をvpeakにする
        self.v_ = v #発火時の電位も含めて記録するための変数
        self.v = v*(1-s) + self.vreset*s  #発火時に膜電位をリセット
        self.tcount += 1
        
        return s 