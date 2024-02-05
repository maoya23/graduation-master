import numpy as np
import matplotlib.pyplot as plt


#ニューロンが２つしかないときの実装
tau_m=tau_p=20
A_p=0.01
A_m=1.05*A_p
dt=np.arange(-50,50,1)
dw=A_p*np.exp(-dt/tau_p)*(dt>0)-A_m*np.exp(dt/tau_p)*(dt<0)

plt.figure(figsize=(10,8))
plt.plot(dt,dw)
plt.hlines(0,-50,50,linestyles='dotted');plt.xlim(-50,50)
plt.xlabel('$\Delta t$(ms)');plt.ylabel('$\Delta w$')
plt.title('Weight values through Time')
plt.savefig('./Images/STDP_weight.png')

#online STDPの実装
dt=1e-3
T=0.5
nt=round(T/dt)
tau_p=tau_m=2e-2
A_p=0.01;A_m=1.05*A_p

spike_pre=np.zeros(nt);spike_pre[[50,200,225,30,425]]=1
spike_post=np.zeros(nt);spike_post[[100,150,250,350,400]]=1

#記録用
x_pre_arr=np.zeros(nt);x_post_arr=np.zeros(nt)
w_arr=np.zeros(nt)
dw_arr=np.zeros(nt)

x_pre=x_post=0;w=0

for i in range(nt):
    x_pre=x_pre*(1-dt/tau_p)+spike_pre[i]
    x_post=x_post*(1-dt/tau_m)+spike_post[i]
    dw=A_p*x_pre*spike_post[i]-A_m*x_post*spike_pre[i]
    w+=dw

    x_pre_arr[i]=x_pre
    x_post_arr[i]=x_post
    w_arr[i]=w
    dw_arr[i]=dw

# 描画
time = np.arange(nt)*dt*1e3
def hide_ticks(): #上と右の軸を表示しないための関数
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().yaxis.set_ticks_position('left')
    plt.gca().xaxis.set_ticks_position('bottom')
plt.figure(figsize=(8, 8))
plt.subplot(6,1,1)
plt.plot(time, x_pre_arr, color="k")
plt.ylabel("$x_{pre}$"); hide_ticks(); plt.xticks([])
plt.subplot(6,1,2)
plt.plot(time, spike_pre, color="k")
plt.ylabel("pre- spikes"); hide_ticks(); plt.xticks([])
plt.subplot(6,1,3)
plt.plot(time, spike_post, color="k")
plt.ylabel("post- spikes"); hide_ticks(); plt.xticks([])
plt.subplot(6,1,4)
plt.plot(time, x_post_arr, color="k")
plt.ylabel("$x_{post}$"); hide_ticks(); plt.xticks([])
plt.subplot(6,1,5)
plt.plot(time, w_arr, color="k")
plt.xlabel("$t$ (ms)"); plt.ylabel("$w$"); hide_ticks()
plt.subplot(6,1,6)
plt.plot(time,dw_arr,color='k')
plt.savefig('./Images/STDP_online.png')