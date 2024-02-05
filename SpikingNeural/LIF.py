import numpy as np
import matplotlib.pyplot as plt


def lif(currents,time:int,dt:float=1.0,rest=-65,th=-40,ref=3,tc_decay=100):
    '''
    dtは微少時間変化
    restは静止膜電位
    thは発火の閾値でこれをvが超えると発火する
    tc_decayは時定数
    '''
    time=int(time/dt)

    t_last=0
    vpeak=20
    spikes=np.zeros(time)
    v=rest

    monitor=[]
    
    for t in range(time):
        dv=((t+dt) > (ref + t_last)) * (-v + rest + currents[t])/tc_decay
        v = v + dv*dt

        t_last=t_last + (t*dt - t_last) * (v >=th)
        #vが'hをこえたらt*dt - t_lastが実行される。そうでないとt_lastのまま
        v = v+ (vpeak - v) * (v >= th)

        monitor.append(v)

        spikes[t] = (v >= th )*1

        v= v + (rest -v) * (v >= th)

    return spikes,monitor

if __name__ == '__main__':
    duration = 500  # ms
    dt = 0.5  # time step

    time = int(duration / dt)

    # Input data
    # 適当なサインカーブの足し合わせで代用
    input_data_1 = 10 * np.sin(0.1 * np.arange(0, duration, dt)) + 50
    input_data_2 = -10 * np.cos(0.05 * np.arange(0, duration, dt)) - 10

    input_data = input_data_1 + input_data_2

    spikes, voltage = lif(input_data, duration, dt)

    # Plot
    plt.figure(figsize=(12, 6))
    plt.subplot(2, 1, 1)
    plt.ylabel('Input Current')
    plt.plot(np.arange(0, duration, dt), input_data)

    plt.subplot(2, 1, 2)
    plt.ylabel('Membrane Voltage')
    plt.xlabel('time [ms]')
    plt.plot(np.arange(0, duration, dt), voltage)
    plt.savefig('Images/LIF_input.png')
    plt.show()