import numpy as np
import matplotlib.pyplot as plt
import os
import math as m

def bimag(file):
    input=open('C:/Users/tgrie/Desktop/3TM_results/bilayers/sum/' + str(file) + '.dat', 'r')
    content=input.read().split('\n')
    toplot=[line for line in content if not str(line).startswith('#')]
    columns=[line.split() for line in toplot]
    del columns[-1]
    t=np.array([-10+float(line[0]) for line in columns])
    mag1=np.array([float(line[1]) for line in columns])
    mag2=np.array([float(line[2]) for line in columns])
    mfe=mag1
    mgd=mag2
    #plt.plot(t,mfe,color='red')
    plt.plot(t,mgd,color='blue')
    return


def bimageq():
    
    path='C:/Users/tgrie/Desktop/3TM_results/bilayers/eq/fe110'
    files=os.listdir(path)
    names=[str(i).replace('.dat','').replace('h','') for i in files]
    
    for file in files:
        raw=open(path+'/'+file, 'r').readlines()
        data=[line for line in raw if not line.startswith('#')]
        eq=data[-1]
        mfe=eq.split('\t')[1]
        mgd=eq.split('\t')[2]
        mfelist=mfe.split(',')
        mgdlist=mgd.split(',')
        puremfe=[float(i.replace('[','').replace(']','')) for i in mfelist]
        puremgd=[float(i.replace('[','').replace(']','')) for i in mgdlist]
        mag=puremfe+puremgd
        mfe=sum(puremfe)*2.2/m.sqrt(2)/(2.86)**2
        mgd=sum(puremgd)*7.5/3/(3.69)**2
        print(mfe, mgd)
        plt.scatter(np.arange(1,len(mag)+1), mag)
        plt.grid(True)
    plt.legend(names)
    
        



bimageq()
plt.xlabel('layer')
plt.ylabel('$M/M_0$')
plt.show()