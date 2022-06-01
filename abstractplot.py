import math
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnchoredOffsetbox, TextArea, HPacker, VPacker
import os
from scipy import constants as sp
from scipy.optimize import curve_fit
from scipy import interpolate as ipl
from matplotlib import gridspec
import matplotlib.patches as patches


cmap=matplotlib.cm.get_cmap('inferno')

max=37.5

red=(0.8,0,0)
blue=(0,0.2,0.4)
green=(0.4,0.6,0.4)


color1=cmap(1/max)
color2=cmap(6/max)
color3=cmap(10/max)
color4=cmap(12/max)
color5=cmap(14/max)
color6=cmap(18/max)
color7=cmap(22/max)
color8=cmap(26/max)
color9=cmap(30/max)
color1p=cmap(0.8/max)
color2p=cmap(7.5/max)
color3p=cmap(12.5/max)
color4p=cmap(15/max)
color5p=cmap(17.5/max)
color6p=cmap(22.5/max)
color7p=cmap(27.5/max)

colors=[color1,color2,color3,color4,color5,color6,color7,color8,color9,color1p,color2p,color3p,color4p,color5p,color6p,color7p]


def ownplot(datei, tom, color, mean, ls, label):
    input=open('C:/Users/tgrie/Desktop/3TM_results/'+str(datei),'r')
    content=input.read().split('\n')
    toplot=[line for line in content if not str(line).startswith('#')]
    columns=[line.split() for line in toplot]
    del columns[-1]
    times=np.array([-10+float(line[0]) for line in columns])
    te=[float(line[2]) for line in columns]
    tp=[float(line[3]) for line in columns]
    mag=np.array([float(line[1]) for line in columns])
    nmag=np.array([float(line[1])/mean for line in columns])
    if tom=='tem':
        #plt.plot(times, te, color=color)
        #plt.plot(times,tp, color=color, linestyle='dashed')

        #plt.plot(times, te, color='orange')
        #plt.plot(times,tp, color='purple')
        return(times, te, tp)

    elif tom=='mag':
        if ls=='dashed':
            #logmask=times>-0.
            #times=times[logmask]
            #print(logmask)
            #print(times)
            #print(nmag)
            #nmag=nmag[logmask]
            #plt.xscale('log')
            #plt.plot(times+15, nmag, color=color, linewidth=2.0, linestyle=ls)
            return(times, nmag)
        else:
            #plt.plot(times+15, nmag, color=color, linewidth=2.0, linestyle=ls, label=label)
            return(times, nmag)

def convolute(sig, dat):
    tmin=-8*sig
    time=dat[0]
    tp=dat[2]
    dt=0.01
    trange=np.arange(-8*sig,8*sig, dt)
    conv=[]
    gaussian=1/math.sqrt(2*math.pi*sig**2)*np.exp(-trange**2/2/sig**2)
    for i in range(int((np.amax(time)+10-8*sig)*100)):
        if time[i]<tmin:
            conv.append(tp[i])
        else:
            conv.append(np.sum(dt*gaussian*[tp[int(i-tau/dt)] for tau in trange]))
    return(time[:1500],conv[:1500])

def lattdyn(file):
    f=open('C:/Users/tgrie/Desktop/3TM_results/lattice/' + str(file),'r')
    fr=f.readlines()
    if file.startswith('F'):
        fn=[line for line in fr if not line.startswith('#')]
        fn1=[line.split() for line in fn]
        times=[float(line[0])*1e-3 for line in fn1]
        tp=[float(line[1]) for line in fn1]
    else:
        fn=[line for line in fr if not line.startswith('D')]
        fn1=[line.split() for line in fn]
        times=[float(line[0]) for line in fn1]
        tp=[float(line[1]) for line in fn1]
    return(np.array(times), np.array(tp))

def dirplot(sample, offset, colors, ax):
    path= 'C:/Users/tgrie/Desktop/3TM_results/MOKE_Unai/' + str(sample)
    files=os.listdir(path)
    i=0
    for file in files[:5]:
        dat=open(path+'/'+file, 'r')
        vals=dat.readlines()
        t=np.array([offset+float(i.split()[0]) for i in vals[1:]])
        mask=t>50
        m=np.array([float(line.split()[1]) for line in vals[1:]])
        thint=[t[i] for i in range(len(t)) if i%5==0]
        thinm=[m[i] for i in range(len(m)) if i%5==0]
        ax.scatter(thint, thinm, color=colors[i], marker='+')
        i+=2
    return

####COBALT LATTICE DATA#####
co_530=ownplot('Cobalt/lattice/530.dat', 'tem', blue, 1, 'solid', None)
co_700=ownplot('Cobalt/lattice/700.dat', 'tem', blue, 1, 'solid', None)
co_110=ownplot('Cobalt/lattice/110.dat', 'tem', blue, 1, 'solid', None)
co_290=ownplot('Cobalt/lattice/290.dat', 'tem', blue, 1, 'solid', None)

cov530=convolute(0.125,co_530)
cov700=convolute(0.125,co_700)
cov110=convolute(0.125,co_110)
cov290=convolute(0.125,co_290)

eco530=lattdyn('co_530.txt')
eco700=lattdyn('co_700.txt')
eco110=lattdyn('co_110.txt')
eco290=lattdyn('co_290.txt')

####IRON LATTICE DATA####
fe_550=ownplot('Iron/lattice/550.dat', 'tem', blue, 1, 'solid', None)
fe_800=ownplot('Iron/lattice/800.dat', 'tem', blue, 1, 'solid', None)
fe_230=ownplot('Iron/lattice/230.dat', 'tem', blue, 1, 'solid', None)
fe_390=ownplot('Iron/lattice/390.dat', 'tem', blue, 1, 'solid', None)

cov550=convolute(0.105,fe_550)
cov800=convolute(0.105,fe_800)
cov230=convolute(0.105,fe_230)
cov390=convolute(0.105,fe_390)

eco550=lattdyn('fe_550.txt')
eco800=lattdyn('fe_800.txt')
eco230=lattdyn('fe_230.txt')
eco390=lattdyn('fe_390.txt')

####NICKEL LATTICE DATA#####

tem6=ownplot('Nickel/lattice/f6.dat', 'tem', blue, 1, 'solid', None)
tem3=ownplot('Nickel/lattice/f3.dat', 'tem', blue, 1, 'solid', None)
tem2=ownplot('Nickel/lattice/f2.dat', 'tem', blue, 1, 'solid', None)
tem1=ownplot('Nickel/lattice/f1.dat', 'tem', blue, 1, 'solid', None)
tem4=ownplot('Nickel/lattice/f4.dat', 'tem', blue, 1, 'solid', None)
tem5=ownplot('Nickel/lattice/f5.dat', 'tem', blue, 1, 'solid', None)

f6=lattdyn('F6.dat')
f3=lattdyn('F3.dat')
f2=lattdyn('F2.dat')
f1=lattdyn('F1.dat')
f4=lattdyn('F4.dat')
f5=lattdyn('F5.dat')

conf3=convolute(0.08, tem3)
conf6=convolute(0.08, tem6)
conf2=convolute(0.08, tem2)
conf1=convolute(0.08, tem1)
conf4=convolute(0.08, tem4)
conf5=convolute(0.08, tem5)

####ALL MAG DATA####
ni1=ownplot('Nickel/c1.dat', 'mag', blue, 1, 'solid', None)
ni2=ownplot('Nickel/c2.dat', 'mag', blue, 1, 'solid', None)
ni3=ownplot('Nickel/c3.dat', 'mag', blue, 1, 'solid', None)
ni4=ownplot('Nickel/c4.dat', 'mag', blue, 1, 'solid', None)
ni5=ownplot('Nickel/c5.dat', 'mag', blue, 1, 'solid', None)
ni6=ownplot('Nickel/c6.dat', 'mag', blue, 1, 'solid', None)

fe1=ownplot('Iron/c1.dat', 'mag', blue, 1, 'solid', None)
fe2=ownplot('Iron/c2.dat', 'mag', blue, 1, 'solid', None)
fe3=ownplot('Iron/c3.dat', 'mag', blue, 1, 'solid', None)
fe4=ownplot('Iron/c4.dat', 'mag', blue, 1, 'solid', None)
fe5=ownplot('Iron/c5.dat', 'mag', blue, 1, 'solid', None)
fe6=ownplot('Iron/c6.dat', 'mag', blue, 1, 'solid', None)

co1=ownplot('Cobalt/c1.dat', 'mag', blue, 1, 'solid', None)
co2=ownplot('Cobalt/c2.dat', 'mag', blue, 1, 'solid', None)
co3=ownplot('Cobalt/c3.dat', 'mag', blue, 1, 'solid', None)
co4=ownplot('Cobalt/c4.dat', 'mag', blue, 1, 'solid', None)
co5=ownplot('Cobalt/c5.dat', 'mag', blue, 1, 'solid', None)
co6=ownplot('Cobalt/c6.dat', 'mag', blue, 1, 'solid', None)

cov=convolute(0.105,ownplot('Iron/lattice/390_1.dat', 'tem', blue, 1, 'solid', None))
cov1=convolute(0.105,ownplot('Iron/lattice/390_05.dat', 'tem', blue, 1, 'solid', None))
cov2=convolute(0.105,ownplot('Iron/lattice/390_0.dat', 'tem', blue, 1, 'solid', None))

#plt.plot(cov[0],cov[1],color='red')
#plt.plot(cov1[0],cov1[1],color='purple')
#plt.plot(cov2[0],cov2[1],color='blue')
#plt.scatter(eco390[0]-0.1, eco390[1])
#plt.xlim(-1,3)
#plt.show()

def plotter():
    fig=plt.figure(figsize=(3.5,4))
    dx=0.35
    dy=0.26
    dist=0.04

    ax1=fig.add_axes([3*dist, 3*dist, dx, dy])
    ax2=fig.add_axes([4*dist+dx, 3*dist, dx, dy], sharex=ax1)
    ax3=fig.add_axes([3*dist, 4*dist+dy, dx, dy], sharex=ax1)
    ax4=fig.add_axes([4*dist+dx, 4*dist+dy, dx, dy], sharex=ax1)
    ax5=fig.add_axes([3*dist, 5*dist+2*dy, dx, dy], sharex=ax1)
    ax6=fig.add_axes([4*dist+dx,5*dist+2*dy, dx, dy], sharex=ax1)


    ####### MODIFY SUBPLOTS#########
    axs=[ax1,ax2,ax3,ax4,ax5,ax6]
    laxs=[ax1,ax3,ax5]
    raxs=[ax2,ax4,ax6]

    ax1.set_xlim(-0.1,2.5)

    for a in axs:
        a.tick_params(axis='both', which='major', labelsize=6)

    for a in raxs:
        a.yaxis.set_label_position("right")
        a.yaxis.tick_right()

    ax1.set_xlabel(r'delay [ps]', fontsize=8)
    ax2.set_xlabel(r'delay [ps]', fontsize=8)

    ax3.set_ylabel(r'normalized magnetization', fontsize=8)
    ax4.set_ylabel(r'Phonon temperature [K]', fontsize=8)

    for a in axs[2:]:
        plt.setp(a.get_xticklabels(), visible=False)

    ax1.set_yticks([0.9,1])
    ax3.set_yticks([0.9,1])
    ax5.set_yticks([0.4,0.6,0.8,1])

    ##########PLOT STUFF##############

    ax1.plot(fe1[0], fe1[1], color=color1, linewidth=1.5)
    ax1.plot(fe2[0], fe2[1], color=color3, linewidth=1.5)
    ax1.plot(fe3[0], fe3[1], color=color5, linewidth=1.5)
    ax1.plot(fe4[0], fe4[1], color=color7, linewidth=1.5)
    ax1.plot(fe5[0], fe5[1], color=color9, linewidth=1.5)

    dirplot('Iron', 0.12, colors, ax1)

    ax3.plot(co1[0], co1[1], color=color1, linewidth=1.5)
    ax3.plot(co2[0], co2[1], color=color3, linewidth=1.5)
    ax3.plot(co3[0], co3[1], color=color5, linewidth=1.5)
    ax3.plot(co4[0], co4[1], color=color7, linewidth=1.5)
    ax3.plot(co5[0], co5[1], color=color9, linewidth=1.5)

    dirplot('Cobalt', -0.1, colors, ax3)

    ax5.plot(ni1[0], ni1[1], color=color1, linewidth=1.5)
    ax5.plot(ni2[0], ni2[1], color=color3, linewidth=1.5)
    ax5.plot(ni3[0], ni3[1], color=color5, linewidth=1.5)
    ax5.plot(ni4[0], ni4[1], color=color7, linewidth=1.5)
    ax5.plot(ni5[0], ni5[1], color=color9, linewidth=1.5)

    dirplot('Nickel', 0.12, colors, ax5)

    ####Tp Cobalt####
    ax4.plot(cov110[0],cov110[1], color='black', linewidth=2.0)
    ax4.plot(cov290[0],cov290[1], color='purple', linewidth=2.0)
    ax4.plot(cov530[0],cov530[1], color='navy', linewidth=2.0)
    ax4.plot(cov700[0],cov700[1], color='slateblue', linewidth=2.0)

    ax4.scatter(np.array(eco110[0])-0.06,eco110[1], color='black', marker='+')
    ax4.scatter(np.array(eco290[0])-0.06,eco290[1], color='purple', marker='+')
    ax4.scatter(np.array(eco530[0])-0.06,eco530[1], color='navy', marker='+')
    ax4.scatter(np.array(eco700[0])-0.06,eco700[1], color='slateblue', marker='+')

    ######Tp Iron####

    ax2.plot(cov230[0],cov230[1], color='black', linewidth=2.0)
    ax2.plot(cov390[0],cov390[1], color='purple', linewidth=2.0)
    ax2.plot(cov550[0],cov550[1], color='navy', linewidth=2.0)
    ax2.plot(cov800[0],cov800[1], color='slateblue', linewidth=2.0)

    ax2.scatter(np.array(eco230[0])-0.08,eco230[1], color='black', marker='+')
    ax2.scatter(np.array(eco390[0])-0.08,eco390[1], color='purple', marker='+')
    ax2.scatter(np.array(eco550[0])-0.08,eco550[1], color='navy', marker='+')
    ax2.scatter(np.array(eco800[0])-0.08,eco800[1], color='slateblue', marker='+')

    ######Tp Nickel####

    ax6.plot(conf4[0],conf4[1], color='slateblue')
    ax6.plot(conf3[0],conf3[1], color='navy')
    ax6.plot(conf2[0],conf2[1], color='purple')
    ax6.plot(conf1[0],conf1[1], color= 'black')

    ax6.scatter([a for a in f4[0]],f4[1], marker='+', color='slateblue')
    ax6.scatter([a for a in f3[0]],f3[1], marker='+', color='navy')
    ax6.scatter([a for a in f2[0]],f2[1], marker='+', color='purple')
    ax6.scatter([a+0.1 for a in f1[0]],f1[1], marker='+', color='black')

    #######ANNOTATIONS######

    ax1.annotate(r'c)', (2.1, 0.895), fontsize=10)
    ax3.annotate(r'b)', (2.1, 0.9), fontsize=10)
    ax5.annotate(r'a)', (2.1, 0.4), fontsize=10)
    ax2.annotate(r'f)', (0, 475), fontsize=10)
    ax4.annotate(r'e)', (0, 450), fontsize=10)
    ax6.annotate(r'd)', (0, 410), fontsize=10)



    fig.savefig('C:/Users/tgrie/Desktop/magplot/figure.pdf')


