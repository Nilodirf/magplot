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
color1p=cmap(31/max)
color2p=cmap(33/max)
color3p=cmap(34/max)
color4p=cmap(35/max)
color5p=cmap(36/max)
color6p=cmap(37/max)
color7p=cmap(37.5/max)

colors=np.array([color1,color2,color3,color4,color5,color6,color7,color8,color9,color1p,color2p,color3p,color4p,color5p,color6p,color7p])
#colors=[color2, color6, color9]

def ownplot(datei, tom):
    input=open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), '3TM_results/'+ str(datei)),'r')
    content=input.read().split('\n')
    toplot=[line for line in content if not str(line).startswith('#')]
    columns=[line.split() for line in toplot]
    del columns[-1]
    times=np.array([-10+float(line[0]) for line in columns])
    te=[float(line[2]) for line in columns]
    tp=[float(line[3]) for line in columns]
    nmag=np.array([float(line[1]) for line in columns])

    if tom=='tem':
        return(times, te, tp)

    elif tom=='mag':
        return(times, nmag)

    elif tom=='gep':
        gep=np.array([float(line[8]) for line in columns])

    elif tom=='both':
        fig, ax1=plt.subplots()

        ax1.set_xlabel('delay [ps]')
        ax1.set_ylabel('temperature [K]')
        ax1.plot(times, te, color='b')
        ax1.plot(times, tp, color='r')

        plt.legend(['Te','Tp'])

        ax2=ax1.twinx()
        ax2.set_ylabel('magnetization', color='g')
        ax2.plot(times, mag, linestyle='dashed', color='g')
        ax2.tick_params(axis='y', labelcolor='g')

    
        fig.tight_layout()

    elif tom=='en':
        et=np.array([float(line[4]) for line in columns])
        ee=np.array([float(line[5]) for line in columns])
        ep=np.array([float(line[6]) for line in columns])
        es=np.array([-float(line[7]) for line in columns])
        return(times, et, ee, ep, es)
    return
    
n1=ownplot('Nickel/c1.dat', 'mag')
n2=ownplot('Nickel/c2.dat', 'mag')
n3=ownplot('Nickel/c3.dat', 'mag')
n4=ownplot('Nickel/c4.dat', 'mag')
n5=ownplot('Nickel/c5.dat', 'mag')
n6=ownplot('Nickel/c6.dat', 'mag')
n7=ownplot('Nickel/c7.dat', 'mag')
ns=[n1,n2,n3,n4,n5,n6,n7]

f1=ownplot('Iron/c1.dat', 'mag')
f2=ownplot('Iron/c2.dat', 'mag')
f3=ownplot('Iron/c3.dat', 'mag')
f4=ownplot('Iron/c4.dat', 'mag')
f5=ownplot('Iron/c5.dat', 'mag')
f6=ownplot('Iron/c6.dat', 'mag')
f7=ownplot('Iron/c7.dat', 'mag')
f8=ownplot('Iron/c8.dat', 'mag')
f9=ownplot('Iron/c9.dat', 'mag')
fs=[f1,f2,f3,f4,f5,f6,f7,f8,f9]

c1=ownplot('Cobalt/c1.dat', 'mag')
c2=ownplot('Cobalt/c2.dat', 'mag')
c3=ownplot('Cobalt/c3.dat', 'mag')
c4=ownplot('Cobalt/c4.dat', 'mag')
c5=ownplot('Cobalt/c5.dat', 'mag')
c6=ownplot('Cobalt/c6.dat', 'mag')
c7=ownplot('Cobalt/c7.dat', 'mag')
c8=ownplot('Cobalt/c8.dat', 'mag')
c9=ownplot('Cobalt/c9.dat', 'mag')
cs=[c1,c2,c3,c4,c5,c6,c7,c8,c9]

cen=ownplot('Cobalt/lattice/530.dat', 'en')
cennoq=ownplot('Cobalt/lattice/530_noq.dat', 'en')
cennoqad=ownplot('Cobalt/lattice/530_noqad.dat', 'en')
cnoqt=ownplot('Cobalt/lattice/530_noq.dat', 'tem')
cnoqtad=ownplot('Cobalt/lattice/530_noqad.dat', 'tem')

n1t=ownplot('Nickel/lattice/f1.dat', 'tem')
n2t=ownplot('Nickel/lattice/f2.dat', 'tem')
n3t=ownplot('Nickel/lattice/f3.dat', 'tem')
n4t=ownplot('Nickel/lattice/f4.dat', 'tem')
n5t=ownplot('Nickel/lattice/f5.dat', 'tem')
n6t=ownplot('Nickel/lattice/f6.dat', 'tem')

nite=ownplot('Nickel/lattice/tefit.dat', 'tem')
nitenoq=ownplot('Nickel/lattice/tefitnoq.dat', 'tem')
n3noqt=ownplot('Nickel/lattice/f3noq.dat', 'tem')

f1t=ownplot('Iron/lattice/230.dat', 'tem')
f2t=ownplot('Iron/lattice/390.dat', 'tem')
f3t=ownplot('Iron/lattice/550.dat', 'tem')
f4t=ownplot('Iron/lattice/800.dat', 'tem')

c1t=ownplot('Cobalt/lattice/110.dat', 'tem')
c2t=ownplot('Cobalt/lattice/290.dat', 'tem')
c3t=ownplot('Cobalt/lattice/530.dat', 'tem')
c4t=ownplot('Cobalt/lattice/700.dat', 'tem')



def floplot(datei, tom):
    input=open('C:/Users/tgrie/Desktop/3TM_results/Ni_sim_flo/'+str(datei),'r')
    content=input.read().split('\n')
    toplot=[line for line in content if not str(line).startswith('#')]
    columns=[line.split() for line in toplot]
    times=[-1.5+float(line[0]) for line in columns]
    tel=[10**2*float(line[1])/8.617 for line in columns]
    tp=[10**2*float(line[2])/8.617 for line in columns]
    mag=[float(line[5]) for line in columns]
    if tom=='tem':
        #plt.plot(times, tel, 'b', linestyle='dashed',)
        plt.plot(times, tp, 'r', linestyle='dashed')
    elif tom=='mag':
        plt.plot(times, mag, 'k')
        plt.ylabel('magnetization')
    return




def expplot(datei):
    input=open('C:/Users/tgrie/Desktop/3TM_results/Ni_exp_flo/'+str(datei),'r')
    content=input.read().split('\n')
    toplot=[line for line in content if not str(line).startswith('D')]
    columns=[line.split() for line in toplot]
    times=[float(line[0]) for line in columns]
    t=[float(line[1]) for line in columns]
    #terr=[float(line[2]) for line in columns]
    plt.scatter(times,t,s=3,c='black')
    plt.xlabel('delay [ps]')
    plt.ylabel('temperature [K]')




def mtplot(dat):
    input=open('C:/Users/tgrie/Desktop/3TM_results/'+str(dat),'r')
    content=input.readlines()
    columns=[line.split() for line in content if not line.startswith('#')]
    mag=[float(line[1])/0.969 for line in columns]
    times=[-10+float(line[0]) for line in columns]
    te=[float(line[2]) for line in columns]
    tp=[float(line[3]) for line in columns]
    input.close()

    fig=plt.figure(figsize=(4,2.5))
    ax1=fig.add_axes([0.1, 0.1, 0.8, 0.85])


    ax1.set_xlabel(r'delay [ps]', fontsize=22)
    ax1.set_ylabel(r'Temperature [K]', size=22)
    #ax1.axhline(y=525.686, color=(0.8,0.2,0.4), linestyle='dashed')
    ax1.plot(times, te, color=blue, linewidth=2.0)
    ax1.plot(times, tp, color=red, linewidth=2.0)
    ax1.set_xlim((-0.3,2.5))
    ax1.tick_params(axis='both', which='major', labelsize=20)

    #plt.vlines(0.24, 280, 1000, color='black')

    plt.legend([r'$T_e$',r'$T_p$'], fontsize=20)

    ax2=ax1.twinx()
    ax2.set_ylabel(r'$m/m_0$', color=green, fontsize=22)
    ax2.plot(times, mag, color=green, linewidth=2.0)
    ax2.tick_params(axis='y', labelcolor=green, labelsize=20)

    ax1.hlines(633,-0.3,4, color='black', linewidth=1.5, linestyle='dashed')
    
    ax1.set_xticks([0,0.33,0.5,1,1.5,2,2.5])
    ax1.set_xticklabels([r'0', r'$t_m$', r'0.5',r'1',r'1.5',r'2',r'2.5'])
    ax1.set_yticks([300,500,633,700,900,1000])
    ax1.set_yticklabels([r'300', r'500', r'$T_C$', r'700', r'900', r'1000'])
    ax2.set_yticks([0.42,0.5,0.7,0.9,1])
    ax2.set_yticklabels([r'$m_{\rm{min}}$', r'0.5', r'0.7', r'0.9', r'1'])
    ax1.vlines(0.33, 250, 633, linewidth=1.5, linestyle='dashed', color=green)
    ax1.hlines(315, 0.33, 2.5, linewidth=1.5, linestyle='dashed', color=green)

    gausstimes=np.arange(-0.3,0.08,0.001)
    ax1.fill_between(gausstimes, [295+ 33/math.sqrt(2*math.pi)/0.018*math.exp(-t**2/2/0.018**2) for t in gausstimes], [295 for t in gausstimes], color=color9, alpha=0.5)


    ax1.set_ylim((280,1050))
    fig.tight_layout()
    plt.show()

#mtplot('Nickel/calib.dat')


def dftplot(file):
    dat=open('C:/Users/tgrie/Desktop/3TM_Data/' + str(file),'r')
    vals=dat.readlines()
    x=np.array([float(i.split()[0]) for i in vals[1:]])
    y=np.array([float(line.split()[1]) for line in vals[1:]])
    plt.plot(x,y, 'tab:cyan')
    plt.xlabel('T [K]')
    

def fillplot(datei, datei2, col):
    input=open('C:/Users/tgrie/Desktop/3TM_results/'+str(datei),'r')
    content=input.read().split('\n')
    toplot=[line for line in content if not str(line).startswith('#')]
    columns=[line.split() for line in toplot]
    del columns[-1]
    times=[-10+float(line[0]) for line in columns]
    mag=[float(line[1]) for line in columns]
    usetime=[]
    usemag=[]
    usemag2=[]

    input=open('C:/Users/tgrie/Desktop/3TM_results/'+str(datei2),'r')
    content2=input.read().split('\n')
    toplot2=[line for line in content2 if not str(line).startswith('#')]
    columns2=[line.split() for line in toplot2]
    del columns2[-1]
    mag2=[float(line[1]) for line in columns2]

    for i in range(int(len(times)/2)):
        if i%5==0:
            usetime.append(times[i])
            usemag.append(mag[i])
            usemag2.append(mag2[i])

    plt.plot(usetime, usemag, color=col)
    plt.plot(usetime, usemag2, color=col)
    plt.fill_between(usetime, usemag, usemag2, color=col, edgecolors=col, alpha=0.5)


def meqplot(sample, s, p, tc):
    temp=np.arange(0,tc,1)
    tred=temp/tc
    func=np.power(1-s*np.power(tred,(3/2))-(1-s)*np.power(tred,p),1/3)

    path='C:/Users/tgrie/Desktop/bestfits/' + str(sample)
    files=os.listdir(path)
    list=[[],[]]
    if sample=='Nickel':
        farben=colors[9:]
    else:
        farben=colors[:9]
    for file in files:
        dat=open(path+ '/' + file, 'r')
        dataline=dat.readlines()[-1]
        data=dataline.split()
        list[0].append(float(data[2])/tc)
        list[1].append((float(data[1])))
    list=np.array(list)
    plt.plot(tred, func, color='black')
    plt.scatter(list[0], list[1], c=farben)
    return

#meqplot('Nickel', 0.15, 5/2, 633)

def dbrillouin(x,spin):
    c1=(2*spin+1)/(2*spin)
    c2=1/(2*spin)
    dfb=c2**2/(np.sinh(c2*x))**2-c1**2/(np.sinh(c1*x))**2
    return(dfb)

def interpol(file):
    dat=open('C:/Users/tgrie/Desktop/3TM_Data/Ab-initio-parameters2/' + str(file),'r')
    lines=dat.readlines()[2:]
    #vals=[line for line in lines if not line.startswith('M') or line.startswith('T')]
    t=np.array([float(i.split()[0]) for i in lines])
    y=np.array([float(line.split()[1]) for line in lines])
    if np.amin(t)>290:
        t[0]=290.
    fit=ipl.interp1d(t,y)
    return(fit)



def cseqplot(sample, S, tc):
    path='C:/Users/tgrie/Desktop/bestfits/' + str(sample)
    files=os.listdir(path)
    list=[[],[],[]]
    if sample=='Nickel':
        farben=colors[9:]
    else:
        farben=colors[:9]
    for file in files:
        dat=open(path+ '/' + file, 'r')
        dataline=dat.readlines()[-1]
        data=dataline.split()
        list[0].append(float(data[2]))
        list[1].append(float(data[-2]))
        list[2].append(float(data[5]))
    list=np.array(list)
    q=3*S/(S+1)*tc/list[0]
    #func=1/2*q**2*dbrillouin(q*list[1],S)/(1-q*dbrillouin(q*list[1],S))
    cs=-list[1]/list[2]*interpol('Ab-initio-Co/Co_c_e.txt')(list[0])
    #plt.scatter(list[0],func, c='black')
    plt.scatter(list[0], cs, c=farben)
    return


#cseqplot('Cobalt', 2, 1041)
#plt.show()


def tepplot():
    dat=open('C:/Users/tgrie/bestfits/Nickel/allnew_c4_lower.dat', 'r')
    content=dat.readlines()
    data=[line.split() for line in content if not str(line).startswith('#')]
    tempe=[float(line[2]) for line in data]
    tampp=[float(line[3]) for line in data]
    dT=max(tempe)-295
    maxtempe=tempe=max(tempe)
    tempp0=tempp[maxtempe]
    tnext=np.roll(maxtempe,1)
    tempenext=tempe[tnext]
    temppnext=tempp[tnext]
    dtte=(tempenext-maxtempe)*1e14
    dttp=(temppnext-tempp0)*1e14



def dirplot(sample, offset, size, colors, ax, min, max, exc):
    path= os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), '3TM_results/MOKE_Unai/') + str(sample)
    files=os.listdir(path)
    i=0
    for  j, file in enumerate(files[min:max]):
        if min+j in exc:
            i+=1
        else:
            dat=open(path+'/'+file, 'r')
            vals=dat.readlines()
            t=np.array([offset+float(i.split()[0]) for i in vals[1:]])
            mask=t>50
            m=np.array([float(line.split()[1]) for line in vals[1:]])
            ax.scatter(t,m, s=size, marker='o', color=colors[2*(j+min-i)], alpha=0.5)
    return

def newfeplot(file):
    file=open('C:/Users/tgrie/Desktop/3TM_results/MOKE_Unai/Iron_new/'+str(file)+'.txt','r')
    if file=='dM':
        dm=np.array([1+float(i) for i in file.readlines()])
        t=(np.arange(len(dm))*2e-14-4.2e-13)*1e12
    else:
        dm=np.array([1+float(i) for i in file.readlines()])
        t=(np.arange(len(dm))*2e-14-4.2e-13)*1e12
    return(t,dm)

def pumpplot():
    sig=0.04
    pumpfile=open('C:/Users/tgrie/Desktop/3TM_Data/Pump.txt','r')
    dpump=np.array([56e21*float(i)**2 for i in pumpfile.readlines()])
    t=(np.arange(len(dpump))*2e-14-2.04e-12)*1e12
    pump=ipl.interp1d(t,dpump, fill_value=(0,0), bounds_error=False)
    gaussian=np.array(np.exp(-(t**2)/(2*sig**2)))*0.02
    return(t,pump(t), gaussian)

def ellrot(g):
    return(trot-g*tell/(1-g))

def eldyn():
    dat=open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), '3TM_results/Nickel/electrons/Tengdin_data.txt'),'r')
    vals=dat.readlines()
    x=np.array([float(i.split()[0])*1e-3 for i in vals[2:]])
    y=np.array([float(line.split()[1]) for line in vals[2:]])
    return(x,y)

nit=eldyn()

def lattdyn(file):
    f=open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), '3TM_results/lattice/') + str(file),'r')
    fr=f.readlines()
    if file.startswith('F'):
        fn=[line for line in fr if not line.startswith('#')]
        fn1=[line.split() for line in fn]
        times=[float(line[0])*1e-3 for line in fn1]
        tp=[float(line[1]) for line in fn1]
    else:
        fn=[line for line in fr if not line.startswith('D')]
        fn1=[line.split() for line in fn]
        times=np.array([float(line[0]) for line in fn1])
        tp=np.array([float(line[1]) for line in fn1])
    return(np.array(times), np.array(tp))

def gadmag(file):
    dat=open('C:/Users/tgrie/Desktop/3TM_results/Gadolinium/'+str(file)+'.txt','r')
    vals=dat.readlines()
    x=np.array([float(i.split()[0]) for i in vals[2:]])
    y=np.array([float(line.split()[1]) for line in vals[2:]])
    return(x,y)

n1td=lattdyn('F1.dat')
n2td=lattdyn('F2.dat')
n3td=lattdyn('F3.dat')
n4td=lattdyn('F4.dat')
n5td=lattdyn('F5.dat')
n6td=lattdyn('F6.dat')

f1td=lattdyn('fe_230.txt')
f2td=lattdyn('fe_390.txt')
f3td=lattdyn('fe_550.txt')
f4td=lattdyn('fe_800.txt')

c1td=lattdyn('co_110.txt')
c2td=lattdyn('co_290.txt')
c3td=lattdyn('co_530.txt')
c4td=lattdyn('co_700.txt')


def convolute(sig, dat):  #sig in ps
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
    return(np.array(time[:1500]), np.array(conv[:1500]))

n1tc=convolute(0.065, n1t)
n2tc=convolute(0.065, n2t)
n3tc=convolute(0.065, n3t)
n4tc=convolute(0.065, n4t)
n5tc=convolute(0.065, n5t)
n6tc=convolute(0.065, n6t)
ntc=[n1tc, n2tc, n3tc, n4tc, n5tc, n6tc]

f1tc=convolute(0.106, f1t)
f2tc=convolute(0.106, f2t)
f3tc=convolute(0.106, f3t)
f4tc=convolute(0.106, f4t)
ftc=[f1tc, f2tc, f3tc, f4tc]

c1tc=convolute(0.106, c1t)
c2tc=convolute(0.106, c2t)
c3tc=convolute(0.106, c3t)
c4tc=convolute(0.106, c4t)
ctc=[c1tc, c2tc, c3tc, c4tc]

cnoqtc=convolute(0.106, cnoqt)
cnoqtadc=convolute(0.106, cnoqtad)
n3noqtc=convolute(0.065, n3noqt)

#fe_ex=ownplot('Iron/c5.dat', 'mag', green, 1, 'solid', 'bla')
#fe_ex_2=ownplot('Iron/lattice/c5_0.dat', 'mag' , green, 1, 'solid', 'bla')

#plt.plot(fe_ex[0], fe_ex[1], color='red')
#plt.plot(fe_ex_2[0], fe_ex_2[1]/0.9842, color='blue')
#plt.show()

def interpol(file):
    dat=open('C:/Users/tgrie/Desktop/3TM_Data/Ab-initio-parameters2/' + str(file),'r')
    lines=dat.readlines()[2:]
    #vals=[line for line in lines if not line.startswith('M') or line.startswith('T')]
    t=np.array([float(i.split()[0]) for i in lines])
    y=np.array([float(line.split()[1]) for line in lines])
    if np.amin(t)>290:
        t[0]=290.
    fit=ipl.interp1d(t,y)
    return(fit)

def plotter(x, y, xnum, ynum, xdist, ydist, trange, xmin, xtra, ytra, sharex, hidex):

    fig=plt.figure(figsize=(x,y))
    dx=(1-2*xmin-(xnum+ytra-1)*xdist)/(xnum+ytra)
    dy=(0.8-(ynum+xtra-1)*ydist)/(ynum+xtra)
    subs=[]
    for i in range(xtra):
        subs.append(fig.add_axes([xmin+i*dx+i*xdist, 0.1, 1-2*xmin, dy]))
    for i in range(xnum):
        for j in range(ynum):
            subs.append(fig.add_axes([ytra+xmin+i*dx+i*xdist, dy*xtra+0.1+j*dy+(j)*ydist, dx, dy]))
    for i in range(len(subs)):
        if i+1-xtra>len(subs)/2 and xnum>1:
            subs[i].yaxis.set_label_position("right")
            subs[i].yaxis.tick_right()
    if len(subs)>1:
        for i, pic in enumerate(subs[xtra:]):
            if i in sharex[0]:
                pic.sharex(subs[sharex[1][0]])
            if i in hidex:
                plt.setp(subs[i].get_xticklabels(), visible=False)
    subs[xtra+sharex[1][0]].set_xlim(trange)
    return(fig, subs)

def labeler(pics, ylabels):
    for i in range(len(pics)):
        pics[i].set_ylabel(ylabels[i], fontsize=14)
    pics[0].set_xlabel(r'delay [ps]', fontsize=14)

def highflu(fsz, txtx):

    figure=plotter(7, 9, 2, 3, 0.05, 0.05, (-0.1,2.4), 0.12, 0, 0, [[1,2,3,4,5],[0]], [1,2,4,5])
    fig=figure[0]
    pics=figure[1]
    labeler(pics, [None, r'$M/M_0$', None, None, r'$T_p$', None])

    ##NICKEL MAG###
    for i in range(0,6):
        pics[0].plot(ns[i][0], ns[i][1], color=colors[2*i], lw=3.0)
    dirplot('Nickel', 0.1, 10, colors, pics[0], 0, 6, [])
    #pics[0].annotate(r'758', (txtx, 0.975), fontsize=fsz, color=colors[2])
    #pics[0].annotate(r'915', (txtx, 0.9), fontsize=fsz, color=colors[4])
    #pics[0].annotate(r'1069', (txtx, 0.825), fontsize=fsz, color=colors[6])
    #pics[0].annotate(r'$\frac{\rm{J}}{\rm{cm}^3}$', (txtx+0.45, 0.9), fontsize=fsz, color=(0,0,0))
    pics[0].set_xlabel(r'delay [ps]', fontsize=fsz)

    ##IRON MAG###
    counter=0
    for i in range(0,6):
        pics[1].plot(fs[i][0], fs[i][1], color=colors[2*i], lw=3.0)
    dirplot('Iron', 0.12, 10, colors, pics[1], 0, 6, [])
    #pics[1].annotate(r'600', (txtx, 1), fontsize=fsz, color=colors[2])
    #pics[1].annotate(r'839', (txtx, 0.98), fontsize=fsz, color=colors[4])
    #pics[1].annotate(r'1078', (txtx, 0.96), fontsize=fsz, color=colors[6])
    #pics[1].annotate(r'$\frac{\rm{J}}{\rm{cm}^3}$', (txtx+0.45, 0.98), fontsize=fsz, color=(0,0,0))
    pics[1].set_ylabel(r'Norm. magnetization', fontsize=fsz)
    pics[1].set_yticks([0.85, 0.9, 0.95, 1])

    ###COBALT MAG###
    counter=0
    for i in range(0,6):
        pics[2].plot(cs[i][0], cs[i][1], color=colors[2*i], lw=3.0)
    dirplot('Cobalt', -0.09, 10, colors, pics[2], 0, 6, [])
    #pics[2].annotate(r'600', (txtx, 0.9), fontsize=fsz, color=colors[2])
    #pics[2].annotate(r'839', (txtx, 0.88), fontsize=fsz, color=colors[4])
    #pics[2].annotate(r'1078', (txtx, 0.86), fontsize=fsz, color=colors[6])
    #pics[2].annotate(r'$\frac{\rm{J}}{\rm{cm}^3}$', (txtx+0.45, 0.88), fontsize=fsz, color=(0,0,0))
    pics[2].set_ylim(0.85, 1.01)
    pics[2].set_yticks([0.9, 0.95, 1])

    ###NICKEL TEMP###
    pics[3].plot(ntc[0][0], ntc[0][1], color=colors[0], lw=3.0)
    pics[3].plot(ntc[1][0], ntc[1][1], color=colors[1], lw=3.0)
    pics[3].plot(ntc[2][0], ntc[2][1], color=colors[2], lw=3.0)
    pics[3].plot(ntc[3][0], ntc[3][1], color=colors[3], lw=3.0)
    pics[3].scatter(n1td[0], n1td[1], color=colors[0], s=10)
    pics[3].scatter(n2td[0], n2td[1], color=colors[1], s=10)
    pics[3].scatter(n3td[0]+0.09, n3td[1], color=colors[2], s=10)
    pics[3].scatter(n4td[0]+0.09, n4td[1], color=colors[3], s=10)
    pics[3].annotate(r'Nickel', (0, 420), fontsize=16, color=(0,0,0))
    pics[3].set_xlabel(r'delay [ps]', fontsize=fsz)
    pics[3].set_ylim(bottom=290)

    ###IRON TEMP###
    pics[4].plot(ftc[0][0],ftc[0][1], color=colors[1], lw=3.0)
    pics[4].plot(ftc[1][0],ftc[1][1], color=colors[3], lw=3.0)
    pics[4].plot(ftc[2][0],ftc[2][1], color=colors[4], lw=3.0)
    pics[4].plot(ftc[3][0],ftc[3][1], color=colors[9], lw=3.0)
    pics[4].scatter(np.array(f1td[0])-0.15,f1td[1], s=10, color=colors[1])
    pics[4].scatter(np.array(f2td[0])-0.1,f2td[1], s=10, color=colors[3])
    pics[4].scatter(np.array(f3td[0])-0.12,f3td[1], s=10, color=colors[4])
    pics[4].scatter(np.array(f4td[0])-0.1,f4td[1], s=10, color=colors[9])
    pics[4].annotate(r'Iron', (0, 490), fontsize=16, color=(0,0,0))
    pics[4].set_ylabel(r'Lattice temperature [K]', fontsize=fsz)
    pics[4].set_ylim(290, 530)

    ###COBALT TEMP###
    pics[5].plot(ctc[0][0], ctc[0][1], color=colors[1], lw=3.0)
    pics[5].plot(ctc[1][0], ctc[1][1], color=colors[2], lw=3.0)
    pics[5].plot(ctc[2][0], ctc[2][1], color=colors[4], lw=3.0)
    pics[5].plot(ctc[3][0], ctc[3][1], color=colors[6], lw=3.0)
    pics[5].scatter(np.array(c1td[0])-0.1, c1td[1], s=10, color=colors[1])
    pics[5].scatter(np.array(c2td[0])-0.1, c2td[1], s=10, color=colors[2])
    pics[5].scatter(np.array(c3td[0])-0.1, c3td[1], s=10, color=colors[4])
    pics[5].scatter(np.array(c4td[0])-0.1, c4td[1], s=10, color=colors[6])
    pics[5].annotate(r'Cobalt', (0, 460), fontsize=16, color=(0,0,0))
    pics[5].set_ylim(290, 500)

    #fig.savefig('C:/Users/tgrie/Desktop/Madrid Physik/paperpics/allflu.pdf')

    pics[2].annotate(r'(a)', (-0.05,0.86), fontsize=16)
    pics[1].annotate(r'(c)', (-0.05,0.86), fontsize=16)
    pics[0].annotate(r'(e)', (-0.05,0.25), fontsize=16)

    pics[5].annotate(r'(b)', (-0.05,425), fontsize=16)
    pics[4].annotate(r'(d)', (-0.05,440), fontsize=16)
    pics[3].annotate(r'(f)', (-0.05,390), fontsize=16)

def energy(fs, col):
    fig, plots=plotter(10, 6, 2, 2, 0.03, 0.05, (-0.3,4), 0.1, 0, 0, [[3],[2]], [1,3])

    n3noq=ownplot('Nickel/c3_noq.dat', 'mag')

    ###TP DYN###
    plots[2].plot(n3tc[0], n3tc[1], color=col, lw=3.0, label=r'e2TM')
    plots[2].plot(n3noqtc[0], n3noqtc[1], color=color6, lw=3.0, ls='--', label=r'2TM')
    plots[2].scatter(np.array(n3td[0])+0.05, n3td[1], s=80, marker='+', lw=2.0, label=r'Zahn et al.')
    plots[2].vlines(0.19, 250, 480, linewidth=1, color=col)
    plots[2].set_ylim(290, 385)
    plots[2].set_xlabel(r'delay [ps]', fontsize=fs)
    plots[2].set_ylabel(r'Lat. temperature [K]', fontsize=fs)
    plots[2].legend(fontsize=fs-1)
    plots[2].annotate(r'332 $\frac{\rm{J}}{\rm{cm}^3}$', (2.7,346), fontsize=fs, fontweight='bold')
    plots[2].annotate(r'd)', (-0.15, 370), fontsize=fs, fontweight='bold')


    ####TE DYN####
    plots[0].plot(nite[0], nite[1], color=col, lw=3.0)
    plots[0].plot(nitenoq[0], nitenoq[1], color=color6, lw=3.0, ls='--')
    plots[0].scatter(nit[0], nit[1], lw=2.0, label=r'Tengdin et al.')
    plots[0].vlines(0.35, 200, 1600, linewidth=1, color=col)
    plots[0].set_ylim(230, 1550)
    plots[0].set_xlabel(r'delay [ps]', fontsize=fs)
    plots[0].set_ylabel(r'El. temperature [K]', fontsize=fs)
    plots[0].set_xlim((-0.1, 0.8))
    plots[0].legend(fontsize=fs-1)
    plots[0].annotate(r'677 $\frac{\rm{J}}{\rm{cm}^3}$', (0.55, 1000), fontsize=fs, fontweight='bold')
    plots[0].annotate(r'I', (0.3, 1000), fontsize=fs, fontweight='bold')
    plots[0].annotate(r'II', (0.38, 1000), fontsize=fs, fontweight='bold')
    plots[0].annotate(r'c)', (-0.08, 1300), fontsize=fs, fontweight='bold')


    ###MAG DYN###
    plots[3].plot(n3[0], n3[1], color=col, lw=3.0)
    plots[3].plot(n3noq[0], n3noq[1], color=color6, lw=3.0, ls='--')
    dirplot('Nickel', 0.1, [None], plots[3], 2, 3, [])
    plots[3].vlines(0.35, 0.5, 1.2, linewidth=1, color=col)
    plots[3].set_ylim(0.55, 1.02)
    plots[3].set_ylabel(r'Norm. magnetization', fontsize=fs)
    plots[3].legend(fontsize=fs-1)
    plots[3].annotate(r'758 $\frac{\rm{J}}{\rm{cm}^3}$', (2.7, 0.7), fontsize=fs, fontweight='bold' )
    plots[3].annotate(r'b)', (-0.15,0.58), fontsize=fs, fontweight='bold' )
    
    #plots[0].plot(cen[0], cen[1]*1e-6, color=color5)
    #plots[0].plot(cennoqad[0], cennoqad[1]*1e-6, color=color8, ls='dashed', lw=2.0)
    #plots[0].plot(cen[0], cen[4]*1e-6, color=color5)
    #plots[0].plot(cen[0], -(cen[2]+cen[3]-cennoqad[2]-cennoqad[3])*1e-6, color=color8, ls='-.', lw=2.0)
    ##plots[0].plot(cen[0], (cen[4]+(cen[2]+cen[3]-cennoqad[2]-cennoqad[3]))*1e-6, linestyle='dotted')
    #plots[0].vlines(0.19, -50, 600, linewidth=1, color='black')
    #plots[0].set_ylim(-10, 590)
    #plots[0].fill_between(cen[0], -(cen[2]+cen[3]-cennoqad[2]-cennoqad[3])*1e-6, cen[4]*1e-6, facecolor=color8, alpha=0.5)
    #plots[0].fill_between(cen[0], cen[1]*1e-6, cennoqad[1]*1e-6, facecolor=color8, alpha=0.5)
    #plots[0].set_ylabel(r'absorbed energy [$\frac{\rm{J}}{\rm{cm}^2}$]', fontsize=fs)
    #plots[0].set_xlabel(r'delay [ps]', fontsize=fs)
    #plots[0].annotate(r'total', (1.25, 460), fontsize=fs-1)
    #plots[0].annotate(r'$\rm{E}_{s}$', (1.2, 110), fontsize=fs-1, color=color5)
    #plots[0].annotate(r'$\Delta(\rm{E}_e+\rm{E}_p)$', (1.35, 110), fontsize=fs-1, color=color8)

    plots[1].yaxis.set_label_position("right")
    plots[3].yaxis.set_label_position("right")
    plots[1].yaxis.tick_right()
    plots[3].yaxis.tick_right()

    plt.setp(plots[1].get_xticklabels(), visible=False)
    plt.setp(plots[1].get_yticklabels(), visible=False)

    plots[1].set_xticks([])
    plots[1].set_yticks([])
    plots[1].set_xlim(-0.1,0.8)
    plots[1].set_ylim(0,1)
    plots[1].vlines(0.35, 0, 1, linewidth=1, color=col)
    plots[1].annotate(r'I', (0.3, 0.4), fontsize=fs, fontweight='bold')
    plots[1].annotate(r'II', (0.38, 0.4), fontsize=fs, fontweight='bold')
    plots[1].annotate(r'a)', (-0.08, 0.06), fontsize=fs, fontweight='bold')

    fig.savefig('C:/Users/tgrie/Desktop/Madrid Physik/paperpics/energy.pdf')

def koopmans(fs):
    
    n3tt=ownplot('Nickel/c3.dat', 'tem')
    n3testt=ownplot('Nickel/lattice/c3.dat', 'tem')
    n3koop=ownplot('Nickel/lattice/c3_koopmans.dat', 'mag')
    n3koopce=ownplot('Nickel/lattice/c3_koopmansce.dat', 'mag')
    n3koopt=ownplot('Nickel/lattice/c3_koopmans.dat', 'tem')
    n3kooptce=ownplot('Nickel/lattice/c3_koopmansce.dat', 'tem')

    fig, plots=plotter(8, 5, 1, 3, 0.0, 0.05, (-0.1,4.5), 0.1, 0, 0, [1,2])

    #nice=interpol('Ab-initio-Ni/Ni_c_e.txt')
    #nicp=interpol('Ab-initio-Ni/Ni_c_p.txt')
    #nigep=interpol('Ab-initio-Ni/Ni_G_ep.txt')

    #fece=interpol('Ab-initio-Fe/Fe_c_e.txt')
    #fecp=interpol('Ab-initio-Fe/Fe_c_p.txt')
    #fegep=interpol('Ab-initio-Fe/Fe_G_ep.txt')

    #coce=interpol('Ab-initio-Co/Co_c_e.txt')
    #cocp=interpol('Ab-initio-Co/Co_c_p.txt')
    #cogep=interpol('Ab-initio-Co/Co_G_ep.txt')

    plots[0].plot(n3[0], n3[1], color=color5, lw=2.0)
    plots[0].plot(n3koop[0], n3koop[1], color=color8, lw=2.0, ls='dashed')
    plots[0].plot(n3koopce[0], n3koopce[1], color=color8, lw=2.0, ls='dotted')
    
    plots[1].plot(n3tt[0], n3tt[1], color=color5, lw=2.0)
    plots[1].plot(n3koopt[0], n3koopt[1], color=color8, lw=2.0, ls='-')
    plots[1].plot(n3koopt[0], n3kooptce[1], color=color8, lw=2.0, ls='dotted')
    
    #plots[0].scatter(nit[0], nit[1], s=40, marker='x')
    #plots[0].plot(nite[0], nite[1], lw=2.0)

    plots[2].plot(n3tt[0], n3tt[2], color=color5, lw=2.0)
    plots[2].plot(n3koopt[0], n3koopt[2], color=color8, lw=2.0, ls='dashed')
    return(fig, plots)

def gadplot():
    fig, plots=plotter(8, 5, 1, 2, 0, 0.05, (-5,600), 0.1, 0, 0, [[],[0]], [])

    #READ exp mag data#
    dat0=gadmag('1.9')
    dat1=gadmag('3.7')
    dat2=gadmag('5.6')
    dat3=gadmag('7.4')
    dat4=gadmag('9.3')
    dat5=gadmag('11.2')

    #READ sim data#
    g0=ownplot('Gadolinium/sim1.9.dat', 'mag')
    g1=ownplot('Gadolinium/sim3.7.dat', 'mag')
    g1b=ownplot('Gadolinium/sim3.7b.dat', 'mag')
    g2=ownplot('Gadolinium/sim5.6.dat', 'mag')
    g2b=ownplot('Gadolinium/sim5.6b.dat', 'mag')
    g3=ownplot('Gadolinium/sim7.4.dat', 'mag')
    g3b=ownplot('Gadolinium/sim7.4b.dat', 'mag')
    g4=ownplot('Gadolinium/sim9.3.dat', 'mag')
    g4b=ownplot('Gadolinium/sim9.3b.dat', 'mag')
    g5=ownplot('Gadolinium/sim11.2.dat', 'mag')
    g5b=ownplot('Gadolinium/sim11.2b.dat', 'mag')
    #g2ot=ownplot('Gadolinium/sim7.4_other.dat', 'mag')

    gt=ownplot('Gadolinium/temp.dat', 'tem')
    #gtps=ownplot('Gadolinium/temp_gps.dat', 'tem')
    #gtce=ownplot('Gadolinium/temp_cehigh.dat', 'tem')
    #gtasf=ownplot('Gadolinium/temp_cehasfh.dat', 'tem')

    #PLOT exp mag data#
    plots[0].scatter(dat0[0]+0.1, dat0[1], s=40, marker='o', color=color1)
    plots[0].scatter(dat1[0]+0.1, dat1[1], s=40, marker='o', color=color4)
    plots[0].scatter(dat2[0]+0.1, dat2[1], s=40, marker='o', color=color6)
    plots[0].scatter(dat3[0]+0.1, dat3[1], s=40, marker='o', color=color8)
    plots[0].scatter(dat4[0]+0.1, dat4[1], s=40, marker='o', color=colors[9])
    plots[0].scatter(dat5[0]+0.1, dat5[1], s=40, marker='o', color=colors[11])

    #PLOT sim mag data#
    plots[0].plot(g0[0]-30, g0[1]/0.948, lw=3.0, color=color1)
    plots[0].plot(g1[0]-30, g1[1]/0.948, lw=3.0, color=color4)
    plots[0].plot(g2[0]-30, g2[1]/0.948, lw=3.0, color=color6)
    plots[0].plot(g3[0]-30, g3[1]/0.948, lw=3.0, color=color8)
    plots[0].plot(g4[0]-30, g4[1]/0.948, lw=3.0, color=colors[9])
    plots[0].plot(g5[0]-30, g5[1]/0.948, lw=3.0, color=colors[11])

    #READ exp temp data#
    dat=open('C:/Users/tgrie/Desktop/3TM_results/Gadolinium/gdtemp.txt','r')
    vals=dat.readlines()
    x=np.array([float(i.split()[0]) for i in vals[1:]])
    tl=np.array([float(line.split()[4]) for line in vals[1:]])
    te=np.array([float(line.split()[5]) for line in vals[1:]])

    #PLOT exp temp data#
    plots[1].scatter(x-0.02, te, s=40, marker='o')

    #PLOT sim tem data#
    plots[1].plot(gt[0]-30, gt[1], lw=3.0, label=r'mag sim')
    #plots[1].plot(gtps[0]-30, gtps[1], lw=3.0, color='blue', ls='dashed', label=r'\sigma higher')
    #plots[1].plot(gtce[0]-30, gtce[1], lw=3.0, color='red', ls='solid', label=r'ce higher')
    #plots[1].plot(gtasf[0]-30, gtasf[1], lw=3.0, color='green', ls='dashed', label=r'ce higher, asf higher')

    #modify plot#
    plots[0].set_xlabel(r'delay [ps]')
    plots[0].set_ylabel(r'Normalized magnetization')
    #plots[0].set_xscale('log')

    plots[1].set_ylabel(r'Electron temperature [K]')
    plots[1].set_xlabel(r'delay [ps]')
    plots[1].set_xlim(-0.2, 5)
    plt.legend()
    #fig.savefig('C:/Users/tgrie/Desktop/3TM_results/Gadolinium/solocomp.pdf')


def niplot(fs):
    fig, pics=plotter(8,6, 1, 2, 0, 0.05, (-0.1, 2.5), 0.2, 0, 0, [[1],[0]], [1])

    for i, dat in enumerate(ns):
        t=dat[0]-0.003*i
        pics[1].plot(t, dat[1], color=colors[i+1], lw=3.0)


    pics[1].legend([r'76', r'456', r'758', r'916', r'1069', r'1377', r'1737'], loc='center left', bbox_to_anchor=(1.01, 0.5),
          ncol=1, fancybox=True, shadow=True, fontsize=fs-4)

    dirplot('Nickel', 0.1, 20, colors, pics[1], 0, 7, [])


    pics[0].plot(ntc[5][0], ntc[5][1], color=colors[7], lw=3.0)
    pics[0].plot(ntc[4][0], ntc[4][1], color=colors[6], lw=3.0)
    pics[0].plot(ntc[3][0], ntc[3][1], color=colors[3], lw=3.0)
    pics[0].plot(ntc[2][0], ntc[2][1], color=colors[2], lw=3.0)
    pics[0].plot(ntc[1][0], ntc[1][1], color=colors[1], lw=3.0)
    pics[0].plot(ntc[0][0], ntc[0][1], color=colors[0], lw=3.0)

    pics[0].legend([r'1449', r'1108', r'605', r'332', r'94', r'26'], loc='center left', bbox_to_anchor=(1.01, 0.5),
          ncol=1, fancybox=True, shadow=True, fontsize=fs-4)
    
    pics[0].scatter(n3td[0], n3td[1], color=colors[2], s=40)
    pics[0].scatter(n4td[0], n4td[1], color=colors[3], s=40)
    pics[0].scatter(n5td[0], n5td[1], color=colors[6], s=40)
    pics[0].scatter(n6td[0]+0.15, n6td[1], color=colors[7], s=40)
    pics[0].scatter(n2td[0], n2td[1], color=colors[1], s=40)
    pics[0].scatter(n1td[0], n1td[1], color=colors[0], s=40)

    pics[0].set_xlabel(r'delay [ps]', fontsize=fs)
    pics[0].set_ylabel(r'Lat. temperature [K]', fontsize=fs)
    pics[1].set_ylabel(r'Norm. magnetization', fontsize=fs)

    pics[0].annotate(r'(b)', (-0.05, 550), fontsize=fs)
    pics[1].annotate(r'(a)', (-0.05, 0.1), fontsize=fs)

    #fig.savefig('C:/Users/tgrie/Desktop/Madrid Physik/paperpics/Nickel.pdf')
    #plt.xscale('log')

    
#energy(17, color8)
#koopmans(10)
highflu(16, 1.5)
#gadplot()
#niplot(17)

plt.show()