from __future__ import print_function
import cv2 
import argparse
import math
import matplotlib.colors as colors
import matplotlib.cm as cmx
from matplotlib import gridspec
from matplotlib.backends.backend_pdf import PdfPages
import os
import datetime
import graphical
import numpy as np
import matplotlib.pyplot as plt
#from mpl_toolkits.basemap import Basemap
#from mpl_toolkits.axes_grid.inset_locator import inset_axes

# all the info
#iev,mags,tms,loc = readTRbostock.readTRbostockall()

# RTR info
#trlm, szs = readTRbostock.readTRrtrtime()

def locfigure():

    from mpl_toolkits.axes_grid.inset_locator import inset_axes
    # read data
    iev,mags,tms,loc = readTRbostockall()
    trlm, szs = readTRrtrtime()
    
    # just pick some
    trlm = trlm[:,28:]

    # initialize
    plt.close()
    f = plt.figure(figsize=(9,6.5))
    #gs=gridspec.GridSpec(1,1)
    #p = plt.subplot(gs[0])
    p  = plt.axes()
    
    # limits
    ylm = np.array([47.8, 49.1])
    xlm = np.array([-125.,-122.8])

    ylm2 = np.array([41, 55.])
    xlm2 = np.array([-131.,-111.])

    # basemap
    m = Basemap(llcrnrlon=xlm[0],llcrnrlat=ylm[0],urcrnrlon=xlm[1],urcrnrlat=ylm[1],
                projection='lcc',resolution='i',lat_0=ylm.mean(),lon_0=xlm.mean(),
                suppress_ticks=True)
    m.drawlsmask(land_color='whitesmoke',ocean_color='aliceblue',lakes=True,
                 resolution='i',grid=1.25)
    m.drawcoastlines()
    xsp = roundsigfigs(np.diff(xlm)/4,1)
    lontk = np.arange(round(xlm[0],1),round(xlm[1],1),xsp)
    m.drawmeridians(lontk,labels=[1,0,0,1])

    ysp = roundsigfigs(np.diff(ylm)/4,1)
    lattk = np.arange(round(ylm[0],1),round(ylm[1],1),ysp)
    m.drawparallels(lattk,labels=[1,0,0,1])

    ps = p.get_position()
    p2 = plt.axes([.57,.65,0.3,0.3])
    f.sca(p2)

    # basemap
    m2 = Basemap(llcrnrlon=xlm2[0],llcrnrlat=ylm2[0],urcrnrlon=xlm2[1],urcrnrlat=ylm2[1],
                 projection='lcc',resolution='l',lat_0=ylm2.mean(),lon_0=xlm2.mean(),
                 suppress_ticks=True)
    m2.drawlsmask(land_color='whitesmoke',ocean_color='aliceblue',lakes=True,
                  resolution='f',grid=1.25)
    m2.drawcoastlines()
    m2.drawstates()
    m2.drawcountries()

    xsp = roundsigfigs(np.diff(xlm2)/4,1)
    lontk = np.arange(round(xlm2[0],1),round(xlm2[1],1),xsp)
    m2.drawmeridians(lontk,labels=[0,0,0,0])

    ysp = roundsigfigs(np.diff(ylm2)/4,1)
    lattk = np.arange(round(ylm2[0],1),round(ylm2[1],1),ysp)
    m2.drawparallels(lattk,labels=[0,0,0,0])


    bxx = xlm[np.array([0,0,1,1,0])]
    bxy = ylm[np.array([0,1,1,0,0])]
    bxx,bxy =  m2(bxx,bxy)
    plt.plot(bxx,bxy,color='blue',linewidth=2)

    f.sca(p)

    fl = os.path.join(os.environ['WRITTEN'],'RTRStrain','cont-32p5')
    vls = np.loadtxt(fl)
    x,y = m(vls[:,0],vls[:,1])
    plt.plot(x,y,color='k')

    fl = os.path.join(os.environ['WRITTEN'],'RTRStrain','cont-43')
    vls = np.loadtxt(fl)
    x,y = m(vls[:,0],vls[:,1])
    plt.plot(x,y,color='k')


    # find all the values in an rtr
    inrtr = np.zeros(tms.shape)
    for k in range(0,trlm.shape[1]):
        inrtr[np.logical_and(tms>=trlm[0,k],tms<=trlm[1,k])]=1

    # count number in RTR time intervals
    nper = np.bincount(iev,inrtr,loc.shape[1])

    # plot with coloring
    x,y = m(loc[0,:],loc[1,:])
    vls = plt.scatter(x,y,c=nper,s=60,vmin=0.,vmax=max(nper))
    plt.set_cmap('Reds')
    cb = plt.colorbar(vls,fraction=0.03, pad=0.14)
    cb.set_label('detections at RTR times')

    # stations
    st = np.array(['B003','B004'])
    sloc = np.array([[-124.140861511, 48.062358856],[-124.427009583,48.201931000]])
    shf = np.diff(xlm)[0]*0.02

    for k in range(0,len(st)):
        x,y = m(sloc[k,0],sloc[k,1])
        plt.scatter(x,y,marker='^',color='k',facecolor='k',s=50)
        x,y = m(sloc[k,0]+shf,sloc[k,1])
        plt.text(x,y,st[k],fontsize=14,verticalalignment='top')

    x,y=m(-124.3,48.9)
    plt.text(x,y,'Vancouver Island',verticalalignment='center',
             fontsize=14,horizontalalignment='center',
             backgroundcolor='whitesmoke')
    x,y=m(-123.8,47.9)
    plt.text(x,y,'Olympic Peninsula',verticalalignment='center',
             fontsize=14,horizontalalignment='center',
             backgroundcolor='whitesmoke')

    prt = True
    if prt:
        fname='SRlocation_figures'
        fname=os.path.join(os.environ['FIGURES'],fname+'.pdf')
        pp=PdfPages(fname)
        pp.savefig(f)
        pp.close()
        plt.clf()
        plt.close(f)
    else:
        plt.show()


def roundsigfigs(x,n):
    from math import log10,floor
    nr = int(floor(log10(abs(x))))
    x = round(x,n-nr-1)

    return x
    

def sstimes(yr,tms):
    t1 = datetime.datetime(yr,1,1)
    t2 = datetime.datetime(yr+1,1,1)
    iok = np.logical_and(tms>t1,tms<t2)
    tmn = np.sort(tms[iok])
    ii = int(len(tmn)/2)
    tmn = tmn[ii]

    return(tmn)


def readTRbostockall():
    
    import numpy as np

    # LFE locations
    evn,snm,loc = readTRbostockloc()

    # LFE times
    iev,mags,nm,tms = readTRbostock()
    # change mapping
    iev = np.array(ismember(iev,snm))

    # subset
    ix = iev!=0
    iev,mags,nm,tms=iev[ix],mags[ix],nm[ix],tms[ix]

    return iev,mags,tms,loc

def ismember(a, b):
    bind = {}
    for i, elt in enumerate(b):
        if elt not in bind:
            bind[elt] = i
    return [bind.get(itm, 0) for itm in a]

def readTRbostockloc():

    import os,datetime
    import numpy as np

    fdir = os.path.join(os.environ['DATA'],'TREMOR','Bostock')
    fname = 'svi_sta.dat'
    fname = os.path.join(fdir,fname)
    vl = np.genfromtxt(fname,dtype=None)

    # event numbers
    evn=np.array([x[0] for x in vl])
    snm=np.array([int(x[1][1:]) for x in vl])
    lat=np.array([x[2] for x in vl])
    lon=np.array([x[3] for x in vl])
    dep=np.array([x[4] for x in vl])
    
    evn=np.array([x[0] for x in vl])
    loc=np.array([lon,lat,dep])

    return evn,snm,loc 

def readTRbostock():
    
    import os,datetime
    import numpy as np
    import pytz

    fdir  = os.path.join(os.environ['DATA'],'TREMOR','Cascadia')
    fdir2 = os.path.expanduser('~/data/tremors')
    fname = 'total_mag_detect_0000_cull.txt'
    
    utc=pytz.timezone('UTC')


    try:
        fl = open(os.path.join(fdir,fname),'r')
    except:
        fl = open(os.path.join(fdir2,fname),'r')
        
    # event index
    iev = [];

    # times
    tms = []

    # magnitudes
    mags = [];

    # some number
    nm = []

    for line in fl:
        # read
        vl = line.split()

        yr=int(vl[1][0:2])
        if yr < 90:
            yr = yr + 2000
        else:
            yr = yr + 1900

        sc = vl[3].split('.')
        ms = int(sc[1])
        sc = int(sc[0])
        hr=int(vl[2]) - 1

        dt=datetime.datetime(year=yr,month=int(vl[1][2:4]),
                             day=int(vl[1][4:6]),hour=hr)
        dt=dt+datetime.timedelta(minutes=0,seconds=sc,milliseconds=ms)
        #dt=dt.astimezone(utc)

        # add to list
        nm.append(int(vl[5]))
        mags.append(float(vl[4]))
        tms.append(dt)
        iev.append(int(vl[0]))

    tms = np.array(tms)
    mags = np.array(mags)
    iev = np.array(iev)
    nm = np.array(nm)

    fl.close()

    return iev,mags,nm,tms


def readTRrtrtime():
    
    # OUTPUT
    #
    # trlm           [start, stop] times
    # szs            sizes: [length,width,velocity]

    import os
    import datetime
    import numpy as np

    fdir = os.path.join(os.environ['DATA'],'TREMOR','RoyerRTR')
    fname = 'Royer2014'
    fname = os.path.join(fdir,fname)

    vl = np.genfromtxt(fname,dtype=None,skiprows=1)

    rt1,rt2,ln,wd,ve = [],[],[],[],[]
    for x in vl:
        dt=datetime.datetime(x[0],int(x[2][3:]),int(x[2][0:2]),int(x[3][0:2]),int(x[3][3:]))
        rt1.append(dt)
        dt=datetime.datetime(x[0],int(x[4][3:]),int(x[4][0:2]),int(x[5][0:2]),int(x[5][3:]))
        rt2.append(dt)
        ln.append(x[6])
        wd.append(x[7])
        ve.append(x[8])
    
    trlm = np.array([rt1,rt2])
    szs = np.array([ln,wd,ve])

    return trlm, szs

def plotTRratevtime(iev,tms,loc,tlm,tsp,ievi=None,plot=2):

    # INPUT
    # 
    # iev         event indices
    # tms         times 
    # evn         numbers for locations
    # loc         locations
    # tlm         time limits to plot
    # tsp         time spacing for averaging
    # ievi        indices to use
    # plot        plot? (default: True)
    
    import numpy as np
    import matplotlib.pyplot as plt
    from mpl_toolkits.basemap import Basemap
    import math
    import datetime

    # extract the times of interest
    tbuf=datetime.timedelta(hours=2)
    ix=[tms[k]>tlm[0]-tbuf and tms[k]<tlm[1]+tbuf for k in range(0,len(tms))]
    ix = np.array(ix)
    tmi = tms[ix]
    tmi=[(tm-tlm[0]).total_seconds()/3600 for tm in tmi]
    tmi=np.array(tmi)
    iev = iev[ix]

    # maximum time
    mxt = (tlm[1]-tlm[0]).total_seconds()/3600

    # if template isn't specified, use the ones with many event
    if ievi is None:
        ix = np.logical_and(tmi>=0,tmi<mxt)
        nper=np.bincount(iev[ix])
        ievi = np.where(nper > max(nper)*0.)[0]
    
    # number of templates
    Nev = len(ievi)

    # grid times
    tsph=tsp/60.
    tim = int(mxt/tsph*10)
    tim = np.linspace(0,mxt,tim)

    # initialize
    rate = np.ndarray([len(tim),Nev])

    # just count the stupid way
    for k in range(0,Nev):
        ixe = iev==ievi[k]
        tmh=np.sort(tmi[ixe])

        # find where the indices fit
        i1 = np.searchsorted(tmh,tim-tsph/2.)
        i2 = np.searchsorted(tmh,tim+tsph/2.)
        
        # number of detections
        i2=(i2-i1).astype(float)
        rate[:,k] = i2
            

    if plot==1:
        f = plt.figure(figsize=(10,10))
        gs,p=gridspec.GridSpec(Nev,1),[]
        for gsi in gs:
            p.append(plt.subplot(gsi))

        for k in range(0,Nev):
            f.sca(p[k])
            plt.plot(tim,rate[:,k])
            
        p[Nev-1].set_xlabel('time since '+str(tlm[0])+' (hours)')
        lb = 'detections per '+str(tsp)+' minutes'
        p[int(Nev/2.-1)].set_ylabel(lb)

        plt.show()
    elif plot==2:
        # strike direction
        stk = 160.
        stkd = np.array([math.sin(stk*math.pi/180),math.cos(stk*math.pi/180)])
        dipd = np.array([math.sin((stk-90)*math.pi/180),math.cos((stk-90)*math.pi/180)])
    
        # project
        xl,yl=loc[0][ievi],loc[1][ievi]
        # reference
        xlr,ylr=np.mean(xl),np.mean(yl)
        xlr,ylr=-124.25,49.
        xl,yl=xl-xlr,yl-ylr
        rt=math.cos(ylr*math.pi/180)
        sk=stkd[0]*xl*rt+stkd[1]*yl
        dp=dipd[0]*xl*rt+dipd[1]*yl
        sk,dp=sk*110,dp*110

        # to scale the amplitude
        ylm = np.array([min(sk),max(sk)])
        clm = np.array([min(dp),max(dp)])
        scl = np.diff(ylm)*.1
        ylm = ylm + np.array([-.5,1.5])*scl
        scl = scl/np.max(rate)

        jet = cm = plt.get_cmap('jet') 
        cNorm  = colors.Normalize(vmin=clm[0],vmax=clm[1])
        scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=jet)
        scalarMap.set_array([])

        # # Using contourf to provide my colorbar info, then clearing the figure
        # Z = [[0,0],[0,0]]
        # levels = np.arange(clm[0],clm[1],1)
        # CS3 = plt.contourf(Z, levels, cmap=jet)
        # plt.clf()

        f = plt.figure(figsize=(10,10))
        gs=gridspec.GridSpec(2,1)
        p=plt.subplot(gs[0])

        for k in range(0,Nev):
            colorVal = scalarMap.to_rgba(dp[k])
            plt.plot(tim,rate[:,k]*scl+sk[k],color=colorVal)
            
        p.set_xlabel('time since '+str(tlm[0])+' (hours)')
        lb = 'detections per '+str(tsp)+' minutes'
        p.set_ylabel(lb)
        p.set_ylim(ylm)
        p.set_xlim([0,mxt])

        cb = plt.colorbar(scalarMap)
        cb.set_label('distance along dip (km)')

        #plt.show()

    prt=plot
    if prt:
        # for output
        fname=tlm[0].strftime('%Y-%b-%d-%H-%M')+'_'+str(tsp)
        fname=os.path.join(os.environ['FIGURES'],'ratewtime_'+fname+'.pdf')
        pp=PdfPages(fname)
        pp.savefig(f)
        pp.close()
        plt.clf()
        plt.close(f)
        
    return tim,rate,ievi


def plotTRstrike(iev,tms,loc,tlm):

    # INPUT
    # 
    # iev         event indices
    # tms         times 
    # evn         numbers for locations
    # loc         locations
    # tlm         time limits to plot

    import numpy as np
    import matplotlib.pyplot as plt
    from mpl_toolkits.basemap import Basemap
    import math
    import datetime

    # extract the times of interest
    tbuf=datetime.timedelta(hours=2)
    ix=[tms[k]>tlm[0]-tbuf and tms[k]<tlm[1]+tbuf for k in range(0,len(tms))]
    ix = np.array(ix)
    tmi = tms[ix]
    tmi=[(tm-tlm[0]).total_seconds()/3600 for tm in tmi]
    tmi=np.array(tmi)

    # want to plot locations
    ixi=iev[ix]
    xl = loc[0,ixi]
    yl = loc[1,ixi]

    # reference
    xlr,ylr=np.mean(xl),np.mean(yl)
    xlr,ylr=-124.25,49.
    xl,yl=xl-xlr,yl-ylr

    # strike direction
    stk = 160.
    stkd = np.array([math.sin(stk*math.pi/180),math.cos(stk*math.pi/180)])
    dipd = np.array([math.sin((stk-90)*math.pi/180),math.cos((stk-90)*math.pi/180)])
    
    # project
    rt=math.cos(ylr*math.pi/180)
    sk=stkd[0]*xl*rt+stkd[1]*yl
    dp=dipd[0]*xl*rt+dipd[1]*yl
    sk,dp=sk*110,dp*110


    f = plt.figure(figsize=(10,10))

    vmin,vmax=min(dp),max(dp)
    h = plt.scatter(tmi,sk,c=dp,s=20,vmin=vmin,vmax=vmax)
    plt.jet()
    plt.xlabel('time since '+str(tlm[0])+' (hours)')
    plt.ylabel('distance along strike (km)')
    cb = plt.colorbar()
    cb.set_label('distance along dip (km)')


    # for output
    fname=tlm[0].strftime('%Y-%b-%d-%H-%M')
    fname=os.path.join(os.environ['FIGURES'],'strikewtime_'+fname+'.pdf')
    pp=PdfPages(fname)
    pp.savefig(f)
    pp.close()

    plt.close()
    #    plt.show()

def plotTRmeanvtime(iev,tms,loc,tlm,tsp,Nbin=8):

    # INPUT
    # 
    # iev         event indices
    # tms         times 
    # evn         numbers for locations
    # loc         locations
    # tlm         time limits to plot
    # tsp         time spacing for averages, in minutes
    # Nbin        number of bins

    import numpy as np
    import matplotlib.pyplot as plt
    from mpl_toolkits.basemap import Basemap
    import datetime

    # times of interest
    mxt = (tlm[1]-tlm[0]).total_seconds()/3600
    tbuf = datetime.timedelta(days=0,hours=2)
    tlmi = tlm.copy()
    tlmi[0],tlmi[1]=tlm[0]-tbuf,tlm[1]+tbuf

    # get rates through time
    ievi = np.unique(iev)
    tim,rate,ievi=plotTRratevtime(iev,tms,loc,tlmi,tsp,ievi,0)
    
    # shift
    tim = tim - tbuf.total_seconds()/3600.
    tlmp = np.array([-1,1])*tbuf.total_seconds()/3600. + np.array([0,mxt])

    # strike direction
    stk = 140.
    stkd = np.array([math.sin(stk*math.pi/180),math.cos(stk*math.pi/180)])
    dipd = np.array([math.sin((stk-90)*math.pi/180),math.cos((stk-90)*math.pi/180)])
    
    # project
    xl,yl=loc[0][ievi],loc[1][ievi]
    # reference
    xlr,ylr=np.mean(xl),np.mean(yl)
    xlr,ylr=-124.25,49.
    xl,yl=xl-xlr,yl-ylr
    rt=math.cos(ylr*math.pi/180)
    sk=stkd[0]*xl*rt+stkd[1]*yl
    dp=dipd[0]*xl*rt+dipd[1]*yl
    sk,dp=sk*110,dp*110

    xlm = np.array([sk.min(), sk.max()])
    xlm = xlm+np.array([-1,1])*5
    ylm = np.array([dp.min(), dp.max()])

    lm = int(np.diff(ylm)/5.)
    lm=Nbin
    lm = np.linspace(ylm[0],ylm[1],lm+1)
    yc = (lm[0:-1]+lm[1:])/2.
    dyc = np.mean(np.diff(lm))

    # weighting along strike
    ski = np.arange(xlm[0],xlm[1],1)
    wgts = np.ndarray([len(sk),len(ski)])

    rt = np.ndarray([len(tim),len(ski),len(yc)])
    vmax=0
    ibel=[]
    for m in range(0,len(yc)):
        lmi = yc[m]+np.array([-1,1])*dyc
        ii=np.logical_or(dp<lmi[0],dp>lmi[1])
        ibel.append(np.where(~ii))
        for k in range(0,len(ski)):
            # weighting
            wgt=np.exp(-((sk-ski[k])/5)**2)
            wgt[ii]=0
            wgt=wgt/sum(wgt)
            wgts[:,k]=wgt
        
        rt[:,:,m] = np.dot(rate,wgts)

    vmax=np.max(rt)
    # identify 90% values
    crt = rt.cumsum(axis=1)
    # maxima and repeat
    mx = crt[:,len(ski)-1,:]
    inan=mx<np.max(mx)*0.1
    mx = mx.reshape([len(tim),1,len(yc)])
    mx = mx.repeat(repeats=len(ski),axis=1)
    i1 = (crt < mx*0.15).sum(axis=1)
    i2 = (crt <= mx*0.85).sum(axis=1)-1
    i1,i2=ski[i1],ski[i2]
    i1[inan] = float('nan')    
    i2[inan] = float('nan')    


    plot=1
    tpl=np.array([min(tim),max(tim)])
    N=len(yc)
    if plot:
        f = plt.figure(figsize=(20,10))
        if N>1:
            gs,p=gridspec.GridSpec(2,N/2),[]
        else:
            gs,p=gridspec.GridSpec(1,1),[]
        for gsi in gs:
            p.append(plt.subplot(gsi))
        plt.set_cmap('Reds')
        for k in range(0,N):
            f.sca(p[k])
            for m in ibel[k]:
                plt.plot([sk[m],sk[m]],tpl,color='lightgray',linestyle=':')
            lmi = yc[k]+np.array([-1,1])*dyc
            extent = np.append(xlm,tpl)
            plt.plot(xlm,[0,0],color='lightgray',linestyle=':')
            plt.plot(xlm,[mxt,mxt],color='lightgray',linestyle=':')
            plt.imshow(rt[:,:,k],vmin=0,vmax=vmax,extent=extent,aspect='auto',
                       origin='lower')
            lb = ('%0.0f - ' % lmi[0])+('%0.0f km' % lmi[1])
            p[k].set_title(lb)
            plt.plot(i1[:,k],tim,color='gray')
            plt.plot(i2[:,k],tim,color='gray')
            p[k].set_xlim(xlm)
            p[k].set_ylim([min(tim),max(tim)])
        for k in range(0,N/2):
            p[k].set_xticklabels([])
        for k in range(N/2,N):
            p[k].set_xlabel('distance along strike (km)')
        for k in range(1,N/2):
            p[k].set_yticklabels([])
        for k in range(N/2+1,N):
            p[k].set_yticklabels([])
        p[0].set_ylabel('time since '+str(tlm[0])+' (hours)')

        #plt.show()

        wrt = False
        if wrt:
            fdir = os.path.join(os.environ['DATA'],'TREMOR','LOCWTIME')
            fname='locs-'+tlm[0].strftime('%Y')+'-'+str(tsp)
            fl = open(os.path.join(fdir,fname),'w')
            fl.write(tlm[0].strftime('%Y-%b-%d-%H-%M')+'\n')
            for k in range(0,len(tim)):
                fl.write(str(tim[k])+','+str(i1[k][0])+','+str(i2[k][0])+'\n')
            fl.close()

        # for output
        fname=tlm[0].strftime('%Y-%b-%d-%H-%M')
        fname=os.path.join(os.environ['FIGURES'],'meanwtime_'+fname+'.pdf')
        pp=PdfPages(fname)
        pp.savefig(f)
        pp.close()

        plt.close()

    return


def plotTRlocrate(iev,tms,loc,tlm,tsp):

    # INPUT
    # 
    # iev         event indices
    # tms         times 
    # evn         numbers for locations
    # loc         locations
    # tlm         time limits to plot
    # tsp         time spacing for averages, in minutes

    import numpy as np
    import matplotlib.pyplot as plt
    from mpl_toolkits.basemap import Basemap
    import datetime

    # times of interest
    mxt = (tlm[1]-tlm[0]).total_seconds()/3600

    xlm = np.array([loc[0].min(), loc[0].max()])
    xlm = xlm + np.array([-1,1])*np.diff(xlm)*0.1
    ylm = np.array([loc[1].min(), loc[1].max()])
    ylm = ylm + np.array([-1,1])*np.diff(ylm)*0.1

    # get rates through time
    ievi = np.unique(iev)
    tim,rate,ievi=plotTRratevtime(iev,tms,loc,tlm,tsp,ievi,0)


    # beforehand
    ibf = np.logical_and(tms>=(tlm[0]-datetime.timedelta(days=3)),tms<tlm[0])
    ievbf = np.unique(iev[ibf])
    nbf = np.bincount(iev[ibf],minlength=np.max(iev)+1)


    plt.close()

    f = plt.figure(figsize=(20,7))
    gs=gridspec.GridSpec(2,1,height_ratios=[1,10],width_ratios=[0.9,1])
    p=plt.subplot(gs[1])
    pb=plt.subplot(gs[0])
    f.sca(p)

    m = Basemap(llcrnrlon=xlm[0],llcrnrlat=ylm[0],urcrnrlon=xlm[1],urcrnrlat=ylm[1],
                projection='lcc',resolution='i',lat_0=ylm.mean(),lon_0=xlm.mean(),
                suppress_ticks=True)
    m.drawlsmask(land_color='lightgray',ocean_color='skyblue',lakes=True,
                 resolution='i',grid=1.25)
    m.drawcoastlines()
    xsp = roundsigfigs(np.diff(xlm)/4,1)
    lontk = np.arange(round(xlm[0],1),round(xlm[1],1),xsp)
    m.drawmeridians(lontk,labels=[1,0,0,1])

    ysp = roundsigfigs(np.diff(ylm)/4,1)
    lattk = np.arange(round(ylm[0],1),round(ylm[1],1),ysp)
    m.drawparallels(lattk,labels=[1,0,0,1])

    # locations
    x,y = m(loc[0][ievi],loc[1][ievi])

    # to map rates
    vmin = 0
    vmax = np.max(rate)

    hoth = m.scatter(x,y,c=nbf[ievi],s=180,vmin=0,vmax=max(nbf))

    # number of differences
    ndf = 20

    # to start
    hbf = m.scatter(x,y,c=rate[0,:],s=70,vmin=vmin,vmax=vmax)
    h = m.scatter(x,y,c=rate[ndf,:],s=30,vmin=vmin,vmax=vmax)
    hbf.set_edgecolor('none')
    h.set_edgecolor('none')

    plt.jet()
    cb = m.colorbar()
    cb.set_label('detections per '+str(tsp)+' minutes')

    # indicate timing
    hb,=pb.plot([tim[1],tim[1]],[0,1],linewidth=4,color='blue')
    pb.set_xlim([0,mxt])
    pb.set_ylim([0,1])
    pb.set_yticks([])
    pb.set_xlabel('time since '+str(tlm[0])+' (hours)')
    pb.xaxis.set_label_position('top') 
    pb.xaxis.tick_top()

    # files for movie
    files = []
    fdir = os.environ['FIGURES']
    fnamea = os.path.join(fdir,'locrate')
    fname = fnamea+'_tmp%03d.png' % 0
    print('Saving frame', fname)
    plt.savefig(fname)
    files.append(fname)

    for k in range(ndf+1,len(tim)):
        # remove the last ones
        hbf.remove()
        h.remove()
        hb.remove()
        
        hb,=pb.plot([tim[k],tim[k]],[0,1],linewidth=4,color='blue')
        hbf = m.scatter(x,y,c=rate[k-ndf,:],s=80,vmin=vmin,vmax=vmax) 
        h = m.scatter(x,y,c=rate[k,:],s=70,vmin=vmin,vmax=vmax)

        hbf.set_edgecolor('none')
        h.set_edgecolor('none')

        fname = fnamea+'_tmp%03d.png' % k
        print('Saving frame', fname)
        plt.savefig(fname)
        files.append(fname)


    print('Making movie animation.mpg - this make take a while')
    nm=tlm[0].strftime('%Y-%b-%d-%H-%M')
    nm=os.path.join(fdir,'animation_'+nm+'.mpg')
    nms=os.path.join(fdir,'locrate_tmp*.png')
    os.system("convert "+nms+" "+nm)

    # cleanup
    for fname in files:
        os.remove(fname)

def rewritertr():
    import glob

    fdir = os.path.join(os.environ['DATA'],'TREMOR','RoyerRTR')
    fls=glob.glob(os.path.join(fdir,'locrect*'))

    flw = os.path.join(fdir,'alllocrect')
    flw = open(flw,'w')


    for fname in fls:
        fl = open(fname,'r')
        line = fl.readline()
        line = fl.readline()
        stk = fl.readline()
        dip = fl.readline()
        tim = fl.readline()
        sdir = fl.readline()
        fl.close()

        stk = stk.split()
        dip = dip.split()
        tim = tim.rstrip()
        sdir = sdir.rstrip()
        
        flw.write(tim+','+stk[0]+','+stk[1]+','+dip[0]+','+dip[1]+','+sdir+'\n')

    flw.close()



def plotTRlocsnap(iev,tms,loc,tlm,prt=True):

    # INPUT
    # 
    # iev         event indices
    # tms         times 
    # loc         locations
    # tlm         time limits to plot

    import numpy as np
    import matplotlib.pyplot as plt
    from mpl_toolkits.basemap import Basemap
    import datetime

    # times of interest
    mxt = (tlm[1]-tlm[0]).total_seconds()/3600
    N =6
    tsp = mxt/N

    xlm = np.array([loc[0].min(), loc[0].max()])
    xlm = xlm + np.array([-1,1])*0.02
    xlm = np.array([-124.45,-123.])
    ylm = np.array([loc[1].min(), loc[1].max()])
    ylm = ylm + np.array([-1,1])*0.02
    ylm = np.array([48.,48.9])

    # get rates through time
    ievi = np.unique(iev)
    tim,rate,ievi=plotTRratevtime(iev,tms,loc,tlm,tsp*60,ievi,0)

    # beforehand
    ibf = np.logical_and(tms>=(tlm[0]-datetime.timedelta(days=3)),
                         tms<tlm[0])
    ievbf = np.unique(iev[ibf])
    nbf = np.bincount(iev[ibf],minlength=np.max(iev)+1)

    # load a contour
    fl = os.path.join(os.environ['WRITTEN'],'RTRStrain','cont-37')
    vls = np.loadtxt(fl)

    # median latitude
    mlat = np.mean(loc[1][nbf>10])
    ix,=np.where(np.sum(rate,axis=0)>5)
    mlat = np.mean(loc[1][ix])
    ix=np.logical_and(vls[:,1]>=mlat-.25,vls[:,1]<=mlat+0.25)
    ix,=np.where(ix)
    dx = np.cos(mlat*math.pi/180.)
    dx = dx * (vls[ix[0],0]-vls[ix[-1],0])
    dy = (vls[ix[0],1]-vls[ix[-1],1])
    global stk
    stk = math.atan(dx/dy)*180./math.pi
    stk = stk % 180.
    print(mlat)
    print(stk)

    #x,y = m(vls[:,0],vls[:,1])
    #plt.plot(x,y,color='k')

    fnamew=tlm[0].strftime('%Y-%b-%d-%H-%M')
    fnamew=os.path.join(os.environ['DATA'],'TREMOR','RoyerRTR',
                        'locrect_'+fnamew)
    if os.path.isfile(fnamew):
        fl = open(fnamew,'r')
        line = fl.readline()
        line = fl.readline()
        line = fl.readline()
        xsp = line.split()
        xsp = np.array([float(xsp[0]),float(xsp[1])])
        line = fl.readline()
        xdp = line.split()
        xdp = np.array([float(xdp[0]),float(xdp[1])])
        fl.close()
        
        xsp=xsp[np.array([0,0,1,1,0])]
        xdp=xdp[np.array([0,1,1,0,0])]
        xsp,xdp = skdptolatlon(xsp,xdp,stk)
    else:
        xdp = []
        xsp = []
     
    # times
    tm = np.linspace(0,mxt,N+1)
    tm = (tm[:-1]+tm[1:])/2.
    
    # indices of certain times
    ix = []
    for k in range(0,N):
        ix.append(np.argmin(np.abs(tim-tm[k])))
    ix = np.array(ix)

    tim = tim[ix]
    rate = rate[ix,:]

    # strike direction
    stkd = np.array([math.sin(stk*math.pi/180),
                     math.cos(stk*math.pi/180)])
    dipd = np.array([math.sin((stk-90)*math.pi/180),
                     math.cos((stk-90)*math.pi/180)])
    
    # reference
    xlr,ylr=-124.25,49.
    rt=math.cos(ylr*math.pi/180)
    sk=np.arange(-200,200,10)
    dp=np.arange(-100,100,10)

    global m,p,f
    plt.close()
    f = plt.figure(figsize=(9.2,5.5))
    
    #width_ratios=[.3,.3,.3,.1]
    gs,p=gridspec.GridSpec(2,N/2),[]
    for k in range(0,N):
        p.append(plt.subplot(gs[k]))
    pm=np.array(p)
    pm=pm.reshape((N/2,2))
        
    m = Basemap(llcrnrlon=xlm[0],llcrnrlat=ylm[0],
                urcrnrlon=xlm[1],urcrnrlat=ylm[1],
                projection='lcc',resolution='i',
                lat_0=ylm.mean(),lon_0=xlm.mean(),
                suppress_ticks=True)

    gs.update(wspace=0.2,hspace=0.05)
    gs.update(left=0.07,right=0.88)
    gs.update(bottom=0.05,top=0.9)


    xsp,xdp = m(xsp,xdp)
    xs,ys=[],[]
    xd,yd=[],[]
    dlat=110.
    for k in range(0,len(dp)):
        xi=sk/dlat/rt*stkd[0]+dp[k]/dlat/rt*dipd[0]+xlr
        yi=sk/dlat*stkd[1]+dp[k]/dlat*dipd[1]+ylr
        xi,yi=m(xi,yi)
        xs.append(xi)
        ys.append(yi)

    for k in range(0,len(sk)):
        xi=sk[k]/dlat/rt*stkd[0]+dp/dlat/rt*dipd[0]+xlr
        yi=sk[k]/dlat*stkd[1]+dp/dlat*dipd[1]+ylr
        xi,yi=m(xi,yi)
        xd.append(xi)
        yd.append(yi)

        #import code
        #code.interact(local=locals())

    for k in range(0,N):
        xx = k % (N/2)
        yy = k / (N/2)

        f.sca(p[k])
        m.drawlsmask(land_color='whitesmoke',ocean_color='aliceblue',
                     lakes=True,
                     resolution='i',grid=1.25)
        m.drawcoastlines(color='dimgray')
        xspi = roundsigfigs(np.diff(xlm)/2,1)
        lontk = np.arange(round(xlm[0]+xspi/4,1),round(xlm[1],1),xspi)
        if yy==1:
            lbl = [1,0,0,1]
        else:
            lbl = [0,0,0,0]
        m.drawmeridians(lontk,labels=lbl,fontsize=11)

        plt.plot(xsp,xdp,color='m',linewidth=1)

        if xx==0:
            lbl = [1,0,0,1]
        else:
            lbl = [0,0,0,0]
        ysp = roundsigfigs(np.diff(ylm)/2,1)
        lattk = np.arange(round(ylm[0]+ysp/4,1),round(ylm[1],1),ysp)
        m.drawparallels(lattk,labels=lbl,fontsize=11)

        for n in range(0,len(xs)):
            plt.plot(xs[n],ys[n],color='k',linestyle=':',linewidth=0.25)
        for n in range(0,len(xd)):
            plt.plot(xd[n],yd[n],color='k',linestyle=':',linewidth=0.25)

        # contours
        x,y = m(vls[:,0],vls[:,1])
        plt.plot(x,y,color='k',linewidth=.5)

        # locations
        x,y = m(loc[0][ievi],loc[1][ievi])
        
        # to map rates
        vmin = 0
        vmax = np.max(rate)*0.75

        # all locations
        xi,yi = m(loc[0],loc[1])
        hall = m.scatter(xi,yi,color='none',s=40)
        hall.set_edgecolor('gray')

        # background beforehand
        ibf = nbf>=10
        hoth = m.scatter(xi[ibf],yi[ibf],color='none',s=40)
        hoth.set_edgecolor('mediumslateblue')

        # current
        ii = sum(rate)>0
        hbf = m.scatter(x[ii],y[ii],c=rate[k,ii],s=30,
                        vmin=vmin,vmax=vmax,cmap='Reds')
        hbf.set_edgecolor('none')

        # ii = np.sum(rate)==0
        # hnu = m.scatter(x[ii],y[ii],color='none',marker='o',s=40)
        # hnu.set_edgecolor('black')


        t1 = roundsigfigs(tim[k]-tsp/2,2)
        t2 = roundsigfigs(tim[k]+tsp/2,2)
        t1 = round((tim[k]-tsp/2)*10.)/10
        t2 = round((tim[k]+tsp/2)*10.)/10
        titl = str(t1) + ' to ' +str(t2) + ' hours';
        plt.title(titl,fontsize=12)

        
    tstr=tlm[0].strftime('%e %h %Y, %H:%M')
    f.suptitle('RTR starting '+tstr,fontsize=13)
    f.subplots_adjust(right=0.87,left=0.1)
    f.subplots_adjust(bottom=0.03,top=0.92)
    cbs = f.add_axes([.9,.3,.02,.4])
    cb = f.colorbar(hbf,cax=cbs)
    cbs.tick_params(axis='y',labelsize=11)
    cb.set_label('number of detections',fontsize=11)

    graphical.delticklabels(pm)
    
    global xl,yl
    xl=[]
    yl=[]
    cid=f.canvas.mpl_connect('button_press_event',onclick)

    # labels
    graphical.cornerlabels(p,loc='ll',fontsize=11)

    if prt:
        # for output
        fname=tlm[0].strftime('%Y-%b-%d-%H-%M')
        fname=os.path.join(os.environ['FIGURES'],'locsnap_'+fname+'.pdf')
        pp=PdfPages(fname)
        pp.savefig(f)
        pp.close()
    else:
        plt.show()

    if len(xl)>2:
        xl = np.array(xl)
        yl = np.array(yl)
        xl,yl=m(xl,yl,inverse=True)

        xs=(xl-xlr)*rt*stkd[0]+(yl-ylr)*stkd[1]
        xd=(xl-xlr)*rt*dipd[0]+(yl-ylr)*dipd[1]
        xs = np.array([min(xs),max(xs)])*dlat
        xd = np.array([min(xd),max(xd)])*dlat

        fl = open(fnamew,'w')
        for xi in xl:
            fl.write(str(xi)+'\t')
        fl.write('\n')
        for xi in yl:
            fl.write(str(xi)+'\t')
        fl.write('\n')
        for xi in xs:
            fl.write(str(xi)+'\t')
        fl.write('\n')
        for xi in xd:
            fl.write(str(xi)+'\t')
        fl.write('\n')
        fl.write(str(tlm[0]))
        fl.write('\n')
        print(stk)
        fl.write(str(stk))
        fl.write('\n')
        fl.close()

        # create a rectangle
        xsp = xs[np.array([0,0,1,1,0])]
        xdp = xd[np.array([0,1,1,0,0])]

        xsp,xdp=skdptolatlon(xs,xd,stk)
        xsp,xdp=m(xsp,xdp)

        #plt.plot(xsp,xdp,color='k')
        


def skdptolatlon(xs,xd,stk=160.):
    xlr,ylr=-124.25,49.
    rt=math.cos(ylr*math.pi/180)
    dlat = 110.
    
    # strike and dip directions
    stkd = np.array([math.sin(stk*math.pi/180),math.cos(stk*math.pi/180)])
    dipd = np.array([math.sin((stk-90)*math.pi/180),math.cos((stk-90)*math.pi/180)])

    lon = xlr+(xs*stkd[0]+xd*dipd[0])/rt/dlat
    lat = ylr+(xs*stkd[1]+xd*dipd[1])/dlat

    return lon,lat

def latlontoskdp(lon,lat,stk=160.):
    xlr,ylr=-124.25,49.
    rt=math.cos(ylr*math.pi/180)
    dlat = 110.

    # strike and dip directions
    stkd = np.array([math.sin(stk*math.pi/180),math.cos(stk*math.pi/180)])
    dipd = np.array([math.sin((stk-90)*math.pi/180),math.cos((stk-90)*math.pi/180)])

    # differences
    lon = (lon-xlr)*rt*dlat
    lat = (lat-ylr)*dlat

    sk = lon*stkd[0] + lat*stkd[1]
    dp = lon*dipd[0] + lat*dipd[1]

    return sk,dp

def onclick(event):
    global xl,yl
    global m
    global p,f
    global stk

    ix=event.xdata
    iy=event.ydata
    xl.append(ix)
    yl.append(iy)

    if len(xl)>2:
        xli,yli=m(xl,yl,inverse=True)
        xli=np.array(xli)
        yli=np.array(yli)

        xs,xd=latlontoskdp(xli,yli,stk)

        xs = np.array([min(xs),max(xs)])
        xd = np.array([min(xd),max(xd)])

        # create a rectangle
        xsp = xs[np.array([0,0,1,1,0])]
        xdp = xd[np.array([0,1,1,0,0])]

        xsp,xdp=skdptolatlon(xsp,xdp)
        xsp,xdp=m(xsp,xdp)


    return ix,iy



def plotTRloc(iev,tms,loc,tlm,tsp):

    # INPUT
    # 
    # iev         event indices
    # tms         times 
    # evn         numbers for locations
    # loc         locations
    # tlm         time limits to plot
    # tsp         time spacing for averages, in minutes

    import numpy as np
    import matplotlib.pyplot as plt
    from mpl_toolkits.basemap import Basemap
    import datetime

    # extract the times of interest
    tbf = datetime.timedelta(days=3)
    ix=[tms[k]>(tlm[0]-tbf) and tms[k]<tlm[1] for k in range(0,len(tms))]
    ix = np.array(ix)
    tmi = tms[ix]

    # want to plot locations
    xl = loc[0,iev[ix]]
    yl = loc[1,iev[ix]]

    dft = tmi-tlm[0]
    dfti = np.array([x.total_seconds()/3600 for x in dft])
    mxt = tlm[1]-tlm[0]
    mxt = mxt.total_seconds()/3600
    mnt = (min(tmi)-tlm[0]).total_seconds()/3600

    xlm = np.array([xl.min(), xl.max()])
    xlm = xlm + np.array([-1,1])*np.diff(xlm)*0.1
    ylm = np.array([yl.min(), yl.max()])
    ylm = ylm + np.array([-1,1])*np.diff(ylm)*0.1

    f = plt.figure(figsize=(10,10))
    m = Basemap(llcrnrlon=xlm[0],llcrnrlat=ylm[0],urcrnrlon=xlm[1],urcrnrlat=ylm[1],
                projection='lcc',resolution='i',lat_0=ylm.mean(),lon_0=xlm.mean(),
                suppress_ticks=True)
    m.drawlsmask(land_color='lightgray',ocean_color='skyblue',lakes=True,
                 resolution='i',grid=1.25)
    m.drawcoastlines(color='gray')
    xsp = roundsigfigs(np.diff(xlm)[0]/4,1)
    lontk = np.arange(round(xlm[0],1),round(xlm[1],1),xsp)
    m.drawmeridians(lontk,labels=[1,0,0,1])

    ysp = roundsigfigs(np.diff(ylm)[0]/4,1)
    lattk = np.arange(round(ylm[0],1),round(ylm[1],1),ysp)
    m.drawparallels(lattk,labels=[1,0,0,1])

    # divide times
    tdiv = np.arange(0,mxt,tsp/60.)
    tmn = (tdiv[0:-1] + tdiv[1:])/2.

    tdivb = np.arange(mnt,0,30/60.)
    tmnb = (tdivb[0:-1] + tdivb[1:])/2.

    # initialize
    mloc = np.zeros([2,len(tmn)])
    mlocb = np.zeros([2,len(tmnb)])

    # mean in specified times
    for k in range(0,len(tmn)):
        ix=np.logical_and(dfti>tdiv[k],dfti<tdiv[k+1])
        mloc[0,k] = xl[ix].mean(0)
        mloc[1,k] = yl[ix].mean(0)
    ixp = ~np.isnan(mloc[1,:])

    # mean in specified times
    for k in range(0,len(tmnb)):
        ix=np.logical_and(dfti>tdivb[k],dfti<tdivb[k+1])
        mlocb[0,k] = xl[ix].mean(0)
        mlocb[1,k] = yl[ix].mean(0)
    ixb = ~np.isnan(mlocb[1,:])

    x,y = m(loc[0,:],loc[1,:])
    hb = m.scatter(x,y,color='k',marker='^')

    x,y = m(mlocb[0,ixb],mlocb[1,ixb])
    h1 = m.scatter(x,y,color='gray',s=20)

    x,y = m(mloc[0,ixp],mloc[1,ixp])
    h = m.scatter(x,y,c=tmn[ixp],s=50,vmin=0,vmax=mxt)

    plt.jet()
    if len(x):
        cb = m.colorbar()
        cb.set_label('time since '+str(tlm[0])+' (hours)')

    # for output
    fname=tlm[0].strftime('%Y-%b-%d-%H-%M')
    fname=os.path.join(os.environ['FIGURES'],'locwtime_'+fname+'.pdf')
    pp=PdfPages(fname)
    pp.savefig(f)
    pp.close()

    #plt.show()
