#!/usr/bin/env python3.6

import numpy as np
import datetime
import obspy
import glob
import sys
import os
import time
import pyproj as pp
import matplotlib.pyplot as plt
from waveformdb import opendatabase
from waveformtable import Waveform

from os.path import join

def readdata(t1,t2):
    '''
    Get waveforms beteen two dates (obspy.UTCDateTime) t1 and t2.
    '''

    # Get correponding waveforms
    session = opendatabase('cascadia')
    q = session.query(Waveform)
    res = []
    t = 0; c = 0
    while len(res)==0:
        res = q.filter((Waveform.starttime>=t1.timestamp-t) & (Waveform.endtime<=t2.timestamp)).all()
        t += 1*3600.
        c += 1
    
    S = obspy.Stream()
    for r in res:
        S.append(r.waveform()[0]) # Put everything in a Stream
    
    # All done
    return S


# -------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------
def readtemplate(N):
    '''
    Get templates number N and return a Stream
    '''
    paf = os.path.join(os.environ['SDATA'],'TEMPLATES')
    files = glob.glob(join(paf,'*/Template.{:03d}.*.SAC'.format(N)))
    S = obspy.Stream()
    for f in files:
        S.append(obspy.read(f)[0])
    
    S.merge()

    # All done
    return S

# -------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------
def readlfeloc():
    '''
    read lfe loc
    Return:
            * ditionnary with lon/lat and UTM coordinates of LFE templates
    '''

    
    # Create converter from ll to utm
    string = '+proj=utm +lat_0={} +lon_0={} +ellps={}'.format(48.5, -123.6, 'WGS84')
    putm = pp.Proj(string)

    # Path to sta loc
    #paf = os.path.expanduser('~/Projects/LFEs/BostockTemplates/lfeloc_300.txt')
    paf = os.path.join(os.environ['DATA'],'TREMOR','Cascadia','lfeloc_300.txt')
    locs = {}

    # Read file
    with open(paf,'r') as f:
        M = f.readlines()
        for line in M:
            line = line.split()
            name = line[0]
            lon = float(line[2])
            lat = float(line[1])
            dep = float(line[3])

            locs[name]={}
            locs[name]['ll'] = np.array((lon,lat))
            locs[name]['xy'] = np.array(putm(lon,lat))
            locs[name]['z'] = dep

    # All done
    return locs


# -------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------
def readstaloc():
    '''
    read station loc
    Return:
        * ditionnary with lon/lat and UTM coordinates of stations
    '''
    
    # Create converter from ll to utm
    string = '+proj=utm +lat_0={} +lon_0={} +ellps={}'.format(48.5, -123.6, 'WGS84')
    putm = pp.Proj(string)

    # Path to sta loc
    paf = os.path.join(os.environ['DATA'],'TREMOR','Cascadia','statloc')
    locs = {}

    # Read file
    with open(paf,'r') as f:
        M = f.readlines()
        for line in M:
            line = line.split(',')
            name = line[0]
            lon = float(line[2])
            lat = float(line[1])
            
            locs[name]={}
            locs[name]['ll'] = np.array((lon,lat))
            locs[name]['xy'] = np.array(putm(lon,lat))

    # All done
    return locs

# -------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------
def getLFESTAdist(lfeloc,staloc,lfe,sta,vert=True):
    '''
    Get distance between a lfe template and a station
    Args:
        * lfeloc: Diction. conainting every LFE template location (created by readlfeloc())
        * staloc: Diction. conainting every station location (created by readstationloc())
        * lfe: Template id (str or int)
        * sta: Name of station (str)
        * vert: If True (default), compute 3d distance. Only horizontal if False
    Returns:
        * distance
    '''

    # Convert to string of necessary
    if type(lfe)==int:
        lfe = '{:03d}'.format(lfe)
    
    # Get lfe coordinates
    xl,yl = lfeloc[lfe]['xy']
    zl    = lfeloc[lfe]['z']*1000.

    # Get station coordinates
    xs,ys = staloc[sta]['xy']

    if vert:
        d = np.sqrt((xs-xl)**2+(ys-yl)**2+zl**2)
    else:
        d = np.sqrt((xs-xl)**2+(ys-yl)**2)

    return d

# -------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------
def getclosestsLFE(lfe,nt=None):
    '''
    Get closest LFEs to selected one
    Args:
        * lfe: Template id (str or int)
        * nt:  how many templates to return 
    Returns:
        * dist: array containing names of closest LFE
        * names: array containing distances (in m) 
    '''

    # Convert to string of necessary
    if type(lfe)==int:
        lfe = '{:03d}'.format(lfe)

    lfeloc = readlfeloc()

    # Get lfe coordinates
    xl,yl = lfeloc[lfe]['xy']
    zl    = lfeloc[lfe]['z']*1000.

    dists = []
    names = []

    for k in lfeloc.keys():
        # skip if target lfe
        if k == lfe:
            continue

        # get lfe loc
        x2,y2 = lfeloc[k]['xy']
        z2    = lfeloc[k]['z']*1000.
        
        # Compute distance
        d = np.sqrt((x2-xl)**2+(y2-yl)**2+(z2-zl)**2)
        
        dists.append(d)
        names.append(k)

    # Convert to array
    dists = np.array(dists)
    names = np.array(names)

    # Get nt closest
    ix = np.argsort(dists)#[:nt]

    if nt is not None:
        ix = ix[:nt]

    # All done
    return dists[ix], names[ix]     


# -------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------
def getclosestsLFE2point(xl,yl,zl):
    '''
    Get closest LFEs to a point
    Args:
        * x,y,z position of the point (in m)
    Returns:
        * d: istance to closest LFE template
        * name: Which one it is 
    '''

    dmin = 99999999.
    name = 'ERR'
    lfeloc = readlfeloc()

    for k in lfeloc.keys():
        # get lfe loc
        x2,y2 = lfeloc[k]['xy']
        z2    = lfeloc[k]['z']*1000.
        
        # Compute distance
        d = np.sqrt((x2-xl)**2+(y2-yl)**2+(z2-zl)**2)
        
        if d<dmin:
            dmin = d
            name = k        

    # All done
    return dmin, name

# -------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------
def getT0vsDist_LFE(T):
    
    
    # Get lfe coordinates
    ps = getpicks(T)
    ds = getdists(T)
    
    for k in ps.keys():
        plt.plot(ds[k],ps[k],'ko')

# -------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------
def getpicks(T):
    '''
    Get picks for a given LFE
    Args:
        * lfe: Template id (str or int)
    Returns:
        * dic of pic
    '''

    staloc = readstaloc()
    T = removeunpicked(T)

    pick = {} 
    for tr in T:
        pick[tr.stats.station] = tr.stats.t0

    # All done
    return pick

# -------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------
def getdists(T):
    '''
    Get dists for a given LFE
    Args:
        * lfe: Template id (str or int)
    Returns:
        * dic of pic
    '''

    staloc = readstaloc()
    T = removeunpicked(T)

    dist = {} 
    for tr in T:
        dist[tr.stats.station] = tr.stats.sac['dist']

    # All done
    return dist


# -------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------
def sortbydist(T,key='dist'):
    '''
    Take a obspy.Stream of a template 
    as an input and return it sorted 
    Args:
        * T         : Obspy Stream of a LFE template
        * [OPT] key : what to sort by (def=dist for station distance)
    '''
    
    return T.traces.sort(key=lambda x: x.stats.sac[key])

# -------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------
def removeunpicked(T,copy=False):
    '''
    Remove traces in Stream without picks
    Args:
            * T          : Obspy Stream of a LFE template
            * copy [OPT] : If True, leave input untouched. Def=False 
    Return:
            * T : Stream without unpicked traces       
    '''

    # Copy or not the old stream
    if copy:
        T2 = copy.deepcopy(T)
    else:
        T2 = T

    # Loop on traces
    for tr in T2:
        if not hasattr(tr.stats.sac,'t0'): # If no 't0'
            if not hasattr(tr.stats.sac,'a'): # If no 'a' either 
                T2.remove(tr) 
            else:
                tr.stats.t0 = tr.stats.sac['a']
        else:
            tr.stats.t0 = tr.stats.sac['t0']

    return T2


# -------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------
def getCommonTraces(D,T):

    '''
    Get common traces (i.e. same station ans channel) between 
    the template and station

    Args:
            * D: Data stream
            * T: Template stream
    Return:
            * data
            * template
            * stations : dictionnary containing some variables used after
                -> Nc       : number of distinct components (1, 2, ou 3)
                -> icmp     : For each trace, indice if which compo it is
                -> ins      : For each trace, get station number
                -> nper     : For each trace, get number of traces
                -> nscT     : Traces in template
                -> nscD     : Traces in data
                -> CommonTr : List of traces in both template and data
    '''

    # Networks, stations, components to compare
    nsc1=np.array([tr.stats.network+'.'+tr.stats.station+'.'+
                   tr.stats.channel[-1] for tr in T])
    nsc2=np.array([tr.stats.network+'.'+tr.stats.station+'.'+
                   tr.stats.channel[-1] for tr in D])
    nsc=np.intersect1d(nsc1,nsc2)
    nsc=np.sort(nsc)

    # split by station
    ns=np.array([vl.split('.')[0]+'.'+vl.split('.')[1] for vl in nsc])
    nsa,ins,nper=np.unique(ns,return_inverse=True,return_counts=True)

    # split by component
    # define groups
    cmps = np.array(['E1','N2','Z3'])

    # initialize
    icmp = np.ndarray(len(nsc),dtype=int)

    for k in range(0,len(nsc)):
        # split component
        vl = nsc[k].split('.')
        vl = vl[-1][-1]
        
        for m in range(0,len(cmps)):
            if vl in cmps[m]:
                icmp[k]=m

    # only components that were used
    ii,icmp=np.unique(icmp,return_inverse=True)
    cmps=cmps[ii]

    Nc=len(cmps)

    stations = {'Nc':Nc, 'icmp':icmp, 'ins':ins, 'nper':nper, \
                'nscT':nsc1, 'nscD':nsc2, 'CommonTr':nsc} 
   

    # Create empty obspy stream to fill with template
    st1i = obspy.Stream()

    # Select only used traces
    bo1 = np.isin(stations['nscT'],stations['CommonTr'])
    ix1 = np.where(bo1)[0]
    [st1i.append(T[i]) for i in ix1]

    # Create empty obspy stream to fill with data
    st2i = obspy.Stream()

    # Select only used traces
    bo2 = np.isin(stations['nscD'],stations['CommonTr'])
    ix2 = np.where(bo2)[0]
    [st2i.append(D[i]) for i in ix2]

    # Put them in current streams
    template = st1i
    data     = st2i
    
    # all done
    return data,template,stations

# -------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------
def plottemplate(T,offset=0.04,color=None,title=None,text=True):
    '''
    Make a quick and dirty plot of many waveforms
    Args:
            * T            : obspy stream of template
            * [OPT] offset : vertical offset betzeen traces
            * [OPT] color  : None (default), str, or list
            * [OPT] title  : None (default), str, or list
            * [OPT] text   : True (default) print station name
    '''

    # Get time vector
    time = T[0].times() 

    k=0
    plt.figure(facecolor='w')
    
    # loop on traces       
    for tr,c in zip(T,range(len(T))):
        if color is None:
            plt.plot(time,tr.data+k)
        elif type(color) is str:
            plt.plot(time,tr.data+k,color=color)
        else:
            plt.plot(time,tr.data+k,color=color[c])
        
        if text:
            plt.text(-2,k,tr.stats.station)
        k -= offset

    # Cosmetics
    plt.xlabel('Time (sec)')

    if title is not None:
        plt.title(title)



    # All done
    return


# -------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------
def getAlongStrikeDist(x,y,utm=True,contour=40):
    '''
    Return distance along-strike for set of X,Y values
    Return:
            * x,y  : vectors of coordinates you want the slab
            * utm [opt] : if False, coords are in lon/lat
            * contour : Which contour to use (default 40km) 
    '''
    

    # Read slab 40km depth contour
    fid = os.path.join(os.environ['DATA'],'SLAB','contour{}km.txt'.format(contour))
    S = np.loadtxt(fid,comments='>')    

    # Create converter from ll to utm
    string = '+proj=utm +lat_0={} +lon_0={} +ellps={}'.format(48.5, -123.6, 'WGS84')
    putm = pp.Proj(string)

    if utm==False:
        x,y = putm(x,y)

    # Limit to northern part
    ix = np.where((S[:,1]>=48.)&(S[:,1]<=50.))[0]
    Sx,Sy = putm(S[ix,0],S[ix,1])

    # Interpolate
    Syn = np.linspace(Sy.min(),Sy.max(),20000)
    Sxn = np.interp(Syn,Sy[::-1],Sx[::-1]) # from north to south to have increasing xp

    # Sum distance over path
    dx = Sxn[1:]-Sxn[:-1]
    dx = np.cumsum(np.append(0.,dx))
    dy = Syn[1:]-Syn[:-1]
    dy = np.cumsum(np.append(0.,dy))
    d_as = np.hypot(dx,dy) # along strike distance

    ASD = np.zeros(len(x))

    for ix,iy,k in zip(x,y,range(len(x))): 
        
        # compute distance to closest point in 40km iso-depth contour
        dists = np.hypot(Sxn-ix,Syn-iy)
        # Get closest point
        ind = dists.argmin()

        # Get and store closest along-strike distance
        ASD[k] = d_as[ind]

    # All done
    return ASD

# -------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------
def getLFEAlongStrikeDist():
    '''
    For each tamplate, get the along-strike distance
    Return:
            * dist  : dictionnary of distamce along-strike 
    '''
    
    # Create empty dictionnaty of along-strike distance
    ASD = {}

    # Read slab 40km depth contour
    fid = os.path.join(os.environ['DATA'],'SLAB','contour40km.txt')
    S = np.loadtxt(fid,skiprows=1)    

    # Create converter from ll to utm
    string = '+proj=utm +lat_0={} +lon_0={} +ellps={}'.format(48.5, -123.6, 'WGS84')
    putm = pp.Proj(string)

    # Limit to northern part
    ix = np.where((S[:,1]>=48.)&(S[:,1]<=50.))[0]
    Sx,Sy = putm(S[ix,0],S[ix,1])

    # Interpolate
    Syn = np.linspace(Sy.min(),Sy.max(),1000)
    Sxn = np.interp(Syn,Sy[::-1],Sx[::-1]) # from north to south to have increasing xp

    # Sum distance over path
    dx = Sxn[1:]-Sxn[:-1]
    dx = np.cumsum(np.append(0.,dx))
    dy = Syn[1:]-Syn[:-1]
    dy = np.cumsum(np.append(0.,dy))
    d_as = np.hypot(dx,dy) # along strike distance

    lfeloc = readlfeloc() # Get lfe locations

    for k in lfeloc.keys(): 
        x,y = lfeloc[k]['xy']
        
        # compute distance to closest point in 40km iso-depth contour
        dists = np.hypot(Sxn-x,Syn-y)
        # Get closest point
        ind = dists.argmin()

        # Get and store closest along-strike distance
        ASD[k] = d_as[ind]

    # All done
    return ASD

# -------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------
def getSlabDepth(x,y,utm=True):
    '''
    Get the slab depth for any point
    Args:   
        * x, y : coordinates of the point(s) you want to have the depth
        * [OPT] utm: if True (default), give UTM coordinates. Else lon/lat
    return:
        * z : depth of the slab    
    '''

    # import some stuffs
    from scipy.interpolate import interp2d
    from netCDF4 import Dataset as netcdf

    # Read slab
    fid = os.path.join(os.environ['DATA'],'SLAB','SlabMrCrory.grd')
    fin = netcdf(fid, 'r', format='NETCDF4')
    los = fin.variables['x'][:]
    las = fin.variables['y'][:]
    zs = fin.variables['z'][:]
    zs = np.ma.filled(zs,fill_value=0)

    #mask = ~zs.mask.flatten()

    # Get dimensions
    nx = len(los)
    ny = len(las)

    LO,LA = np.meshgrid(los,las)
    LO = LO.flatten()
    LA = LA.flatten()
    
    # Create converter from ll to utm
    string = '+proj=utm +lat_0={} +lon_0={} +ellps={}'.format(48.5, -123.6, 'WGS84')
    putm = pp.Proj(string)
    
    # Create interpolator
    #f = interp2d(LO[mask],LA[mask],zs[~zs.mask])
    f = interp2d(los,las,zs)

    # convert to lon lat if necessary
    if utm:
        lo,la = putm(x,y,inverse=True)
        lo,la = np.array((lo)),np.array((la))
    else:
        lo,la = np.array((x)),np.array((y))
    
    # get depth
    if lo.size==1:
        z = -1*f(lo,la)
    else:
        z = np.zeros((len(lo)))
        for i in range(len(lo)):
            z[i] = -1*f(lo[i],la[i])[0] 

    # all done
    return z

# -------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------
def InterpolateLFEDepth(x,y,utm=True):
    '''
    Interpolate depth between LFEs
    Args:   
        * x, y : coordinates of the point(s) you want to have the depth
        * [OPT] utm: if True (default), give UTM coordinates. Else lon/lat
    return:
        * z : depth     
    '''

    # import some stuffs
    from scipy.interpolate import griddata

    # Get LFE locs
    lfeloc = readlfeloc()
    lfex = []; lfey = []; lfez = [] 
    for k in lfeloc.keys():  
        lfex.append(lfeloc[k]['xy'][0]) 
        lfey.append(lfeloc[k]['xy'][1]) 
        lfez.append(lfeloc[k]['z']) 
    lfex = np.array(lfex)
    lfey = np.array(lfey) 
    lfez = np.array(lfez) 

    # Convert to UTM if necessary
    if utm:
        x,y = np.array((x)),np.array((y)) # Comvert to UTM because why not
    else:
        # Create converter from ll to utm
        string = '+proj=utm +lat_0={} +lon_0={} +ellps={}'.format(48.5, -123.6, 'WGS84')
        putm = pp.Proj(string)
        # Convert
        x,y = putm(x,y,inverse=True)
        x,y = np.array((x)),np.array((y))

    # Interpolate that shit
    zi = griddata((lfex,lfey),lfez,(x,y),method='linear')

    # All done
    return zi
