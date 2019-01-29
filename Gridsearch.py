#!/usr/bin/env python3.6

import numpy as np
import scipy
import datetime
import obspy
import glob
import sys
import os
import time
import copy
import pyproj as pp
import matplotlib.pyplot as plt

from os.path import join

# Internal
from CascadiaUtils import *
import general
 
# -------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------
def computevelocity(lfe):
    '''
    For a given LFE, returns the S-wave velocity
    Args:
        * lfe: Template id (str or int)
    Returns:
        * slope : apparent veocity associated with nearby LFEs
    '''
    # Convert to string of necessary
    if type(lfe)==int:
        lfe = '{:03d}'.format(lfe)    
        T = readtemplate(int(lfe))
    else:
        assert(type(lfe)==obspy.core.stream.Stream),'1st arg must be int, str, or obspy.Stream'
        T = lfe 
          
    # Get lfe dist to sta and picks time
    ps = getpicks(T)
    ds = getdists(T)
    
    t0 = []
    d  = []
    for sta in ps.keys():
        t0.append(ps[sta])
        d.append(ds[sta])

    t0 = np.array(t0)
    d  = np.array(d)

    # Get speed
    slope, a, _,_,_ = scipy.stats.linregress(t0,d)
        
    return slope

# -------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------
def computevelocities(lfe,nt=5):
    '''
    For a given LFE, returns the S-wave velocities associated with nearby LFEs
    Args:
        * lfe: Template id (str or int)
    Returns:
        * slopes : apparent veocities associated with nearby LFEs
    '''


    # Convert to string of necessary
    if type(lfe)==int:
        lfe = '{:03d}'.format(lfe)    

    # Get lfe coordinates
    lfeloc = readlfeloc()
    xl,yl = lfeloc[lfe]['xy']
    zl    = lfeloc[lfe]['z']*1000.

    # Get closest LFEs
    dists, names = getclosestsLFE(lfe,nt=None)

    # set empty slopes vector 
    slopes = np.zeros((nt))

    # set empty counter
    count = 0
    k     = 0

    while count<nt:
        T = readtemplate(int(names[k]))
        if len(T) == 0:
            k += 1
            continue
        
        else:    
            # Get lfe dist to sta and picks time
            ps = getpicks(T)
            ds = getdists(T)
            
            t0 = []
            d  = []
            for sta in ps.keys():
                t0.append(ps[sta])
                d.append(ds[sta])

            t0 = np.array(t0)
            d  = np.array(d)

            # Get speed
            slope, a, _,_,_ = scipy.stats.linregress(t0,d)
            slopes[count] = slope

            count +=1
            k +=1 
        
    return slopes  


# -------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------
def computeshifts(lfe,x,y,z,speed):
    '''
    From a point position, returns expected shift compared to a given (nearby) LFE
    Args:
        * lfe  : Template id (str or int)
        * x,y,z: position of the point of interest, in m UTM
        * speed: velocity of S-wave, in m/s !!
    Returns:
        * shifts: dictionnary of time shifts
    '''

    # Read template is arg is not 
    if type(lfe) in [str,int]:        
        T = readtemplate(int(lfe))
    else:
        assert(type(lfe)==obspy.core.stream.Stream),'1st arg must be int, str, or obspy.Stream'
        T = lfe

    # Read station location
    staloc = readstaloc()
    dists = getdists(T)

    # empty dico
    shifts = {}

    for sta in dists.keys():
        # get lfe loc
        x2,y2 = staloc[sta]['xy']
        z2    = 0.
        d2 = np.sqrt((x2-x)**2+(y2-y)**2+(z2-z)**2) # Compoute point/ sta distance
        d1 = dists[sta]*1000

        DELTA = d2-d1 # in meters!!!
        shift = DELTA/speed
        shifts[sta] = shift

    # All done
    return shifts


# -------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------
def makevelocitymodel(Vs,depth,fname,base='iasp91'):
    '''
    Add a layer to a preexisting velocity model
    and create a .npz file to be read with obspy.taup
    Args:
            * Vs    : S-wave velocuty (in km/s)
            * depth : Depth of the layer (km)
            * fname : name of npz file to write
            * base  : preexisting model to use for larger depths
    '''

    # Read base velocity file
    if base == 'iasp91':
        fid = os.path.join(os.environ['DATA'],'VELMODEL/iasp91.tvel')
        Mod = np.loadtxt(fid)
    else:
        print('To implement')
        sys.exit(0)

    # Delete first layer
    D = Mod[:,0]
    if depth>35.:
        ix = np.where(D>=depth)[0]
    else:
        ix = np.where(D>35.)[0]

    Mod = Mod[ix,:]

    # Get VpVs ratio and Vp
    VpVs = Mod[0,1]/Mod[0,2]   
    Vp = VpVs*Vs    

    Density = 2.92

    # Make lines
    l0 = np.array((0.,Vp,Vs,Density))
    l1 = np.array((depth,Vp,Vs,Density))
    l2 = np.array((depth,Mod[0,1],Mod[0,2],Mod[0,3]))
    l = np.vstack((l0,l1,l2))
    lines = np.vstack((l,Mod))
    header = '#\n'

    np.savetxt('tmp.tvel',lines,header=header,fmt='%1.4f')
    
    # Build NPZ file 
    from obspy.taup.taup_create import build_taup_model
    build_taup_model('./tmp.tvel','./') 
    
    # Clean up
    os.remove('tmp.tvel')
    os.rename('tmp.npz',fname)

    # All done
    return


# -------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------
def computeTAUPshifts(lfe,x,y,z,vmod='iasp91'):
    '''
    From a point position, returns expected shift compared to a given (nearby) LFE
    Args:
        * lfe  : Template id (str or int)
        * x,y,z: position of the point of interest, in m UTM
        * speed: velocity of S-wave, in m/s !!
    Returns:
        * shifts: dictionnary of time shifts
    '''
    
    # Convert to km
    x=copy.deepcopy(x)/1000.
    y=copy.deepcopy(y)/1000.
    z=copy.deepcopy(z)/1000.

    # Read station location
    staloc = readstaloc()

    # Read lfe loc
    lfeloc = readlfeloc()
    key = '{:03d}'.format(int(lfe))
    xlfe,ylfe = lfeloc[key]['xy']/1000.
    zlfe = lfeloc[key]['z']
    
    
    # Get stations of interest
    T = readtemplate(int(lfe))
    dists = getdists(T)
    
    # empty dico
    shifts = {}

    for sta in dists.keys():
        # get lfe loc
        x2,y2 = staloc[sta]['xy']/1000.
        z2    = 0.
        t1 = calcspectrav(x-x2,y-y2,z-z2,vmod=vmod)
        t0 = calcspectrav(xlfe-x2,ylfe-y2,zlfe-z2,vmod=vmod)

        shifts[sta] = t1-t0

    # All done
    return shifts

# -------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------
def calcspectrav(xi,yi,zi,Ni=6,usevmod=True,vmod='iasp91',phsarv=['s','S']):
    """
    calculate travel times for a set of points,
    interpolating to reduce the number of calculations
    :param      xi:  x-indices in km
    :param      yi:  y-indices in km
    :param      zi:  depths in km
    :param      Ni:  number of interpolation points to use
    :param usevmod:  use a velocity model  (default: True)
    :param    vmod:  velocity model to use (default: 'iasp91')
    :param  phsarv:  which phases to use (default: ['S','s'])
    :return  ttrav:  travel times
    """

    # want them to be at least 2-d
    xi,yi,zi=np.atleast_1d(xi),np.atleast_1d(yi),np.atleast_1d(zi)
    if xi.ndim==1:
        xi=np.atleast_2d(xi).transpose()
    if yi.ndim==1:
        yi=np.atleast_2d(yi).transpose()
    if zi.ndim==1:
        zi=np.atleast_2d(zi).transpose()

    if usevmod:
        from obspy.taup import TauPyModel
        try:
            # check if it exists
            model = TauPyModel(model=vmod)
        except:
            # read from an npz file
            #fdir=os.path.join(os.environ['DATA'],'VELMODELS',vmod)
            #fname=os.path.join(fdir,vmod+'.npz')
            fname=vmod
            model = TauPyModel(model=fname)

        # for each station
        dhor = np.power(xi,2)+np.power(yi,2)
        dhor = np.power(dhor,0.5)

        # initialize
        ttrav = np.zeros(xi.shape)

        for m in range(0,xi.shape[1]):
            dhtry=np.linspace(np.min(dhor[:,m]),np.max(dhor[:,m]),Ni)
            ztry=np.linspace(np.min(zi),np.max(zi),Ni)
            dhtry,ztry=np.unique(dhtry),np.unique(ztry)
            ttravi=np.zeros([len(dhtry),len(ztry)],dtype=float)
            
            for kd in range(0,len(dhtry)):
                for kz in range(0,len(ztry)):
                    arrivals=model.get_travel_times(
                        distance_in_degree=dhtry[kd]/111.,
                        source_depth_in_km=ztry[kz],
                        phase_list=phsarv)
                    ttravi[kd,kz]=np.min([arr.time for arr in arrivals])

            # interpolate to these points
            dhoru,zu=np.unique(dhor[:,m]),np.unique(zi[:,m])
            if len(ztry)>1 and len(dhtry)>1:
                f=scipy.interpolate.RectBivariateSpline(dhtry,ztry,ttravi)
                ttravj=f(dhoru,zu)
                # f=scipy.interpolate.interp2d(ztry,dhtry,ttravi,kind='cubic')
                # ttravj=f(zu,dhoru)
            elif len(ztry)>1:
                f=scipy.interpolate.interp1d(ztry,ttravi.flatten(),kind='cubic')
                ttravj=f(zu).reshape([len(dhoru),len(zu)])
            elif len(dhtry)>1:
                f=scipy.interpolate.interp1d(dhtry,ttravi.flatten(),kind='cubic')
                ttravj=f(dhoru).reshape([len(dhoru),len(zu)])
            else:
                ttravj=ttravi.flatten().reshape([len(dhoru),len(zu)])

            # pick the relevant points
            ih = general.closest(dhoru,dhor[:,m])
            iz = general.closest(zu,zi[:,m])

            ttravs=[ttravj[ih[n],iz[n]] for n in range(0,len(ih))]
            ttrav[:,m]=ttravs

    else:

        # total distance and travel time
        spd = 3.
        ttrav = np.power(xi,2)+np.power(yi,2)+np.power(zi,2)
        ttrav = np.power(ttrav,0.5)
        ttrav = ttrav/spd

    return ttrav
    
# -------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------
def makegrid(gridtype,dim,npoints,removecenter=False,returnLL=False,center=(0.,0.,0.)):
    '''
    Make a grid given shape and center position
    Args:
        * gridtype : either 'circle' or 'square'
        * dim      : how much it extands. Either radius or half-side
        * npoints  : Number of points
        * removecenter [OPT] : Remove point in the center (def=False)
        * center [OPT]       : Center of the grid (x,y,z UTM), typically a LFE position
        * returmLL [OPT]     : If True, also return lon/lat location (def=False) 

    Return:
        * X,Y     : 2 arrays of X and Y horixontal positions
    '''
    
    # Check arguments
    assert(gridtype in ['circle','square']),'gridtype must be circle or square'


    X = []
    Y = []

    if gridtype is 'circle': # we look in a square and throw away if not in circle
        R = dim # radius
        # compute number of points in each direction
        nx = round(np.sqrt(npoints*(4./np.pi))/2.)
        dx = float(R)/nx

        nx = round(np.sqrt(npoints))
        dx = 2. * R / nx
        for x in np.arange(-nx,nx+1):
            for y in np.arange(-nx,nx+1):
                d = np.sqrt((x*dx)**2+(y*dx)**2) 
                if (d>R):
                    continue
                elif (d==0)&(removecenter):
                    continue
                else:
                    X.append(x)
                    Y.append(y)
        X = np.array(X)*dx+center[0]
        Y = np.array(Y)*dx+center[1]

    elif gridtype is 'square':
        nx = round(np.sqrt(npoints))
        dx = float(dim)/nx
        x = np.linspace(-nx*dx,(nx+0)*dx,nx)

        X,Y = np.meshgrid(x,x)
        X = X.flatten()
        Y = Y.flatten()
        if not removecenter and 0. not in x:
            X = np.append(X,0.)
            Y = np.append(Y,0.)
        X += center[0]
        Y += center[1]

    if returnLL:
        # Create converter from ll to utm
        string = '+proj=utm +lat_0={} +lon_0={} +ellps={}'.format(48.5, -123.6, 'WGS84')
        putm = pp.Proj(string)
        LO,LA = putm(X,Y,inverse=True)
        
        return X,Y,LO,LA
    
    else:
        return X,Y
    


# -------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------
def makeblanketgrid(dx,mdist,removecenter=False,returnLL=False):
    '''
    Blanket cascadia with points and remove those too far from template
    Args:
        * dx       : Spacing between points (in m) 
        * mdist    : Maximum distance to template
        * removecenter [OPT] : Remove point in the center (def=False)
        * center [OPT]       : Center of the grid (x,y,z UTM), typically a LFE position
        * returmLL [OPT]     : If True, also return lon/lat location (def=False) 

    Return:
        * X,Y     : 2 arrays of X and Y horixontal positions
    '''
    
    # Create converter from ll to utm
    string = '+proj=utm +lat_0={} +lon_0={} +ellps={}'.format(48.5, -123.6, 'WGS84')
    putm = pp.Proj(string)

    # Define bounds
    minlo = -124.5; minla = 48.0
    maxlo = -123.0; maxla = 49.0
    minx,miny = putm(minlo,minla)
    maxx,maxy = putm(maxlo,maxla)

    # Make Grid
    x = np.arange(minx,maxx,dx)
    y = np.arange(miny,maxy,dx)
    X,Y = np.meshgrid(x,y)
    X = X.flatten()
    Y = Y.flatten()

    # Initialize final grids
    Xs = []
    Ys = []
    Zs = []
    Ns = [] # Names of closest LFE
    # get LFE loc
    lfeloc = readlfeloc()

    for i in range(len(X)):
        z = InterpolateLFEDepth(X[i],Y[i],utm=True)*1000. # Get depth
        d,n = getclosestsLFE2point(X[i],Y[i],z)
        
        # If close enough
        if d<=mdist:
            Xs.append(X[i])
            Ys.append(Y[i])
            Zs.append(z)
            Ns.append(n)

    # Make arrays
    Xs = np.array(Xs)
    Ys = np.array(Ys)
    Zs = np.array(Zs)
    Ns = np.array(Ns)

    # All done
    return Xs,Ys,Zs,Ns
    
# -------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------

