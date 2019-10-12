#!/usr/bin/env python3

'''
Small module made by B. Gombert to make
the moment-duration plot of slow slip events
B. Gombert - 2019
'''

import numpy as np
import ezodf
import sys
import os
from os.path import join,expanduser

# Plot stuff
import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms
from mpl_toolkits.axes_grid1.parasite_axes import SubplotHost
import matplotlib

fontname='times new roman'
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
matplotlib.rcParams['xtick.labelsize'] = 11
matplotlib.rcParams['ytick.labelsize'] = 11
matplotlib.rc('font',family=fontname)

fontname='times new roman'
hfont_mainticks = {'fontname':fontname,'fontsize':12}
hfont_cbarticks = {'fontname':fontname,'fontsize':15}
hfont_cbartitle = {'fontname':fontname,'fontsize':19}
hfont_stanames  = {'fontname':fontname,'fontsize':12}


class Bleteryevents:
    '''
    Class to store results from Bletery et al. 2016
    '''
    def __init__(self):
        # Supp from Bletery
        self.fid = join(os.environ['DATA'],'ssf-lfe_V2.txt')
        M = np.loadtxt(self.fid)
        self.duration   = M[:,10] *60.*60.
        self.M0         = M[:,16]


class Gaoevents:
    '''
    Class to store results of Gao et al. 2012
    '''
    def __init__(self):
        # S1 file from Gao
        self.fid = join(os.environ['DATA'],'GaoSSE.ods')
        # Open it
        sh = ezodf.opendoc(self.fid).sheets[0]
        A = np.array(list(sh.rows()))
        # Get rid of header
        A = A[2:,]
     
        # Initialise empty lists
        region     = []
        date       = []
        length     = []
        width      = []
        Mw         = []
        duration   = []
        risetime   = []
        Vr         = []
        slip       = []
        stressdrop = []
        reference  = []
        tokeep     = []

        # Store results
        for i in range(len(A)):
            # REGION
            if A[i,0].value is None:
                region.append(region[-1])
            else:
                region.append(A[i,0].value)
            # DATE
            if A[i,1].value is None:    
                date.append(date[-1])
            else:
                date.append(A[i,1].value)
            # LENGTH
            try:
                length.append(float(A[i,2].value))
            except:
                length.append(np.nan)
            # WIDTH
            try:
                width.append(float(A[i,3].value))
            except:
                width.append(np.nan)       
            # Mw
            try:
                Mw.append(float(A[i,4].value))
            except:
                Mw.append(np.nan)
            # DURATION
            try:
                duration.append(float(A[i,5].value)*86400.0)
            except:
                duration.append(np.nan)
            # RISE TIME 
            try:
                risetime.append(float(A[i,6].value)*86400.0)
            except:
                risetime.append(np.nan)        
            #  Vr 
            try:
                Vr.append(float(A[i,7].value)*1000.0)
            except:
                Vr.append(np.nan)                
            #  SLIP
            try:
                slip.append(float(A[i,8].value)*100.0)
            except:
                slip.append(np.nan)               
            #  STRESS DROP
            try:
                stressdrop.append(float(A[i,9].value))
            except:
                stressdrop.append(np.nan)            
            # REFERENCE
            if A[i,10].value is None:    
                reference.append(reference[-1])
            else:
                reference.append(A[i,10].value)
    
        # Convert to arrays
        self.Np = len(region)
        self.region     = np.array(region)
        self.date       = np.array(date)
        self.length     = np.array(length)
        self.width      = np.array(width) 
        self.Mw         = np.array(Mw)
        self.M0         = 10.**(1.5*self.Mw+9.1)
        self.duration   = np.array(duration)
        self.risetime   = np.array(risetime)
        self.Vr         = np.array(Vr)
        self.slip       = np.array(slip)
        self.stressdrop = np.array(stressdrop)
        self.reference  = np.array(reference)
 
        # All done
        return

    def getPoints(self):
        '''
        Return dictionnary of results
        '''
        
        points = {}
        for i in range(Np):
            key = '#{:03d}'.format(i)
            
            points[key]['region'] = self.region[i]
            points[key]['date'] = self.date[i]       
            points[key]['length'] = self.length[i]     
            points[key]['width'] = self.width[i]     
            points[key]['Mw'] = self.Mw[i]         
            points[key]['M0'] = self.M0[i]         
            points[key]['duration'] = self.duration[i]   
            points[key]['risetime'] = self.risetime[i]   
            points[key]['Vr'] = self.Vr[i]         
            points[key]['slip'] = self.slip[i]       
            points[key]['stressdrop'] = self.stressdrop[i] 
            points[key]['reference'] = self.reference[i]  

        # All done
        return points

# ----------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------
class MDplot():
    '''
    Class to build the moment-duration scale, as 
    shown in Ide et al. 2007, Peng and Gomberg 2010,
    or Gao et al. 2012
    '''
    
    def __init__(self,show=False):
        # Create figure
        #fig,ax = plt.subplots(facecolor='w',figsize=(10.5,9))
        fig,ax = plt.subplots(facecolor='w',figsize=(5.25,4.5))
         
        # Make titles
        ax.set_xlabel('Seismic moment (N.m/s)',fontsize=11)
        ax.set_ylabel('Duration (s)',fontsize=11)

        # Save fig
        self.fig = fig
        self.ax = ax

        if show:
            plt.ion()
            plt.show()


        self.addGaodata()
        self.addBleterydata()
        self.addLFEs()
        self.mwaxis()
        self.addisotimelines()
        self.addscalinglines()
        self.addeqscalinglines()
        
        # All done
        return 

# ----------------------------------------------------------------------------------------------
    def addGaodata(self,onlyETS=True):
        '''
        Plot Gao et al. data (2012)
        '''

        # Get data points
        data = Gaoevents()
        M0 = data.M0
        duration = data.duration
        if onlyETS:
            ix = np.where(duration>1000)[0]
            duration = duration[ix]
            M0 = M0[ix]

        # Plot them
        self.ax.loglog(M0,duration,marker='^',linestyle='',c='navy',label='Gao et al. (2012)')

        return

# ----------------------------------------------------------------------------------------------
    def addBleterydata(self):
        '''
        Plot Bletery et al. data (2016)
        '''

        # Get data points
        data = Bleteryevents()

        # Plot them
        self.ax.loglog(data.M0,data.duration,marker='.',linestyle='',c='k',label='Bletery et al. (2016)')

        return

# ----------------------------------------------------------------------------------------------
    def addLFEs(self):
        '''
        Plot Ide et a. 2007 LFE MD
        '''
        self.ax.scatter(10**(11.4),0.35,color='teal',s=1000,marker='s')
        #self.ax.scatter(10**(11.4),0.35,color='teal',s=2000,marker='s')
        self.ax.scatter(10**(11.4),0.35,color='teal',s=50,marker='s',label='Ide et al. (2007)')
        self.ax.scatter(10**(13.9),20,color='dodgerblue',s=150,marker='s')
        #self.ax.scatter(10**(13.9),20,color='dodgerblue',s=300,marker='s')
        self.ax.scatter(10**(13.9),20,color='dodgerblue',s=50,marker='s',label='Ito et al. (2007)')       
        return

# ----------------------------------------------------------------------------------------------
    def mwaxis(self):
        '''
        Put axis to the top with Magnitude
        '''
        # Moment to magnitude
        m0_to_mw = lambda M0: 2./3 * (np.log10(M0)-9.1)
        # Get limits
        xmin, xmax = self.ax.get_xlim()
        # Make twin axis
        ax2 = self.ax.twiny()
        # set lims
        ax2.set_xlim((m0_to_mw(xmin),m0_to_mw(xmax)))
        ax2.set_xlabel('Moment magnitude',fontsize=11)
        self.ax2 = ax2

        return

# ----------------------------------------------------------------------------------------------
    def addisotimelines(self):
        ''' 
        Add the 1s, 1min, 1hour, 1day, 1month and 1year isolines
        '''
        # Get axes
        ax = self.ax
        
        # Get lims for annotation postion
        xlims = np.log10(self.ax.get_xlim())
        xpos =  10**(xlims[0] + 0.07*np.diff(xlims))

        lines = [1.,60.,3600.,3600*24.,3600.*24*30.5,3600.*24*365]
        labels = ['1 second','1 minute','1 hour','1 day','1 month','1 year']
        for li,la in zip(lines,labels):
            ax.axhline(li,linestyle='--',color='k',lw=0.5)
            ax.text(xpos,li,la,va='center',ha='center',backgroundcolor='w',fontsize=11)

        ax.set_xlim(10**xlims)

        return


# ----------------------------------------------------------------------------------------------
    def addscalinglines(self):
        ''' 
        Add the logMo = log T +12 
        '''
        
        # Get axes
        ax = self.ax
        
        # Get lims for annotation postion
        xlims = np.array(self.ax.get_xlim())
        mo = np.logspace(np.log10(xlims[0]),np.log10(xlims[1]),100)
        l1 = 10**(np.log10(mo)-12)
        l2 = 10**(np.log10(mo)-13)
        ax.fill_between(mo,l1,l2,color='gray',alpha=0.5)
        #ax.loglog(mo,10**(np.log10(mo)-12),'k')
        #ax.loglog(mo,10**(np.log10(mo)-13),'k')

        ax.set_xlim(xlims)

        return

# ----------------------------------------------------------------------------------------------
    def addeqscalinglines(self):
        ''' 
        Add the scaling law for earthquakes 
        logMo = log T +12 
        '''

        # Get axes
        ax = self.ax
        
        # Get lims for annotation postion
        xlims = np.array(self.ax.get_xlim())
        mo = np.logspace(np.log10(xlims[0]),np.log10(xlims[1]),100)

        l1 = 10**(1./3.*np.log10(mo)-(16/3.))
        l2 = 10**(1./3.*np.log10(mo)+(np.log10(0.02)-10/3.))
        ax.fill_between(mo,l1,l2,color='gray',alpha=0.5)
        
        #ax.loglog(mo,10**(1./3.*np.log10(mo)-(16/3.)),'k')
        #ax.loglog(mo,10**(1./3.*np.log10(mo)+(np.log10(0.02)-10/3.)),'k')

        ax.set_xlim(xlims)

        return


# ----------------------------------------------------------------------------------------------
    def addpoints(self,m0,T,m0err=None,Terr=None,marker='+',markersize=10,color='C3',label='This study'):
        '''
        Add points on the plot
        Args:
            * m0     : moment values
            * T      : Duration
            * m0err  : std of m0 values (def='None')
            * Terr   : std of T values (def='None')
            * marker : markertype (def='+')
            * color  : color of marker (def='orange')
            * label  : Label for the legend (def='This study')
        '''

        # Plot that shit
        if (m0err is None)&(Terr is None):
            self.ax.loglog(m0,T,marker=marker,linestyle='',c=color,markersize=markersize,label=label) 
        else:
            self.ax.errorbar(m0,T,xerr=m0err,yerr=Terr,marker=marker,linestyle='',markersize=markersize,c=color,label='This study') 
        
        # All done   
        return
