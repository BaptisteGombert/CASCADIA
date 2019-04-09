#!/usr/bin/env python3

import numpy as np
import ezodf
import sys
import os
from os.path import join,expanduser

# Plot stuff
import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms
from mpl_toolkits.axes_grid1.parasite_axes import SubplotHost

class slowevents:
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
        fig,ax = plt.subplots(facecolor='w')
         
        # Make titles
        ax.set_xlabel('Seismic moment (N.m/s)')
        ax.set_ylabel('Duration (s)')

        # Save fig
        self.fig = fig
        self.ax = ax

        if show:
            plt.ion()
            plt.show()


        self.addGaodata()
        self.mwaxis()
        self.addisotimelines()
        self.addscalinglines()
        
        # All done
        return 

# ----------------------------------------------------------------------------------------------
    def addGaodata(self):
        '''
        Plot Gao et al. data (2012)
        '''

        # Get data points
        data = slowevents()

        # Plot them
        #self.ax.plot(np.log10(data.M0),np.log10(data.duration),'k.')
        self.ax.loglog(data.M0,data.duration,marker='^',linestyle='',c='navy',label='Gao et al. (2012)')

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
        ax2.set_xlabel('Moment magnitude')
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
            ax.axhline(li,linestyle='--',color='k')
            ax.text(xpos,li,la,va='center',ha='center',backgroundcolor='w')

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

        ax.loglog(mo,10**(np.log10(mo)-12),'k')
        ax.loglog(mo,10**(np.log10(mo)-13),'k')

        ax.set_xlim(xlims)

        return

# ----------------------------------------------------------------------------------------------
    def addpoints(self,m0,T,marker='+',color='orange'):
        '''
        Add points on the plot
        Args:
            * m0     : moment values
            * T      : Duration
            * marker : markertype (def='+')
            * color  : color of marker (def='orange')
        '''

        # Plot that shit
        self.ax.loglog(m0,T,marker=marker,linestyle='',c=color,label='This study') 

        # return
