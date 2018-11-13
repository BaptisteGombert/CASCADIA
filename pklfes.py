import obspy,string
from eventtable import Event
import phasecoh
from sqlalchemy import or_,and_
import corrhrsn
import matplotlib
import scipy
import seisproc
import pickle
import general
import responsecorr
import shutil
import syntheq
import numpy as np
from scipy.special import erf
import pksdrops
import math
from matplotlib.patches import Polygon
from matplotlib.backends.backend_pdf import PdfPages
import graphical
import sqlalchemy as sa
import os,datetime,glob
import waveformdb
from waveformtable import Waveform
import phscoh
import spectrum
import matplotlib.pyplot as plt
from matplotlib import gridspec
from pairtable import Xcpair
from empca import empca


class procsearch:
    def __init__(self,fnum=37102,sta=None,xc=None,xcu=None,xci=None,ps=None):
        # initialize with family number
        self.fnum = fnum
        self.flmget = np.array([2.,30.])
        self.samp20=False
        self.flmamp = np.array([2.,8.])
        self.flmtshf = np.array([2.,5.])
        self.eqloc = lfeloc(self.fnum)
        self.minsnr=3.
        self.blim=np.array([2.,10.])
        self.mxmad = 10.
        self.pk='t3'
        self.ppk='t2'
        self.checkp=False
        self.wgttype='bymax'
        self.rnoise=False
        self.txc = np.array([-.1,2.9])
        self.amplim = [1.,float('inf')]
        self.maxstd=10.
        self.realign=False
        self.tshfumax=float('inf')
        self.ampstack=None
        self.ampmean=None
        self.usemshf=True
        self.rscl=True
        self.tlmxc=[-.2,2.8]
        self.flmxc=[2.,6]

        self.twin=np.array([-1.,1])*3
        self.tms,self.xcvl = lfetimes(self.fnum)
        #self.splitevents()
        self.allevents()
        self.trange = np.array([-.1,2.9])
        #self.trange = np.array([-.4,4.6])
        self.tget = np.array([-10.,10.])
        self.wlen=np.diff(self.trange)[0]
        self.cmpuse = 'both_horizontals'
        self.cmpuse = 'all'
        self.csyn = None
        self.blimloc = np.array([0.,15])
        self.vmod = 'iasp91'
        self.medscale=None
        self.randxshf=0.
        self.randzshf=0.

        
        self.xcsyn=False
        self.useloc='saved'
        
        # parameters for collecting LFEs to a single waveform
        self.tshf=4.
        self.tadd=np.array([-1.,7.])
        self.rnshf=np.array([3.,0.5,1.])
        self.tpr=1.

        # save extra information if given
        if sta is not None:
            self.sta=sta
        else:
            self.sta = initstacks()
            readpicks(fnum=self.fnum,sta=self.sta)
            readppicks(fnum=self.fnum,sta=self.sta)
            for tr in self.sta:
                tr.stats.t9=self.fnum

        self.iterstack=-1
        self.iterloc=0
        self.iteramp=0
        self.stam=self.sta.copy()
        self.staw=None

        # initalize time shifts
        ids = [tr.get_id() for tr in self.sta]
        tref=obspy.UTCDateTime(2000,1,1)
        # shfs=dict((tr.get_id(),tr.stats.starttime-tref+tr.stats.t3)
        #           for tr in self.sta)
        # self.tshfs=dict((idi,np.ones(len(self.tms))*shfs[idi]) for
        #                 idi in ids)
        self.tshfu=dict((idi,np.zeros(len(self.tms))) for
                        idi in ids)
        self.tshfs=dict((idi,np.zeros(len(self.tms))) for
                        idi in ids)


        if xc is not None:
            self.xc,self.xcu,self.xci=xc,xcu,xci

        if ps is not None:
            for ky in ps.__dict__.keys():
                self.__setattr__(ky,ps.__getattribute__(ky))

#--------COMPONENT OF INTEREST-----------------------------

    def pickcomp(self):
        """
        set stam to just the specified components
        """

        if self.cmpuse=='one_horizontal':
            cmpos=['DP2','EH1']

        elif self.cmpuse=='first_horizontal':
            cmpos=['DP2','EH1']

        elif self.cmpuse=='second_horizontal':
            cmpos=['DP3','EH2']

        elif self.cmpuse=='both_horizontals':
            cmpos=['DP[23]','?H[12]']

        elif self.cmpuse=='all':
            cmpos = np.unique([tr.stats.channel for tr in self.sta])


        if self.cmpuse=='maxsnr_horizontal':
            # max signal to noise horizontals
            self.findmaxcomp()
        
            # project to these
            self.projmaxcomp()

        else:
            self.stam = obspy.Stream()
            for cmpi in cmpos:
                self.stam=self.stam+self.sta.select(channel=cmpi).copy()
            
            # set the weighting to 1 for each of these
            self.cwgts = {}
            for tr in self.stam:
                nwst=tr.stats.network+'.'+tr.stats.station+'.'+\
                    tr.stats.channel
                self.cwgts[nwst]={tr.stats.channel:1.}


            
#----------GETTING THE LFE TIMES------------------------------



    def readdetections(self,xvli=3.,zvli=5,flm=[2,8],odur=0.2,ndur=0.5):
        """
        write the identified detections
        """

        if xvli is not None:
            self.xvli=float(xvli)
        if zvli is not None:
            self.zvli=float(zvli)
        if flm is not None:
            self.flmdet=np.atleast_1d(flm).astype(float)
        if odur is not None:
            self.odurdet=odur
        if ndur is not None:
            self.ndurdet=ndur

        fname=detectionsfilename(self.fnum,xvli,zvli,flm,odur,ndur)
        fdir=os.path.join(os.environ['DATA'],'KLFESEARCH','DETECTIONS')
        fname=os.path.join(fdir,fname)
        vls=np.loadtxt(fname,dtype=bytes,delimiter=',').astype(str)
        # times
        tms=np.array([datetime.datetime.strptime(tm,'%Y-%b-%d-%H-%M-%S-%f')
                      for tm in vls[:,0]])
        self.tms=np.array([obspy.UTCDateTime(tm) for tm in tms])

        self.xc=vls[:,1].astype(float)


    def allevents(self):
        # define all events in each stack and comparison

        # times for stack
        self.tstack = self.tms.copy()

        # times for comparison
        self.tcomp = self.tms.copy()
        
        # save indices
        self.istack = np.arange(0,self.tms.size)
        self.icomp = np.arange(0,self.tms.size)

        self.tsv = self.tms.copy()
        self.isv = np.arange(0,self.tms.size)

    def splitevents(self):
        # split into two groups---one for stack, one for comparison
        Ntot = self.tms.size
        ix = np.random.choice(Ntot,Ntot/2,replace=False)
        ix.sort()
        ix2 = np.ones(Ntot,dtype=bool)
        ix2[ix] = False

        # times for stack
        self.tstack = self.tms[ix]

        # times for comparison
        self.tcomp = self.tms[ix2]
        
        # save indices
        self.istack = ix
        self.icomp = ix2

        self.tsv = self.tcomp.copy()
        self.isv = self.icomp.copy()

    def readeqloc(self):
        # note the LFE location
        self.eqloc = lfeloc(self.fnum)

    def exctimes(self,ids=None):
        """
        to identify the times with poor data
        :param    ids: the intervals noted as having poor amplitudes
        """

        # default ids
        if ids is None:
            ids=np.array([tr.get_id() for tr in self.sta])

        # find the relevant times
        self.exct = exctimes(self.tms,self.fnum,ids)
        

#----------GETTING ALL THE DATA------------------------------


    def readmanylfes(self,lbl='',detname=False):
        # write extracted LFEs
        lbl=lbl+'_'+'{:g}'.format(general.roundsigfigs(self.flmget[0],3))+'_'+\
             '{:g}'.format(general.roundsigfigs(self.flmget[1],3))+'Hz'
        lbl='Family'+str(self.fnum)+lbl

        # write extracted LFEs
        if detname:
            fname=self.detfilename()
            lbl=lbl+'_'+fname

        print('Reading data '+lbl)

        self.stsv,self.tsv,self.stsvn,self.ilmsv,self.tget=\
            readmanylfes(lbl)
        self.isv = eventindex(self.tms,self.tsv)

        # correct problematic intervals
        corrsaved([self.stsv]+self.stsvn,self.ilmsv,self.tget,self.tsv)
        
    # def writemanylfes(self,lbl=''):
    #     # write extracted LFEs
    #     lbl='Family'+str(self.fnum)+lbl        
    #     lbl=lbl+'_'+str(int(self.flmget[0]))+'_'+\
    #         str(int(self.flmget[1]))+'Hz'

    #     writemanylfes(lbl,self.stsv,self.tsv,self.stsvn,
    #                   self.ilmsv,self.tget)

    def copymanylfes(self):

        # copy stsv to a saved set
        self.stsvsv=self.stsv.copy()

    def noisemanylfes(self,ins=0):

        # replace stsv with noise
        self.stsv=self.stsvn[ins].copy()

    def addranddurtemp(self):
        """
        add  one of the duration templates
        """

        # to keep the original
        stam=self.stam.copy()

        # which template is assigned
        Ndur = len(self.durtemp)
        self.iduras=np.random.choice(Ndur,self.tms.size,replace=True)

        # add for each set
        for k in range(0,Ndur):
            ii,=np.where(self.iduras==k)
            self.stam=self.durtemp[k]
            self.addtemp(iadd=ii)
        
        # to keep the original
        stam=self.stam.copy()
        
    def addtemp(self,iadd=None):
        """
        add scaled versions of the template in ps.stam
        to the noise signals in stsv
        :param    iadd: which LFE numbers to add to
        """

        if iadd is None:
            ilmsv = self.ilmsv[0:-1]
        else:
            iadd = np.atleast_1d(iadd)
            ilmsv = self.ilmsv[iadd]

        nper=int(np.median(np.diff(self.ilmsv)))
        # indices within template, of earthquake
        i1,ieq=np.meshgrid(np.arange(0,nper),np.arange(0,ilmsv.size))
        i1,i2=np.meshgrid(np.arange(0,nper),ilmsv)
        i1,i2,ieq=i1.flatten(),i2.flatten(),ieq.flatten()

        # indices to add to
        ix=i1+i2

        # the relevant amplitudes
        amps=self.meanamp[ieq]

        # go through the channels
        for tr in self.stam:
            idi=tr.get_id()

            # the average scaling
            lbl='.'.join([tr.stats.network,tr.stats.station,tr.stats.channel])
            amp=self.medscale[lbl]

            # get the part of the template
            tref=tr.stats.starttime+tr.stats.t3
            tri=tr.copy().trim(starttime=tref+self.tget[0],
                               endtime=tref+self.tget[1]+3*tr.stats.delta,
                               pad=True,fill_value=0.)
            vls=tri.data # *amp

            # what to add to
            trs = self.stsv.select(id=idi)[0]
            trs.data[ix]=trs.data[ix]+np.multiply(amps,vls[i1])

    def changetempdur(self,odur=0.2,ndur=0.3):

        # change the template durations
        extendtemp(self.stam,odur=odur,ndur=ndur)

        self.odur=odur
        self.ndur=ndur

    def createlongtemp(self,odur=0.2,ndur=[0.1,0.2,0.3]):

        # keep track of values used
        self.ndurs=ndur
        self.odur=odur
        self.durtemp=[]

        for nduri in ndur:
            # templates for each
            sta = self.stam.copy()
            extendtemp(sta,odur=odur,ndur=nduri)
            self.durtemp.append(sta)

    def stackbydurs(self):

        # to save
        sta = self.sta.copy()
        Ndur = len(self.durtemp)

        self.stackduras=[]
        self.stackdurmx=[]
        
        for k in range(0,Ndur):
            # no time shifts
            self.zerotshfs()
            self.calcmedtshf()
            
            # only for this set, for assigned durations
            istack=self.iduras==k
            self.stackmanylfes(istack=istack)

            # only keep high SNR
            self.checksnr()
            self.stackduras.append(self.sta.copy())

            # only for this set, for identified xc
            istack=self.idurmx==k

            # assign alignment
            for idi in self.tshfs.keys():
                self.tshfs[idi]=self.tswd[:,k]
            
            # stack
            self.stackmanylfes(istack=istack)

            # only keep high SNR
            self.checksnr()
            self.stackdurmx.append(self.sta.copy())

        # replace original
        self.sta = sta

    def xcwdurs(self):
        # save the originals
        stam = self.stam.copy()
        tshfs = self.tshfs.copy()

        self.xcwd=np.ndarray([self.tms.size,len(self.durtemp)])
        self.xcwd=np.ma.masked_array(self.xcwd,mask=False)
        self.tswd=np.ndarray([self.tms.size,len(self.durtemp)])
        self.tswd=np.ma.masked_array(self.tswd,mask=False)
        
        # cross-correlate with each of the proposed templates
        for k in range(0,len(self.durtemp)):
            self.stam = self.durtemp[k]

            # cross-correlations
            self.besttshf()

            # take the median
            xc = calcmedtshf(self.xcshf)
            self.xcwd[:,k]=xc
            self.xcwd.mask[:,k]=xc.mask

            # take the median time shift too
            tm = calcmedtshf(self.tshfs)
            self.tswd[:,k]=tm
            self.tswd.mask[:,k]=tm.mask

        # replace original
        self.stam = stam
        self.tshfs = tshfs

        # figure out which is max
        msk = np.sum(self.xcwd.mask,axis=1).astype(bool)
        self.idurmx = np.argmax(self.xcwd,axis=1)
        self.idurmx = np.ma.masked_array(self.idurmx,mask=msk)
        
        
        
#----------AMPLITUDE SCALING------------------------------


    def scalemanyamps(self):
        # need to pre-filter  for the first stack
        sta=self.sta.copy()
        if self.iterstack==0:
            sta.filter('bandpass',freqmin=self.flmget[0],
                       freqmax=self.flmget[1])
        
        # copy just the relevant times shifts
        tshfs=self.tshfs.copy()
        for ky in tshfs.keys():
            tshfs[ky]=tshfs[ky][self.isv]

        # filtering for x-c
        xcflm=np.array(self.flmamp).copy()
        if xcflm[0]==self.flmget[0]:
            xcflm[0]=0.
        if xcflm[1]==self.flmget[1]:
            xcflm[1]=float('inf')
            
        # get the amplitudes
        amps=scalemanyamps(sta,self.stsv,self.ilmsv,self.tget,
                           tshfs,xcflm=xcflm)

        # and place in the right locations
        N=len(self.tms)
        self.ampscl=dict((idi,np.ma.masked_array(np.zeros(N,dtype=float),
                                                 np.ones(N,dtype=bool)))
                         for idi in amps.keys())
        for idi in amps.keys():
            self.ampscl[idi].data[self.isv]=amps[idi]
            self.ampscl[idi].mask[self.isv]=amps[idi].mask

        # make sure they're masked
        for idi in self.ampscl.keys():
            if not isinstance(self.ampscl[idi],np.ma.masked_array):
                self.ampscl[idi]=np.ma.masked_array(self.ampscl[idi],mask=False)
            
        # note the stack used for calculation
        self.iteramp=self.iterstack

        # amplitude normalization
        self.ampok = np.ones(self.tms.size,dtype=bool)

    def meanamps(self):
        # calculate the mean amplitudes

        # just use the relevant stations
        stns = [tr.stats.network+'.'+tr.stats.station+'.'+
                tr.stats.channel for tr in self.stam]
        # stns = list(self.ampscl.keys())
        
        # average over these stations
        self.meanamp = meanamps(self.ampscl,stns=stns,minfrc=0.8,
                                medscale=self.medscale,minstat=5)

    def ampsok(self,minfrc=0.9,tlm=None):
        """
        identify some amplitudes to use
        :param     minfrc: minimum fraction of stations to use
        :param        tlm: time limit to use
        """

        if tlm is None:
            tlm=[obspy.UTCDateTime(2007,1,1),obspy.UTCDateTime(2012,1,1)]

        # just use the relevant stations
        stns = [tr.stats.network+'.'+tr.stats.station+'.'+
                tr.stats.channel for tr in self.sta]
        # stns = list(self.ampscl.keys())
        
        # check availability at these stations
        self.ampok = ampsok(self.ampscl,stns=stns,minfrc=minfrc,
                             medscale=self.medscale)

        # also consider time range
        self.ampok=np.logical_and(self.ampok,self.tms>=tlm[0])
        self.ampok=np.logical_and(self.ampok,self.tms<=tlm[1])

    def newfun(self):

        print('new')
        
    def scaleamps(self):
        """
        scale to the median amplitudes
        """

        for tr in self.sta:
            # find the right one
            lbl=tr.stats.network+'.'+tr.stats.station+'.'+tr.stats.channel
            if lbl in self.medscale.keys():
                tr.data=tr.data*self.medscale[lbl]
            else:
                # if there's no data remove this station from the waveforms
                self.sta.remove(tr)

        for tr in self.stam:
            # find the right one
            lbl=tr.stats.network+'.'+tr.stats.station+'.'+tr.stats.channel
            if lbl in self.medscale.keys():
                tr.data=tr.data*self.medscale[lbl]
            else:
                # if there's no data remove this station from the waveforms
                self.stam.remove(tr)
                
    def amppatterns(self):

        # just stations in stack
        stns=[tr.stats.network+'.'+tr.stats.station+'.'+tr.stats.channel
              for tr in self.sta]
        stns = np.unique(np.array(stns))

        
        self.pamp=amppatterns(self.ampscl,stns,self.medscale)

    def calcmedscale(self,niter=5,ampl=[-float('inf'),float('inf')]):
        """
        determine the median amplitude for each station
        """

        # start with everything in range
        iok=np.ones(list(self.ampscl.values())[0].size,dtype=bool)
        try:
            iok=self.ampok.copy()
        except:
            self.ampsok(minfrc=0.7)
            iok=self.ampok.copy()
            
        # initialize median scalings
        self.medscales = []
        self.medscale = {}

        for lbl in self.ampscl.keys():
            # identify median value for this component from the original
            mn = self.ampscl[lbl]
            if not isinstance(mn,np.ma.masked_array):
                mn=np.ma.masked_array(mn,mask=False)
            jok=np.logical_and(~np.isnan(mn),iok)
            jok=np.logical_and(jok,~np.ma.getmaskarray(mn))
            mn = mn[jok]
            if mn.any():
                self.medscale[lbl] = np.median(mn)
                

        # recompute the median amplitudes per event
        self.meanamps()

        # a scaling factor to keep the mean near 1
        scl = np.median(self.meanamp[~self.meanamp.mask])
        self.meanamp=self.meanamp/scl
        for ky in self.medscale.keys():
            self.medscale[ky]=self.medscale[ky]*scl

        # copy to see how things change
        self.medscales=self.medscale.copy()

        for k in range(0,niter):
            # start with a blank set each time
            self.medscale = {}

            # a scaling factor to avoid throwing away bits
            scl = np.median(self.meanamp[~self.meanamp.mask])

            # only use a subset of the events
            iok = np.logical_and(self.meanamp>=ampl[0]*scl,
                                 self.meanamp<=ampl[1]*scl)

            for lbl in self.ampscl.keys():
                # estimate the amplitude ratios for this component
                mn = np.divide(self.ampscl[lbl],self.meanamp)

                # and take the median
                jok=np.logical_and(~np.isnan(mn),iok)
                jok=np.logical_and(jok,~self.ampscl[lbl].mask)
                jok=np.logical_and(jok,~self.meanamp.mask)
                mn = mn[jok]
                if mn.any():
                    self.medscale[lbl] = np.median(mn) / scl


            # recompute the median amplitudes per event
            self.meanamps()

            # a scaling factor to keep the mean near 1
            scl = np.median(self.meanamp[~self.meanamp.mask])
            self.meanamp=self.meanamp/scl
            for ky in self.medscale.keys():
                self.medscale[ky]=self.medscale[ky]*scl

            # copy to see how things change
            for lbl in self.medscale.keys():
                self.medscales[lbl]=np.append(self.medscales[lbl],
                                              self.medscale[lbl])

    def calcmedscalenomean(self,niter=5,ampl=[-float('inf'),float('inf')]):
        """
        determine the median amplitude for each station,
        but don't recompute the mean
        """

        # start with everything in range
        iok=np.ones(list(self.ampscl.values())[0].size,dtype=bool)
        iok=self.ampok.copy()


        # start with a blank set
        self.medscale = {}

        # a scaling factor to avoid throwing away bits
        scl = np.median(self.meanamp[~self.meanamp.mask])

        # only use a subset of the events
        iok = np.logical_and(self.meanamp>=ampl[0]*scl,
                             self.meanamp<=ampl[1]*scl)

        for lbl in self.ampscl.keys():
            # estimate the amplitude ratios for this component
            mn = np.divide(self.ampscl[lbl],self.meanamp)

            # and take the median
            jok=np.logical_and(~np.isnan(mn),iok)
            jok=np.logical_and(jok,~self.ampscl[lbl].mask)
            jok=np.logical_and(jok,~self.meanamp.mask)
            mn = mn[jok]
            if mn.any():
                self.medscale[lbl] = np.median(mn) / scl
                
#--------THE CROSS-CORRELATION----------------------------------------

    def taperbydur(self):
        """
        taper the templates to avoid extra values
        """

        taperbydur(self.sta+self.stam)

    def calcstd(self,usenshf=False):
        """
        calculate standard deviations for the saved data
        :param   usenshf: use the shifted values
        """
        
        # for all the data
        ids=np.array([tr.get_id() for tr in self.stsv])

        # timing of the data
        ilmsv1,ilmsv2=self.ilmsv[0:-1],self.ilmsv[1:]
        twin=np.array([-1,5])-self.tget[0]
        iwin=np.round(twin/self.sta[0].stats.delta).astype(int)
        ilmsv1,ilmsv2=ilmsv1+iwin[0],ilmsv1+iwin[1]

        # calculate
        ststd,trash=stsvstd(self.stsv,ilmsv1,ilmsv2,mxmad=4.,ids=ids)

        # create a dictionary
        self.ststd=dict((ids[k],ststd[k,:]) for k in range(0,len(ids)))


        if usenshf:
            # use the pre-extracted noise intervals
            stsvn = [self.stsvn[k] for k in ins]
            zshf = 0
        else:
            # or just an earlier portion of the main record
            stsvn = [self.stsv]
            zshf = int(np.round(-8./self.stsv[0].stats.delta))

        # create a dictionary for the noise
        N = ststd.shape[1]
        Nn = len(stsvn)
        self.ststdn=dict((ids[k],np.ndarray([N,Nn],dtype=float)) 
                         for k in range(0,len(ids)))
        
        # for each noise realization
        for m in range(0,Nn):
            ststd,trash=stsvstd(stsvn[m],ilmsv1+zshf,ilmsv2+zshf,mxmad=4.,ids=ids)
            for k in range(0,len(ids)):
                self.ststdn[ids[k]][:,m] = ststd[k,:]

        # timing of the data for a longer interval
        ilmsv1,ilmsv2=self.ilmsv[0:-1],self.ilmsv[1:]
        twin=np.array([-4.5,-0.5])-self.tget[0]
        iwin=np.round(twin/self.sta[0].stats.delta).astype(int)
        ilmsv1,ilmsv2=ilmsv1+iwin[0],ilmsv1+iwin[1]

        # also a longer interval
        ststd,trash=stsvstd(self.stsv,ilmsv1,ilmsv2,mxmad=4.,ids=ids)

        # create a dictionary
        self.ststda=dict((ids[k],ststd[k,:]) for k in range(0,len(ids)))


    def xcmanylfes(self,ixc=None,bystat=True,usenshf=False):

        # keep track of whether stations are averaged
        self.xcbystat=bystat
        
        # need to identify noisy data
        try:
            stds = self.ststd
        except:
            self.calcstd()
            
        # compute with synthetics?
        if self.xcsyn:
            stsv = self.stsvs
            ins = ins[ins!=self.isyn]
            stds = dict((idi,self.ststdn[idi][:,self.isyn]) 
                        for idi in self.ststdn.keys())
        else:
            stsv = self.stsv
            stds = self.ststd.copy()

        if usenshf:
            # use the pre-extracted noise intervals
            stsvn = [self.stsvn[k] for k in ins]
            zshf = 0
        else:
            # or just an earlier portion of the main record
            stsvn = [stsv]
            zshf = int(np.round(-8./stsv[0].stats.delta))
        ins = np.arange(0,len(stsvn))

        # add noise to the stds
        for idi in stds.keys():
            stds[idi]=np.append(stds[idi].reshape([stds[idi].size,1]),
                                self.ststdn[idi][:,ins],axis=1)
        

        # station ids
        ids=np.array([tr.get_id() for tr in self.stam])
        self.ids = ids

        # figure out which events to use
        iok,jok=pickstsv(stsv,stsvn,self.stam,
                         self.ilmsv,self.maxstd,ids,
                         stds)

        iok=np.sum(jok,axis=0)/float(jok.shape[0])>0.7
        iok=np.sum(jok,axis=0)>=3
        iok=np.atleast_1d(iok).astype(bool)
        print(str(np.sum(iok))+' LFEs with acceptable std')

        # also require that the time shift uncertainty be acceptable
        tshfok=np.vstack([self.tshfu[idi][self.isv] for idi in ids])<\
            self.tshfumax
        jok = np.logical_and(jok,tshfok)

        # std scaling
        scls=[np.max(stds[idi][:,1:],axis=1) for idi in ids]
        scls=[self.ststda[idi] for idi in ids]
        scls=np.vstack(scls)
        scls=np.exp(scls)

        # save the median
        mscl=np.array([np.median(scls[k,jok[k,:]]) for
                       k in range(0,scls.shape[0])])
        mscl=np.divide(scls,mscl.reshape([scls.shape[0],1]))
        mscl=np.array([np.median(mscl[jok[:,k],k]) for
                       k in range(0,scls.shape[1])])
        self.mscl=mscl

        # scaling
        if self.rscl:
            pass
        else:
            scls=np.ones(jok.shape,dtype=float)

        # check scaling is okay
        jok=np.logical_and(jok,scls<1.e20)
            
        # keep events with some fraction of the stations
        iok=np.sum(jok,axis=0)/float(jok.shape[0])>0.
        iok=np.atleast_1d(iok).astype(bool)

        # a bound on the LFE amplitudes to use
        if self.amplim is not None:
            self.meanamps()
            iok=np.logical_and(iok,self.meanamp[self.isv]>=
                               self.amplim[0])
            iok=np.logical_and(iok,self.meanamp[self.isv]<=
                               self.amplim[1])

        # normalize scaling
        mscl=np.array([np.median(scls[k,np.logical_and(jok[k,:],iok)])
                       for k in range(0,scls.shape[0])])
        scls=np.divide(scls,mscl.reshape([mscl.size,1]))
            
        # if one or more events were specified
        if ixc is not None:
            iok=np.logical_and(iok,ixc)

        if isinstance(iok,np.ma.masked_array):
            iok=np.logical_and(iok.data,~iok.mask)

        # index data to use by station (and component) as well
        jok = jok[:,iok]
        scls = scls[:,iok]
        self.tmxc = np.atleast_1d(self.tsv[iok])

        print(str(np.sum(iok))+' LFEs used')

        # which indices contain these LFEs
        i1,i2=self.ilmsv[0:-1],self.ilmsv[1:]
        i1,i2=i1[iok],i2[iok]
        
        # get the indices of the zero time
        izero = -self.tget[0]/stsv[0].stats.delta
        izero = int(np.round(izero))
        izero = i1+izero
        self.izero=izero

        # to keep track of the range of allowable values
        irange = np.vstack([i1,i2]).transpose()

        # filtering for the stack only required if it
        # hasn't been created from the filtered data
        if self.iterstack==0:
            flmget = self.flmget
        else:
            flmget = None

        if self.useloc=='saved':
            tshfs = self.tshfs.copy()
            for ky in tshfs.keys():
                tshfs[ky]=tshfs[ky][self.isv][iok]

        # with actual times
        self.cp,self.en,self.freq,self.tshfsxc,self.cpt,self.fxc,\
            self.ampt,self.xz,self.Nucp,self.tprs,self.xcsave=\
            calcwlk(stsv,self.stam,izero=self.izero,
                    trange=self.txc,tshfs=tshfs,
                    iok=jok,twin=self.twin,mxmad=self.mxmad,
                    eqloc=self.eqloc,flm=flmget,irange=irange,
                    bystat=self.xcbystat,usemshf=self.usemshf,
                    scls=scls)

        # with shifted times
        self.cpn,self.enn=[],[]
        for st in stsvn:
            cp,en,freq,tsh,cpt,tsh,tsh,tsh,tsh,tsh,tsh=\
                calcwlk(st,self.stam,izero=self.izero+zshf,
                        trange=self.txc,eqloc=self.eqloc,
                        iok=jok,twin=self.twin,mxmad=self.mxmad,
                        tshfs=self.tshfsxc,flm=flmget,irange=irange,
                        bystat=self.xcbystat,usemshf=self.usemshf,
                        scls=scls)

            self.cpn.append(cp)
            self.enn.append(en)

#--------TIME SHIFTS---------------------------------

    def calcmedtshf(self,settshf=True):

        # only use some shifts
        ids=np.array([tr.get_id() for tr in self.stam])
        tshfs=dict([(idi,self.tshfs[idi]) for idi in ids])
        
        # calculate median shifts
        self.mtshf = calcmedtshf(tshfs)
        self.mtshf.mask = np.logical_or(self.mtshf.mask,
                                        np.abs(self.mtshf)>0.05)

        # set the values to ttshfs
        try:
            ids=[tr.get_id() for tr in self.stsv]
        except:
            ids=list(self.tshfs.keys())

        self.tshfs=dict([(idi,self.mtshf.copy()) for idi in ids])

    def comparemedtshf(self):

        # split into two sets of stations
        stns=np.unique([tr.stats.station for tr in self.stam])
        st1,st2=obspy.Stream(),obspy.Stream()
        i1=np.random.choice(len(stns),int(len(stns)/2),replace=False)
        i2=np.array(list(set(np.arange(0,len(stns)))-set(i1)))
        for stn in stns[i1]:
            st1=st1+self.stam.select(station=stn)
        for stn in stns[i2]:
            st2=st2+self.stam.select(station=stn)
        id1=[tr.get_id() for tr in st1]
        id2=[tr.get_id() for tr in st2]

        # for the first set of stations
        tshfs=dict([(idi,self.tshfs[idi]) for idi in id1])
        self.mtshf1 = calcmedtshf(tshfs)
        self.mtshf1.mask = np.logical_or(self.mtshf1.mask,
                                         np.abs(self.mtshf1)>0.05)

        # for the first set of stations
        tshfs=dict([(idi,self.tshfs[idi]) for idi in id2])
        self.mtshf2 = calcmedtshf(tshfs)
        self.mtshf2.mask = np.logical_or(self.mtshf2.mask,
                                         np.abs(self.mtshf2)>0.05)

        # to compare
        try:
            ixc = general.closest(self.tms,self.tmxc)
            ixc = ixc[ps.Nucp>=5]
        except:
            ixc = np.arange(0,self.tms.size)

        # differences
        df=self.mtshf1[ixc]-self.mtshf2[ixc]
        df=df.data[~df.mask]
        df.sort()

        # find the relevant values
        print('{:0.0f}'.format(100*np.sum(np.abs(df)<=0.01)/df.size)+
              '% smaller than 0.01')
        print('{:0.0f}'.format(100*np.sum(np.abs(df)<=0.02)/df.size)+
              '% smaller than 0.02')
        
        prc=np.array([0.05,0.15,0.5,0.85,0.95])
        prcl=['{:g}'.format(vl) for vl in prc]
        vls =np.interp(prc*df.size,np.arange(0,df.size),df)
        #print('Percentiles: [
            
    def besttshf(self):

        # which indices contain these LFEs
        i1,i2=self.ilmsv[0:-1],self.ilmsv[1:]
        i1=i1.flatten()
        
        # get the indices of the zero time
        izero = -self.tget[0]/self.stsv[0].stats.delta
        izero = int(np.round(izero))
        izero = i1+izero

        # filtering for x-c
        xcflm=np.array(self.flmtshf).copy()
        if xcflm[0]==self.flmget[0]:
            xcflm[0]=0.
        if xcflm[1]==self.flmget[1]:
            xcflm[1]=float('inf')
        
        # x-c to get prefered shift
        self.tshfs,self.xcshf=\
                besttshf(self.stsv,self.stam,izero,
                         trange=self.txc,tlk=[-0.5,0.5],
                         xcflm=xcflm)

    def zerotshfsmany(self):
        # initalize time shifts
        ids = [tr.get_id() for tr in self.stsv]
        self.tshfu=dict((idi,np.zeros(len(self.tms))) for
                        idi in ids)
        self.tshfs=dict((idi,np.zeros(len(self.tms))) for
                        idi in ids)

        
    def zerotshfs(self):
        # set all the shifts to zero
        for idi in self.tshfs:
            self.tshfs[idi]=np.zeros(self.tms.size,dtype=float)

        # note that there's no shift
        self.randxshf=0.
        self.randzshf=0.

    def randtshfs(self,xshf=0.5,zshf=None,addold=False):
        """
        :param      xshf: half-width for x shifts
        :param      xshf: half-width for z shifts
        :param    addold: add existing time shifts?
        """
        if zshf is None:
            zshf=xshf
        xshf=float(xshf)
        zshf=float(zshf)

        if self.eqloc is None:
            self.eqloc=lfeloc(self.fnum)
        
        # compute shifts
        shfs,self.xsh,self.zsh=\
            addrandshifts(self.eqloc,self.sta,Nev=self.tms.size,
                          hhwd=xshf,vhwd=zshf)

        # add existing values
        if addold:
            for idi in shfs:
                shfs[idi]=shfs[idi]+self.tshfs[idi]

        # copy to save (overwrites old values)
        self.tshfs = shfs

        # note that there's no shift
        self.randxshf=xshf
        self.randzshf=zshf

    def calctakeang(self):

        if self.eqloc is None:
            self.eqloc=lfeloc(self.fnum)

        # stations of interest
        stns=np.unique([tr.stats.station for tr in self.stam])
        
        self.staz={}
        self.tkang={}
        for stn in stns:
            tr=self.stam.select(station=stn)[0]
            sloc=[tr.stats.sac['stlo'],tr.stats.sac['stla'],0]
            tkang,tms,az=pksdrops.calctakeang(eloc=self.eqloc,oloc=sloc,
                                              mdl='iasp91',phsarv='sS')
            lbl=tr.stats.network+'.'+tr.stats.station
            self.staz[lbl]=(az-135) % 360
            self.tkang[lbl]=tkang

    def writetkang(self):

        fname='Stations_'+'{:d}'.format(self.fnum)
        fdir=os.path.join(os.environ['DATA'],'TREMORAREA','SimLFEs')
        fname=os.path.join(fdir,fname)

        fl=open(fname,'w')
        for lbl in list(self.tkang.keys()):
            fl.write(lbl+','+'{:0.1f}'.format(self.staz[lbl])+','+\
                     '{:0.1f}'.format(self.tkang[lbl])+'\n')
        fl.close()

#--------SIGNAL TO NOISE RATIO---------------------------------

    def xcsnr(self):

        # get signal to noise ratio relevant for templates
        self.snrfreq,self.snr,self.snrreal=tempsnr(self.stam,self.txc,self.twin)


    def checksnr(self):
        # also check in case it's okay on P but not S
        if self.checkp:
            self.stap=checksnr(self.sta,minsnr=self.minsnr,blim=self.blim,
                               pk=self.ppk,wlen=self.trange,cmps=self.cmpuse)
        else:
            self.stap = obspy.Stream()

        # check the signal to noise ratio for the S arrival
        self.sta=checksnr(self.sta,minsnr=self.minsnr,blim=self.blim,
                          pk=self.pk,wlen=self.trange,cmps=self.cmpuse)
        

        if self.stam:
            # also check in case it's okay on P but not S
            if self.checkp:
                self.stamp=checksnr(self.stam,minsnr=self.minsnr,blim=self.blim,  
                                    pk=self.ppk,wlen=self.trange,cmps=self.cmpuse)
            else:
                self.stamp = obspy.Stream()


            self.stam=checksnr(self.stam,minsnr=self.minsnr,
                               blim=self.blim,
                               pk=self.pk,wlen=self.trange,
                               cmps=self.cmpuse)

#----------THE DATA-----------------------------------
    def dataprep(self):
        """
        use the first stack to collect data
        """

        # apply a low threshold for accepting stacks
        #self.minsnr = 1.
        #self.blim = np.array([2.,8])
        #self.checksnr()

        # consider all components
        self.cmpuse = 'all'
        self.pickcomp()

        print('Collecting LFE data')
        self.tget = np.array([-10.,10])
        self.grabmanylfes()
        self.writemanylfes()

    def grabmanylfesold(self):
        self.stsv,self.tsv,self.stsvn,self.ilmsv= \
            grabmanylfes(self.stam,self.tsv,rnoise=self.rnoise,
                         csyn=self.csyn,tget=self.tget,
                         realign=self.realign,flm=self.flmget)
        self.isv = eventindex(self.tms,self.tsv)

    def grabmanylfes(self):

        import klfesearch as klfe
        self.tsv=self.tms.copy()
        self.stsv,self.ilmsv= \
            klfe.collectlfedata(self.tsv,fnum=self.fnum,flm=self.flmget,
                                trange=self.tget,samp20=self.samp20)
        self.isv = eventindex(self.tms,self.tsv)

        self.stsvn=[]

        # to grab picks
        sta = initstacks()
        readpicks(fnum=self.fnum,sta=sta,pk='t3',pkref='t1')
        
        # set reference time shifts
        for st in [self.stsv]+self.stsvn:
            for tr in st:
                #trr=sta.select(id=tr.get_id())[0]
                trr=sta.select(station=tr.stats.station,network=tr.stats.network)[0]
                tshf=trr.stats.starttime+trr.stats.t3-obspy.UTCDateTime(2000,1,1)
                tr.stats.t7 = tshf


    def detfilename(self):
        # write extracted LFEs
        fname=detectionsfilename(self.fnum,self.xvli,self.zvli,
                                 self.flmdet,self.odurdet,self.ndurdet)

        return fname
        
    def writemanylfes(self,lbl='',detname=True):
        
        lbl=lbl+'_'+'{:g}'.format(general.roundsigfigs(self.flmget[0],3))+'_'+\
             '{:g}'.format(general.roundsigfigs(self.flmget[1],3))+'Hz'
        lbl='Family'+str(self.fnum)+lbl

        # write extracted LFEs
        if detname:
            fname=self.detfilename()
            lbl=lbl+'_'+fname

        writemanylfes(lbl,self.stsv,self.tsv,self.stsvn,
                      self.ilmsv,self.tget)


            
#----------STACKING AND LOCATING------------------------------------


    def firststacks(self,ntry=4):
        """
        iterate over locations and stack ntry times
        assumes one initial stack already exists
        """

        print('Reading the data')
        self.readmanylfes()

        print('Making the first stack')
        # the first stack
        self.stackmanylfes()
        self.iterstack=0
        self.checkstackdata()
        self.writestacks()

        print('Computing initial amplitude scaling')
        self.scalemanyamps()
        self.iteramp = self.iterstack
        self.writeamps()

        # for locations
        self.minsnr = 3.
        self.blim = np.array([2.,8.])
        self.checkp = False
        self.trange=np.array([-0.1,2.9])
        
        for k in range(0,ntry):
            print('Iteration '+str(k))

            print('New stack')
            self.readstacks(itn=self.iterstack)
            self.readamps(itn=self.iterstack)
            self.checkstackdata()
            self.cmpuse = 'all'
            self.pickcomp()

            # only good stations for the scaling
            print('Scaling stack')
            self.checksnr()
            self.calcmedscale()
            self.ampsok(minfrc=0.7)
            ampok=self.ampok.copy()

            print('New alignment')
            self.besttshf()
            self.calcmedtshf()
            
            print('Rescaling stacks')
            # but then everything
            self.readstacks(itn=self.iterstack)
            self.readamps(itn=self.iterstack)
            self.checkstackdata()
            self.cmpuse = 'all'
            self.pickcomp()
            self.ampok=ampok
            self.calcmedscalenomean()
            
            # stack
            print('Stacking')
            self.ampstack=[0.2,6]
            self.stackmanylfes()

            # note the iteration and write to file
            print('Writing stacks')
            self.iterstack = self.iterstack+1
            self.writestacks()

            # compute the appropriate amplitude scaling
            print('Computing amplitude scaling')
            self.scalemanyamps()
            self.iteramp = self.iterstack
            self.writeamps()

    def stacknext(self,ntry=4):
        """
        iterate over locations and stack ntry times
        assumes one initial stack already exists
        """

        # for locations
        self.minsnr = 2.
        self.blim = np.array([2.,10.])
        self.checkp = True
        self.trange=np.array([-0.1,1.9])

        for k in range(0,ntry):
            # stacks and waveforms
            self.readstacks()
            self.checkstackdata()

            # first set of amplitudes, in case it wasn't done before
            if k==0:
                print('Computing initial amplitude scaling')
                self.cmpuse = 'all'
                self.pickcomp()
                self.scalemanyamps()
                self.writeamps()

            self.cmpuse = 'all'
            self.pickcomp()
            self.checksnr()
            
            # locations
            print('Locating')
            self.locmanylfes()

            # write the value
            print('Writing locations')
            self.iterloc = self.iterloc+1
            self.iterloc = self.iterstack+1
            self.writetshfs()

            # re-read stacks and extrapolate time shifts 
            print('Re-reading stacks and time shifts')
            self.readstacks()
            self.checkstackdata()
            self.cmpuse = 'all'
            self.pickcomp()
            self.readtshfs(itn=self.iterloc)

            # stack
            print('Stacking')
            self.stackmanylfes()

            # note the iteration and write to file
            print('Writing stacks')
            self.iterstack = self.iterstack+1
            self.writestacks()

            # compute the appropriate amplitude scaling
            print('Computing amplitude scaling')
            self.scalemanyamps()
            self.writeamps()

    def checkstackdata(self):
        # remove any stacks with nans
        for tr in self.sta:
            if np.sum(np.isnan(tr.data)):
                self.sta.remove(tr)
            elif isinstance(tr.data,np.ma.masked_array):
                if np.sum(~tr.data.mask)==0:
                    self.sta.remove(tr)


    def stackmanylfes(self,istack=None,pweight=True):
        # staion ids
        ids=np.array([tr.get_id() for tr in self.stsv])
        if self.ampstack is not None:
            ids=np.array([tr.get_id() for tr in self.sta])
            ids2=np.array(['.'.join([tr.stats.network,tr.stats.station,
                                     tr.stats.channel]) for tr in self.sta])
        
        # also require that the time shift uncertainty be acceptable
        jok=np.vstack([self.tshfu[idi][self.isv] for idi in ids])<\
            self.tshfumax
        jok=jok.transpose()

        # if only a subset of the events was specified
        if istack is not None:
            istack=istack.reshape([istack.size,1])
            jok=np.logical_and(jok,istack)
        
        # and that they not be marked as intervals with poor data
        #self.exctimes(ids=ids)
        #jok=np.logical_and(jok,~self.exct[self.isv,:])
        # also require that the amplitude be reasonable
        if self.ampmean is not None:
            aok=np.logical_and(self.meanamp>=self.ampmean[0],
                               self.meanamp<=self.ampmean[1])
            jok=np.logical_and(jok,aok.reshape([aok.size,1]))

        # also require that the amplitude be reasonable
        if self.ampstack is not None:
            amps=np.vstack([self.ampscl[idi][self.isv]/self.medscale[idi]
                            for idi in ids2]).T
            # amps[self.exct[self.isv,:]] = 0
            # mamps=np.array([np.median(amps[amps[:,k]!=0,k]) for
            #                 k in range(0,amps.shape[1])])
            # mamps=mamps.reshape([1,mamps.size])
            aok=np.logical_and(amps>=self.ampstack[0],
                               amps<=self.ampstack[1])
            jok=np.logical_and(jok,aok)

        
        # change time shifts to be relative to pick
        # tshfs = self.tshfs.copy()
        # for tr in self.stam:
        #     dtim=tr.stats.starttime+tr.stats.t3-obspy.UTCDateTime(2000,1,1)
        #     tshfs[tr.get_id()]=tshfs[tr.get_id()]-dtim
            
        # also remove the median times to avoid a gradual shift
        tmn=np.median(np.median(np.vstack(self.tshfs.values()),axis=0))
        for ky in self.tshfs.keys():
            self.tshfs[ky] = self.tshfs[ky]-tmn
            
        # just the events of interest
        tshfs = self.tshfs.copy()
        for ky in tshfs.keys():
            tshfs[ky] = tshfs[ky][self.isv]

        # stack
        self.sta,self.staw=stackmanylfes(self.stsv,self.ilmsv,self.tget,
                               jok,tshfs,ids,self.xcvl[self.isv],
                               nmtype=self.wgttype,pweight=pweight)

        self.pweight=pweight
        
        # add time picks
        for tr in self.sta+self.staw:
            tr.stats.t9=self.fnum
            tr.stats.t1=obspy.UTCDateTime(2000,1,1)-tr.stats.starttime
        readpicks(sta=self.sta+self.staw)

        if 'shifted' not in self.wgttype:
            if np.array(list(self.tshfs.values())).any():
                self.wgttype = self.wgttype + '_shifted'


    def readstacks(self,lbstack=None,itn=None):
        """
        read the stacks and the relevant times
        :param   lbstack:   a label for this stack
        :param       itn:   iteration number (default: maximum)
        """

        # directory to read from
        if lbstack is None:
            lbstack = self.wgttype
        if self.flmget[0]!=2. or self.flmget[1]!=30.:
            lbstack=lbstack+'_'+\
                     '{:g}'.format(general.roundsigfigs(self.flmget[0],3))+'_'+\
                     '{:g}'.format(general.roundsigfigs(self.flmget[1],3))+'Hz'

        fname= 'Family'+'{:d}'.format(self.fnum)+lbstack
        fdir = os.path.join(os.environ['TREMORAREA'],'SavedStacks')
        if not os.path.exists(fdir):
            fdir = os.path.join(os.environ['DATA'],'TREMORAREA','SavedStacks')
        fdir=os.path.join(fdir,fname)
        print(fdir)

        # find the latest iteration if not given
        if itn is None:
            fls=glob.glob(fdir+'_itn*')
            itn=np.array([int(fl.split('_itn')[-1]) for fl in fls])
            itn=np.max(itn)
        self.iterstack = itn

        # add iteration number
        fdir = fdir+'_itn'+str(itn)
        lbstack = lbstack+'_itn'+str(itn)

        # read stack times
        fname = os.path.join(fdir,'stacktimes')
        vls = np.loadtxt(fname,dtype=int)
        self.tstack=[obspy.UTCDateTime(vls[k,0],vls[k,1],vls[k,2],
                                       vls[k,3],vls[k,4],vls[k,5]) + 
                     float(vls[k,6])/1.e6 for k in range(0,vls.shape[0])]
        self.tstack=np.array(self.tstack)
        self.istack=eventindex(self.tms,self.tstack)

        # read comparison times
        fname = os.path.join(fdir,'comptimes')
        vls = np.loadtxt(fname,dtype=int)
        self.tcomp=[obspy.UTCDateTime(vls[k,0],vls[k,1],vls[k,2],
                                      vls[k,3],vls[k,4],vls[k,5]) + 
                    float(vls[k,6])/1.e6 for k in range(0,vls.shape[0])]
        self.tcomp=np.array(self.tcomp)
        self.icomp=eventindex(self.tms,self.tcomp)

        # second value
        self.sta = readstacks(self.fnum,lbstack=lbstack)

            
    def writestacks(self,lbstack=None):
        """
        # write stacked templates and their indices
        :param   lbstack:   a label for this stack
        """

        # the label
        if lbstack is None:
            lbstack = self.wgttype
        if self.flmget[0]!=2. or self.flmget[1]!=30.:
            lbstack=lbstack+'_'+\
                     '{:g}'.format(general.roundsigfigs(self.flmget[0],3))+'_'+\
                     '{:g}'.format(general.roundsigfigs(self.flmget[1],3))+'Hz'

        # add the iteration number
        lbstack = lbstack+'_itn'+str(self.iterstack)
        
        # directory to write to
        fdir = 'Family'+str(self.fnum)+lbstack
        fdir = os.path.join(os.environ['TREMORAREA'],
                            'SavedStacks',fdir)
        if not os.path.exists(fdir):
            os.makedirs(fdir)

        # write stack times
        fname = os.path.join(fdir,'stacktimes')
        fl = open(fname,'w')
        for tm in self.tstack:
            fl.write(tm.strftime("%Y %m %d %H %M %S %f\n"))
        fl.close()

        # write comparison times
        fname = os.path.join(fdir,'comptimes')
        fl = open(fname,'w')
        for tm in self.tcomp:
            fl.write(tm.strftime("%Y %m %d %H %M %S %f\n"))
        fl.close()

        # write stack
        print(lbstack)
        writestacks(self.sta,lbstack=lbstack)

    def writecomplete(self,lbstack=None):
        """
        # write stacked templates and their indices
        :param   lbstack:   a label for this stack
        """

        # the label
        if lbstack is None:
            lbstack = self.wgttype

        # add the iteration number
        lbstack = lbstack+'_itn'+str(self.iterstack)

        # directory to write to
        fdir = 'Family'+str(self.fnum)+lbstack
        fdir = os.path.join(os.environ['TREMORAREA'],
                            'SavedXC',fdir)
        if not os.path.exists(fdir):
            os.makedirs(fdir)

        # write stack times
        fname = os.path.join(fdir,'stacktimes')
        fl = open(fname,'w')
        for tm in self.tstack:
            fl.write(tm.strftime("%Y %m %d %H %M %S %f\n"))
        fl.close()

        # write comparison times
        fname = os.path.join(fdir,'comptimes')
        fl = open(fname,'w')
        for tm in self.tcomp:
            fl.write(tm.strftime("%Y %m %d %H %M %S %f\n"))
        fl.close()

        # write stack
        writestacks(self.sta,lbstack=lbstack)

        
    def writeamps(self,lbstack=None):
        """
        # write stacked templates and their indices
        :param   lbstack:   a label for this stack
        """

        if lbstack is None:
            lbstack = self.wgttype

        if self.flmget[0]!=2. or self.flmget[1]!=30.:
            lbstack=lbstack+'_'+\
                     '{:g}'.format(general.roundsigfigs(self.flmget[0],3))+'_'+\
                     '{:g}'.format(general.roundsigfigs(self.flmget[1],3))+'Hz'

        # add the iteration number
        lbstack = lbstack+'_itn'+str(self.iteramp)

        # directory to write to
        fname = 'Family'+str(self.fnum)+lbstack
        fdir = os.path.join(os.environ['TREMORAREA'],
                            'SavedAmps')
        if not os.path.exists(fdir):
            os.makedirs(fdir)

        print('Amplitudes writing '+fname)

        # write amplitudes
        fname = os.path.join(fdir,fname)
        fl = open(fname,'w')
        for ky in self.ampscl.keys():
            fl.write(ky)
            for vl in self.ampscl[ky]:
                fl.write(','+str(vl))
            fl.write('\n')

        fl.close()

    def readamps(self,lbstack=None,itn=None):
        """
        read amplitudes
        :param   lbstack:   a label for this stack
        :param       itn:   iteration number to use
        """

        if lbstack is None:
            lbstack = self.wgttype

        if self.flmget[0]!=2. or self.flmget[1]!=30.:
            lbstack=lbstack+'_'+\
                     '{:g}'.format(general.roundsigfigs(self.flmget[0],3))+'_'+\
                     '{:g}'.format(general.roundsigfigs(self.flmget[1],3))+'Hz'
            
        # directory to write to
        fdir = os.path.join(os.environ['TREMORAREA'],
                            'SavedAmps')
        fname = 'Family'+str(self.fnum)+lbstack

        print('Amplitudes '+fname)

        # use stack or maximum iteration number
        if itn is None:
            itn = self.iterstack
        if itn<0:
            fls=glob.glob(os.path.join(fdir,fname+'_itn*'))
            itn=np.array([int(fl.split('_itn')[-1]) for fl in fls])
            itn=np.max(itn)
        self.iteramp = itn

        # add iteration number
        lbstack = lbstack+'_itn'+str(itn)
        fname = 'Family'+str(self.fnum)+lbstack

        # read amplitudes
        fname = os.path.join(fdir,fname)
        fl = open(fname,'r')

        # assign to amplitude
        self.ampscl={}

        for line in fl:
            # organize
            vls=line.split(',')
            ky=vls[0]
            vls=vls[1:]
            vls=[vl.replace('--','nan') for vl in vls]
            vls=np.array(vls).astype(float)

            msk = np.isnan(vls)

            # add to set
            self.ampscl[ky]=np.ma.masked_array(vls,mask=msk)

        fl.close()


            
#--------LOCATIONS AND TIME SHIFTS-------------------------

    def locmanylfes(self):
        # need an additional filter to apply to the data if
        # it's the first iteration, not created from the stacks
        if self.iterstack==0:
            stam = self.stam.copy()
            stam.filter('bandpass',freqmin=self.flmget[0],
                        freqmax=self.flmget[1])
        else:
            stam = self.stam

        # get LFE locations and time shifts
        self.xzbest,self.tshfs,self.xboot,self.zboot,self.tshfu=\
            locmanylfes(stam,self.stsv,self.ilmsv,
                        self.tget,None,self.eqloc,
                        wlen=ps.trange,blimadd=self.blimloc,
                        vmod=self.vmod,checkp=self.checkp,
                        stap=self.stamp)
        
        # # change the timing to be relative to the event time
        # for tr in self.stam:
        #     dtim=tr.stats.starttime+tr.stats.t3-obspy.UTCDateTime(2000,1,1)
        #     self.tshfs[tr.get_id()]=self.tshfs[tr.get_id()]+dtim


    def writetshfs(self,lbstack=None):
        """
        write time shifts estimate for each one
        :param   lbstack:   a label for this stack
        """
        
        if lbstack is None:
            lbstack = self.wgttype

        # add the iteration number
        lbstack = lbstack+'_itn'+str(self.iterloc)
        
        # directory to write to
        fname = 'Family'+str(self.fnum)+lbstack
        fdir = os.path.join(os.environ['TREMORAREA'],
                            'LOCATIONS',fname)
        if not os.path.exists(fdir):
            os.makedirs(fdir)
            
        # write time shifts
        fname = os.path.join(fdir,'tshfs')
        fl = open(fname,'w')
        for ky in self.tshfs.keys():
            fl.write(ky)
            for vl in self.tshfs[ky]:
                fl.write(','+str(vl))
            fl.write('\n')
        fl.close()

        # write time uncertainty
        fname = os.path.join(fdir,'tshfu')
        fl = open(fname,'w')
        for ky in self.tshfu.keys():
            fl.write(ky)
            for vl in self.tshfu[ky]:
                fl.write(','+str(vl))
            fl.write('\n')
        fl.close()

        # write velocity model used for locations
        fname = os.path.join(fdir,'velocitymodel')
        fl = open(fname,'w')
        fl.write(self.vmod)
        fl.close()
        
        # write locations
        fname = os.path.join(fdir,'xzbest')
        fl = open(fname,'w')
        for k in range(0,self.xzbest.shape[0]):
            fl.write(str(self.xzbest[k,0])+','+
                     str(self.xzbest[k,1])+'\n')
        fl.close()

        # write location uncertainty
        fname = os.path.join(fdir,'xboot')
        fl = open(fname,'w')
        for k in range(0,self.xboot.shape[0]):
            fl.write(str(self.xboot[k,0])+','+
                     str(self.xboot[k,1])+'\n')
        fl.close()
        
        # write location uncertainty
        fname = os.path.join(fdir,'zboot')
        fl = open(fname,'w')
        for k in range(0,self.xboot.shape[0]):
            fl.write(str(self.zboot[k,0])+','+
                     str(self.zboot[k,1])+'\n')
        fl.close()


    def readtshfs(self,lbstack=None,itn=None):
        """
         write stacked templates and their indices
        :param   lbstack:   a label for this stack
        :param       itn:   iteration number (default: maximum)
        """
        
        if lbstack is None:
            lbstack = self.wgttype
        
        # directory to write to
        fname = 'Family'+str(self.fnum)+lbstack
        fdir = os.path.join(os.environ['TREMORAREA'],
                            'LOCATIONS',fname)

        # use maximum iteration number
        if itn is None:
            fls=glob.glob(fdir+'_itn*')
            itn=np.array([int(fl.split('_itn')[-1]) for fl in fls])
            itn=np.max(itn)
        self.iterloc = itn

        # add iteration number
        fdir = fdir+'_itn'+str(itn)
        lbstack = lbstack+'_itn'+str(itn)

        # read time shifts
        fname = os.path.join(fdir,'tshfs')
        fl = open(fname,'r')
        self.tshfs={}
        for line in fl:
            # organize
            vls=line.split(',')
            ky=vls[0]
            vls=vls[1:]
            vls=np.array(vls).astype(float)

            # add to set
            self.tshfs[ky]=vls
        fl.close()

        # read time uncertainty
        fname = os.path.join(fdir,'tshfu')
        fl = open(fname,'r')
        self.tshfu={}
        for line in fl:
            # organize
            vls=line.split(',')
            ky=vls[0]
            vls=vls[1:]
            vls=np.array(vls).astype(float)

            # add to set
            self.tshfu[ky]=vls
        fl.close()

        # read velocity model used for locations
        fname = os.path.join(fdir,'velocitymodel')
        fl = open(fname,'r')
        self.vmod = fl.readline().strip()
        fl.close()

        # read locations
        fname = os.path.join(fdir,'xzbest')
        vls = np.loadtxt(fname,delimiter=',',dtype=float)
        self.xzbest=vls

        # read location uncertainty
        fname = os.path.join(fdir,'xboot')
        vls = np.loadtxt(fname,delimiter=',',dtype=float)
        self.xboot=vls

        # read location uncertainty
        fname = os.path.join(fdir,'zboot')
        vls = np.loadtxt(fname,delimiter=',',dtype=float)
        self.zboot=vls

        # fill time shifts in case there are more stations
        imiss = [not tr.get_id() in self.tshfs.keys()
                 for tr in self.sta]
        if np.sum(imiss):
            self.tshfs,self.tshfu = \
                filltshf(self.xzbest,self.tshfs,
                         self.sta,self.eqloc,
                         self.xboot,self.zboot,
                         vmod=self.vmod)

#----------GRID OF X-C--------------------------------------

    def xcgrid(self,ids=None,usetshf=False):

        # staion ids
        if ids is None:
            ids=np.array([tr.get_id() for tr in self.stsv])
        
        # also remove the median times to avoid a gradual shift
        tmn=np.median(np.median(np.vstack(self.tshfs.values()),axis=0))
        for ky in self.tshfs.keys():
            self.tshfs[ky] = self.tshfs[ky]-tmn

        if usetshf:
            # just the events of interest
            tshfs = self.tshfs.copy()
            for ky in tshfs.keys():
                tshfs[ky] = tshfs[ky][self.isv]
        else:
            # set everything to zeros
            tshfs = dict([(ky,np.zeros(self.isv.size,dtype=float))
                          for ky in ids])

        # stack
        self.xcg=xcgrid(self.stsv,self.ilmsv,tget=self.tget,tlm=self.tlmxc,
                        tshfs=tshfs,ids=ids,flm=self.flmxc)

        # average
        self.xcga=np.ma.mean(self.xcg,axis=2)

            
#----------COHERENCE COMPUTATION-----------------------------

    def calcxc(self):
        # get the whole stack
        self.cpnorm,self.ennorm,self.cpnormstd,self.nsnorm,\
            self.nsnormstd,self.ennnorm,self.ennnormstd,self.xci = \
            xcwstack(self.stam,self.tcomp,rnoise=self.rnoise,
                     csyn=self.csyn,trange=self.trange,
                     realign=self.realign)

    def writexc(self,lbxc=''):
        # write the coherence values to a directory

        lbl='_T'+str(self.trange[0])+'_'+str(self.trange[1])
        if self.realign:
            lbl=lbl+'_realigned'
        else:
            lbl=lbl+'_unshifted'
        lbl=lbl+'_SNR'+str(self.minsnr)
        lbl=lbl+'_'+str(self.blim[0])+'-'+str(self.blim[1])
        lbl=lbl.replace('.','p')

        # directory to write to
        fdir = 'Family'+str(self.fnum)+lbl+lbxc
        fdir = os.path.join(os.environ['TREMORAREA'],
                            'SavedXC',fdir)
        if not os.path.exists(fdir):
            os.makedirs(fdir)

        # values of interest
        tosave=['cpnorm','ennorm','cpnormstd','nsnorm',
                'nsnormstd','ennnorm','ennnormstd','xci']

        # write to files
        for vr in tosave:
            fname = os.path.join(fdir,vr)
            with open(fname,'w') as fl:
                pickle.dump(getattr(self,vr),fl)

        # write comparison times
        fname = os.path.join(fdir,'comptimes')
        fl = open(fname,'w')
        for tm in self.tcomp:
            fl.write(tm.strftime("%Y %m %d %H %M %S %f\n"))
        fl.close()

    def readxc(self,lbxc=''):

        lbl='_T'+str(self.trange[0])+'_'+str(self.trange[1])
        if self.realign:
            lbl=lbl+'_realigned'
        else:
            lbl=lbl+'_unshifted'
        lbl=lbl+'_SNR'+str(self.minsnr)
        lbl=lbl+'_'+str(self.blim[0])+'-'+str(self.blim[1])
        lbl=lbl.replace('.','p')

        # directory to write to
        fdir = 'Family'+str(self.fnum)+lbl+lbxc
        fdir = os.path.join(os.environ['TREMORAREA'],
                            'SavedXC',fdir)

        # values of interest
        tosave=['cpnorm','ennorm','cpnormstd','nsnorm',
                'nsnormstd','ennnorm','ennnormstd','xci']

        # write to files
        for vr in tosave:
            fname = os.path.join(fdir,vr)
            print(fname)
            with open(fname,'r') as fl:
                setattr(self,vr,pickle.load(fl))

        # read comparison times
        fname = os.path.join(fdir,'comptimes')
        vls = np.loadtxt(fname,dtype=int)
        self.tcomp=[obspy.UTCDateTime(vls[k,0],vls[k,1],vls[k,2],
                                      vls[k,3],vls[k,4],vls[k,5]) + 
                   float(vls[k,6])/1.e6 for k in range(0,vls.shape[0])]
        self.tcomp=np.array(self.tcomp)
        self.icomp=eventindex(self.tms,self.tcomp)

            
#----------COHERENCE COMPUTATION-----------------------------

                    
def xcwstack(sta,tms,rnoise=False,cwgts=None,csyn=None,
             trange=None,realign=False):
    """
    :param      sta:  the stack of LFEs
    :param      tms:  the times to consider
    :param   rnoise:  replace all the data with random noise 
                         (default: false)
    :param    cwgts:  to construct a new set of components
    :param     csyn:  a synthetic coherent fraction to use
                         (default: None---use actual data)
    :param   trange:  time range
    :param  realign:  realign with cross-correlation
    :return      xc:  all the normalized cross-correlations
    :return     xcu:  the x-c's standard deviations
    :return     xci:  the cross-correlation function with frequencies, etc
    :return     xcv:  realization of values
    """

    # family number
    try:
        fnum = sta[0].stats.t9
    except:
        fnum=0

    # only read the relevant stations
    stns = [tr.stats.network+'.'+tr.stats.station+'.'+tr.stats.channel
            for tr in sta]
    stns = np.unique(np.array(stns))

    # copy and bandpass filter
    sta=sta.copy()
    msk = seisproc.prepfiltmask(sta,tmask=3.)
    sta.filter('bandpass',freqmin=1.,freqmax=30.)
    seisproc.addfiltmask(sta,msk)

    zeroafter=False
    if zeroafter:
        tlast=trange[1]-0.3
        for tr in sta:
            # data
            data = tr.data
            tm = tr.times()-tr.stats.t3

            # delete later
            data[tm>=tlast]=0.

            # taper
            ii=np.sum(np.logical_and(tm>tlast-0.5,tm<tlast))
            tri=tr.copy()
            tri.data = np.ones(2*ii,dtype=float)
            tri.taper(0.5,type='hann')
            tri.data = tri.data[ii:]

            ix = np.argmin(np.abs(tm-(tlast-0.5)))+1
            ix = np.arange(0,ii)+ix
            tr.data[ix]=np.multiply(tr.data[ix],tri.data)
        
    # if needed to realign
    if realign:
        star = sta.copy()
        msk = seisproc.prepfiltmask(star,tmask=3.)
        star.filter('bandpass',freqmin=2.,freqmax=8.)
        seisproc.addfiltmask(star,msk)
            
    # # resample if necessary
    # samprate=100.
    # for tr in sta:
    #     tr.interpolate(samprate,method='linear')

    # save coherence, standard deviation
    cpnorm,ennorm,cpnormstd=[],[],[]
    nsnorm,nsnormstd,ennnorm,ennnormstd=[],[],[],[]

    # initialize with response corrections
    invi = obspy.Inventory([],'T','T')
    invi = responsecorr.readresponse(sta,invi)

    for tm in tms:
        print(tm)

        # grab the data
        st,invi = grablfedata(tm,stns=stns,invi=invi,remresp=1)

        # alternative components
        if cwgts:
            st = maxcomp(st,cwgts)
            
        # replace with Gaussian noise
        if rnoise:
            for tr in st:
                tr.data=np.random.randn(tr.data.shape[0])
        elif csyn is not None:
            print('Synthetics: '+str(csyn))
            # add the pick data
            for tr in st:
                tr.stats.t1=tm-tr.stats.starttime
            readpicks(fnum=fnum,sta=st,pk='t3',pkref='t1')
            
            csyn = np.atleast_1d(csyn)
            if csyn.size==1:
                csyn=np.append(csyn,100.)

            # filter 
            msk = seisproc.prepfiltmask(st,tmask=3.)
            st.filter('bandpass',freqmin=1,freqmax=30.)
            seisproc.addfiltmask(st,msk)

            # add synthetic signals
            csynshf = 15.

            #for tr in st:
            #tr.data=np.zeros(tr.stats.npts,dtype=float)
            st = makesyndata(sta,st,csyn=csyn[0],tst=-0.3,flm=0.,
                             pk='t3',tlen=0.5,tshf=csynshf,amp=csyn[1],
                             stfl=0.15,tscl=trange)


        if st:
            print('Calculating x-c')
            
            if csyn is None:
                # filter 
                msk = seisproc.prepfiltmask(st,tmask=3.)
                st.filter('bandpass',freqmin=1,freqmax=30.)
                seisproc.addfiltmask(st,msk)

            # add the pick data
            for tr in st:
                tr.stats.t1=tm-tr.stats.starttime
            readpicks(fnum=fnum,sta=st,pk='t3',pkref='t1')
            if csyn is not None:
                for tr in st:
                    tr.stats.t3=tr.stats.t3-csynshf

            # if needed to realign
            if realign:
                # filter to limits of interest
                stre = st.copy()
                msk = seisproc.prepfiltmask(stre,tmask=3.)
                stre.filter('bandpass',freqmin=2.,freqmax=8.)
                seisproc.addfiltmask(stre,msk)

                # new alignment
                realignwv(star,stre,pk='t3')

                # reassign picks to original traces
                for tr in st:
                    tr2=stre.select(station=tr.stats.station,
                                    network=tr.stats.network,
                                    channel=tr.stats.channel)[0]
                    tr.stats.t3=tr2.stats.t3

            # coherence calculation
            xci = calcxc([sta,st],cm='*',sratmin=0.,xcmin=0.,
                         trange=trange)

            # only save if there are enough stations
            if len(xci.igrp)>2:
                # save
                cpnorm.append(xci.cpnorm)
                ennorm.append(xci.ennorm)
                cpnormstd.append(xci.cpnormstd)

                nsnorm.append(xci.nsnorm)
                nsnormstd.append(xci.nsnormstd)

                ennnorm.append(xci.ennnorm)
                ennnormstd.append(xci.ennnormstd)

    # set as numpy arrays
    cpnorm = np.vstack(cpnorm)
    ennorm = np.vstack(ennorm)
    cpnormstd = np.vstack(cpnormstd)
    nsnorm = np.vstack(nsnorm)
    nsnormstd = np.vstack(nsnormstd)
    ennnorm = np.vstack(ennnorm)
    ennnormstd = np.vstack(ennnormstd)

    return cpnorm,ennorm,cpnormstd,nsnorm,nsnormstd,ennnorm,ennnormstd,xci


#----------GETTING ALL THE DATA------------------------------


def readmanylfes(lbl):
    """
    :param     lbl:  name of directory to write to
    :return   stsv:  a list containing LFE waveforms
    :return    tsv:  times of the LFEs considered
    :return  stsvn:  a list of lists containing LFE waveforms at shifted times
    :return  ilmsv:  limiting indices for each event
    :return   tget:  time range saved
    """

    # directory to write to
    fdir = os.path.join(os.environ['TREMORAREA'],
                        'SavedLFEs',lbl)
    
    # saved events
    tsv = np.loadtxt(os.path.join(fdir,'times'),dtype=float)
    tsv = np.array([obspy.UTCDateTime(tm) for tm in tsv])

    # timing and indices
    fname = os.path.join(fdir,'ilimits')

    # timing
    fl=open(fname,'r')
    tget=fl.readline()
    fl.close()
    tget=tget.strip().split(',')
    tget=np.array(tget).astype(float)

    # indices
    ilmsv = np.loadtxt(fname,skiprows=1,dtype=int)

    # arrival times
    fname = os.path.join(fdir,'arrivals')
    artim = {}
    fl = open(fname,'r')
    for line in fl:
        vls = line.split(',')
        artim[vls[0]] = float(vls[1])
    fl.close()

    # original data
    stsv = obspy.read(os.path.join(fdir,'original','*.SAC'))
    for tr in stsv:
        tr.data=np.ma.masked_array(tr.data,mask=(tr.data==-12345.))

        # add the reference time used
        tr.stats.t7 = artim[tr.get_id()]

    # initialize
    nsi = glob.glob(os.path.join(fdir,'noise*'))
    stsvn=[]
    for fnm in nsi:
        st = obspy.read(os.path.join(fnm,'*.SAC'))
        for tr in st:
            tr.data=np.ma.masked_array(tr.data,mask=(tr.data==-12345.))

            # add the reference time used
            tr.stats.t7 = artim[tr.get_id()]

        stsvn.append(st)

    return stsv,tsv,stsvn,ilmsv,tget
    

def writemanylfes(lbl,stsv,tsv,stsvn,ilmsv,tget):
    """
    :param     lbl:  name of directory to write to
    :param    stsv:  a list containing LFE waveforms
    :param     tsv:  times of the LFEs considered
    :param   stsvn:  a list of lists containing LFE waveforms at shifted times
    :param   ilmsv:  limiting indices of values for each event
    :param    tget:  time range for each event relative to pick
    """

    # directory to write to
    fdir = os.path.join(os.environ['TREMORAREA'],
                        'SavedLFEs',lbl)
    print(fdir)
    if os.path.exists(fdir):
        shutil.rmtree(fdir)
    os.makedirs(fdir)

    # arrival times
    fname = os.path.join(fdir,'arrivals')
    fl = open(fname,'w')
    for tr in stsv:
        fl.write(tr.get_id()+','+str(tr.stats.t7)+'\n')
    fl.close()

    # times
    fname = os.path.join(fdir,'times')
    fl = open(fname,'w')
    for k in range(0,len(tsv)):
        nm = str(tsv[k].timestamp)
        fl.write(nm+'\n')
    fl.close()

    # indices
    fname = os.path.join(fdir,'ilimits')
    fl = open(fname,'w')
    fl.write(str(tget[0])+','+str(tget[1])+'\n')
    for ix in ilmsv:
        fl.write(str(ix)+'\n')
    fl.close()
    
    # actual data
    seisproc.copytosacheader(stsv)
    fdirj = os.path.join(fdir,'original')
    os.makedirs(fdirj)

    for tr in stsv.copy():
        if isinstance(tr.data,np.ma.masked_array):
            msk=tr.data.mask
            tr.data=tr.data.data
            tr.data[msk]=-12345
        
        # write each one
        fname=waveformdb.sacfilename(tr)
        fname=os.path.join(fdirj,fname)
        tr.write(fname,'SAC')

    # each noise series
    for m in range(0,len(stsvn)):
        st = stsvn[m]
        seisproc.copytosacheader(st)

        fdirj = os.path.join(fdir,'noise'+str(m))
        os.makedirs(fdirj)

        for tr in st.copy():
            if isinstance(tr.data,np.ma.masked_array):
                msk=tr.data.mask
                tr.data=tr.data.data
                tr.data[msk]=-12345

            # write each one
            fname=seisproc.sacfilename(tr)
            fname=os.path.join(fdirj,fname)
            tr.write(fname,'SAC')


#----------------LFE INFO-------------------------------

def egflist(justhave=True):
    """
    :param  justhave: just the ones in my database
    :return      ids: event ids of earthquakes Amanda used as EGFs
    :return      tms: dictionary of times
    """

    fname=os.path.join(os.environ['DATA'],'TREMORAREA',
                       'Amanda','EGFlist')
    ids=np.loadtxt(fname,dtype=int)

    eqs=pksdrops.pickeqs(10)
    q=eqs.filter(Event.evid.in_(ids.astype(float)))
    idh=np.array([eq.evid for eq in q])

    if justhave:
        ids=idh
    ids=ids.astype(int)

    tms=dict([(idi,float('nan')) for idi in ids])
    for eq in q:
        tms[int(eq.evid)]=obspy.UTCDateTime(eq.time)

    return ids,tms

def lfelocskml():
    """
    create a kml file with the lfe locations
    """

    # preferred indices
    fnums = bestfnum()

    # initialize kml info
    import simplekml
    kml = simplekml.Kml()

    # add the LFE locations
    for fnum in fnums:
        kml.newpoint(name=str(fnum),
                     coords=[lfeloc(fnum)[0:2]])

    # write to file
    fname=os.path.join(os.environ['WRITTEN'],'TremorArea',
                       'LFElocs.kml')
    kml.save(fname)


def lfelocs():
    """
    :param        locs:  LFE locations dictionary
    """

    # read all the info
    fname = os.path.join(os.environ['TREMORAREA'],'lfe_times',
                         'lfelocs')
    if not os.path.exists(fname):
        fname = os.path.join(os.environ['DATA'],'TREMORAREA','lfe_times',
                             'lfelocs')

    vls = np.loadtxt(fname,dtype=float,delimiter=',')

    # indices and locations
    fnums = vls[:,0].astype(int)
    loc = vls[:,1:]

    # create dictionary
    locs=dict((fnums[k],loc[k,:]) for k in range(0,len(fnums)))

    return locs

def lfeloc(fnum):
    """
    :param        fnum:  family number
    :param         loc:  LFE location
    """

    # all the events
    locs = lfelocs()

    # for this family
    loc = locs.get(fnum,np.array([0,0,0],dtype=float)*float('nan'))

    return loc

def biglfetimes(fnum=37102):
    """
    :param     fnum:   family number
                    best are 37102 and 37140w
    :return     tms:   event times
    """

    if fnum==37102:
        tms=[obspy.UTCDateTime(2010,10,31,9,35,31),
             obspy.UTCDateTime(2010,12,6,11,15,34),
             obspy.UTCDateTime(2011,1,28,6,33,11),
             obspy.UTCDateTime(2011,2,3,2,0,0),
             obspy.UTCDateTime(2011,2,8,0,37,32),
             obspy.UTCDateTime(2011,4,3,10,8,20)]
    tms = np.array(tms)

    # for more precise values, grab from full list
    itms,xcvl = lfetimes(fnum=37102)
    ii = np.array([np.argmin(np.abs(itms-tm)) for tm in tms])
    tms,xcvl = itms[ii],xcvl[ii]
    

    return tms,xcvl


def lfetimes(fnum=37102):
    """
    :param     fnum:   family number
                    best are 37102 and 37140w
    :return     tms:   event times
    """

    fdir2 = os.path.join(os.environ['DATA'],'TREMORAREA','lfe_times')
    fdir = os.path.join(os.environ['TREMORAREA'],'lfe_times')
    fname = str(fnum) + '.200.post2014'
    fname = str(fnum) + '.400'
    if os.path.exists(os.path.join(fdir,fname)):
        fname = os.path.join(fdir,fname)
    else:
        fname = os.path.join(fdir2,fname)


    # read from file
    vls = np.loadtxt(fname,dtype=float)

    # the average x-c value (8) or median (9)
    xcvl = vls[:,8]
    
    # arrange
    sc = vls[:,6]
    iget = np.array([0,1,2,4,5])
    vls = vls[:,iget].astype(int)
    
    
    tms = []
    for k in range(0,vls.shape[0]):
        tms.append(obspy.UTCDateTime(vls[k,0],vls[k,1],vls[k,2],
                                     vls[k,3],vls[k,4])+sc[k])
    tms = np.array(tms)


    return tms,xcvl

def eventindex(tms,tstack):
    """
    :param        tms:  times of all events
    :param     tstack:  times of events of interest
    :param         ix:  closest indices of nearby events
    """
    
    # to linear indices
    tref = obspy.UTCDateTime(2000,1,1)
    tmsi = tms - tref
    tstacki = tstack -tref

    # closest indices
    ix = general.closest(tmsi,tstacki)

    return ix



def readpicks(fnum=None,sta=None,pk='t3',pkref='t1'):
    """
    :param     fnum:  family number (default: from t9)
    :param      sta:  waveforms, with picks
    :param       pk:  pick to write
    :param    pkref:  pick with reference
    :return    dpks:  dictionary with picks
    """

    if fnum is None:
        fnum = sta[0].stats.t9

    # files of interest
    fdir=os.path.join(os.environ['TREMORAREA'],
                      'familypicks')
    fname='picks_'+str(fnum)
    fname=os.path.join(fdir,fname)

    if os.path.exists(fname):
        # read
        vls = np.loadtxt(fname,dtype=bytes,delimiter=',').astype(str)
        
        stn=vls[:,0]
        pks=vls[:,1].astype(float)
        
        # place in dictionary
        dpks = {}
        for k in range(0,len(stn)):
            dpks[stn[k]] = pks[k]
    else:
        # just an empty dictionary
        dpks = {}



    # set picks
    if sta:
        # get the station locations
        sti = initstacks()
        lon = dict((tr.stats.network+'.'+tr.stats.station,
                    tr.stats.sac.stlo) for tr in sti)
        lat = dict((tr.stats.network+'.'+tr.stats.station,
                    tr.stats.sac.stla) for tr in sti)

        for tr in sta:
            if tr.stats.network+'.'+tr.stats.station in dpks.keys():
                tr.stats[pk]=tr.stats[pkref]+\
                    dpks.get(tr.stats.network+'.'+tr.stats.station,0.)
            else:
                # get approximate distance from event location
                loc = lfeloc(fnum)
                rlon = math.cos(loc[1]*math.pi/180.)
                try:
                    dst=np.array([(loc[0]-tr.stats.sac.stlo)*rlon,
                                  loc[1]-tr.stats.sac.stla])
                except:
                    lbl=tr.stats.network+'.'+tr.stats.station
                    dst=np.array([(loc[0]-lon[lbl])*rlon,
                                  loc[1]-lat[lbl]])

                dst=dst*111.
                dst=np.append(dst,loc[2])
                dst=np.sum(np.power(dst,2))**0.5
                tim = dst/4.
                tr.stats[pk]=tr.stats[pkref]+tim
                
    return dpks


def plotdursep(ps):

    plt.close()
    f = plt.figure(figsize=(8,3))
    gs,p=gridspec.GridSpec(1,1),[]
    gs.update(left=0.1,right=0.97)
    gs.update(bottom=0.2,top=0.92)
    gs.update(hspace=0.05,wspace=0.25)
    for k in range(0,1):
        p.append(plt.subplot(gs[k]))
    p=np.array(p)
    pm=p.reshape([p.size,1])

    fs = 17

    ii=[0,2]
    cols=graphical.colors(len(ii))
    idi='BP.CCRB..DP3'
    idi='PB.B073..EH2'
    h=[]
    for k in range(0,len(ii)):
        kk=ii[k]

        # plot synthetic
        tr=ps.durtemp[kk].select(id=idi)[0]
        hh,=p[0].plot(tr.times()-tr.stats.t3,tr.data/np.max(np.abs(tr.data)),
                      color=cols[k],linestyle=':')
        h.append(hh)

        # plot stack of assigned values
        tr=ps.stackduras[kk].select(id=idi)[0]
        hh,=p[0].plot(tr.times()-tr.stats.t3,tr.data/np.max(np.abs(tr.data)),
                      color=cols[k],linestyle='--')
        h.append(hh)

        # plot stack of identified values
        tr=ps.stackdurmx[kk].select(id=idi)[0]
        hh,=p[0].plot(tr.times()-tr.stats.t3,tr.data/np.max(np.abs(tr.data)),
                      color=cols[k],linestyle='-')
        h.append(hh)

    p[0].set_xlim([-0.5,3])
    p[0].set_ylim([-1.05,1.05])
    p[0].set_ylabel('ground velocity at '+tr.stats.station+'   ',fontsize=fs)
    p[0].set_yticks([-1.,0,1])
    p[0].set_xticks(np.arange(0,5))
    p[0].set_xticklabels(['0','1','','',''])
    p[0].set_xlabel('time since S arrival (s)',fontsize=fs)
    h=np.array(h).reshape([len(ii),3])
    lbls=['{:g}'.format(ps.ndurs[k]) for k in ii]
    lg = p[0].legend(h[:,-1],lbls,loc='upper right',fontsize=fs)
    ps=p[0].get_position()
    lg.set_bbox_to_anchor((ps.x1+0.02,ps.y1+0.1),transform=plt.gcf().transFigure)    
    lg = p[0].add_artist(lg)
    lg.set_title('LFE duration (s)')
    plt.setp(lg.get_title(),fontsize=fs)

    for ph in p:
        ph.yaxis.set_tick_params(labelsize=fs)
        ph.xaxis.set_tick_params(labelsize=fs)

    lbls=['synthetics','stack, assigned groups','stack, identified groups']
    lg2 = p[0].legend(h[0,:],lbls,loc='lower right',fontsize=fs)
    lg2.set_bbox_to_anchor((ps.x1+0.02,ps.y0-0.15),transform=plt.gcf().transFigure)    
    graphical.printfigure('VLdurtemp',f)


def readppicks(fnum=None,sta=None,pk='t2',pkref='t1'):
    """
    :param     fnum:  family number (default: from t9)
    :param      sta:  waveforms, with picks
    :param       pk:  pick to write
    :param    pkref:  pick with reference
    :return    dpks:  dictionary with picks
    """

    if fnum is None:
        fnum = sta[0].stats.t9

    # files of interest
    fdir=os.path.join(os.environ['TREMORAREA'],
                      'familypicks')
    fname='Ppicks_'+str(fnum)
    fname=os.path.join(fdir,fname)

    if os.path.exists(fname):
        # read
        vls = np.loadtxt(fname,dtype=bytes,delimiter=',').astype(str)
        
        stn=vls[:,0]
        pks=vls[:,1].astype(float)
        
        # place in dictionary
        dpks = {}
        for k in range(0,len(stn)):
            dpks[stn[k]] = pks[k]
    else:
        # just an empty dictionary
        dpks = {}



    # set picks
    if sta:
        # get the station locations
        sti = initstacks()
        lon = dict((tr.stats.network+'.'+tr.stats.station,
                    tr.stats.sac.stlo) for tr in sti)
        lat = dict((tr.stats.network+'.'+tr.stats.station,
                    tr.stats.sac.stla) for tr in sti)

        for tr in sta:
            if tr.stats.network+'.'+tr.stats.station in dpks.keys():
                tr.stats[pk]=tr.stats[pkref]+\
                    dpks.get(tr.stats.network+'.'+tr.stats.station,0.)
            else:
                # get approximate distance from event location
                loc = lfeloc(fnum)
                rlon = math.cos(loc[1]*math.pi/180.)
                try:
                    dst=np.array([(loc[0]-tr.stats.sac.stlo)*rlon,
                                  loc[1]-tr.stats.sac.stla])
                except:
                    lbl=tr.stats.network+'.'+tr.stats.station
                    dst=np.array([(loc[0]-lon[lbl])*rlon,
                                  loc[1]-lat[lbl]])

                dst=dst*111.
                dst=np.append(dst,loc[2])
                dst=np.sum(np.power(dst,2))**0.5
                tim = dst/6.
                tr.stats[pk]=tr.stats[pkref]+tim

    return dpks

def addrandshifts(eqloc,sta,hhwd=0.5,vhwd=None,Nev=100):
    """
    :param         eqloc: LFE location
    :param           sta: waveforms
    :param          hhwd: half-width, horizontally
    :param          vhwd: half-width, vertically (default: hhwd)
    """

    if vhwd is None:
        vhwd = hhwd

    # how much to shift each
    xsh=np.random.randn(Nev)*hhwd
    zsh=np.random.randn(Nev)*vhwd
    xlm=general.minmax(xsh)
    zlm=general.minmax(zsh)

    # compute travel times on a grid
    ttrav,x,z,ids=calcttrav(sta,eqloc,xlm=xlm,zlm=zlm)

    # make  dictionary of shifts
    shfs=dict.fromkeys(ids)
    
    # interpolate to points of interest
    for k in range(0,len(ids)):
        f=scipy.interpolate.RectBivariateSpline(x,z,ttrav[:,:,k].T)
        shfs[ids[k]]=np.array([f(xsh[n],zsh[n])[0,0] for n in range(0,Nev)])

    # subtract a median
    submed=True
    if submed:
        mds = np.vstack(list(shfs.values()))
        mds = np.median(mds,axis=0)
        for idi in ids:
            shfs[idi]=shfs[idi]-mds

    return shfs,xsh,zsh

def detectionsfilename(fnum,xvli,zvli,flm,odur,ndur):
    """
    :param        fnum: family number
    :param        xvli: x location
    :param        zvli: z location
    :param         flm: frequency limit for detection
    :param        odur: assumed original duration
    :param        ndur: assumed new duration
    :return      fname: an initial file name
    """
                     
    fname='{:d}'.format(fnum)

    # location shift
    fname=fname+'_x{:0.2f}'.format(xvli)
    fname=fname+'_z{:0.2f}'.format(zvli)

    # frequency band
    fname=fname+'_f{:0.2f}-{:0.2f}'.format(flm[0],flm[1])

    # template duration
    fname=fname+'_dur{:0.2f}-{:0.2f}'.format(odur,ndur)

    return fname


def calcspectrav(xi,yi,zi,Ni=6,usevmod=True,vmod='iasp91',phsarv=['S','s']):
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
            fdir=os.path.join(os.environ['DATA'],'VELMODELS',vmod)
            fname=os.path.join(fdir,vmod+'.npz')
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
    

def calcttrav(sta,eqloc,vmod='iasp91',phsarv=['S','s'],xlm=[-.5,.5],zlm=[-1.,1.],
              x=None,z=None):
    """
    :param      sta: waveforms, including locations
    :param    eqloc: earthquake locations
    :param     vmod: velocity model to use (default: 'iasp91')
    :param   phsarv: which phases to use (default: ['S','s'])
    :param      xlm: x limits
    :param      zlm: z limits
    :param        x: x-values desired, if known
    :param        z: z-values desired, if known
    :return   ttrav: travel times
    :return      xz: x-z locations
    :return     ids: components indexed in columns
    """

    # a suite of locations in km
    xlm=np.atleast_1d(xlm).astype(float)
    zlm=np.atleast_1d(zlm).astype(float)
    ztox = np.diff(zlm)[0]/np.diff(xlm)[0]
    ztox = 1.5

    # time spacing to consider
    dtim=0.01
    
    # wavespeed in km/s
    spd = 3.

    # grid of locations
    if x is None:
        N = int(np.ceil(np.diff(xlm)[0]/spd/dtim))
        x = np.linspace(xlm[0],xlm[1],N+1)
    else:
        x=np.atleast_1d(x)
        xlm=general.minmax(x)
    if z is None:
        N = int(np.ceil(np.diff(zlm)[0]/spd/dtim/ztox))
        z = np.linspace(zlm[0],zlm[1],N+1)
    else:
        z = np.atleast_1d(z)
        zlm=general.minmax(z)

    # distances from the earthquake center in km
    xx,zz = np.meshgrid(x,z)
    xx,zz = xx.flatten(),zz.flatten()

    # make a grid of locations
    xz = np.vstack([xx,zz])
    xz = xz.transpose()

    # calculate time shifts
    ttravh,trefs = xztotshfs(xz,sta,eqloc,vmod=vmod,phsarv=phsarv)

    # place in grid
    ids = np.array([tr.get_id() for tr in sta])
    ttrav = np.zeros([xz.shape[0],len(sta)])
    for k in range(0,len(sta)):
        ttrav[:,k] = ttravh[ids[k]]

    ttrav=ttrav.reshape([z.size,x.size,len(sta)])

    return ttrav,x,z,ids


def xztotshfs(xz,sta,eqloc,vmod='iasp91',phsarv=['S','s']):
    """
    :param       xz: set of locations along the fault
                      [horizontal offset, depth]
    :param      sta: waveforms, including locations
    :param    eqloc: earthquake locations
    :param     vmod: velocity model to use (default: 'iasp91')
    :param   phsarv: which phases to use (default: ['S','s'])
    :return   tshfs: a dictionary of travel time shifts for each station
    :return    tref: reference times
    """

    # number of estimates
    N = xz.shape[0]

    # initialize values
    tshfs = dict((tr.get_id(),np.zeros(N,dtype=float)) for
                 tr in sta)

    # get the station locations
    sloc = np.zeros([len(sta),2],dtype=float)
    ids = np.array([tr.get_id() for tr in sta])
    for k in range(0,len(sta)):
        tr = sta.select(id=ids[k])[0]
        sloc[k,0]=tr.stats.sac.stlo
        sloc[k,1]=tr.stats.sac.stla
        ids[k]=tr.get_id()

    # subtract earthquake location
    sloc[:,0]=sloc[:,0]-eqloc[0]
    sloc[:,1]=sloc[:,1]-eqloc[1]

    # scale longitudes
    lscl = math.cos(math.pi/180.*eqloc[1])
    sloc[:,0]=sloc[:,0]*lscl

    # and to km
    sloc = sloc * 111.

    # change to E-W-Z
    xx,zz=xz[:,0],xz[:,1]
    stk = math.pi/180*135.
    xi = xx*math.sin(stk)
    yi = xx*math.cos(stk)
    zi = zz.copy()

    xi,yi,zi=np.append(xi,0.),np.append(yi,0.),np.append(zi,0.)

    # finally to distances
    sloc = sloc.transpose()
    xi = sloc[0:1,:] - xi.reshape([xi.size,1])
    yi = sloc[1:2,:] - yi.reshape([yi.size,1])
    zi = eqloc[2]+np.repeat(zi.reshape([zi.size,1]),
                            sloc.shape[1],axis=1)
    
    # travel times
    ttrav=calcspectrav(xi,yi,zi,Ni=7,usevmod=True,vmod=vmod,phsarv=phsarv)

    # reference times
    tref = ttrav[-1,:]
    tref = np.atleast_1d(tref)

    # subtract the reference times
    ttrav=ttrav[:-1,:]
    ttrav=ttrav.reshape([N,tref.size])
    ttrav = ttrav - tref.reshape([1,tref.size])

    # save results
    for k in range(0,tref.size):
        tshfs[ids[k]] = ttrav[:,k]
    
    return tshfs,tref

    

#-----------INITIALIZE THE STATIONS AND DATA TO STACK-----------------------------


def allstations():
    """
    :return   ids: a list of relevant waveform ids
    """

    stpb = ['B072','B073','B075','B076','B078',\
            'B079','B900','B901']
    chpb = ['EH1','EH2','EHZ']
    stbp = ['CCRB','EADB','FROB','GHIB','JCSB',\
            'LCCB','MMNB','RMNB','SCYB','SMNB',\
            'VARB','VCAB','JCNB']
    chbp = ['DP1','DP2','DP3']

    ids = []
    for stn in stpb:
        for ch in chpb:
            ids.append('PB.'+stn+'..'+ch)
    for stn in stbp:
        for ch in chbp:
            ids.append('BP.'+stn+'..'+ch)

    return ids


def addstatloc(sta):
    """
    :param      sta:  traces to add locations for
    """

    fname=os.path.join(os.environ['TREMORAREA'],
                       'familypicks','statloc')
    if not os.path.exists(fname):
        fname=os.path.join(os.environ['DATA'],'TREMORAREA',
                           'familypicks','statloc')
    vls=np.loadtxt(fname,dtype=bytes,delimiter=',').astype(str)


    # stations
    stns = vls[:,0]
    for  k in range(0,len(stns)):
        nw,stn,trs,chn=stns[k].split('.')
        stns[k]=nw+'.'+stn

    # longitudes and latitudes
    lon = vls[:,1].astype(float)
    lat = vls[:,2].astype(float)

    lons = dict((stns[k],lon[k]) for k in range(0,len(stns)))
    lats = dict((stns[k],lat[k]) for k in range(0,len(stns)))

    for tr in sta:
        lbl=tr.stats.network+'.'+tr.stats.station
        try:
            tr.stats.sac['stlo']=lons[lbl]
            tr.stats.sac['stla']=lats[lbl]
        except:
            tr.stats.sac = {}
            tr.stats.sac['stlo']=lons[lbl]
            tr.stats.sac['stla']=lats[lbl]


def initstacks():
    """
    :return   stk:  a set of stacks with all values initialized to zero
    """

    # get all the stations
    ids = allstations()

    # specify a timing and time range
    trange = [-30,50]
    dtim = 1./100
    npts = (trange[1]-trange[0])/dtim
    npts = int(np.round(npts))
    tref = obspy.UTCDateTime(2000,1,1)

    # initialize a trace
    tr = obspy.Trace()
    tr.data = np.zeros(npts,dtype=float)
    tr.stats.delta = dtim
    tr.stats.starttime = tref+trange[0]
    tr.stats.t1 = tref - tr.stats.starttime
    
    # add a trace per observation
    stk = obspy.Stream()
    for idi in ids:
        nw,stn,lc,ch=idi.split('.')
        tri = tr.copy()
        tri.stats.network=nw
        tri.stats.station=stn
        tri.stats.location=lc
        tri.stats.channel=ch
        stk.append(tri)

    # add station location information
    addstatloc(stk)
    
    return stk


#------------STACK AND LOCATE---------------------------------------

def readstacks(fnum=37102,lbstack=''):
    """
    :param    fnum:  family numbers
    :return    sta:  stacked traces
    """

    # directory to read from to
    fnum = int(fnum)
    fname = 'Family'+'{:d}'.format(fnum)+lbstack
    fdir = os.path.join(os.environ['TREMORAREA'],'SavedStacks')
    if not os.path.exists(fdir):
        fdir = os.path.join(os.environ['DATA'],'TREMORAREA','SavedStacks')
    fdir=os.path.join(fdir,fname)

    # relevant files
    fls = glob.glob(os.path.join(fdir,'*SAC'))

    #print(fls)
    # initialize
    sta = obspy.Stream()

    # read each one
    for fl in fls:
        sti = obspy.read(fl)
        sta = sta + sti
    
    seisproc.copyfromsacheader(sta)
    sta = sta.merge()
    
    for tr in sta:
        # set the event time
        tr.stats.t1=obspy.UTCDateTime(2000,1,1)-tr.stats.starttime

        # and the family number
        tr.stats.t9=fnum

    # set the picks
    readpicks(fnum=fnum,sta=sta,pk='t3',pkref='t1')
    readppicks(fnum=fnum,sta=sta,pk='t2',pkref='t1')

    # also set the locations
    sti = initstacks()
    for tr in sta:
        tri=sti.select(station=tr.stats.station,network=tr.stats.network)
        tr.stats.sac.stlo=tri[0].stats.sac.stlo
        tr.stats.sac.stla=tri[0].stats.sac.stla

    return sta

def writecomplete(ps,lbstack=None):
    """
    # write stacked templates and their indices
    :param   lbstack:   a label for this stack
    """
    
    # the label
    if lbstack is None:
        lbstack = ''
        
    # directory to write to
    fdir = 'Family'+str(ps.fnum)
    fdir = os.path.join(os.environ['TREMORAREA'],
                        'SavedXC',fdir)
    if not os.path.exists(fdir):
        os.makedirs(fdir)

    # file name
    fname='xc'+lbstack
    fname=os.path.join(fdir,fname)

    with open(fname,'wb') as fl:
        pickle.dump(ps,fl)


def writestacks(sta,lbstack=''):
    """
    :param     sta:  waveforms, with family numbers saved as t9
    :param lbstack:  a label for this set of waveforms
    """

    # directory to write to
    fnum = int(sta[0].stats.t9)
    fdir = 'Family'+str(fnum)+lbstack
    fdir = os.path.join(os.environ['TREMORAREA'],
                        'SavedStacks',fdir)
    if not os.path.exists(fdir):
        os.makedirs(fdir)

    # remove sac files in that directory
    fls = glob.glob(os.path.join(fdir,'*.SAC'))
    for fl in fls:
        os.remove(fl)

    # exclude any masked portions
    #sta = sta.split()

    for tr in sta:
        if isinstance(tr.data,np.ma.masked_array):
            msk = tr.data.mask
            tr.data = tr.data.data
            tr.data[msk]=999999

    # get the sac headers right
    seisproc.copytosacheader(sta)

    for tr in sta:
        # write each one
        fname=waveformdb.sacfilename(tr)
        fname=os.path.join(fdir,fname)
        tr.write(fname,'SAC')


def xcgrid(stsv,ilmsv,tlm,tget,tshfs,ids=None,flm=[1.,5.]):
    """
    :param       stsv:   target waveforms, concatenated
    :param      ilmsv:   limits of the LFEs
    :param       tget:   time range included in the data
    :param        tlm:   time range for x-c
    :param      tshfs:   time shifts to use
    :param        ids:   stations / components to use
    :param        flm:   frequency limit
    """

    if ids is None:
        ids=np.array([tr.get_id() for tr in stsv])
    Ntim = len(ilmsv)-1
    Nstat = len(ids)

    # acceptable times
    iok = np.ones([Ntim,Nstat],dtype=bool)

    # make sure there are time shifts
    for k in range(0,len(ids)):
        if isinstance(tshfs[ids[k]],np.ma.masked_array):
            iok[:,k]=np.logical_and(iok[:,k],~tshfs[ids[k]].mask)

    # time limits to extract
    dtim = stsv[0].stats.delta
    tmin = np.min([np.min(tshfs[idi]) for idi in ids])
    tmax = np.max([np.max(tshfs[idi]) for idi in ids])
    tlm = np.array([tget[0]-tmin,tget[1]-tmax])
    tlm = tlm + np.array([1.,-1.])*dtim

    # default shifts
    izero = int(np.round(-tget[0]/dtim))
    izero = ilmsv[0:-1]+izero

    # and intervals relative to zero points
    sh1 = int(np.round(tlm[0]/dtim))
    sh2 = int(np.round(tlm[1]/dtim))
    shg = np.arange(sh1,sh2)

    # initialize
    xcg=np.ma.masked_array(np.ndarray([Ntim,Ntim,len(ids)]),mask=False)

    # initialize a set of values for PCA
    
    
    for k in range(0,1): #len(ids)):
        idi = ids[k]
        tr = stsv.select(id=idi)[0]
        wtot = 0.

        # copy the trace here to keep the header
        tri = stsv.select(id=idi)[0].copy()
        tri.data = np.zeros(len(shg),dtype=float)

        if np.sum(iok[:,k])>0:
            # add time shifts for this station
            ishf=tshfs[idi][iok[:,k]]/dtim
            ishf=np.round(ishf).astype(int)
            ishf = ishf+izero[iok[:,k]]
            
            # initialize
            vls = np.zeros(len(shg),dtype=float)
            ntot = np.zeros(len(shg),dtype=float)
            
            # all the points
            i1,ct=np.meshgrid(shg,np.arange(0,ishf.size))
            i1,i2=np.meshgrid(shg,ishf)
            ix=i1+i2
            ix,ct=ix.flatten(),ct.flatten()
            
            # check that there are no masked or nan values
            mlen=ishf.size
            nmsk=np.bincount(ct,weights=tr.data.mask[ix],minlength=mlen)
            vl=np.bincount(ct,weights=tr.data.data[ix],minlength=mlen)
            isok=np.logical_and(nmsk==0,~np.isnan(vl))
            isok=np.logical_and(isok,~np.isinf(vl))

            # # the maximum value
            # if 'bymax' in nmtype:
            #     nml=np.zeros(mlen,dtype=float)
            #     np.maximum.at(nml,ct,np.abs(tr.data[ix]))
            #     isok=np.logical_and(isok,nml>0)
            # else:
            #     nml=np.ones(mlen,dtype=float)
                
            # pick the relevant events
            ix=ix.reshape(i1.shape)[isok,:].flatten()
            ct=ct.reshape(i1.shape)[isok,:].flatten()
            i1=(i1[isok,:]-shg[0]).flatten()

            # create an array
            vlg = tr.data[ix].reshape([shg.size,np.sum(isok)])

            import code
            code.interact(local=locals())
            
            # normalization
            wgts = np.sum(np.power(vlg,2),axis=0)
            wgts = np.power(wgts,-0.5).reshape([1,np.sum(isok)])
            vlg = np.multiply(vlg,wgts)

            # x-c 
            vlg = np.dot(vlg.T,vlg)

            # save
            ii,=np.where(isok)
            for kk in range(0,ii.size):
                xcg.data[ii,ii[kk],k]=vlg[:,kk]
                xcg.mask[ii,ii[kk],k] = False

    return xcg
        
def stackmanylfes(stsv,ilmsv,tget,iok,tshfs,ids,wgt,nmtype='bymax',pweight=True):
    """
    :param       stsv:   target waveforms, concatenated
    :param      ilmsv:   limits of the LFEs
    :param       tget:   time range included in the data
    :param        iok:   a ntime x nstat grid of which values to use
    :param      tshfs:   time shifts to use
    :param        ids:   stations / components to use
    :param        wgt:   how much to weight each stack
    :param     nmtype:   normalization type (default: 'bymax')
    :return       sts:   stacked traces
    :return      stsp:   phase weighting
    """

    if ids is None:
        ids=np.array([tr.get_id() for tr in stsv])
    Ntim = len(ilmsv)-1
    Nstat = len(ids)

    if iok is None:
        iok = np.ones([Ntim,Nstat],dtype=bool)

    # make sure there are time shifts
    for k in range(0,len(ids)):
        if isinstance(tshfs[ids[k]],np.ma.masked_array):
            iok[:,k]=np.logical_and(iok[:,k],~tshfs[ids[k]].mask)

    # time limits to extract
    dtim = stsv[0].stats.delta
    tmin = np.min([np.min(tshfs[idi]) for idi in ids])
    tmax = np.max([np.max(tshfs[idi]) for idi in ids])
    tlm = np.array([tget[0]-tmin,tget[1]-tmax])
    tlm = tlm + np.array([1.,-1.])*dtim

    # default shifts
    izero = int(np.round(-tget[0]/dtim))
    izero = ilmsv[0:-1]+izero

    # and intervals relative to zero points
    sh1 = int(np.round(tlm[0]/dtim))
    sh2 = int(np.round(tlm[1]/dtim))
    shg = np.arange(sh1,sh2)

    # initalize
    sts = obspy.Stream()
    stsp = obspy.Stream()
    tref = obspy.UTCDateTime(2000,1,1)

    for k in range(0,len(ids)):
        idi = ids[k]
        tr = stsv.select(id=idi)[0]
        wtot = 0.

        if pweight:
            # compute the hilbert transform
            trp=tr.copy()
            msk=seisproc.prepfiltmask(trp,tmask=0.)
            trp.filter('bandpass',freqmin=2,freqmax=6.)
            trp.data=scipy.signal.hilbert(trp.data)
            trp.data=trp.data/np.abs(trp.data)
            seisproc.addfiltmask(trp,msk)
        
        # copy the trace here to keep the header
        tri = stsv.select(id=idi)[0].copy()
        tri.data = np.zeros(len(shg),dtype=float)

        if np.sum(iok[:,k])>0:
            # add time shifts for this station
            ishf=tshfs[idi][iok[:,k]]/dtim
            ishf=np.round(ishf).astype(int)
            ishf = ishf+izero[iok[:,k]]
            
            # initialize
            vls = np.zeros(len(shg),dtype=float)
            ntot = np.zeros(len(shg),dtype=float)
            
            # all the points
            i1,ct=np.meshgrid(shg,np.arange(0,ishf.size))
            i1,i2=np.meshgrid(shg,ishf)
            ix=i1+i2
            ix,ct=ix.flatten(),ct.flatten()
            
            # check that there are no masked or nan values
            mlen=ishf.size
            nmsk=np.bincount(ct,weights=tr.data.mask[ix],minlength=mlen)
            vl=np.bincount(ct,weights=tr.data.data[ix],minlength=mlen)
            isok=np.logical_and(nmsk==0,~np.isnan(vl))
            isok=np.logical_and(isok,~np.isinf(vl))
                            
            # the maximum value
            if 'bymax' in nmtype:
                nml=np.zeros(mlen,dtype=float)
                np.maximum.at(nml,ct,np.abs(tr.data[ix]))
                isok=np.logical_and(isok,nml>0)
            else:
                nml=np.ones(mlen,dtype=float)
                
            # pick the relevant events
            ix=ix.reshape(i1.shape)[isok,:].flatten()
            ct=ct.reshape(i1.shape)[isok,:].flatten()
            i1=(i1[isok,:]-shg[0]).flatten()
            
            # weights and normalizations
            wgti = np.divide(wgt[iok[:,k]],nml)
            
            # import code
            # code.interact(local=locals())

            # the whole summation
            vls=np.bincount(i1,weights=np.multiply(tr.data[ix],wgti[ct]),
                            minlength=shg[-1]-shg[0]+1)
            
            # and normalize
            vls=vls/np.sum(wgt[iok[:,k]][isok])

            # phase weighting
            if pweight:
                hsumr=np.bincount(i1,weights=np.real(trp.data[ix]),
                                 minlength=shg[-1]-shg[0]+1)
                hsumi=np.bincount(i1,weights=np.imag(trp.data[ix]),
                                 minlength=shg[-1]-shg[0]+1)
                hsum=np.abs(hsumr+1j*hsumi)
                nper=np.bincount(i1,minlength=shg[-1]-shg[0]+1)
                hsum=np.divide(hsum,nper)

                vls=np.multiply(vls,nper)


        else:
            # if there was no data to stack
            vls=np.zeros(shg[-1]-shg[0]+1,dtype=float)
            vls=np.ma.masked_array(vls,mask=True)
            
        # add this trace
        trj = tri.copy()
        trj.stats.starttime = tref + tlm[0] + tr.stats.t7
        nw,stn,trsh,chn=idi.split('.')
        trj.stats.network = nw
        trj.stats.channel = chn
        trj.stats.station = stn
        trj.data = vls #np.divide(vls,ntot)
        
        sts.append(trj)

        if pweight:
            trj=trj.copy()
            trj.data=hsum
            stsp.append(trj)
        
    return sts,stsp



def locmanylfes(sta,stsv,ilmsv,tget,blim,eqloc,wlen=[-.2,1.8],
                blimadd=None,vmod='iasp91',checkp=True,stap=None):
    """
    :param        sta:   template waveforms
    :param       stsv:   target waveforms, concatenated
    :param      ilmsv:   limits of the LFEs
    :param       tget:   time range used when grabbing the data
    :param       blim:   bandlimit of target data
    :param      eqloc:   earthquake location
    :param       wlen:   limits to use for cross-correlation
    :param    blimadd:   a second bandlimit filter to add
    :param       vmod:   velocity model to use (default: 'iasp91')
    :param     checkp:   also check the P arrivals
    :param       stap:   waveforms to use for P-wave picks
    :return    xzbest:   preferred x,z location
    :return     tshfs:   a dictionary of the time shifts
    :return         x:   the limiting x-locations from bootstrapping
    :return         z:   the limiting z-locations from bootstrapping
    :return     tshfu:   limiting time shifts from bootstrapping
    """

    blimadd = np.array([0.,10.])

    # calculate travel times
    ttrav,xz,trash = calcttrav(sta,eqloc,vmod=vmod,phsarv=['S','s'])

    # calculate travel times for the P waves if appropriate
    if not stap:
        checkp = False
    if checkp:
        # calculate travel times for P waves
        ttravp,xz,trash = calcttrav(stap,eqloc,vmod=vmod,phsarv=['P','p'])

        # reset the picks to the S index
        stap = stap.copy()
        for tr in stap:
            tr.stats.t3 = tr.stats.t2
            
        # combine the relvant P and S arrivals
        sta = sta.copy()+stap
        ttrav = np.append(ttrav,ttravp,axis=1)

    # also repeat to allow time shifts
    tlk = np.arange(-0.3,0.3,0.005)
    #tlk = np.arange(-1.,1.,0.05)
    tlk = np.arange(-0.,0.003,0.05)
    Nlk = tlk.size
    tlk = tlk.reshape([1,tlk.size])
    tlk = np.repeat(tlk,ttrav.shape[0],axis=0)
    tlk = tlk.reshape([tlk.size,1])
    ttrav = np.repeat(ttrav,Nlk,axis=0)
    xz = np.repeat(xz,Nlk,axis=0)
    ttrav = ttrav + tlk

    # standard deviation contributed per station
    stst = (wlen[1]-wlen[0])/sta[0].stats.delta
    stst = (1/(stst-3))**0.5
    stst = stst* 0.
    print('time weighting '+str(np.max(np.abs(stst))))

    # add a small disadvantage to time shifts away from zero
    tlkpn = np.exp(-np.power(tlk/0.2,2))
    tlkpn = stst/5.*tlkpn
    tlkpn = tlkpn.reshape([tlkpn.size,1])

    # compute events in groups to avoid running out of memory
    igrp=np.append(np.arange(0,len(ilmsv)-1,100),len(ilmsv)-1)
    # which indices
    i1,i2=ilmsv[0:-1],ilmsv[1:]
                    
    # preferred shifts
    ibest = np.zeros(len(ilmsv)-1,dtype=int)
    Nb = 15
    ibestb = np.zeros([len(ilmsv)-1,Nb],dtype=int)

    if blimadd is not None:
        sta = sta.copy()
        if blimadd[0]==0.:
            sta.filter('lowpass',freq=blimadd[1])
        elif np.isinf(blimadd[1]):
            sta.filter('highpass',freq=blimadd[0])
        else:
            sta.filter('bandpass',freqmin=blimadd[0],
                       freqmax=blimadd[1])


    # to compute for all time shifts
    dtim = sta[0].stats.delta
    tlm = general.minmax(ttrav,1.)+np.array([-1.,1.])*dtim
    tcalc = np.arange(tlm[0],tlm[1],dtim/10.)

    # identify the relevant times
    ix = general.closest(tcalc,ttrav)


    for m in range(0,len(igrp)-1):
        # initialize x-c for all stations
        jgrp = igrp[m:m+2]
        print('Events '+str(jgrp[0])+' to '+str(jgrp[1])+
              ' of '+str(igrp[-1]))
        xc = np.zeros([ttrav.shape[0],np.diff(jgrp)[0],
                       ttrav.shape[1]])
        xc = np.zeros([tcalc.size,np.diff(jgrp)[0],
                       ttrav.shape[1]])

        # note that it's important to loop through the 
        # sta[k] indices and select from stsv
        # because of how the S and P arrivals are noted
        for k in range(0,len(sta)):
            trr = sta[k]
            trt = stsv.select(id=trr.get_id())
            print(trr.get_id())

            if trt:
                # grab the relevant portion of the template
                trr = trr.copy()
                tref = trr.stats.starttime+trr.stats.t3

                # look for any difference in shifts for this station
                shfsv = trt[0].stats.t7
                shfh=trr.stats.starttime+trr.stats.t3-\
                    obspy.UTCDateTime(2000,1,1)
                ishf=shfh-shfsv

                # to compute for all time shifts
                #tlm = general.minmax(ttrav[:,k],1.01)
                ishf1=(ishf+wlen[0]+tlm[0]-tget[0])/stsv[0].stats.delta
                ishf1=int(np.round(ishf1))
                nget=(wlen[1]-wlen[0]+tlm[1]-tlm[0])/  \
                    trt[0].stats.delta
                nget=int(np.ceil(nget))
                
                # the indices to get
                j1=np.arange(igrp[m],igrp[m+1])
                j1=i1[j1]+ishf1
        
                # extract a portion of the data before filtering to save time
                istart = np.maximum(np.min(j1)-5*nget,0)
                iend = np.minimum(np.max(j1)+6*nget,trt[0].stats.npts)
                trt = trt.copy()
                trt[0].data = trt[0].data[istart:iend]
                j1 = j1 - istart

                # create indices
                k1,k2=np.meshgrid(j1,np.arange(0,nget))
                k1=(k1+k2).flatten()

                # the original filter that was applied to the saved data
                # PROBABLY DON'T NEED THIS IN THE FUTURE 
                # BECAUSE THE STACK IS MADE FROM FILTERED DATA
                if blim is not None:
                    trr.filter('bandpass',freqmin=blim[0],
                               freqmax=blim[1])
                # a second filter to add at this point?
                if blimadd is not None:
                    msk=seisproc.prepfiltmask(trt,tmask=0.)
                    if blimadd[0]==0.:
                        trt.filter('lowpass',freq=blimadd[1])
                    elif np.isinf(blimadd[1]):
                        trt.filter('highpass',freq=blimadd[0])
                    else:
                        trt.filter('bandpass',freqmin=blimadd[0],
                                   freqmax=blimadd[1])
                    seisproc.addfiltmask(trt,msk)


                # extract the relevant portion of the template
                trr = trr.trim(starttime=tref+wlen[0],
                               endtime=tref+wlen[1],pad=True)
                tdata = trr.data.copy()
                tdata = tdata/(np.dot(tdata,tdata))**0.5
                tdata = tdata.reshape([tdata.size,1])
                
                # the relevant target data
                data = trt[0].data[k1]
                data = data.reshape([nget,len(j1)])

                # the cross-correlation
                xci = scipy.signal.correlate(data,tdata,mode='valid')

                # need to normalize the second set
                nml2=np.power(data,2)
                nml2=np.cumsum(nml2,axis=0)
                nml2=np.append(np.zeros([1,nml2.shape[1]]),nml2,
                               axis=0)
                nml2=nml2[tdata.size:,:]-nml2[:-tdata.size,:]
                nml2[nml2==0.] = float('inf')
                nml2=np.power(nml2,0.5)
                xci = np.divide(xci,nml2)

                # which shifts to get
                tms = np.arange(0,xci.shape[0]).astype(float)
                tms = tms*trr.stats.delta+tlm[0]
                
                # interpolate to any intermediate points
                for mm in range(0,xci.shape[1]):
                    #xc[:,mm,k]= np.interp(ttrav[:,k],tms,xci[:,mm])
                    xc[:,mm,k]= np.interp(tcalc,tms,xci[:,mm])

        # identify acceptable time shifts and set the 
        # remaining values to zero
        iok = np.logical_and(xc!=0.,~np.isnan(xc))
        iok = np.logical_and(iok,~np.isinf(xc))
        xc[~iok] = 0.

        # # to the Fourier domain
        # Nf = xc.shape[0]
        # xcf = np.fft.rfft(xc,Nf,axis=0)
        # freq = np.fft.rfftfreq(Nf,d=tcalc[1]-tcalc[0])

        # ttravf = ttrav.transpose().reshape([1,1,ttrav.shape[1],ttrav.shape[0]])
        # ttravf = np.multiply(ttravf,freq.reshape([freq.size,1,1,1]))
        # ttravf = ttravf*(-1j*2.*math.pi)

        # sum by extracting the relevant times
        xcm = np.zeros([ttrav.shape[0],xc.shape[1]])
        nper = np.zeros([ttrav.shape[0],xc.shape[1]])
        for ks in range(0,ttrav.shape[1]):
            xcm = xcm+xc[ix[:,ks],:,ks]
            nper = nper+iok[ix[:,ks],:,ks]

        # to normalize by number of observations
        xcm = np.divide(xcm,nper)

        # add time shift penalty
        #xcm = xcm + tlkpn
        #print('max shift penalty '+str(np.max(tlkpn)))

        # maximum
        ibest[jgrp[0]:jgrp[1]] = np.argmax(xcm,axis=0)

        # also bootstrap by station
        for kb in range(0,Nb):
            print(str(kb)+' of '+str(Nb)+' bootstrap locations')
            ii = np.random.choice(xc.shape[2],xc.shape[2])
            wgts = np.bincount(ii,minlength=xc.shape[2]).astype(float)
            ich, = np.where(wgts)

            # sum by extracting the relevant times
            xcm = np.zeros([ttrav.shape[0],xc.shape[1]])
            nper = np.zeros([ttrav.shape[0],xc.shape[1]])
            for ks in ich:
                xcm = xcm+wgts[ks]*xc[ix[:,ks],:,ks]
                nper = nper+wgts[ks]*iok[ix[:,ks],:,ks]
            xcm = np.divide(xcm,nper)

            # and the maximum x-c
            ibestb[jgrp[0]:jgrp[1],kb] = np.argmax(xcm,axis=0)


    # save the preferred locations
    xzbest = xz[ibest,:]

    # and the preferred time shifts
    stns=[tr.stats.network+'.'+tr.stats.station+'.'+
          tr.stats.channel for tr in sta]
    stns=[tr.get_id() for tr in sta]
    tshfs = dict((stns[k],ttrav[ibest,k]) for k in range(0,len(sta)))

    # 90% intervals
    ilm = np.array([0.05,.95])*Nb
    ilm = ilm.astype(int)

    # also the limiting locations
    x = xz[ibestb,0]
    x.sort(axis=1)
    x = x[:,ilm]

    z = xz[ibestb,1]
    z.sort(axis=1)
    z = z[:,ilm]

    # also get some estimate of variability in the travel time 
    # relative to the median
    ttravm = np.median(ttrav,axis=1)

    tshfu = {}
    for k in range(0,len(sta)):
        # the range of travel times for each station
        tsu = ttrav[ibestb,k]
        tsu = tsu - ttravm[ibestb]
        tsu.sort(axis=1)
        tsu = tsu[:,ilm]
        tsu = np.diff(tsu,axis=1)
        tshfu[stns[k]] = tsu.flatten()

    # but to save, would 

    return xzbest,tshfs,x,z,tshfu


#---------------CHECK SIGNAL TO NOISE RATIO-----------------------

def checksnr(st,minsnr=5.,blim=[2.,8.],pk='t3',wlen=3.,cmps=None):
    """
    :param       st:   waveforms
    :param   minsnr:   minimum signal to noise ratio
    :param     blim:   bandlimits to check
    :param       pk:   reference pick to use (default:'t3')
    :param     wlen:   window length (default: 3)
    :return     sta:   waveforms that pass snr
    """

    # needs to be a list
    if isinstance(blim[0],float) or \
            isinstance(blim[0],int):
        blim=[blim]
        
    # repeat signal fraction to preferred size
    minsnr=np.atleast_1d(minsnr)
    if len(minsnr)==1:
        minsnr=np.repeat(minsnr,len(blim))
        
    blmn = np.min(np.array([np.min(blm) for blm in blim]))
        
    # check signal to noise ratios
    sta = obspy.Stream()
    for tr in st:
        # initial filter
        tri = tr.copy().filter('highpass',freq=blmn/2.)

        sg,ns,snr,freq=pksdrops.getsnr(tri,wlen=wlen,pk=pk,nsi=1)

        iok = True
        for blm in blim:
            ii=np.logical_and(freq>=blm[0],freq<=blm[1])
            #iok=iok and np.mean(snr[ii])>minsnr
            iok=iok and np.min(snr[ii])>minsnr

        if iok:
            sta.append(tr)

    if cmps=='both_horizontals':
        for tr in sta:
            if tr.stats.channel[0]=='D':
                och = {'2':'3','3':'2'}
            elif tr.stats.channel[0] in 'EH':
                och = {'1':'2','2':'1','E':'N','N':'E'}
            else:
                och = {}
            if tr.stats.channel[2] in och.keys():
                och = tr.stats.channel[0:2]+och[tr.stats.channel[2]]

                # look for the other component
                if not sta.select(station=tr.stats.station,
                                  network=tr.stats.network,
                                  channel=och):
                    sta.remove(tr)
                    # # if it's not already included, look in the original
                    # sti = st.select(station=tr.stats.station,
                    #                 network=tr.stats.network,
                    #                 channel=och)
                    # if sti:
                    #     sta = sta+sti
                    # else:
                    #     sta.remove(tr)

    return sta


def scalemanyamps(sta,stsv,ilmsv,tget,tshfs,xcflm=[0.,8.]):
    """
    :param        sta:   template waveforms
    :param       stsv:   target waveforms, concatenated
    :param      ilmsv:   limits of the LFEs
    :param       tget:   time range included in the data
    :param      tshfs:   time shifts to use
    :param      xcflm:   frequency limits to use before computation
    """

    # time interval for cross-correlation
    # relative to t3 in sta
    txc = np.array([-.2,2.8])
    wpk = 't3'

    # certain stations 
    stns=[tr.stats.network+'.'+tr.stats.station+'.'+
          tr.stats.channel for tr in sta]

    # reference time
    reftime=obspy.UTCDateTime(2000,1,1)

    # filter
    print(xcflm)
    stsv = stsv.copy()
    sta = sta.copy()
    msk=seisproc.prepfiltmask(stsv,tmask=0.)
    if xcflm[0]==0.:
        sta.filter('lowpass',freq=xcflm[1])
        stsv.filter('lowpass',freq=xcflm[1])
    elif np.isinf(xcflm[1]):
        sta.filter('highpass',freq=xcflm[0])
        stsv.filter('highpass',freq=xcflm[0])
    elif xcflm[0]>0. and not np.isinf(xcflm[1]):
        sta.filter('bandpass',freqmin=xcflm[0],freqmax=xcflm[1])
        stsv.filter('bandpass',freqmin=xcflm[0],freqmax=xcflm[1])
    seisproc.addfiltmask(stsv,msk)

    # indices to extract the interval of interest
    ix1=int(np.round((txc[0]-tget[0])/sta[0].stats.delta))
    ix2=int(np.round((txc[1]-tget[0])/sta[0].stats.delta))

    # length of interval of interest
    N = ix2-ix1

    # initialize amplitudes
    amp = dict((sti, {}) for sti in stns)
    for lb in stns:
        amp[lb]=np.ones(len(ilmsv)-1,dtype=float)*float('nan')

    for tr in sta:
        # identify the other waveform
        trr = stsv.select(id=tr.get_id())[0]

        # identify any time shift in the reference time
        shfsv=trr.stats.t7
        shfh=tr.stats.starttime+tr.stats.t3-obspy.UTCDateTime(2000,1,1)

        # also time shifts
        tshf = tshfs[tr.get_id()] + (shfh-shfsv)
        ishf=np.round(tshf/tr.stats.delta).astype(int)
        
        # create grid of indices
        i1,i2=np.meshgrid(ishf+ilmsv[0:-1],np.arange(ix1,ix2))
        i1 = i1 + i2
        i1 = i1.flatten()

        # extract the relevant template data
        tr.trim(starttime=tr.stats.starttime+tr.stats.t3+txc[0],
                endtime=tr.stats.starttime+tr.stats.t3+txc[1]+\
                    5*tr.stats.delta)
        tr.data=tr.data[0:N]

        if len(tr.data)==N:
            # reshape for multiplication
            tdata=tr.data.reshape([1,N])

            # and the target data
            odata=trr.data[i1]
            odata=odata.reshape([N,len(ilmsv)-1])

            if not isinstance(odata,np.ma.masked_array):
                odata=np.ma.masked_array(odata,mask=False)
            
            # normalization
            nml1 = np.dot(tr.data,tr.data)**0.5
            nml2 = np.power(np.sum(np.power(odata,2),axis=0),0.5)

            # amplitude
            ampi = np.dot(tdata,odata)
            ampi = ampi / nml1**2
            #ampi = np.divide(ampi,nml2)/nml1

            
            # normalization
            lbl=tr.stats.network+'.'+tr.stats.station+'.'+tr.stats.channel
            amp[lbl]=ampi.flatten()

            # add a mask if appropriate
            msk=np.sum(odata.mask,axis=0)>0
            amp[lbl].mask=np.logical_or(amp[lbl].mask,msk)

    return amp

def stsvstd(stsv,ilmsv1,ilmsv2,mxmad=4.,ids=None):
    """
    :param      stsv: all the waveforms
    :param    ilmsv1: first indices limits to use for std calculation
    :param    ilmsv2: first indices limits to use for std calculation
    :param     mxmad: maximum std---the (log) median + mxmad times
                     the median absolute difference from the (log) median
    :param       ids: station ids to use
    :return     stds: standard deviation [# of stations by # of events]
    :return      iok: acceptable values [# of stations by # of events]
    """
    
    if ids is None:
        ids=np.array([tr.get_id() for tr in stsv])

    # to count
    ii = np.zeros(stsv[0].stats.npts,dtype=int)
    ii[ilmsv1] = 1
    ii = np.cumsum(ii)

    # in case there are dropouts
    jj = np.zeros(stsv[0].stats.npts+1,dtype=int)
    jj[ilmsv2] = -1
    jj = np.cumsum(jj)
    jj = ii+jj[0:-1]
    ii[jj==0] = 0
    
    stds = []
    for idi in ids:
        tr = stsv.select(id=idi)[0]
        stdi = np.power(tr.data,2)
        wgt=np.ma.getmaskarray(stdi)
        stdi[wgt]=0.
        nper = np.bincount(ii,weights=~wgt,minlength=len(ilmsv1)+1)
        stdi = np.bincount(ii,weights=stdi,minlength=len(ilmsv1)+1)
        nper = np.ma.masked_array(nper[1:],mask=nper[1:]==0)
        stdi = np.divide(stdi[1:],nper)
        stdi = np.power(stdi,0.5)
        stds.append(stdi)

    # collect log standard deviations
    stds=np.vstack(stds)        
    stds=np.ma.masked_array(stds,mask=np.isnan(stds))
    stds=np.log(stds)

    # compare to medians
    md=np.array([np.median(stds[k,~stds.mask[k,:]]) 
                 for k in range(0,stds.shape[0])])
    #md = np.median(stds,axis=1)
    md = md.reshape([md.size,1])
    
    stdd = stds-md
    #mad = np.median(np.abs(stdd),axis=1)
    mad=np.array([np.median(stdd[k,~stdd.mask[k,:]]) 
                  for k in range(0,stdd.shape[0])])
    mad = mad.reshape([mad.size,1])
    jok = np.abs(stdd) < np.log(mxmad)

    if isinstance(jok,np.ma.masked_array):
        jok=np.logical_and(jok.data,~jok.mask)

    return stds,jok

def pickstsv(stsv,stsvn,stam,ilmsv,maxstd,ids=None,stds=None):
    """
    :param       stsv:  original
    :param      stsvn:  noise intervals
    :param       stam:  stations we want
    :param      ilmsv:  indices bounding events
    :param     maxstd:  maximum variation in std
    :param        ids:  station ids to use (default: what's there)
    :param       stds:  the standard deviations, if they're pre-calculated
    :return       iok:  indices of events to keep
    :return       jok:  indices of events to keep by station
    """

    # station/channel ids
    if ids is None:
        ids=np.array([tr.get_id() for tr in stam])

    # make sure they're all masked
    for st in (stsvn+[stsv]):
        for tr in st:
            if not isinstance(tr.data,np.ma.masked_array):
                tr.data=np.ma.masked_array(tr.data,mask=False)

    # create masks for each stream
    msk = obspy.Stream()
    for idi in ids:
        msk = msk + stam.select(id=idi).copy()

    for tr in msk:
        tro=stsv.select(station=tr.stats.station,
                        network=tr.stats.network,
                        channel=tr.stats.channel)
        if tro:
            tr.data = tro[0].data.mask.astype(float)
        else:
            tr.data = np.ones(stsv[0].stats.npts,dtype=float)

    for st in stsvn:
        for tr in msk:
            tro=st.select(station=tr.stats.station,
                          network=tr.stats.network,
                          channel=tr.stats.channel)
            if tro:
                tr.data=np.logical_or(tr.data,tro[0].data.mask)

    # collect for comparison
    msks = np.vstack([tr.data for tr in msk])
    msks = np.append(np.zeros([msks.shape[0],1],dtype=float),
                     np.cumsum(msks,axis=1),axis=1)
        
    # number of problematic points per interval
    i1,i2=ilmsv[0:-1],ilmsv[1:]
    nprob = msks[:,i2]-msks[:,i1]
    jok = nprob==0
    iok, = np.where(np.sum(jok,axis=0)>0.*len(stam))

    print(str(iok.size)+' of '+str(i1.size)+' LFEs with available data')
    
    # stds = None
    # pick based on std
    if stds is None:
        stds,istd=stsvstd(stsv,i1[iok],i2[iok],mxmad=maxstd,ids=ids)
        for st in stsvn:
            stds,istdi=stsvstd(st,i1[iok],i2[iok],mxmad=maxstd,ids=ids)
            istd=np.logical_and(istd,istdi)
            jok[:,iok]=np.logical_and(jok[:,iok],istd)
        istd = np.sum(~istd,axis=0)==0

        # the correct values
        iok = iok[istd]
        
    else:

        for k in range(0,len(ids)):
            stdsi = stds[ids[k]]
            if not isinstance(stdsi,np.ma.masked_array):
                stdsi=np.ma.masked_array(stdsi,mask=stdsi==0.)
                stdsi.mask=np.logical_or(stdsi.mask,np.isnan(stdsi))
                stdsi.mask=np.logical_or(stdsi.mask,stdsi==1.)
            stdsi.mask=np.logical_or(stdsi.mask,~jok[k,:].reshape([jok.shape[1],1]))

            # compare to median
            md = np.array([np.median(stdsi[~stdsi.mask[:,k],k])
                           for k in range(0,stdsi.shape[1])])
            md = md.reshape([1,md.size])
            stdd = stdsi-md

            # within range
            joki = np.sum(np.abs(stdd) > np.log(maxstd),axis=1)==0

            # and unmasked
            msk = np.sum(stdsi.mask,axis=1)
            joki = np.logical_and(joki,msk==0)
            jok[k,:] = np.logical_and(jok[k,:],joki)
        
        iok, = np.where(np.sum(jok,axis=0)>0.5*len(stam))

    return iok,jok


def calcmedtshf(tshfs):

    # only use some shifts
    ids=list(tshfs.keys())

    # calculate median shifts
    tshf=np.vstack([tshfs[idi] for idi in ids])

    # mask
    if isinstance(tshfs[ids[0]],np.ma.masked_array):
        msk=np.vstack([tshfs[idi].mask for idi in ids])
        tshf=tshf.data
    else:
        msk=np.zeros(tshf.shape,dtype=bool)

    # medians
    tmd=np.array([np.median(tshf[~msk[:,k],k]) for k in range(0,tshf.shape[1])])
    mmd=np.array([np.sum(~msk[:,k])==0 for k in range(0,tshf.shape[1])])

    tmd[mmd]=0.
    tmd=np.ma.masked_array(tmd,mask=mmd)

    return tmd
    
def besttshf(stsv,sta,izero,trange=None,tlk=[-0.5,0.5],xcflm=[0.,5]):
    """
    :param       stsv:   target data
    :param        sta:   target data
    :param      izero:   indices of zero time to consider
    :param     trange:   time range to use for template
    :param        tlk:   time shift to search ([-0.5,0.5])
    :param      xcflm:   filter to apply
    :return     tshfs:   time shifts for each observation
    """
        
    # indices to extract the interval of interest
    ixo=int(np.round(trange[0]/sta[0].stats.delta))
    ix1=int(np.round((trange[0]+tlk[0])/sta[0].stats.delta))
    ix2=int(np.round((trange[1]+tlk[1])/sta[0].stats.delta))
    
    # length of interval of interest
    N = int(np.round(np.diff(trange)[0]/sta[0].stats.delta))
    Nx = ix2-ix1

    # observed components
    ids = np.array([tr.get_id() for tr in sta])

    #----------go on to extracting the data--------------------

    # create indices
    i1,i2=np.meshgrid(izero,np.arange(ix1,ix2))
    i1 = i1 + i2
    i1 = i1.flatten()
    
    # number of events
    Nev = len(izero)

    # number of stations and components
    Nc = len(sta)

    # initialize data
    tdata=np.zeros([N,Nc],dtype=float)
    odata=np.zeros([Nx,Nc,Nev],dtype=float)

    # template data
    sta = sta.copy()
    print(xcflm)
    if xcflm[0]==0.:
        sta.filter('lowpass',freq=xcflm[1],zerophase=True)
    elif np.isinf(xcflm[1]):
        sta.filter('highpass',freq=xcflm[0],zerophase=True)
    elif xcflm[0]>0. and not np.isinf(xcflm[1]):
        sta.filter('bandpass',freqmin=xcflm[0],freqmax=xcflm[1],
                   zerophase=True)

    for k in range(0,Nc):
        # template waveform
        idi = ids[k]
        tri = sta.select(id=idi)[0]
        trj = tri.copy()

        # target waveform
        tr = stsv.select(id=idi)[0].copy()

        # filter
        print(xcflm)
        msk = seisproc.prepfiltmask(tr,tmask=1.)
        if xcflm[0]==0.:
            tr.filter('lowpass',freq=xcflm[1],zerophase=True)
        elif np.isinf(xcflm[1]):
            tr.filter('highpass',freq=xcflm[0],zerophase=True)
        elif xcflm[0]>0. and not np.isinf(xcflm[1]):
            tr.filter('bandpass',freqmin=xcflm[0],freqmax=xcflm[1],
                      zerophase=True)
        seisproc.addfiltmask(tr,msk)

        # identify any time shift in the reference time
        shfsv=tr.stats.t7
        shfh=tri.stats.starttime+tri.stats.t3-obspy.UTCDateTime(2000,1,1)
        ishf=int(np.round((shfh-shfsv)/tr.stats.delta))

        # trim and extract template data
        tri.trim(starttime=tri.stats.starttime+tri.stats.t3+trange[0],
                 endtime=tri.stats.starttime+tri.stats.t3+trange[1]+
                 5*tri.stats.delta)
        tri.data=tri.data[0:N]
        tdata[:,k] = tri.data

        # extract target data
        datai = tr.data[i1+ishf].reshape([Nx,Nev])
        odata[:,k,:] = datai

    # demean
    mn = np.median(odata,axis=0).reshape([1,Nc,Nev])
    odata = odata - mn
    mn = np.median(tdata,axis=0).reshape([1,Nc])
    tdata = tdata - mn

    # frequencies
    Nf = np.maximum(N,Nx)*2

    # to frequency domain
    ftdata = np.fft.rfft(tdata,Nf,axis=0)
    fodata = np.fft.rfft(odata,Nf,axis=0)

    # cross-correlation FT and back
    xc = np.multiply(fodata,ftdata.conj().reshape([fodata.shape[0],Nc,1]))
    xc = np.fft.irfft(xc,axis=0)

    # to normalize
    nml1=np.power(np.sum(np.power(tdata,2),axis=0),0.5)
    nml1=nml1.reshape([1,Nc,1])
    nml2=np.cumsum(np.power(odata,2),axis=0)
    nml2=np.append(np.zeros([1,Nc,Nev]),nml2,axis=0)
    i1=np.arange(0,Nx-N)
    nml2=nml2[i1+N,:,:]-nml2[i1,:,:]
    nml2=np.power(nml2,0.5)

    # normalize
    xc = np.divide(xc[i1,:,:],nml2)
    xc = np.divide(xc,nml1)
    xc = np.ma.masked_array(xc,mask=np.isnan(xc))

    # preferred shifts
    ishf=(np.argmax(xc,axis=0)+(ix1-ixo))*sta[0].stats.delta
    msk=np.sum(xc.mask,axis=0).astype(bool)
    ishf[msk]=0.
    ishf=np.ma.masked_array(ishf,mask=msk)

    # also save x-c
    xc = np.max(xc,axis=0)
    xc[msk]=0.
    xc=np.ma.masked_array(xc,mask=msk)

    # save and output
    tshfs=dict((ids[k],ishf[k,:]) for k in range(0,len(ids)))
    xc=dict((ids[k],xc[k,:]) for k in range(0,len(ids)))

    return tshfs,xc
    

def calcwlk(stsv,sta,izero,trange=None,mxmad=10.,iok=None,twin=None,
            fullxc=True,eqloc=None,tshfs=None,flm=None,irange=None,
            bystat=True,usemshf=True,scls=None):
    """
    :param       stsv:   target data
    :param        sta:   target data
    :param      izero:   indices of zero time to consider
    :param     trange:   time range to use for template
    :param        iok:   [len(izero) by number of components] 
                          array indicating acceptable data
    :param       twin:   time window to use for computing
                           the cross-correlation coefficients
    :param     fullxc:   compute the full cross-correlation
    :param      eqloc:   earthquake location for possible shifts
    :param      tshfs:   time shifts, if known
    :param        flm:   frequency limits prior to analysis
    :param     irange:   the range of times with data, if applicable
    :param     bystat:   average x-c over stations first
    :return     tsave:   time shifts if any
    """

    if twin is None:
        twin = [-2.,2.]

    # how much to add before
    # 0 to avoid the P arrival
    tbf=0.2
        
    # indices to extract the interval of interest
    ix1i=int(np.round((trange[0])/sta[0].stats.delta))
    ix1=int(np.round((trange[0]-tbf)/sta[0].stats.delta))
    ix2=int(np.round(trange[1]/sta[0].stats.delta))
    
    # length of interval of interest
    N = ix2-ix1i

    # add length if we want the full x-c
    if fullxc:
        ix2=ix2+int(N*1.)
        Nx = ix2-ix1
    else:
        Nx = N

    # observed components
    ids = np.array([tr.get_id() for tr in sta])

    #---------to deal with the data intervals available-------------

    ishfa = np.array([0,0],dtype=int)
    for idi in ids:
        # template waveform
        tri = sta.select(id=idi)[0]
        tref=tri.stats.starttime+tri.stats.t1

        # target waveform
        tr = stsv.select(id=idi)[0]

        # identify any time shift in the reference time
        shfsv=tr.stats.t7
        shfh=tri.stats.starttime+tri.stats.t3-tref #obspy.UTCDateTime(2000,1,1)
        ishf=int(np.round((shfh-shfsv)/tr.stats.delta))
        ishfa = general.minmax(np.append(ishf,ishfa))
    ishfa = ishfa.astype(int)

    if irange is not None:
        ix1=np.maximum(ix1-ishfa[0],np.max(irange[:,0]-izero))
        ix2=np.minimum(ix2-ishfa[1],np.min(irange[:,1]-izero))
    Nx = ix2-ix1

    #----------go on to extracting the data--------------------

    # create indices
    i1,i2=np.meshgrid(izero,np.arange(ix1,ix2))
    i1 = i1 + i2
    i1 = i1.flatten()
    
    # number of events
    Nev = len(izero)

    # number of stations and components
    Nc = len(sta)

    # create taper
    taplen=-trange[0]
    tr = sta[0].copy()
    tr.data = np.ones(N,dtype=float)
    tr.taper(side='both',max_length=taplen,max_percentage=0.5,
             type='hann')
    tpr = tr.data.copy()

    # taper for the full target data
    tr.data = np.ones(Nx,dtype=float)
    tr.taper(side='both',max_length=taplen,max_percentage=0.5,
             type='hann')
    tprt = tr.data.copy()

    # initialize data
    tdata=np.zeros([N,Nc],dtype=float)
    tdataw=np.zeros([Nx,Nc],dtype=float)
    odata=np.zeros([Nx,Nc,Nev],dtype=float)

    # identify acceptable intervals
    if iok is None:
        iok = np.ones([Nc,Nev],dtype=float)
    if scls is None:
        scls = np.ones([Nc,Nev],dtype=float)

    # adjust for median time shift
    tmd=np.vstack([tshfs[ids[k]] for k in range(0,Nc)])
    tmd=np.array([np.median(tmd[iok[:,k],k]) for k in range(0,Nev)])
    tmd[np.isnan(tmd)]=0.

    # median scaling
    mscl=np.array([np.median(scls[k,iok[k,:]]) for k in range(0,Nc)])
    print(mscl)
    mscl=np.ones(mscl.shape,dtype=float)

    # template data
    sta = sta.copy()
    if flm is not None:
        sta.filter('bandpass',freqmin=flm[0],freqmax=flm[1])

    for k in range(0,Nc):
        # template waveform
        idi = ids[k]
        tri = sta.select(id=idi)[0]
        trj = tri.copy()

        # target waveform
        tr = stsv.select(id=idi)[0]

        # identify any time shift in the reference time
        shfsv=tr.stats.t7
        tref=tri.stats.starttime+tri.stats.t1
        shfh=tri.stats.starttime+tri.stats.t3-tref #obspy.UTCDateTime(2000,1,1)
        ishf=int(np.round((shfh-shfsv)/tr.stats.delta))

        # trim and extract longer template data
        dlen=np.diff(trange)[0]
        trj.trim(starttime=tri.stats.starttime+tri.stats.t3+trange[0]-tbf,
                 endtime=tri.stats.starttime+tri.stats.t3+trange[1]+
                 5*tri.stats.delta+dlen)
        trj.data=trj.data[0:Nx]
        tdataw[:,k] = trj.data
        
        # trim and extract template data
        tri.trim(starttime=tri.stats.starttime+tri.stats.t3+trange[0],
                 endtime=tri.stats.starttime+tri.stats.t3+trange[1]+
                 5*tri.stats.delta)
        tri.data=tri.data[0:N]
        tdata[:,k] = tri.data

        # create indices
        if usemshf:
            # use a median value for all stations
            itshf=np.round(tmd/tri.stats.delta).astype(int)
        else:
            # use relative shifts
            itshf=np.round((tshfs[idi]-tmd)/tri.stats.delta).astype(int)
        itshf,tsh=np.meshgrid(itshf,np.arange(ix1,ix2))
        itshf=itshf.flatten()
        
        # extract target data
        datai = tr.data[i1+ishf+itshf].reshape([Nx,Nev])
        odata[:,k,:] = datai

    # demean
    mn = np.median(odata,axis=0).reshape([1,Nc,Nev])
    odata = odata - mn
    
    # taper the edges
    tdata = np.multiply(tdata,tpr.reshape([N,1]))
    odata = np.multiply(odata,tprt.reshape([Nx,1,1]))
    tdataw = np.multiply(tdataw,tprt.reshape([Nx,1]))

    # rescale
    odata=np.divide(odata,scls.reshape([1,Nc,Nev]))
    tdata=np.divide(tdata,mscl.reshape([1,Nc]))
    tdataw=np.divide(tdataw,mscl.reshape([1,Nc]))

    # frequencies
    Nf = 2*np.maximum(N,Nx)
    freq = np.fft.rfftfreq(Nf,d=sta[0].stats.delta)
    ifreq = np.logical_and(freq>0,freq<=30)

    # to frequency domain
    ftdata = np.fft.rfft(tdata,Nf,axis=0)
    ftdataw = np.fft.rfft(tdataw,Nf,axis=0)
    fodata = np.fft.rfft(odata,Nf,axis=0)

    # cross-correlation FT
    fxc = np.multiply(fodata,ftdata.conj().reshape([freq.size,Nc,1]))
    fxct = np.multiply(ftdataw,ftdata.conj())

    # trash any interval with just zeros
    if isinstance(fxc,np.ma.masked_array):
        fxc.mask = np.logical_or(fxc.mask,fxc==0.)
    else:
        fxc = np.ma.masked_array(fxc,fxc==0.)

    # average x-c over components at each station
    if bystat:
        # which stations they belong to
        stn=np.array([idi.split('.')[1] for idi in ids])
        ia,ix,ix2=np.unique(stn,return_inverse=True,return_index=True)
        imap=ix[ix2]
        if iok is None:
            iok = np.ones([Nc,Nev],dtype=bool)
        nok = ~iok
        
        # go through and add
        for k in range(0,ix2.size):
            if imap[k]!=k:
                fxc[:,imap[k],:]=fxc[:,imap[k],:]+fxc[:,k]
                fxct[:,imap[k]]=fxct[:,imap[k]]+fxct[:,k]
                nok[imap[k],:]=nok[imap[k],:]+nok[k,:]
        iok = ~nok[ix,:]
                
        # extract and normalize
        fxc,fxct = fxc[:,ix,:],fxct[:,ix]
        nper=np.bincount(ix2)
        fxc=np.divide(fxc,nper.reshape([1,nper.size,1]))
        fxct=np.divide(fxct,nper.reshape([1,nper.size]))
        Nc = len(ix)

    # if we want to extract a time-domain portion of the FT
    if fullxc:
        #---------CROSS-CORRELATION---------------------------

        # back to the time domain
        xc = np.fft.irfft(fxc,axis=0)
        xct = np.fft.irfft(fxct,axis=0)
        
        # window to extract
        iwin=np.atleast_1d(twin)+tbf
        iwin=np.round(iwin/sta[0].stats.delta).astype(int)
        iwin=np.arange(iwin[0],iwin[1])
        iwin=iwin % xc.shape[0]
        xc = xc[iwin,:,:]
        xct = xct[iwin,:]
        xcsave = xc[:,:,0]

        # taper this
        tr = tr.copy()
        tr.data = np.ones(iwin.size,dtype=float)

        #--------TAPERS--------------------------------------
        ttyp='slepian'
        if ttyp=='edge':
            # just taper the edges
            tr.taper(side='both',max_percentage=0.25,type='hann')
            tprx = tr.data.copy()
            tprx = tprx / np.dot(tprx,tprx)**0.5
            tprs=tprx.reshape([tprx.size,1])
        elif ttyp=='hann':
            # for a hann taper
            tprx = scipy.signal.hann(iwin.size)
            tprx = tprx / np.dot(tprx,tprx)**0.5
            tprs=tprx.reshape([tprx.size,1])
        elif ttyp=='slepianbad':
            tprx = scipy.signal.slepian(iwin.size,3./iwin.size)
            tprx = tprx / np.dot(tprx,tprx)**0.5
            tprs=tprx.reshape([tprx.size,1])
        elif ttyp=='multi' or ttyp=='slepian':
            # decide on the tapers' concentration
            dfres=0.3
            NW = dfres / (1./np.diff(twin)[0]) * 2

            # compute tapers
            [tprs,V] = spectrum.mtm.dpss(iwin.size,NW)
            
            # just select some?
            ii = V>=0.95
            tprs = tprs[:,ii]
            nmax=int(np.maximum(np.floor(tprs.shape[1]*0.75),1))
            if ttyp=='slepian':
                nmax=1
            tprs = tprs[:,0:nmax]

        Ntap=tprs.shape[1]


        #-------TAPER AND SHIFT-----------------------------------

        # one taper at a time
        fxc=np.ndarray([np.sum(ifreq),xc.shape[1],xc.shape[2],Ntap],dtype=complex)

        for kt in range(0,Ntap):
            # reshape
            tprx = tprs[:,kt].reshape([iwin.size,1,1])
            xcj = np.multiply(xc,tprx)

            # move center point to first position
            xcj=np.append(xcj,np.zeros([Nf-iwin.size,xc.shape[1],xc.shape[2]],
                                       dtype=float),axis=0)
            xcj=np.roll(xcj,-int(iwin.size/2),axis=0)
        
            # and back to frequency domain
            xcj=np.fft.rfft(xcj,Nf,axis=0)

            # add to set
            fxc[:,:,:,kt]=xcj[ifreq,:,:]

        #--------TEMPLATE AMPLITUDE FOR NORMALIZATION-----------------

        # use the same normalization for the original
        tprx = tprs.reshape([iwin.size,1,Ntap])
        xct = xct.reshape(np.append(xct.shape,1))
        xct = np.multiply(xct,tprx)

        # move center point to first position
        xct=np.append(xct,np.zeros([Nf-iwin.size,xct.shape[1],Ntap],dtype=float),axis=0)
        xct=np.roll(xct,-int(iwin.size/2),axis=0)

        # and back to frequency domain
        fxci = np.fft.rfft(xct,Nf,axis=0)
        fxci = fxci[ifreq,:,:]

        #--------NORMALIZE------------------------------------------------
        freq=freq[ifreq]
        
        # first average over tapers
        ampi = np.mean(np.power(np.abs(fxci),2),axis=2)
        ampi = np.power(ampi,0.5)
        
        # normalize
        ampi = ampi.reshape([freq.size,Nc,1,1])
        fxc = np.divide(fxc,ampi)
        ampi = ampi.reshape([freq.size,Nc])
        ampj = np.power(np.abs(fxc),2)
        ampj = np.mean(ampj,axis=3)
        
        #--------------ALLOWING FOR VARYING TIME SHIFTS-------------------

        # save the travel times
        tsaves = np.zeros([0,len(ids)],dtype=float)
        xz = None
        ipick = np.zeros(0,dtype=int)

        eqloc=None
        if eqloc is not None:
            # adjust for travel times
            if tshfs is None:
                ttrav,xzi,trash = calcttrav(sta,eqloc,phsarv=['S','s'])
                
                # change to phases
                ii = np.logical_and(freq>=2.,freq<=10.)
                ttrav = ttrav.reshape([ttrav.shape[0],ttrav.shape[1],1,1])
                phs = np.multiply(ttrav,(freq[ii]).reshape([1,1,np.sum(ii),1]))
                phs = (phs % 1.) * (2.*math.pi*1j)
                phs = np.exp(phs)
                phs = phs.transpose([2,1,3,0])

            # need to consider events in groups if there are too many
            igrp = np.arange(0,fxc.shape[2],100)
            igrp = np.append(igrp,fxc.shape[2])

            for kk in range(0,len(igrp)-1):
                jj=np.arange(igrp[kk],igrp[kk+1])
                print(str(jj[0])+' to '+str(jj[-1])+' of '+str(max(igrp)))

                if tshfs is not None:
                    # if the time shifts were specified
                    ttravi=np.vstack([tshfs[idi][jj] for idi in ids])
                else:
                    fmult=fxc[ii,:,:].copy()
                    fmult=fmult[:,:,jj].reshape([np.sum(ii),Nc,jj.size,1])
                    fmult = np.multiply(fmult,phs)
                    
                    fmult = np.divide(fmult,np.abs(fmult))
                    fmult = np.abs(np.sum(fmult,axis=1))
                    fmult = np.mean(fmult,axis=0)
                    
                    # choose the maximum walkout
                    imult = np.argmax(fmult,axis=1)
                    ipick = np.append(ipick,imult)

                    # save time shifts
                    tsaves = np.append(tsaves,ttrav[imult,:,0,0],axis=0)
                    ttravi=ttrav[imult,:,0,0].transpose()    

                # multiply with these travel times
                ttravi=ttravi.reshape([1,ttravi.shape[0],ttravi.shape[1]])
                phsh = np.multiply(ttravi,freq.reshape([freq.size,1,1]))
                phsh = (phsh % 1.) * (2.*math.pi*1j)
                phsh = np.exp(-phsh)
                fxc[:,:,jj] = np.multiply(fxc[:,:,jj],phsh)

            if tshfs is None:
                # average preferred times
                xz = xzi[ipick,:]

                # try shifting all the stations together
                ii = np.logical_and(freq>=2.,freq<=10.)
                vli = np.sum(fxc[ii,:,:],axis=1)
                tlook = np.arange(-.5,.5,0.01)
                tlook = tlook.reshape([1,tlook.size])
                phs = np.multiply(tlook,(freq[ii]).reshape([np.sum(ii),1]))
                phs = np.exp((phs % 1.) * (2.*math.pi*1j))
                vli=np.multiply(vli.reshape([vli.shape[0],1,
                                             vli.shape[1]]),
                                phs.reshape([phs.shape[0],
                                             phs.shape[1],1]))
                vli = np.sum(vli,axis=0)
                ishf = np.argmax(np.real(vli),axis=0)
                tadd = np.atleast_1d(tlook.flatten()[ishf])
                tadd = tadd.reshape([tadd.size,1])
                tsaves = tsaves + tadd
                
                tadd = tadd.reshape([1,1,tadd.size])
                phs = np.multiply(tadd,freq.reshape([freq.size,1,1]))
                phs = np.exp((phs % 1.) * (2.*math.pi*1j))
                fxc = np.multiply(fxc,phs)
    else:

        #----THE CASE WHERE COEFFICIENTS ARE USED DIRECTLY, NOT BACK TO TIME DOMAIN-------

        # normalize
        ampi = np.power(np.abs(ftdata),2).reshape([freq.size,Nc,1])
        fxc = np.divide(fxc,ampi)
        ampj = np.divide(np.power(np.abs(fodata),2),ampi)


    #--------AVOID OUTLIERS WITH EXCEPTIONALLY LARGE AMPLITUDES----------------------

    # don't let the amplitudes be too different from the median
    ii = np.logical_and(freq>2,freq<8)
    amps = np.mean(ampj[ii,:,:],axis=0)
    mamp=np.log10(np.median(amps[amps>0]))
    df = np.abs(np.log10(amps)-mamp)
    mdf = np.median(df[np.abs(df)<1.e10])
    jok = df < 4*mdf

    # or too large---if the amplitude is far larger than the
    # expected LFE, it's unlikely to be useful
    # jok = np.logical_and(jok,amps<5000.)

    #jok = np.ones(jok.shape,dtype=bool)

    # combine with input accepted intervals
    #iok = np.logical_and(iok,jok)

    # delete the problematic values
    msk = (~iok).reshape([1,Nc,Nev])
    msk = np.repeat(msk,fxc.shape[0],axis=0)
    iok = iok.reshape([1,Nc,Nev]).astype(float)
    ampj[msk] = 0.
    ampj = np.ma.masked_array(ampj,mask=msk)
    msk=np.repeat(msk.reshape(np.append(msk.shape,1)),fxc.shape[3],axis=3)
    fxc[msk] = 0.
    fxc = np.ma.masked_array(fxc,mask=msk)

    #-------TO ACTUALLY COMPUTE THE PHASE COHERENCE-------------------------------

    # number of stations used
    Nu = np.sum(iok,axis=1)
    msk = Nu<0.7*fxc.shape[1]
    msk = Nu<3
    Nu = np.ma.masked_array(Nu,mask=msk)
    nml = np.multiply(Nu,Nu-1.)

    # fully normalize phase coherence
    cpt = np.power(ampj.reshape(np.append(ampj.shape,1)),0.5)
    cpt = np.divide(fxc,cpt)
    en = np.sum(np.power(np.abs(cpt),2),axis=1)
    cpt = np.sum(cpt,axis=1)
    cpt = np.power(np.abs(cpt),2)
    cpt = cpt - en

    # average these over tapers
    cpt=np.mean(cpt,axis=2)
    en=np.mean(en,axis=2)

    # total energy
    en = np.sum(ampj,axis=1)

    # energy-normalized phase coherence
    cp = np.sum(fxc,axis=1)
    cp = np.power(np.abs(cp),2)
    # average over tapers
    cp = np.mean(cp,axis=2)
    cp = cp - en

    # normalize by relevant number of stations
    cp = np.divide(cp,nml)
    cpt = np.divide(cpt,nml)
    en = np.divide(en,Nu)

    # mask if there are too few stations
    msk = np.repeat(msk,cp.shape[0],axis=0)
    cp = np.ma.masked_array(cp,mask=msk)
    en = np.ma.masked_array(en,mask=msk)

    # save the time shifts
    if tshfs is None:
        tsave = dict((ids[k],tsaves[:,k]) for k in range(0,len(ids)))
    else:
        tsave = tshfs.copy()

    Nu = Nu.flatten()

    return cp,en,freq,tsave,cpt,fxc,ampi,xz,Nu,tprs,xcsave

def exctimes(tms,fnum,ids):
    """
    :param        tms: times to check
    :param       fnum: event id
    :param        ids: which stations/channels to use
    :return      exct: a grid of the problematic values
    """

    # initialize and go through ids
    exct = np.zeros([len(tms),len(ids)],dtype=bool)

    fname=os.path.join(os.environ['TREMORAREA'],'problem_times',str(fnum))
    if os.path.exists(fname):

        # read the relevant channels and timing
        vls = np.loadtxt(fname,delimiter=',',dtype=bytes).astype(str)
        chn = vls[:,0]
        t1=np.array([datetime.datetime.strptime(vl,'%Y-%b-%d-%H-%M-%S')
                     for vl in vls[:,1]])
        t2=np.array([datetime.datetime.strptime(vl,'%Y-%b-%d-%H-%M-%S')
                     for vl in vls[:,2]])
        
        for k in range(0,len(chn)):
            # exclude this time range
            iex = np.logical_and(tms>=t1[k],tms<=t2[k])
            #from the relevant stations
            exct[iex,ids==chn[k]] = True
        
    return exct

def addexcl(tlm,idi):
    """
    :param    tlm: time limit in datenum time
    :param    idi: network.station.channel
    """
    
    t1=matplotlib.dates.num2date(tlm[0])
    t2=matplotlib.dates.num2date(tlm[1])
    t1w=t1.strftime('%Y-%b-%d-%H-%M-%S')
    t2w=t2.strftime('%Y-%b-%d-%H-%M-%S')
    fname=os.path.join(os.environ['TREMORAREA'],'problem_times',idi)
    fl = open(fname,'a')
    fl.write(t1w+','+t2w+'\n')
    print(t1w+','+t2w)
    fl.close()

def plotamps(psg,idi):

    plt.close('all')

    for ps in psg:
        dts = np.array([matplotlib.dates.date2num(t.datetime) for t in ps.tms])
        plt.figure()

        try:
            plt.plot_date(dts,ps.ampscl[idi]/ps.medscale[idi],
                          marker='o',linestyle='none',color='b')
            
        except:
            print('No data for '+str(ps.fnum))

        plt.plot_date(dts,ps.meanamp,
                      marker='o',linestyle='none',color='r')
        
        plt.title(str(ps.fnum)+', '+idi)
    


def amppatterns(amps,stns=None,mamps=None,alim=[-1,5]):
    """
    :param     amps: the set of amplitudes
    :param     stns: stations to consider
    :param    mamps: median amplitudes per channel
    :param     alim: normalized amplitudes to allow
                      (default: [-1,5])
    """

    # default stations
    if stns is None:
        stns=np.array(list(amps.keys()))
    stns=np.atleast_1d(stns)

    # medians if not givens
    if mamps is None:
        mamps={}
        for ky in stns:
            mamps[ky]=amps[ky][amps[ky]!=0]

    # grab the normalized values
    vls=np.vstack([amps[ky]/mamps[ky] for ky in stns])

    # only non-zero values
    iok = np.sum(vls==0,axis=0)==0
    vls=vls[:,iok]

    # and only within a certain range
    iok = np.logical_and(vls>=alim[0],vls<=alim[1])
    vls = np.ma.masked_array(vls,mask=~iok)

    # do a PCA analysis
    m = empca.empca(vls,nvec=1)
            
    return m


def meanamps(ampscl,stns=None,minfrc=0.8,medscale=None,minstat=5):
    """
    :param      ampscl:  set of amplitudes
    :param        stns:  specified stations and components to use
    :param      minfrc:  minimum fraction of the stations required to
                           give an amplitude
    :param     minstat:  minimum number of channels (overrides minfrc)
    :return    meanamp:  a mean amplitude for each event
    """

    if stns is None:
        stns = list(ampscl.keys())

    # grab the amplitudes, scaled or not
    if medscale:
        meanamp = [ampscl[stn]/medscale[stn] 
                   for stn in stns]
    else:
        meanamp = [ampscl[stn] for stn in stns]
    meanamp = np.vstack(meanamp)
    msk = np.vstack([ampscl[stn].mask for stn in stns])
 
    # in case there's only one station
    if meanamp.ndim==1:
        meanamp=meanamp.reshape([meanamp.size,1])
        msk=msk.reshape([msk.size,1])

    # add to mask
    msk = np.logical_or(msk,np.isnan(meanamp))
    msk = np.logical_or(msk,meanamp==0.)
    meanamp.mask = msk

    # take the median
    mn=[np.median(meanamp[~meanamp.mask[:,k],k])
        for k in range(0,meanamp.shape[1])]

    # which ones have enough channels
    iok = np.sum(~msk,axis=0)
    if minstat is None:
        frcok = iok>=minfrc*float(meanamp.shape[0])
    else:
        frcok = iok>=minstat
    iok = np.maximum(iok,1.)
    
    # add to set mask problematic values
    meanamp = np.ma.masked_array(mn,mask=~frcok)
    meanamp.mask = np.logical_or(meanamp.mask,np.isnan(meanamp))
    
    return meanamp

def ampsok(ampscl,stns=None,minfrc=0.8,medscale=None):
    """
    :param      ampscl:  set of amplitudes
    :param        stns:  specified stations and components to use
    :param      minfrc:  minimum fraction of the stations required to
                           give an amplitude
    :return    meanamp:  a mean amplitude for each event
    """

    if stns is None:
        stns = list(ampscl.keys())

    # figure out the mask
    if medscale:
        meanamp = [ampscl[stn]/medscale[stn] 
                   for stn in stns]
    else:
        meanamp = [ampscl[stn] for stn in stns]
    meanamp = np.vstack(meanamp)
    msk = np.vstack([ampscl[stn].mask for stn in stns])
 
    # in case there's only one station
    if msk.ndim==1:
        msk=msk.reshape([msk.size,1])

    # add to mask
    msk = np.logical_or(msk,np.isnan(meanamp))
    msk = np.logical_or(msk,meanamp==0.)

    # which ones have enough stations
    iok = np.sum(~msk,axis=0)
    frcok = iok>=minfrc*float(meanamp.shape[0])
    iok = np.maximum(iok,1.)

    return frcok

def tempsnr(sta,trange,twin):
    """
    :param       sta:  templates
    :param    trange:  time range to compute signal ratio
    :param      twin:  time window for x-c
    """
    
    # filter to what I've used before
    sta = sta.copy().filter('bandpass',freqmin=1,freqmax=30)

    Nf = np.diff(trange)[0]*2+np.diff(twin)[0]
    Nf = int(2*Nf/sta[0].stats.delta)
    Nf = np.max([tr.stats.npts for tr in sta])*2

    nswin = np.atleast_1d(trange)-trange[1]-1.

    Sa,Saw,Sac = [],[],[]
    for tr in sta:
        # spectra and noise
        sg,ns,snr,freq=pksdrops.getsnr(tr,wlen=trange,pk='t3',nsi=[nswin])        
        ns = np.mean(ns,axis=1)

        # signal fraction
        S = 1.-np.divide(ns,sg.flatten())
        S = np.maximum(S,0.)
        S = np.minimum(S,1.)

        Sa.append(S)

        # template data
        t1=tr.stats.starttime+tr.stats.t3+trange[0]
        t2=tr.stats.starttime+tr.stats.t3+trange[1]
        tri=tr.copy().trim(starttime=t1,endtime=t2)
        tri.taper(max_length=0.5,max_percentage=0.5,
                  type='hann',side='both')

        # similar data
        t2=t2+np.diff(trange)[0]+np.diff(twin)[0]
        trj=tr.copy().trim(starttime=t1,endtime=t2)
        trj.taper(max_length=0.5,max_percentage=0.5,
                  type='hann',side='both')
        trj=tr.copy()

        # cross-correlation
        fxct = np.fft.rfft(tri.data,Nf)
        fxcd = np.fft.rfft(trj.data,Nf)
        fxcd = np.multiply(fxcd,fxct.conj())
        
        # back to the time domain
        xc = np.fft.irfft(fxcd,axis=0)

        # time difference betwen the two windows
        tdiff = trj.stats.starttime-tri.stats.starttime
        
        # window to extract
        iwin=np.round(np.atleast_1d(twin-tdiff)/sta[0].stats.delta).astype(int)
        iwin=np.arange(iwin[0],iwin[1])
        iwin=iwin % xc.shape[0]
        xc = xc[iwin]

        # taper this
        tprx = scipy.signal.hann(iwin.size)
        tprx = tprx / np.dot(tprx,tprx)**0.5
        xc = np.multiply(xc,tprx)
        xc = np.append(xc,np.zeros(Nf-xc.size,dtype=float))
        xc = np.roll(xc,-int(iwin.size/2))

        fxcd = np.real(np.fft.rfft(xc,Nf))
        fxcd = np.power(fxcd,2)

        # save the real fraction
        rrat = np.fft.rfft(xc,Nf)
        freal = np.divide(np.real(rrat),np.abs(rrat))


        # and for noise
        shf = np.diff(trange)[0]+np.diff(twin)[0]
        nsi = phscoh.defnoiseint(trange,3,allshf=-shf)
        fxcn = np.zeros([fxcd.size,len(nsi)])
        for m in range(0,len(nsi)):
            # t1=tr.stats.starttime+tr.stats.t3+nsi[m][0]
            # t2=tr.stats.starttime+tr.stats.t3+nsi[m][1]
            # t2=t2+np.diff(trange)[0]+np.diff(twin)[0]
            # trj=tr.copy().trim(starttime=t1,endtime=t2)
            # trj.taper(max_length=0.5,max_percentage=0.5,
            #           type='hann',side='both')
            trj=tr.copy().trim(starttime=t1,endtime=t2)
            trj.taper(max_length=0.5,max_percentage=0.5,
                      type='hann',side='both')


            fxci = np.fft.rfft(trj.data,Nf)
            fxci = np.multiply(fxci,fxct.conj())
        
            # back to the time domain
            xc = np.fft.irfft(fxci,axis=0)
    
            # window to extract
            iwin = twin - tdiff + (nsi[m][0]-trange[0])
            iwin=np.round(np.atleast_1d(twin-tdiff)/sta[0].stats.delta).astype(int)
            iwin=np.arange(iwin[0],iwin[1])
            iwin=iwin % xc.shape[0]
            xc = xc[iwin]

            # taper this
            xc = np.multiply(xc,tprx)
            xc = np.append(xc,np.zeros(Nf-xc.size,dtype=float))
            xc = np.roll(xc,-int(iwin.size/2))

            rrat = np.fft.rfft(xc,Nf)
            fxcn[:,m] = np.power(np.abs(rrat),2)


        freq = np.fft.rfftfreq(Nf,d=tr.stats.delta)

        # signal fraction
        S = 1.-np.divide(np.mean(fxcn,axis=1),fxcd)
        S = np.maximum(S,0.)
        S = np.minimum(S,1.)

        Saw.append(S)
        Sac.append(freal)

    # stack and average signal fraction
    Sa = np.vstack(Saw)
    
    Sac = np.vstack(Sac)
    Sac = np.mean(Sac,axis=0)

    # multiply relevant values
    ii = np.arange(0,Sa.shape[0])
    ii,jj=np.meshgrid(ii,ii)
    iok=jj>ii
    ii,jj=ii[iok],jj[iok]

    # average
    mred = np.mean(np.multiply(Sa[ii,:],Sa[jj,:]),axis=0)
        
    return freq,mred,Sac


def cpfromamptshf(tshfu=None):
    """
    :param     tshfu:  a set of time shift uncertainties
    """

    plt.close()
    f = plt.figure(figsize=(8,12))
    gs,p=gridspec.GridSpec(2,1),[]
    gs.update(left=0.15,right=0.97)
    gs.update(bottom=0.07,top=0.94)
    gs.update(hspace=0.05,wspace=0.25)
    for k in range(0,2):
        p.append(plt.subplot(gs[k]))
    p=np.array(p)
    pm=p.reshape([p.size,1])


    if tshfu is None:
        tshfu = (np.random.rand(8)-0.5)*0.01+0.01
    dtim = 0.01
    N = int(20./dtim)
    Nshf = 1000
    Nch = 10
    Ns =len(tshfu)

    freq = np.fft.rfftfreq(N,d=dtim)
    
    Nphs = 500
    Ntry = 10
    rdsa = np.zeros([len(freq),Ns],dtype=float)
    rds = np.ndarray([len(freq),Ns],dtype=float)
    cpa = np.ndarray([len(freq),Nphs*Ntry],dtype=float)

    
    for ktry in range(0,Ntry):
        for m in range(0,len(tshfu)):
            rt = np.zeros(freq.size,dtype=float)
            
            for k in range(0,Nch):
                # initialize random set
                vls = np.random.randn(N)
                vlss = np.zeros(vls.shape,dtype=float)
                # time shifts to use
                shf=np.random.randn(Nshf)*tshfu[m]/dtim
                shf=np.round(shf).astype(int)
                # time shift and sum
                for n in range(0,Nshf):
                    vlss=vlss+np.roll(vls,shf[n])
                vlss = vlss/float(Nshf)
            
                fvls = np.fft.rfft(vls)
                fvlss = np.fft.rfft(vlss)
          
                # is the absolute value in the right place????
                rt = rt+np.abs(np.divide(fvlss,fvls))

            rds[:,m] = rt / float(Nch)

        Nn = 10000
        prc = np.array([0.5,0.15,0.85])
        ii = (prc * Nn).astype(int)
        
        # the amplitudes
        vlsi = rds.copy()
        vlsi = vlsi.transpose()
        vlsi = np.divide(1.,vlsi)
        
        tshfu=tshfu.reshape([Ns,1])
        freqh=freq.copy().reshape([1,freq.size])
        for n in range(0,Nphs):
            vlsj = vlsi.copy()

            # time shifts to use
            shf=np.random.randn(Ns).reshape([Ns,1])
            shf=np.multiply(shf,tshfu)
            phs=np.multiply(shf,freq)
            phs = np.exp(phs*math.pi*2*1j)
            vlsj = np.multiply(vlsj,phs)

            # total energy and coherence
            en = np.sum(np.power(np.abs(vlsj),2),axis=0)
            cp = np.power(np.abs(np.sum(vlsj,axis=0)),2)
            cp = (cp-en)/float(Ns*(Ns-1))
            en = en / float(Ns)
            cp = np.divide(cp,en)
            
            cpa[:,ktry*Nphs+n] = np.real(cp)


    prc = np.array([0.5,0.15,0.85])
    ii = (prc * Ntry*Nphs).astype(int)
    cpa.sort(axis=1)

    cols = graphical.colors(Ns)
    for k in range(0,Ns):
        p[0].plot(freq,rds[:,k],color=cols[k])

    p[0].set_xscale('log')
    p[0].set_xlim([0.2,50])
    p[0].set_ylim([-0.03,1.03])
    p[0].set_ylabel('averaged amplitudes adjusted for time shift')

    for k in range(1,ii.size):
        p[1].plot(freq,cpa[:,ii[k]],color='k',linestyle='--')
    p[1].plot(freq,cpa[:,ii[0]],color='k')
    p[1].set_xlabel('frequency (Hz)')
    p[1].set_xscale('log')
    p[1].set_ylabel('$E_c / E_t$ adjusted for amplitudes and time shifts')
    p[1].set_xlim([0.2,50])
    p[1].set_ylim([np.min(cpa.flatten())-0.03,1.03])

    graphical.delticklabels(pm,'x')
    graphical.printfigure('PKLcpfromamptshf',f)


def cpfromamptshf(tshfu=None):
    """
    :param     tshfu:  a set of time shift uncertainties
    :return      cpa:  the adjust phase coherences [Nfreq x 3]
    :return     freq:  the frequencies
    :return      prc:  the percentiles extracted
    """

    if tshfu is None:
        tshfu = (np.random.rand(8)-0.5)*0.01+0.01
    dtim = 0.01
    N = int(20./dtim)
    Nshf = 1000
    Nch = 5
    Ns =len(tshfu)

    freq = np.fft.rfftfreq(N,d=dtim)
    
    Nphs = 50
    Ntry = 5
    rdsa = np.zeros([len(freq),Ns],dtype=float)
    rds = np.ndarray([len(freq),Ns],dtype=float)
    cpa = np.ndarray([len(freq),Nphs*Ntry],dtype=float)

    
    for ktry in range(0,Ntry):
        for m in range(0,len(tshfu)):
            rt = np.zeros(freq.size,dtype=float)
            
            for k in range(0,Nch):
                # initialize random set
                vls = np.random.randn(N)
                vlss = np.zeros(vls.shape,dtype=float)
                # time shifts to use
                shf=np.random.randn(Nshf)*tshfu[m]/dtim
                shf=np.round(shf).astype(int)
                # time shift and sum
                for n in range(0,Nshf):
                    vlss=vlss+np.roll(vls,shf[n])
                vlss = vlss/float(Nshf)
            
                fvls = np.fft.rfft(vls)
                fvlss = np.fft.rfft(vlss)
                
                rt = rt+np.abs(np.divide(fvlss,fvls))

            rds[:,m] = rt / float(Nch)

        Nn = 10000
        prc = np.array([0.5,0.15,0.85])
        ii = (prc * Nn).astype(int)
        
        # the amplitudes
        vlsi = rds.copy()
        vlsi = vlsi.transpose()
        vlsi = np.divide(1.,vlsi)
        
        tshfu=tshfu.reshape([Ns,1])
        freqh=freq.copy().reshape([1,freq.size])
        for n in range(0,Nphs):
            vlsj = vlsi.copy()

            # time shifts to use
            shf=np.random.randn(Ns).reshape([Ns,1])
            shf=np.multiply(shf,tshfu)
            phs=np.multiply(shf,freq)
            phs = np.exp(phs*math.pi*2*1j)
            vlsj = np.multiply(vlsj,phs)

            # total energy and coherence
            en = np.sum(np.power(np.abs(vlsj),2),axis=0)
            cp = np.power(np.abs(np.sum(vlsj,axis=0)),2)
            cp = (cp-en)/float(Ns*(Ns-1))
            en = en / float(Ns)
            cp = np.divide(cp,en)
            
            cpa[:,ktry*Nphs+n] = np.real(cp)

    # identify the intervals of interest
    prc = np.array([0.5,0.15,0.85])
    ii = (prc * Ntry*Nphs).astype(int)
    cpa.sort(axis=1)
    cpa = cpa[:,ii]

    return cpa,freq,prc


#-----------PLOT THE STACKED LFES-----------------------------------

def pickdur(sta,frcsig=0.95,fmax=20.,twin=[0.1,3.9]):
    """
    :param       sta: duration
    :param    frcsig: fraction of the power to identify
    :param      fmax: lowpass filter to apply
    :return     durs: a dictionary of the durations with some 
                       fraction of the signal
    """

    kys=[tr.get_id() for tr in sta]
    durs=dict.fromkeys(kys)

    sta=sta.copy().filter('lowpass',freq=fmax)

    twin=np.atleast_1d(twin)
    tns=twin-6.
    
    for tr in sta:
        # grab the intervals
        tref=tr.stats.starttime+tr.stats.t3
        tri=tr.copy().trim(starttime=tref+twin[0],
                           endtime=tref+twin[1])
        trn=tr.copy().trim(starttime=tref+tns[0],
                           endtime=tref+tns[1])

        # the average noise power
        npow=np.median(np.power(trn.data,2))

        # the cumulative data power
        dpow=np.cumsum(np.power(tri.data,2)-npow)
        mpow=np.max(dpow)
        dpow=dpow/mpow


        # where to stop
        istop,=np.where(dpow>=frcsig)
        ii=istop[0]+np.arange(-1,1)
        
        # timing
        duri=np.interp(frcsig,dpow[ii],tri.times()[ii])
        durs[tr.get_id()]=duri

        #print((npow/tr.stats.delta)/(mpow*frcsig/duri))

    return durs


def taperbydur(sta,tdec=0.3,durs=None):
    """
    :param    sta: templates to taper
    :param   tdec: apply a cosine taper over this interval
    :param   durs: dictionary of durations, if known
    """

    # compute durations
    twin=[-.1,3.9]
    if durs is None:
        durs=pickdur(sta,twin=twin)

    for tr in sta:
        # timing
        tms=tr.times()-tr.stats.t3

        # end of template
        tmax=twin[0]+durs[tr.get_id()]

        print(tmax)
        # zero after
        tr.data[tms>(tmax+tdec)]=0.

        # taper inside
        ii=np.logical_and(tms>=tmax,tms<=tmax+tdec)
        scl=(tms[ii]-tmax)/tdec
        scl=(np.cos(scl*np.pi)+1)/2.
        tr.data[ii]=np.multiply(tr.data[ii],scl)

def extendtemp(sta,odur=0.2,ndur=0.3):
    """
    :param      sta: templates to filter
    :param     odur: the original earthquake duration
    :param     ndur: the new earthquake duration
    """

    # get transfer function
    dtim = sta[0].stats.delta
    tr = scalefun(odur=odur,ndur=ndur,dtim=dtim)
    tr = tr.select(channel='TFUN')[0]
    tfun = tr.data

    # shift to center and buffer
    N = tfun.size+np.max([tr.stats.npts for tr in sta])
    tfun=np.append(tfun,np.zeros(N-tr.stats.npts))
    tfun=np.roll(tfun,-int(tr.stats.t0/tr.stats.delta))
    tfun=tfun/np.sum(np.power(tfun,2))**0.5

    # to frequency domain
    tfun=np.fft.rfft(tfun,n=N*2)

    scls = []
    
    for tr in sta:
        # convolve with each template
        Nh = tr.stats.npts
        data = np.fft.rfft(tr.data,n=N*2)

        # to cross-correlate with original
        xc = data.copy()

        # modify
        data = np.multiply(data,tfun)

        # finish x-c
        xc = np.multiply(xc,data.conj())
        xc = np.fft.irfft(xc)
            
        # back to time domain
        data = np.fft.irfft(data)[0:Nh]

        # to scale
        nml1 = np.sum(np.power(tr.data,2))**0.5
        nml2 = np.sum(np.power(data,2))**0.5
        scl = np.max(xc)**0.5/nml1

        scl = nml2 / nml1
        scls.append(scl)

        # save
        tr.data = data

    scls = np.median(scls)
    for tr in sta:
        tr.data = tr.data / scls
        

def scalefunspec(odur=0.2,ndur=0.3,flm=[2,30],dtim=0.01):
    """
    :param     odur: the original earthquake duration
    :param     ndur: the new earthquake duration
    :param      flm: the frequency limits to filter to
    :param     dtim: time spacing
    """

    # upsample for the beginning
    dtim2=dtim/10.

    # make Hann windows for the two values
    iodur=int(np.round(odur/dtim2))
    indur=int(np.round(ndur/dtim2))
    iodur=iodur + (iodur % 2) - 1
    indur=indur + (indur % 2) - 1
    vlo = np.hanning(iodur)
    vln = np.hanning(indur)
    vlo = vlo/np.sum(vlo)
    vln = vln/np.sum(vln)
    
    # to buffer with a long window
    mdur=np.maximum(odur,ndur)
    N = int(mdur*2./dtim2)
    N = N + (N % 2) - 1
    vlo=np.hstack([np.zeros(int((N-iodur)/2)),vlo,
                   np.zeros(int((N-iodur)/2))])
    vln=np.hstack([np.zeros(int((N-indur)/2)),vln,
                   np.zeros(int((N-indur)/2))])
    

    # compute tapers
    [tprs,V] = spectrum.mtm.dpss(vlo.size,NW=6)
            
    # just select some?
    ii = V>=0.95
    tprs = tprs[:,ii]

    # spectra
    Nf=vlo.size*2
    vlfo=np.fft.rfft(np.multiply(tprs,vlo.reshape([vlo.size,1])),
                     axis=0,n=Nf)
    vlfn=np.fft.rfft(np.multiply(tprs,vln.reshape([vln.size,1])),
                     axis=0,n=Nf)
    vlfo=np.power(np.mean(np.power(vlfo,2),axis=1),0.5)
    vlfn=np.power(np.mean(np.power(vlfn,2),axis=1),0.5)
    freq=np.fft.rfftfreq(n=Nf,d=dtim2)
    plt.loglog(freq,np.abs(vlfo))
    plt.loglog(freq,np.abs(vlfn))
    
    # water level in frequency band
    ii=np.logical_and(freq>=flm[0],freq<=flm[1])
    scl=np.abs(vlfo)
    mscl=np.max(scl[ii])*1e-2
    scl=np.maximum(np.divide(mscl,scl),1)
    vlfo=np.multiply(vlfo,scl)

    plt.loglog(freq,np.abs(vlfo))

    # deconvolve the original
    vl = np.divide(vlfn,vlfo)
    vl[np.isnan(vl)] = 0.
    vlp=np.multiply(vlfo,vl)

    plt.loglog(freq,np.abs(vl))

    vl = np.fft.irfft(vl)
    vlp = np.fft.irfft(vlp)

    # shift middle to zero
    vl = np.roll(vl,int((N-1)/2))

    trt=obspy.Trace(data=vl.copy())
    trt.stats.delta=dtim2
    trt.stats.channel='TFUN'

    trp=obspy.Trace(data=vlp.copy())
    trp.stats.delta=dtim2
    trp.stats.channel='NPRD'
    
    st = obspy.Stream([trt,trp])
    #trt.filter('bandpass',freqmin=flm[0],freqmax=10.,zerophase=True)
    #trt.filter('lowpass',freq=8.)

    # and downsample
    st.resample(sampling_rate=1./dtim,no_filter=True)

    # set zero
    for tr in st:
        tr.stats.t0=N/2*dtim2
    
    return st

def scalefun(odur=0.2,ndur=0.3,flm=[2,30],dtim=0.01):
    """
    :param     odur: the original earthquake duration
    :param     ndur: the new earthquake duration
    :param      flm: the frequency limits to filter to
    :param     dtim: time spacing
    """

    # upsample for the beginning
    dtim2=dtim/10.

    # make Hann windows for the two values
    iodur=int(np.round(odur/dtim2))
    indur=int(np.round(ndur/dtim2))
    iodur=iodur + (iodur % 2) - 1
    indur=indur + (indur % 2) - 1
    vlo = np.hanning(iodur)
    vln = np.hanning(indur)
    vlo = vlo/np.sum(vlo)
    vln = vln/np.sum(vln)
    
    # to buffer with a long window
    N = int(20./dtim2)
    N = N + (N % 2) - 1
    vlo=np.hstack([np.zeros(int((N-iodur)/2)),vlo,
                   np.zeros(int((N-iodur)/2))])
    vln=np.hstack([np.zeros(int((N-indur)/2)),vln,
                   np.zeros(int((N-indur)/2))])
    
    tro=obspy.Trace(data=vlo.copy())
    tro.stats.delta=dtim2
    tro.stats.channel='OSTF'
    trn=obspy.Trace(data=vln.copy())
    trn.stats.delta=dtim2
    trn.stats.channel='NSTF'
    tro.filter('bandpass',freqmin=flm[0],freqmax=flm[1],zerophase=True)
    trn.filter('bandpass',freqmin=flm[0],freqmax=flm[1],zerophase=True)
    
    # FT with a long window
    vlo,vln=tro.data.copy(),trn.data.copy()
    vlo=np.fft.rfft(vlo,n=N)
    vln=np.fft.rfft(vln,n=N)

    # water level in frequency band
    freq=np.fft.rfftfreq(n=N,d=dtim2)
    ii=np.logical_and(freq>=flm[0],freq<=flm[1])
    scl=np.abs(vlo)
    mscl=np.max(scl[ii])*1e-2
    scl=np.maximum(np.divide(mscl,scl),1)
    vlo=np.multiply(vlo,scl)

    # deconvolve the original
    vl = np.divide(vln,vlo)
    vl[np.isnan(vl)] = 0.
    vlp=np.multiply(vlo,vl)

    vl = np.fft.irfft(vl)
    vlp = np.fft.irfft(vlp)

    # shift middle to zero
    vl = np.roll(vl,int((N-1)/2))

    trt=obspy.Trace(data=vl.copy())
    trt.stats.delta=dtim2
    trt.stats.channel='TFUN'

    trp=obspy.Trace(data=vlp.copy())
    trp.stats.delta=dtim2
    trp.stats.channel='NPRD'
    
    st = obspy.Stream([tro,trn,trt,trp])
    trt.filter('bandpass',freqmin=flm[0],freqmax=8.,zerophase=True)
    #trt.filter('lowpass',freq=8.)

    # and downsample
    st.resample(sampling_rate=1./dtim,no_filter=True)

    # set zero
    for tr in st:
        tr.stats.t0=N/2*dtim2
    
    return st

def plottemplates(ps,xc=None,cm='*',pk='t3'):

    # copy the templates
    st = ps.stam.copy()
    
    # component
    st1=st.select(channel=cm)

    # plot
    plt.close()
    f = plt.figure(figsize=(10,8))
    gs,p=gridspec.GridSpec(1,1),[]
    gs.update(left=0.03,right=0.46)
    gs.update(bottom=0.06,top=0.97)
    gs.update(hspace=0.1,wspace=0.2)
    p1=plt.subplot(gs[0])
    
    gs,pm=gridspec.GridSpec(2,1),[]
    gs.update(left=0.55,right=0.98)
    gs.update(bottom=0.06,top=0.97)
    gs.update(hspace=0.07,wspace=0.2)
    for gsi in gs:
        pm.append(plt.subplot(gsi))
    p=np.append([p1],pm)
    pm=np.array(pm).reshape([2,1])

    fs,fs2 = 'large','medium'

    # filter
    #st2 = st1.copy().filter('bandpass',freqmin=2.,freqmax=8.)
    st2 = st1.copy().filter('lowpass',freq=15.)
    N = len(st2)
    cols = graphical.colors(N,lgt=False)
    cols2 = graphical.colors(N,lgt=True)

    # vertical shifts
    shfs=-np.arange(0.,float(N),1.)*2.*1.1
    ylm = general.minmax(shfs,bfr=1.5)
    ylm = general.minmax(shfs)+np.array([-1.,1.])*\
        np.median(np.abs(np.diff(shfs)))
    trange=np.array([-.2,2.8])
    trange=np.array([-.4,4.6])
    trange=np.array(ps.trange)
    trange=np.array(ps.txc)
    xx = trange[np.array([0,1,1,0,0])]
    yy = general.minmax(ylm,1.5)[np.array([0,0,1,1,0])]
    ply = Polygon(np.vstack([xx,yy]).transpose())
    ply.set_edgecolor('none')
    ply.set_color('gray')
    ply.set_alpha(0.3)
    p[0].add_patch(ply)
    p[0].set_ylim(ylm)
    p[2].plot(ps.blim,np.array([1.,1.])*ps.minsnr,
              color='gray',linewidth=6,zorder=0)

    # limits
    sglm = np.array([],dtype=float)
    snrlm = np.array([],dtype=float)

    nswin = np.atleast_1d(trange)-trange[1]-1.
    tlm = np.array([-2,5.])
    tlmp = np.array([-4,5.])
    xvl = general.minmax(tlmp,0.9)[0]

    lat = np.array([tr.stats.sac.stla for tr in st2])
    lon = np.array([tr.stats.sac.stlo for tr in st2])
    refloc=np.array([lon[0],lat[0]])
    dx=-(lat-refloc[1])+np.cos(refloc[1]*np.pi/180)*(lon-refloc[1])
    dx=dx*111.
    ixp = np.argsort(dx)

    durs = pickdur(st1)
    
    for k in range(0,N):
        tr = st2[ixp[k]]
        tri = st1.select(id=tr.get_id())[0]

        # for normalization
        tchk = [tlm[0],1.]
        tchk = trange
        tpk = tr.stats.starttime+tr.stats[pk]
        trn = tr.copy().trim(starttime=tr.stats.starttime+
                             tr.stats[pk]+tchk[0],
                             endtime=tr.stats.starttime+
                             tr.stats[pk]+tchk[1])
        trn = np.max(np.abs(trn.data))

        #trange=trange[0]+np.array([0,durs[tr.get_id()]])
        
        # plot
        p[0].plot(tr.times()-tr.stats[pk],tr.data/trn+shfs[k],
                  color=cols[k])

        p[0].text(xvl,shfs[k]-np.median(np.abs(np.diff(shfs)))*0.5,
                  tr.stats.network+'.'+tr.stats.station+'.'+tr.stats.channel,
                  backgroundcolor='none')

        # spectra and noise
        sg,ns,snr,freq=pksdrops.getsnr(tri,wlen=trange,pk=pk,nsi=[nswin])        
        ns = np.mean(ns,axis=1)

        ns=np.power(ns,0.5)
        sg=np.power(sg,0.5)
        snr=np.power(snr,0.5)

        ii = np.logical_and(freq>=0.5,freq<=50.)
        nm = np.max(sg[ii])

        sglm = general.minmax(np.append(sglm,sg[ii]/nm))
        sglm = general.minmax(np.append(sglm,ns[ii]/nm))
        snrlm = general.minmax(np.append(snrlm,snr[ii]))
        p[1].plot(freq,ns/nm,color=cols2[k],zorder=2)
        p[1].plot(freq,sg/nm,color=cols[k],zorder=3)
        p[2].plot(freq,snr,color=cols[k])


    sglm = np.exp(general.minmax(np.log(sglm),1.1))
    sglm = np.array([1.e-3,2.])
    snrlm = np.exp(general.minmax(np.log(snrlm),1.1))
    snrlm = np.array([0.1,100])
    p[1].set_xscale('log')
    p[1].set_yscale('log')
    xlm=np.array([0.7,30])
    p[1].set_xlim(xlm)
    p[2].set_xlim(xlm)
    p[2].set_xscale('log')
    p[2].set_yscale('log')
    for ph in p:
        ph.yaxis.set_tick_params(labelsize=fs2)
        ph.xaxis.set_tick_params(labelsize=fs2)

    p[1].set_ylim(sglm)
    p[2].set_ylim(snrlm)
    p[0].set_xlim(tlmp)
    p[0].set_xlabel('time since pick (s)',fontsize=fs)
    p[0].set_yticklabels([])
    #p[1].set_ylabel('normalized power spectra (dB)')
    p[1].set_ylabel('normalized amplitude',fontsize=fs)
    p[1].set_xlabel('frequency (Hz)',fontsize=fs)
    p[2].set_xlabel('frequency (Hz)',fontsize=fs)
    p[2].set_ylabel('observed amplitude / noise amplitude',fontsize=fs)
    graphical.delticklabels(pm)

    nm = "{0:0.0f}".format(tr.stats.t9)
    graphical.cornerlabels(p[0:1],'ul',fontsize=fs2,xscl=0.03,yscl=0.015)
    graphical.cornerlabels(p[1:],'ul',fontsize=fs2,scl=0.03,lskip=1)
    graphical.printfigure('PKLstacks_'+nm,plt.gcf())


def calcvals(ps,plotnoise=False,flm=None,mxdf=None):
    if not plotnoise:
        # coherent energy
        cp = ps.cp
        
        # direct coherence
        dcp = np.real(ps.fxc[:,:,:,0])
        dcp=np.mean(dcp,axis=1)
        dcp=np.multiply(np.power(dcp,2),np.sign(dcp))
        
        # total energy
        en = ps.en
        
        # noise
        ns=[vl.reshape([1,vl.shape[0],vl.shape[1]]) for vl in ps.enn]
        ns = np.vstack(ns)
        ns = np.mean(ns,axis=0)
        
    else:
        k=2
        cp = ps.cpn[k]
        en = ps.enn[k]
        ns = np.ones(cp.shape)*1.e-8
        dcp = np.ones(cp.shape)*1.e-8

    # scaling?
    rscl = True
    if rscl:
        ixc=general.closest(ps.tms,ps.tmxc)
        rscl=ps.mscl[ixc]
        rscl=rscl.reshape([1,rscl.size])
        cp=np.divide(cp,rscl)
        en=np.divide(en,rscl)
        ns=np.divide(ns,rscl)
        dcp=np.divide(dcp,rscl)
        
    # frequencies
    freq = ps.freq
    
    # have enough stations
    ii = ps.Nucp>=5
    cp,en,ns,dcp=cp[:,ii],en[:,ii],ns[:,ii],dcp[:,ii]
    
    # just some frequencies
    ii = np.logical_and(freq>=flm[0],freq<=flm[1])
    freq,cp,en,ns,dcp=freq[ii],cp[ii,:],en[ii,:],ns[ii,:],dcp[ii,:]

    # just some events, with enough stations 
    if isinstance(cp,np.ma.masked_array):
        ii = np.sum(cp.mask,axis=0)
        ii = ii==0
        cp,en,ns,dcp=cp[:,ii],en[:,ii],ns[:,ii],dcp[:,ii]
        
    # check variability
    enl = np.log(en)
    enm = np.median(enl,axis=1)
    iok = np.abs(enl-enm.reshape([enm.size,1]))<np.log(mxdf)
    iok = np.sum(iok.astype(int),axis=0)
    ii = iok>=freq.size*0.7
    
    enl = np.log(ns)
    enm = np.median(enl,axis=1)
    iok = np.abs(enl-enm.reshape([enm.size,1]))<np.log(mxdf)
    iok = np.sum(iok.astype(int),axis=0)
    ii = np.logical_and(ii,iok>=freq.size*0.7)
    
    cp,en,ns,dcp=cp[:,ii],en[:,ii],ns[:,ii],dcp[:,ii]
    
    # only pick events with low amplitudes
    ix = np.logical_and(freq>=2,freq<=10)
    eni = np.mean(en[ix,:],axis=0)
    ii = eni<=float('inf')
    cp,en,ns,dcp=cp[:,ii],en[:,ii],ns[:,ii],dcp[:,ii]
    
    # only pick events with significant signal
    ix = np.logical_and(freq>=2,freq<=10)
    eni = np.sum(en[ix,:],axis=0)
    nsi = np.sum(ns[ix,:],axis=0)
    snr = np.divide(eni,nsi)
    
    
    ii = np.logical_and(snr>2,snr<=6)
    #cp,en,ns,dcp=cp[:,ii],en[:,ii],ns[:,ii],dcp[:,ii]
    
    nused = cp.shape[1]
    print(str(cp.shape[1])+ ' events included')
    
    # bootstrap values
    Nb = 100
    cpb,enb=np.zeros([freq.size,Nb]),np.zeros([freq.size,Nb])
    nsb,dcpb=np.zeros([freq.size,Nb]),np.zeros([freq.size,Nb])
    for k in range(0,Nb):
        ii = np.random.choice(cp.shape[1],int(cp.shape[1]*1.0))
        cpb[:,k]=np.mean(cp[:,ii],axis=1)
        enb[:,k]=np.mean(en[:,ii],axis=1)
        nsb[:,k]=np.mean(ns[:,ii],axis=1)
        dcpb[:,k]=np.mean(dcp[:,ii],axis=1)
        
    cp = np.mean(cp,axis=1)
    en = np.mean(en,axis=1)
    ns = np.mean(ns,axis=1)
    dcp = np.mean(dcp,axis=1)

    # smooth?
    fsmth = 0.5
    if fsmth:
        dfreq = np.median(np.diff(freq))
        nwl = int(np.round(fsmth*3*2/dfreq))
        nwl = nwl + 1 - (nwl%2)
        gwin=scipy.signal.gaussian(nwl,fsmth/dfreq)
        gwin=gwin/np.sum(gwin)
        cp=scipy.signal.convolve(cp,gwin,mode='same')
        en=scipy.signal.convolve(en,gwin,mode='same')
        ns=scipy.signal.convolve(ns,gwin,mode='same')
        dcp=scipy.signal.convolve(dcp,gwin,mode='same')

        for k in range(0,Nb):
            cpb[:,k]=scipy.signal.convolve(cpb[:,k],gwin,mode='same')
            enb[:,k]=scipy.signal.convolve(enb[:,k],gwin,mode='same')
            nsb[:,k]=scipy.signal.convolve(nsb[:,k],gwin,mode='same')
            dcpb[:,k]=scipy.signal.convolve(dcpb[:,k],gwin,mode='same')

    lfb = enb-nsb
    rtb = np.divide(cpb,lfb)
    drtb = np.divide(dcpb,lfb)
    cpb.sort(axis=1)
    dcpb.sort(axis=1)
    enb.sort(axis=1)
    nsb.sort(axis=1)
    lfb.sort(axis=1)
    rtb.sort(axis=1)
    drtb.sort(axis=1)
    
    return freq,cp,en,ns,dcp,cpb,enb,nsb,dcpb,lfb,rtb,drtb,nused
    

def plotxccomb(ps,prt=False,mxdf=5,plotnoise=False,adjsum=False,ps2=None):

    xccomb = False
    flm = np.array([2,20/.85])

    # extract the relevant values
    freq,cp,en,ns,dcp,cpb,enb,nsb,dcpb,lfb,rtb,drtb,nused=\
       calcvals(ps=ps,flm=flm,plotnoise=plotnoise,mxdf=mxdf)

    # extract the relevant values for a second series
    if ps2 is not None:
        freq2,cp2,en2,ns2,dcp2,cpb2,enb2,nsb2,dcpb2,lfb2,rtb2,drtb2,nused2=\
          calcvals(ps=ps2,flm=flm,plotnoise=plotnoise,mxdf=mxdf)

    Nb = cpb.shape[1]
    ifrc = 0.5+np.array([-.5,.5])*0.95
    ifrc = (ifrc*Nb).astype(int)
    imn= int(0.5*Nb)
        
    # plot
    plt.close()
    f = plt.figure(figsize=(11.5,10))
    gs,p=gridspec.GridSpec(2,1),[]
    gs.update(left=0.08,right=0.98)
    gs.update(bottom=0.07,top=0.93)
    gs.update(hspace=0.1,wspace=0.1)
    for gsi in gs:
        p.append(plt.subplot(gsi))
    p=np.array(p)
    pm=p.reshape([2,1])

    h,hbl,hbu=[],[],[]
    lw = 2.

    hh,=p[0].plot(freq,dcp,label='$P_d$: directly coherent',
                  color='darkgoldenrod',linewidth=lw)
    h.append(hh)
    x = np.append(freq,np.flipud(freq))
    y = np.append(np.minimum(dcpb[:,ifrc[1]],1e5),
                  np.flipud(np.maximum(dcpb[:,ifrc[0]],1e-5)))
    x,y=np.append(x,x[0]),np.append(y,y[0])
    y = np.minimum(np.maximum(y,1e-5),1e5)
    ply = Polygon(np.vstack([x,y]).transpose())
    ply.set_edgecolor('none')
    ply.set_color('darkgoldenrod')
    ply.set_alpha(0.3)
    p[0].add_patch(ply)
    # hh,=p[0].plot(freq,dcpb[:,ifrc[0]],linestyle='--',
    #           color='darkgoldenrod')
    # hbl.append(hh)
    # hh,=p[0].plot(freq,dcpb[:,ifrc[1]],linestyle='--',
    #           color='darkgoldenrod')
    hbu.append(hh)

    hh,=p[0].plot(freq,cp,label='$P_c$: inter-station coherent',color='r',
                  linewidth=lw)
    h.append(hh)
    x = np.append(freq,np.flipud(freq))
    y = np.append(np.minimum(cpb[:,ifrc[1]],1e5),
                  np.flipud(np.maximum(cpb[:,ifrc[0]],1e-5)))
    x,y=np.append(x,x[0]),np.append(y,y[0])
    y = np.minimum(np.maximum(y,1e-5),1e5)
    ply = Polygon(np.vstack([x,y]).transpose())
    ply.set_edgecolor('none')
    ply.set_color('r')
    ply.set_alpha(0.3)
    p[0].add_patch(ply)
    # hh,=p[0].plot(freq,cpb[:,ifrc[0]],linestyle='--',color='r')
    # hbl.append(hh)
    # hh,=p[0].plot(freq,cpb[:,ifrc[1]],linestyle='--',color='r')
    # hbu.append(hh)

    hh,=p[0].plot(freq,en,label='$P_t$: total (incl. noise)',color='green',
                  linewidth=lw)
    h.append(hh)
    x = np.append(freq,np.flipud(freq))
    y = np.append(np.minimum(enb[:,ifrc[1]],1e5),
                  np.flipud(np.maximum(enb[:,ifrc[0]],1e-5)))
    x,y=np.append(x,x[0]),np.append(y,y[0])
    y = np.minimum(np.maximum(y,1e-5),1e5)
    ply = Polygon(np.vstack([x,y]).transpose())
    ply.set_edgecolor('none')
    ply.set_color('green')
    ply.set_alpha(0.3)
    p[0].add_patch(ply)
    # hh,=p[0].plot(freq,enb[:,ifrc[0]],linestyle='--',color='green')
    # hbl.append(hh)
    # hh,=p[0].plot(freq,enb[:,ifrc[1]],linestyle='--',color='green')
    # hbu.append(hh)

    hh,=p[0].plot(freq,ns,label='$P_n$: noise',color='gray',linewidth=lw)
    h.append(hh)
    x = np.append(freq,np.flipud(freq))
    y = np.append(np.minimum(nsb[:,ifrc[1]],1e5),
                  np.flipud(np.maximum(nsb[:,ifrc[0]],1e-5)))
    x,y=np.append(x,x[0]),np.append(y,y[0])
    y = np.minimum(np.maximum(y,1e-5),1e5)
    ply = Polygon(np.vstack([x,y]).transpose())
    ply.set_edgecolor('none')
    ply.set_color('gray')
    ply.set_alpha(0.3)
    p[0].add_patch(ply)
    # hh,=p[0].plot(freq,nsb[:,ifrc[0]],linestyle='--',color='gray')
    # hbl.append(hh)
    # hh,=p[0].plot(freq,nsb[:,ifrc[1]],linestyle='--',color='gray')
    # hbu.append(hh)

    hh,=p[0].plot(freq,en-ns,label='$P_l$: LFE (total - noise)',color='b',linewidth=lw)
    h.append(hh)
    x = np.append(freq,np.flipud(freq))
    y = np.append(np.minimum(lfb[:,ifrc[1]],1e5),
                  np.flipud(np.maximum(lfb[:,ifrc[0]],1e-5)))
    y = np.minimum(np.maximum(y,1e-5),1e5)
    x,y=np.append(x,x[0]),np.append(y,y[0])
    ply = Polygon(np.vstack([x,y]).transpose())
    ply.set_edgecolor('none')
    ply.set_color('b')
    ply.set_alpha(0.3)
    ply.set_zorder(0)
    p[0].add_patch(ply)
    # hh,=p[0].plot(freq,lfb[:,ifrc[0]],linestyle='--',color='b')
    # hbl.append(hh)
    # hh,=p[0].plot(freq,lfb[:,ifrc[1]],linestyle='--',color='b')
    # hbu.append(hh)

    fs = 14
    lg1=p[0].legend(loc='upper left',fontsize=fs-1)
    p[0].set_yscale('log')
    p[0].set_ylim([0.1,3.e4])
    
    # signal to noise
    ps.xcsnr()

    h2,hbl2,hbu2=[],[],[]
    hh,=p[1].plot(ps.snrfreq,np.power(ps.snr,1),color='gray',
                 linewidth=lw)
    h2.append(hh)

    hrefs=[]
    hh,=p[1].plot(flm,[0,0],color='k',linestyle=':')
    hrefs.append(hh)
    hh,=p[1].plot(flm,[0.6,0.6],color='k',linestyle=':')
    hrefs.append(hh)
    hh,=p[1].plot(flm,[1,1],color='k',linestyle=':')
    hrefs.append(hh)

    # estimate the scatter from the estimated locations
    tshfs=np.vstack([ps.tshfs[tr.get_id()] for tr in ps.stam])
    mds = np.median(tshfs,axis=0).reshape([1,tshfs.shape[1]])
    tshfs = tshfs - mds
    tshfu = np.std(tshfs,axis=1)
    #tshfu=(np.random.rand(len(ps.stam))-0.5)*0.01+0.01
    cpa,freqc,prc = cpfromamptshf(tshfu)


    hh,=p[1].plot(freqc,cpa[:,0]*0.75**2,label='time shifted',color='gray',
                  linewidth=lw)
    h2.append(hh)

    hh,=p[1].plot(flm,np.ones(2)*0.75**2,color='gray',linewidth=lw)
    h2.append(hh)

    hh,=p[1].plot(freq,np.divide(cp,en-ns),color='r',linewidth=lw)
    h2.append(hh)
    x = np.append(freq,np.flipud(freq))
    y = np.append(np.minimum(rtb[:,ifrc[1]],5),
                  np.flipud(np.maximum(rtb[:,ifrc[0]],-1)))
    y = np.minimum(np.maximum(y,-1),5)
    x,y=np.append(x,x[0]),np.append(y,y[0])
    ply = Polygon(np.vstack([x,y]).transpose())
    ply.set_edgecolor('none')
    ply.set_color('r')
    ply.set_alpha(0.3)
    ply.set_zorder(2)
    p[1].add_patch(ply)
    # hh,=p[1].plot(freq,rtb[:,ifrc[0]],linestyle='--',color='r')
    # hbl2.append(hh)
    # hh,=p[1].plot(freq,rtb[:,ifrc[1]],linestyle='--',color='r')
    # hbu2.append(hh)


    hh,=p[1].plot(freq,np.divide(dcp,en-ns),color='darkgoldenrod',
                 linewidth=lw)
    h2.append(hh)
    x = np.append(freq,np.flipud(freq))
    y = np.append(np.minimum(drtb[:,ifrc[1]],5),
                  np.flipud(np.maximum(drtb[:,ifrc[0]],-1)))
    y = np.minimum(np.maximum(y,-1),5)
    x,y=np.append(x,x[0]),np.append(y,y[0])
    ply = Polygon(np.vstack([x,y]).transpose())
    ply.set_edgecolor('none')
    ply.set_color('darkgoldenrod')
    ply.set_alpha(0.3)
    ply.set_zorder(1)
    p[1].add_patch(ply)
    # hh,=p[1].plot(freq,drtb[:,ifrc[0]],linestyle='--',color='darkgoldenrod')
    # hbl2.append(hh)
    # hh,=p[1].plot(freq,drtb[:,ifrc[1]],linestyle='--',color='darkgoldenrod')
    # hbu2.append(hh)

    h2=np.array(h2)[np.array([4,3,0,1,2])]
    
    # p[1].legend(h2,['direct','inter-station','effect of noise'],
    #             fontsize=fs-1,loc='lower left')
    lg=p[1].legend(h2[np.array([0,1])],['$P_d/P_l$: directly coherent',
                                        '$P_c/P_l$: inter-station coherent'],
                   fontsize=fs-1,loc='lower left')
    h2[2].remove()
    h2[4].remove()
    h2[3].remove()
    lg.set_alpha(1.)

    xlm = np.array([flm[0],flm[1]*.85])
    p[0].set_yscale('log')
    xtk=np.arange(2,xlm[1]+0.0001)
    xtkl=['']*xtk.size
    ix,=np.where([xtki in [2,5,10,15,20] for xtki in xtk])
    for ixi in ix:
        xtkl[ixi]='{:0.0f}'.format(xtk[ixi])
    for ph in p:
        ph.set_xscale('log')
        ph.set_xlim(xlm)
        #ph.set_yscale('log')
        ph.yaxis.set_tick_params(labelsize=fs)
        ph.xaxis.set_tick_params(labelsize=fs)
        ph.set_xticks(xtk)
        ph.minorticks_off()
        ph.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
        ph.set_xticklabels(xtkl)
        
    p[0].set_xticklabels([])

    p[1].set_ylim(np.array([-0.1,1.3]))
    #p[0].set_xlabel('frequency (Hz)',fontsize=fs)
    p[1].set_xlabel('frequency (Hz)',fontsize=fs)
    p[1].set_ylabel('coherent power / LFE power',fontsize=fs)
    p[0].set_ylabel('template-normalized power',fontsize=fs)
    #p[0].set_ylim([1,1000])

    xtk=np.arange(1,20)*100.
    xtkl=['']*xtk.size
    ix,=np.where([xtki in [300,500,1000] for xtki in xtk])
    for ixi in ix:
        xtkl[ixi]='{:0.0f}'.format(xtk[ixi])

    # add diameters to plot
    raxa=[]
    for k in range(0,2):
        rax = p[k].twiny()
        rax.set_xlim(np.divide(1.,np.flipud(xlm))*2.1*4000.)
        rax.set_xscale('log')
        rax.tick_params(axis='x',labelsize=fs)
        rax.invert_xaxis()
        rax.set_xticks(xtk)
        rax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
        rax.xaxis.set_tick_params(labelsize=fs)
        rax.yaxis.set_tick_params(labelsize=fs)
        if k==0:
            rax.set_xticklabels(xtkl)
            rax.set_xlabel('maximum diameter (m)',fontsize=fs)
        else:
            rax.set_xticklabels([])
        raxa.append(rax)

    hb=graphical.cornerlabels(p[0:1],'ll',fontsize=fs-2,xscl=0.01,yscl=0.03)
    hbi=graphical.cornerlabels(p[1:2],'ul',fontsize=fs-2,
                               xscl=0.01,yscl=0.03,lskip=1)
    hb=np.append(hb,hbi)
    lbl = str(nused)+' LFEs from family '+str(ps.fnum)
    xvl=np.exp(general.minmax(np.log(p[0].get_xlim()),0.93)[0])
    yvl=np.exp(general.minmax(np.log(p[0].get_ylim()),0.9)[0])
    ht=p[0].text(xvl,yvl,lbl,fontsize=fs-2,horizontalalignment='left',
                 verticalalignment='center')
                 
    chlb=np.unique(np.array([tr.stats.channel for tr in ps.stam]))
    chlb='-'.join(chlb)
    lbl=''
    if ps.amplim is not None:
        lbl=lbl+'amps'+str(ps.amplim[0])+'-'+str(ps.amplim[1])
    lbl=lbl+'_txc'+str(ps.txc[0])+'-'+str(ps.txc[1])
    #lbl=lbl+'_Nev'+str(nused)
    lbl=lbl+'_'+chlb
    lbl=lbl+'_'+str(mxdf)
    if ps.eqloc is None:
        lbl=lbl+'_noshift'
    else:
        lbl=lbl+'_shifted'
    fname='PCxccomb_'+str(ps.fnum)+'_'+lbl
    fname=fname+'_stack'+str(ps.iterstack)
    if ps.usemshf:
        fname=fname+'_mshf'
    elif ps.randxshf or ps.randzshf:
        fname=fname+'_shf{:0.2f}-{:0.2f}'.format(ps.randxshf,ps.randzshf)
    if not ps.rscl:
        fname=fname+'_uscl'
    fname=fname.replace('.','p')
    

    if ps2 is not None:

        hold2=[]
        hh,=p[1].plot(freq,np.divide(dcp2,en2-ns2),color='darkgoldenrod',
                      linewidth=lw,linestyle='--',zorder=0)
        hold2.append(hh)

        hh,=p[1].plot(freq,np.divide(cp2,en2-ns2),color='r',
                      linewidth=lw,linestyle='--',zorder=0)
        hold2.append(hh)

        hold=[]
        hh,=p[0].plot(freq2,dcp2,label='$P_d$: directly coherent',
                      color='darkgoldenrod',linewidth=lw,linestyle='--',
                      zorder=0)
        hold.append(hh)

        hh,=p[0].plot(freq2,cp2,
                      color='r',linewidth=lw,linestyle='--',
                      zorder=0)
        hold.append(hh)

        hh,=p[0].plot(freq2,en2,
                      color='g',linewidth=lw,linestyle='--',
                      zorder=0)
        hold.append(hh)

        hh,=p[0].plot(freq2,ns2,
                      color='gray',linewidth=lw,linestyle='--',
                      zorder=0)
        hold.append(hh)

        hh,=p[0].plot(freq2,en2-ns2,
                      color='blue',linewidth=lw,linestyle='--',
                      zorder=0)
        hold.append(hh)

        lg = p[1].add_artist(lg)
        #ax.set_bbox_to_anchor((ps.x1,ps.y0),transform=plt.gcf().transFigure)
        lg2=p[1].legend([h2[1],hold2[1]],['shifted LFE locations','original'],
                        loc='upper right',fontsize=fs-1)

    print(fname)
    if prt:        
        graphical.printfigure(fname,f)

        fname=os.path.join(os.environ['DATA'],'TREMORAREA','Results',fname)
        fl = open(fname,'w')
        # write the number of LFEs
        vls='{:0.0f}'.format(nused)
        fl.write(vls+'\n')
        
        # write the frequencies
        vls=['{:0.3f}'.format(vl) for vl in freq]
        fl.write(','.join(vls)+'\n')

        # write the inter-station coherence
        vls=['{:0.3f}'.format(vl) for vl in np.divide(cp,en-ns)]
        fl.write(','.join(vls)+'\n')
        vls=['{:0.3f}'.format(vl) for vl in rtb[:,ifrc[0]]]
        fl.write(','.join(vls)+'\n')
        vls=['{:0.3f}'.format(vl) for vl in rtb[:,ifrc[1]]]
        fl.write(','.join(vls)+'\n')
        
        # write the direct coherence
        vls=['{:0.3f}'.format(vl) for vl in np.divide(dcp,en-ns)]
        fl.write(','.join(vls)+'\n')
        vls=['{:0.3f}'.format(vl) for vl in drtb[:,ifrc[0]]]
        fl.write(','.join(vls)+'\n')
        vls=['{:0.3f}'.format(vl) for vl in drtb[:,ifrc[1]]]
        fl.write(','.join(vls)+'\n')
        fl.close()

        # h[0].remove()
        # h[2].remove()
        # h[3].remove()

        # hbl[0].remove()
        # hbl[2].remove()
        # hbl[3].remove()

        # #hbu[0].remove()
        # #hbu[2].remove()
        # #hbu[3].remove()


        # p[0].legend(loc='upper left',fontsize=fs-1)

        # # the noise
        # h2[0].remove()        
        # #h2[2].remove()        
        # hbl2[1].remove()        
        # #hbu2[1].remove()        
        # #h2[3].remove()        
        # #h2[4].remove()        


        # #p[1].legend([h2[1]],['inter-station coherent'],fontsize=fs-1)    
        # # p[1].legend([h2[1],h2[3]],
        # #             ['inter-station coherent',
        # #              'point source adjusted for noise'],fontsize=fs-1)    
        
        # fname = fname+'_simp'
        # graphical.printfigure(fname,f)
    if adjsum:
        for hh in hb:
            hh.remove()
        for hh in hrefs:
            hh.remove()
        lg1.remove()
        #lg1=p[0].legend(loc='upper left',fontsize=8)
        lg.remove()
        ht.remove()

    return f,p,raxa,fname,nused

def corrsaved(stsv,ilmsv,tget,tms):
    """
    :param      stsv: saved dataset
    :param     ilmsv: indices dividing intervals
    :param      tget: the time limits
    :param       tms: event times
    """

    if isinstance(stsv,obspy.Stream):
        stsv=[stsv]
    
    tref = obspy.UTCDateTime(2010,1,1)
    tdf = tms-tref

    # map to event time and varying time within
    ilm=int(np.median(np.diff(ilmsv)))
    tvl=tget[0]+ilm*np.arange(0,ilm)*stsv[0][0].stats.delta
    i1,i2=np.meshgrid(np.arange(0,ilm),np.arange(0,ilmsv.size-1))
    i1,i2=i1.flatten(),i2.flatten()

    # the time of each point
    tm=tdf[i2]+tvl[i1]

    # create time waveforms
    tms = stsv[0].copy()
    for tr in tms:
        tr.data=tm.copy()

    for stsvi in stsv:
        corrhrsn.corrdata(stsvi,tref=tref,tms=tms)
        
def grabmanylfes(sta,tms,rnoise=False,csyn=None,cwgts=None,
                 tget=None,realign=False,flm=None,samp20=False):
    """
    :param      sta:  the stack of LFEs
    :param      tms:  the times to consider
    :param   rnoise:  replace all the data with random noise 
                         (default: false)
    :param     csyn:  a synthetic coherent fraction to use
                         (default: None---use actual data)
    :param    cwgts:  to construct a new set of components
    :param     tget:  time range to get for each lfe
    :param  realign:  realign with cross-correlation
    :param      flm:  frequencies to filter to before extracting
    :return    stsv:  all the LFE waveforms
    :return     tsv:  times of the LFEs considered
    :return   stsvn:  a list of all the LFE waveforms at shifted times
    """

    # family number
    try:
        fnum = sta[0].stats.t9
    except:
        fnum=0

    # only read the relevant stations
    if not samp20:
        stns = [tr.stats.network+'.'+tr.stats.station+'.'+tr.stats.channel
                for tr in sta]
    else:
        stns = [tr.stats.network+'.'+tr.stats.station+'.'+\
                tr.stats.channel.replace('DP','BP')
                for tr in sta]
        stns = stns+ [tr.stats.network+'.'+tr.stats.station+'.'+\
                      tr.stats.channel.replace('DP','SP')
                      for tr in sta]
    stns = np.unique(np.array(stns))

    # if needed to realign
    if realign:
        star = sta.copy()
        msk = seisproc.prepfiltmask(star,tmask=3.)
        star.filter('bandpass',freqmin=2.,freqmax=8.)
        seisproc.addfiltmask(star,msk)
    
    # initialize with response corrections
    invi = obspy.Inventory([],'T','T')
    invi = responsecorr.readresponse(sta,invi)

    # pick noise intervals
    noiseint=phscoh.defnoiseint(trange=tget,N=3,allshf=-5.)

    # spacing of intervals to add
    tget=np.atleast_1d(tget)
    Nper=int(np.round(np.diff(tget)[0]/sta[0].stats.delta))
    Ntot=Nper*len(tms)
    ilm = np.arange(0,Ntot+1,Nper)

    # initialize set to output
    stsv = sta.copy()
    if samp20:
        for tr in stsv:
            tr.stats.channel.replace('DP','BP')
            tr.stats.channel.replace('SP','BP')
    
    for tr in stsv:
        tr.data=np.ma.masked_array(np.zeros(Ntot,dtype=float),mask=True)

    # and for noise
    stsvn = []
    for k in range(0,len(noiseint)):
        stsvn.append(stsv.copy())

    for itm in range(0,len(tms)):
        tm=tms[itm]
        i1,i2=ilm[itm],ilm[itm+1]
        print(tm)

        # grab the data
        st,invi = grablfedata(tm,stns=stns,invi=invi,remresp=1)

        # alternative components
        if cwgts:
            st = maxcomp(st,cwgts)

        # replace with Gaussian noise
        if rnoise:
            for tr in st:
                tr.data=np.random.randn(tr.data.shape[0])
        elif csyn is not None:
            print('Synthetics: '+str(csyn))
            # add the pick data
            for tr in st:
                tr.stats.t1=tm-tr.stats.starttime
            readpicks(fnum=fnum,sta=st,pk='t3',pkref='t1')
            
            csyn = np.atleast_1d(csyn)
            if csyn.size==1:
                csyn=np.append(csyn,100.)

            # filter 
            msk = seisproc.prepfiltmask(st,tmask=3.)
            st.filter('bandpass',freqmin=flm[0],freqmax=flm[1])
            seisproc.addfiltmask(st,msk)

            # add synthetic signals
            csynshf = 15.

            #for tr in st:
            #tr.data=np.zeros(tr.stats.npts,dtype=float)
            tscl=np.array([-.2,2.8])
            st = makesyndata(sta,st,csyn=csyn[0],tst=-0.3,flm=0.,
                             pk='t3',tlen=1.,tshf=csynshf,amp=csyn[1],
                             stfl=0.15,tscl=tscl)


        if st:
            print('Grabbing LFE data')
            
            if csyn is None:
                # filter 
                msk = seisproc.prepfiltmask(st,tmask=3.)
                st.filter('bandpass',freqmin=flm[0],freqmax=flm[1])
                seisproc.addfiltmask(st,msk)

            # add the pick data
            for tr in st:
                tr.stats.t1=tm-tr.stats.starttime
            readpicks(fnum=fnum,sta=st,pk='t3',pkref='t1')
            if csyn is not None:
                for tr in st:
                    tr.stats.t3=tr.stats.t3-csynshf

            # if needed to realign
            if realign:
                # filter to limits of interest
                stre = st.copy()
                msk = seisproc.prepfiltmask(stre,tmask=3.)
                stre.filter('bandpass',freqmin=2.,freqmax=8.)
                seisproc.addfiltmask(stre,msk)

                # new alignment
                realignwv(star,stre,pk='t3')

                # reassign picks to original traces
                for tr in st:
                    tr2=stre.select(station=tr.stats.station,
                                    network=tr.stats.network,
                                    channel=tr.stats.channel)[0]
                    tr.stats.t3=tr2.stats.t3

            # get rid of location labels
            for tr in st:
                tr.stats.location=''
            
            # trim as necessary for noise
            for k in range(0,len(noiseint)):
                stn = st.copy()
                for tr in stn:
                    t1=tr.stats.starttime+tr.stats.t3+noiseint[k][0]
                    t2=tr.stats.starttime+tr.stats.t3+\
                        noiseint[k][1]+3*tr.stats.delta
                    seisproc.trimandshift(tr,starttime=t1,endtime=t2,
                                          pad=True)
                    tr.data=tr.data[0:Nper]
                    tr.stats.t3=tr.stats.t3+noiseint[k][0]-tget[0]
                trashmasked(stn)

                # add to the output trace
                for tr in stn:
                    tro=stsvn[k].select(station=tr.stats.station,
                                        channel=tr.stats.channel,
                                        network=tr.stats.network)[0]
                    tro.data.mask[i1:i2]=False
                    tro.data[i1:i2]=tr.data

            # trim as necessary 
            for tr in st:
                t1=tr.stats.starttime+tr.stats.t3+tget[0]
                t2=tr.stats.starttime+tr.stats.t3+tget[1]+\
                    3*tr.stats.delta
                seisproc.trimandshift(tr,starttime=t1,endtime=t2,pad=True)
                tr.data=tr.data[0:Nper]
            trashmasked(st)

            # add to the output trace
            for tr in st:
                tro=stsv.select(station=tr.stats.station,
                                channel=tr.stats.channel,
                                network=tr.stats.network)[0]
                tro.data.mask[i1:i2]=False
                tro.data[i1:i2]=tr.data

    tsv = tms.copy()

    # set reference time shifts
    for st in [stsv]+stsvn:
        for tr in st:
            trr=sta.select(id=tr.get_id())[0]
            tshf=trr.stats.starttime+trr.stats.t3-obspy.UTCDateTime(2000,1,1)
            tr.stats.t7 = tshf

    return stsv,tsv,stsvn,ilm


def identeqfiles(eq,trange=[-5.,15],tbuf=5.,datab='pksdrop'):
    """
    :param      eq:  earthquake info
    :param  trange:  time range to get relative to eq
    :param    tbuf:  buffer time for reading
    :param   datab:  database with seismogram info
    :return    fls:  a list of files to consider
    """

    # time limits
    t1 = eq.time+trange[0]
    t2 = eq.time+trange[1]      
    t1g,t2g=t1-tbuf,t2+tbuf

    # open the database
    session = waveformdb.opendatabase(datab)

    # identify the files of interest
    fls = []
    flsa=session.query(Waveform).yield_per(1000)

    print(flsa.count())

    fls=fls+flsa.filter(or_(Waveform.net=='BP',Waveform.net=='PB',Waveform.net=='NC'),
                        Waveform.starttime<=t2g,
                        Waveform.endtime>=t1g).all()


    # fls=fls+flsa.filter(Waveform.net=='PB',
    #                     Waveform.starttime<=t2g,
    #                     Waveform.endtime>=t1g).all()

    # close file database
    session.close()

    return fls

def trashmasked(st):
    """
    remove the masks or trash the traces with anything masked
    :param       st:  waveforms
    """

    # check for masks and remove
    for tr in st:
        if isinstance(tr.data,np.ma.masked_array):
            if np.sum(tr.data.mask):
                st.remove(tr)
            else:
                tr.data=tr.data.data

    # also remove empty traces
    for tr in st:
        if tr.stats.npts==0:
            st.remove(tr)

def cohwlfe(idi=None,ps=None):

    if idi is None:
        ids,tms=egflist(justhave=True)
        idi=ids[3]
        idi=tms[idi]
    elif isinstance(idi,int):
        ids,tms=egflist(justhave=True)
        idi=tms[idi]

    if ps is None:
        ps = initstack(37140,juststack=True)
        
    # get the data
    tm=idi
    stns=[tr.stats.network+'.'+tr.stats.station+'.'+tr.stats.channel
          for tr in ps.stam]
    st,invi = grablfedata(tm,stns=stns,invi=None,remresp=1)
    
    # filter
    msk=seisproc.prepfiltmask(st,0)
    st.filter('bandpass',freqmin=ps.flmget[0],freqmax=ps.flmget[1])
    seisproc.addfiltmask(st,msk)

    # align
    pksdrops.pickarvkt(st,mkref='t1',tmin=4.,tmax=8.,twin=0.5,mkset='t3')
    mdf=[]
    for tr in st:
        tri=ps.sta.select(station=tr.stats.station,channel=tr.stats.channel)[0]
        mdf.append(tr.stats.t3-tr.stats.t1 - (tri.stats.t3-tri.stats.t1))
    mdf=np.median(mdf)
    for tr in st:
        tr.stats.t1=tr.stats.t1+mdf
    
    readpicks(fnum=ps.fnum,sta=st,pkref='t1',pk='t3')

    pksdrops.pickarvkt(st,mkref='t3',tmin=-1.,tmax=1.,twin=1.,mkset='t3')

    # unfiltered
    sto=st.copy()
    
    # convolve the EGF with a longer STF
    nper=int(np.round(0.2/ps.sta[0].stats.delta))
    tprx = scipy.signal.hann(nper)
    tprx = tprx / np.dot(tprx,tprx)**0.5
    tprx = tprx / np.sum(tprx)

    Nf=np.max([tr.stats.npts for tr in st])*2
    ftpr=np.fft.rfft(tprx,n=Nf)

    nshf=int(nper/2)
    for tr in st:
        nper=tr.stats.npts
        data=np.fft.rfft(tr.data,n=Nf)
        data=np.multiply(data,ftpr)
        data=np.fft.irfft(data)
        tr.data=data[nshf:(nshf+nper)]

    # and align
    pksdrops.pickxcorr(ps.sta,st,pk1='t3',pk2='t3',twin=[-.5,3.5],
                       shflm=[-.5,.5],flm=[0.,8],
                       mkset1='t5',mkset2='t3',refeq=1)

    # remove anything too poor
    for tr in st:
        tri=sto.select(id=tr.get_id())[0]
        if tr.stats.xcmax<0.2:
            st.remove(tr)
            sto.remove(tri)
        else:
            tri.stats.t3=tr.stats.t3

    return st,ps,sto

def ploteqlfe(sta,sto):

    trlfe=sta.select(station='CCRB',channel='DP2')[0]
    treq=sto.select(station='CCRB',channel='DP2')[0]

    tmlfe=trlfe.times()-trlfe.stats.t3
    tmeq=treq.times()-treq.stats.t3

    f = plt.figure(figsize=(10,4))
    p = plt.axes([0,0,1,1])

    p.plot(tmeq,treq.data/np.max(np.abs(treq.data)),color='r',
           linewidth=1.5)
    
    p.plot(tmlfe,trlfe.data/np.max(np.abs(trlfe.data))-0.7,color='b',
           linewidth=2.5)

    p.set_xlim([-3,9])

    plt.setp(p.spines.values(), color='whitesmoke')
    #plt.box(on=None)

    p.set_facecolor('whitesmoke')
    p.xaxis.set_visible(False)
    p.yaxis.set_visible(False)
    
    graphical.printfigure('seiscompare',f,tight=False,ftype='png')


def grablfedata(tm,trange=[-50.,50.],datab='pksdrop',fnum=None,
                stns=None,remresp=0,invi=None,checkzero=True):
    """
    :param       tm:  time of LFE
    :param   trange:  time range relative to event to get
                        (default: [-140,90])
    :param    datab:  database name (default: 'pksdrop')
    :param     fnum:  family number to set picks
    :param     stns:  stations (default: get them all)
    :param  remresp:  remove the response (0-no, 1-yes, 2-just gain)
    :param     invi:  inventory with responses already available
    :return      st:  set of waveforms
    :return    invi:  inventory with added responses
    """

    # time limits
    t1 = tm+trange[0]
    t2 = tm+trange[1]
        
    if remresp is True:
        remresp = 1

    # open the database
    session = waveformdb.opendatabase(datab)
    fls=session.query(Waveform).yield_per(1000)
    session.close()
    
    # identify the files of interest
    fls=fls.filter(Waveform.starttime<=t2.timestamp,
                   Waveform.endtime>=t1.timestamp).all()

    if stns is not None:
        # only read some of the data
        flsi = []
        for fl in fls:
            nm=fl.net+'.'+fl.sta+'.'+fl.chan
            if nm in stns:
                flsi.append(fl)
        fls=flsi

    # # check for zeros
    # if checkzero and len(fls):
    #     # the files
    #     flsi=np.array([os.path.join(fl.dir,fl.dfile) for
    #                    fl in fls])
    #     # check for zeros
    #     iszero=databseis.checkzeros(flsi,mnfrc=0.03)
    #     ix,=np.where(~iszero)
    #     fls=[fls[ixi] for ixi in ix]

    #     # delete the files
    #     #databseis.delfile(flsi[iszero],datab)

    # # check for additional files
    # checkexist=False
    # if checkexist:
    #     # the files
    #     flsi=np.array([os.path.join(fl.dir,fl.dfile) for
    #                    fl in fls])
    #     flsi=np.unique(flsi)
    #     for flsj in flsi:
    #         if not os.path.exists(flsj):
    #             print('To delete '+flsj)
    #             # databseis.delfile(flsj,datab,dfile=False)

    # and read
    st=obspy.Stream()
    for fl in fls:
        try:
            st=st+fl.waveform(t1=t1,t2=t2)
        except:
            print(fl.filename())

    # only some networks and channels
    st=st.select(channel='?H?')+st.select(channel='DP?')
    st=st.select(network='PB')+st.select(network='BP')

    # set the earthquake time
    st = st.split().merge()
    for tr in st:
        tr.stats.t1=obspy.UTCDateTime(tm)-tr.stats.starttime

    # check for data
    sta = obspy.Stream()
    frcneed=0.95
    tch = np.array([-5.,20.])
    for tra in st:
        # overall
        if not isinstance(tra.data,np.ma.masked_array):
            tra.data=np.ma.masked_array(tra.data,mask=False)
        dtim = tra.stats.delta
        nexp=np.diff(trange)[0]/dtim
        npts=np.sum(~tra.data.mask)
        iok=npts>frcneed*nexp
        # and near earthquake
        tr=tra.copy().trim(starttime=tm+tch[0],endtime=tm+tch[1])
        if not isinstance(tr.data,np.ma.masked_array):
            tr.data=np.ma.masked_array(tr.data,mask=False)
        npts=np.sum(~tr.data.mask)
        nexp=np.diff(tch)[0]/dtim
        iok=np.logical_and(iok,npts>frcneed*nexp)
        if iok:
            sta.append(tra)
    st = sta

    # detrend
    st = st.split().detrend().merge()

    # create a mask before filtering
    msk=seisproc.prepfiltmask(st,tmask=2.)

    # apply a broad filter
    st.taper(type='cosine',max_percentage=0.5,max_length=2.,
             side='both')

    if remresp == 1 and len(st):
        print('Removing response')
        pre_filt = [0.2,0.4,30.,40.]
        # remove responses
        trdel,msdel = obspy.Stream(),obspy.Stream()
        if invi is None:
            # one at a time
            for k in range(0,len(st)):
                tr = st[k]
                try:
                    # response corrections
                    responsecorr.removeresponsepz(tr,water_level=100,
                                                  pre_filt=pre_filt)
                except:
                    print('Response correction failed for ')
                    print(tr)
                    # if it doesn't work, trash this trace
                    trdel.append(tr)
                    msdel.append(msk[k])

        else:
            #print('Reading response')
            # collect all responses first
            invi = responsecorr.readresponse(st,invi)
            #print('Removing response')
            # then remove response

            for k in range(0,len(st)):
                tr = st[k]
                try:
                    responsecorr.removeresponsepz(tr,invi,water_level=100,
                                               pre_filt=pre_filt)
                    # response corrections
                    # tr.remove_response(pre_filt=pre_filt,
                    #                    output='VEL',water_level=100,
                    #                    inventory=invi)
                except:
                    print('Response correction failed for ')
                    print(tr)
                    trdel.append(tr)
                    msdel.append(msk[k])
        # traces trackes that don't work
        for tr in trdel:
            # if it doesn't work, trash this trace
            st.remove(tr)
        for tr in msdel:
            msk.remove(tr)
    elif remresp == 2:
        print('Removing sensitivity')
        # remove responses
        trdel,msdel = obspy.Stream(),obspy.Stream()
        if invi is None:
            # just consider one at a time
            for k in range(0,len(st)):
                tr = st[k]
                try:
                    # response corrections
                    responsecorr.removesensitivity(tr)
                except:
                    print('Response correction failed for ')
                    print(tr)
                    trdel.append(tr)
                    msdel.append(msk[k])
        else:
            # collect all responses first
            invi = responsecorr.readresponse(st,invi)
            # then remove sensitivity
            for k in range(0,len(st)):
                tr = st[k]
                try:
                    # response corrections
                    tr.remove_sensitivity(invi)
                except:
                    print('Response correction failed for ')
                    print(tr)
                    # if it doesn't work, trash this trace
                    trdel.append(tr)
                    msdel.append(msk[k])
        for tr in trdel:
            # if it doesn't work, trash this trace
            st.remove(tr)
        for tr in msdel:
            msk.remove(tr)
        print('Filtering')
        st.filter('bandpass',freqmin=0.3,freqmax=40.)
    elif not remresp:
        #st.filter('highpass',freq=0.3)
        st.filter('bandpass',freqmin=0.3,freqmax=40.)

    # put the filtering mask on
    seisproc.addfiltmask(st,msk)

    # resample to a specified timing
    st = st.split()
    samprate = 100.
    for tr in st:
        tr.interpolate(samprate,method='weighted_average_slopes')
    
    # reset the earthquake time
    st=st.merge()
    for tr in st:
        tr.stats.t1=obspy.UTCDateTime(tm)-tr.stats.starttime

    # set picks
    if fnum:
        readpicks(fnum=fnum,sta=st,pk='t3',pkref='t1')

    return st,invi

def checksimilarity(i1=37140,i2=37102):

    # get the two stacks
    ps1 = initps(i1,juststack=True)
    ps2 = initps(i2,juststack=True)

    ttrav1={}
    for tr in ps1.stam:
        oloc=[tr.stats.sac['stlo'],tr.stats.sac['stla'],0]
        tkang,tms,az=pksdrops.calctakeang(eloc=ps1.eqloc,oloc=oloc,
                                          mdl='iasp91',phsarv='sS')
        ttrav1[tr.get_id()]=tms

    ttrav2={}
    for tr in ps2.stam:
        oloc=[tr.stats.sac['stlo'],tr.stats.sac['stla'],0]
        tkang,tms,az=pksdrops.calctakeang(eloc=ps2.eqloc,oloc=oloc,
                                          mdl='iasp91',phsarv='sS')
        ttrav2[tr.get_id()]=tms

    tref=ps1.stam[0].stats.starttime
    tmn1=tref + np.mean([tr.stats.starttime-tref+tr.stats.t3
                         for tr in ps1.stam])
    tmn2=tref + np.mean([tr.stats.starttime-tref+tr.stats.t3
                         for tr in ps2.stam])

    for tr in (ps1.stam+ps2.stam):
        tr.stats.t4=tr.stats.t3
        tr.stats.t5=tr.stats.t3

    # shift the times
    tshfs=[]
    for tr2 in ps2.stam:
        tr1=ps1.stam.select(id=tr2.get_id())
        if tr1:
            tr1=tr1[0]
            tshf=ttrav2[tr2.get_id()]-ttrav1[tr2.get_id()]
            t1=tr1.stats.starttime+tr1.stats.t3-tmn1
            tshfs.append(tshf+t1)
            t2=tmn2+t1+tshf
            tr2.stats.t4=t2-tr2.stats.starttime


    tshf=np.mean(tshfs)
    for tr in ps2.stam:
        tr.stats.t4=tr.stats.t4-tshf

    # just x-c to match
    flm=[2,6]
    pksdrops.pickxcorr(ps1.stam,ps2.stam,
                       pk1='t3',pk2='t4',
                       twin=[-.2,2.8],shflm=[-.5,.5],
                       mkset1='t5',mkset2=None,flm=flm,refeq=1)

    # computation parameters
    trange = [-.2,3.8]
    twin = [-2.,2.]
    dfres = 1./1.

    # compute cross-spectra
    xc=phscoh.calcxctim(ps1.stam,ps2.stam,trange,mk1='t3',mk2='t5',
                        nsint=1,fmax=100.,dfres=dfres,
                        tpr='slepian',twin=twin)
    freq=xc.freq

    # just want the normalized coherence
    xc=np.mean(xc.xc,axis=2)
    xc=np.divide(np.real(xc),np.abs(xc))
    
    return xc,freq,ps1.stam,ps2.stam



# def writemanylfes(lbl,stsv,tsv,stsvn,ilmsv,tget):
#     """
#     :param     lbl:  name of directory to write to
#     :param    stsv:  a list containing LFE waveforms
#     :param     tsv:  times of the LFEs considered
#     :param   stsvn:  a list of lists containing LFE waveforms at shifted times
#     :param   ilmsv:  limiting indices of values for each event
#     :param    tget:  time range for each event relative to pick
#     """

#     # directory to write to
#     fdir = os.path.join(os.environ['DATA'],'TREMORAREA',
#                         'SavedLFEs',lbl)
#     if os.path.exists(fdir):
#         shutil.rmtree(fdir)
#     os.makedirs(fdir)

#     # arrival times
#     fname = os.path.join(fdir,'arrivals')
#     fl = open(fname,'w')
#     for tr in stsv:
#         fl.write(tr.get_id()+','+str(tr.stats.t7)+'\n')
#     fl.close()

#     # times
#     fname = os.path.join(fdir,'times')
#     fl = open(fname,'w')
#     for k in range(0,len(tsv)):
#         nm = str(tsv[k].timestamp)
#         fl.write(nm+'\n')
#     fl.close()

#     # indices
#     fname = os.path.join(fdir,'ilimits')
#     fl = open(fname,'w')
#     fl.write(str(tget[0])+','+str(tget[1])+'\n')
#     for ix in ilmsv:
#         fl.write(str(ix)+'\n')
#     fl.close()
    
#     # actual data
#     seisproc.copytosacheader(stsv)
#     fdirj = os.path.join(fdir,'original')
#     os.makedirs(fdirj)

#     for tr in stsv.copy():
#         if isinstance(tr.data,np.ma.masked_array):
#             msk=tr.data.mask
#             tr.data=tr.data.data
#             tr.data[msk]=-12345
        
#         # write each one
#         fname=waveformdb.sacfilename(tr)
#         fname=os.path.join(fdirj,fname)
#         tr.write(fname,'SAC')

#     # each noise series
#     for m in range(0,len(stsvn)):
#         st = stsvn[m]
#         seisproc.copytosacheader(st)

#         fdirj = os.path.join(fdir,'noise'+str(m))
#         os.makedirs(fdirj)

#         for tr in st.copy():
#             if isinstance(tr.data,np.ma.masked_array):
#                 msk=tr.data.mask
#                 tr.data=tr.data.data
#                 tr.data[msk]=-12345

#             # write each one
#             fname=seisproc.sacfilename(tr)
#             fname=os.path.join(fdirj,fname)
#             tr.write(fname,'SAC')


def plotexamp(ps=None,tm=None):
    
    if ps is None:
        ps = procsearch(37102)
        #ps.stackprep()
        ps.wgttype='bymax'
        ps.readstacks(itn=0)
        ps.checksnr()
        ps.sta=ps.sta.select(channel='*2')
        ps.stam=ps.sta.copy()

        if tm is None:
            tm,xc = biglfetimes()
            tm,xc = tm[1],xc[1]
        tm = np.atleast_1d(tm)
        ps.tms = tm
        ps.allevents()

        ps.grabmanylfes()
        ps.amplim = None
        ps.tshfumax = float('inf')
        ps.maxstd = float('inf')
        ps.stsvn = ps.stsvn[0:1]
        ps.xcmanylfes()
    elif tm is not None:
        ixc=np.argmin(np.abs(ps.tsv-tm))
        ixc=np.bincount([ixc],minlength=ps.tsv.size).astype(bool)
        ps.xcmanylfes(ixc=ixc)
    else:
        tm = ps.tmxc[0]
        
        
    # timing
    dtim = ps.stam[0].stats.delta

    # Fourier coefficients
    fxc = ps.fxc[:,:,0,0]
    ampt = ps.ampt[:,:,0]

    # un-normalize
    fxc = np.multiply(fxc,ampt)


    # back to time domain
    xc = np.fft.irfft(fxc,axis=0)

    iwin=np.round(np.atleast_1d(ps.twin)/dtim).astype(int)
    iwin=np.arange(iwin[0],iwin[1])
    iwin=iwin % xc.shape[0]
    xc = xc[iwin,:]

    # un-taper this
    tprx = scipy.signal.hann(iwin.size)
    tprx = tprx / np.dot(tprx,tprx)**0.5
    xc = np.divide(xc,tprx.reshape([xc.shape[0],1]))
    xc[tprx==0.,:]=0.

    # times of x-c
    iwin=np.round(np.atleast_1d(ps.twin)/dtim).astype(int)
    tim = np.arange(iwin[0],iwin[1]).astype(float)*dtim
    
    # central portion
    xc = xc[20:-20,:]
    tim = tim[20:-20]

    # normalize
    xcm = np.max(np.abs(xc),axis=0)
    xc = np.divide(xc,xcm.reshape([1,xcm.size]))

    # which to plot
    iplt = np.arange(0,7,2)
    iplt,=np.where(np.sum(np.abs(xc),axis=0)>0)
    iplt = np.random.choice(iplt,np.minimum(4,iplt.size),replace=False)
    print(iplt)

    plt.close()
    f = plt.figure(figsize=(12,9))
    gs,p=gridspec.GridSpec(2,3),[]
    gs.update(left=0.07,right=0.97)
    gs.update(bottom=0.07,top=0.94)
    gs.update(hspace=0.25,wspace=0.25)
    for k in range(0,5):
        p.append(plt.subplot(gs[k]))
    p=np.array(p)
    #pm=p.reshape([2,3])

    lw = 1.5
    fsmth = 0.5
    dfreq = np.median(np.diff(ps.freq))
    nwl = int(np.round(fsmth*3*2/dfreq))
    nwl = nwl + 1 - (nwl%2)
    gwin=scipy.signal.gaussian(nwl,fsmth/dfreq)
    gwin=gwin/np.sum(gwin)
    gwin=gwin.reshape([gwin.size,1])
    cp=scipy.signal.convolve(ps.cp,gwin,mode='same').flatten()
    en=scipy.signal.convolve(ps.en,gwin,mode='same').flatten()
    ns = np.hstack(ps.enn)
    ns=scipy.signal.convolve(ns,gwin,mode='same')
    nsu = np.std(ns,axis=1)
    ns = np.mean(ns,axis=1)
    
    cp[cp<0]=float('nan')
    ns[ns<0]=float('nan')
    en[en<0]=float('nan')
    
    h=[]
    
    #p[2].plot(ps.freq,ns-2*nsu,color='gray',linestyle='--')
    #p[2].plot(ps.freq,ns+2*nsu,color='gray',linestyle='--')
    hh,=p[2].plot(ps.freq,ns,color='gray',linewidth=lw)
    h.append(hh)
    lbl=['direct coherent','inter-station coherent',
         'total','noise','LFE']


    #p[2].plot(ps.freq,en-2*nsu,color='green',linestyle='--')
    #p[2].plot(ps.freq,en+2*nsu,color='green',linestyle='--')
    hh,=p[2].plot(ps.freq,en,color='green',linewidth=lw)
    h.append(hh)

    #p[2].plot(ps.freq,en-ns-2*nsu,color='blue',linestyle='--')
    #p[2].plot(ps.freq,en-ns+2*nsu,color='blue',linestyle='--')
    hh,=p[2].plot(ps.freq,en-ns,color='blue',linewidth=lw)
    h.append(hh)
    
    cpu = np.std(np.hstack(ps.cpn),axis=1)
    #p[2].plot(ps.freq,cp-2*cpu,color='r',linestyle='--')
    #p[2].plot(ps.freq,cp+2*cpu,color='r',linestyle='--')
    hh,=p[2].plot(ps.freq,cp,color='r',linewidth=lw)
    h.append(hh)

    ten = np.real(ps.fxc)
    ten=np.mean(ten,axis=1)
    ten[ten<0] = 0.
    ten = np.power(ten,2)
    hh,=p[2].plot(ps.freq,ten,color='darkgoldenrod',linewidth=lw)
    h.append(hh)

    h2 = []
    hh,=p[4].plot(ps.freq,np.divide(ten.flatten(),(en.flatten()-ns.flatten())),
                  color='darkgoldenrod',linewidth=lw)
    h2.append(hh)

    hh,=p[4].plot(ps.freq,np.divide(cp,en-ns),color='red',linewidth=lw)
    h2.append(hh)

    isp = int(fsmth/dfreq)
    isp = np.arange(0,ps.freq.size,isp)
        
    cols = graphical.colors(len(iplt))
    shm=2.5
    for k in range(0,len(iplt)):
        ix=iplt[k]
        
        p[1].plot(tim,xc[:,ix]+shm*k,color=cols[k],linewidth=lw)
        
        tr=ps.stam.select(id=ps.ids[ix])[0].copy()
        #tr.filter('bandpass',freqmin=ps.flmget[0],freqmax=ps.flmget[1])
        timi=tr.times()-tr.stats.t3
        ii=np.logical_and(timi>-1.,timi<6)
        data=tr.data[ii]
        data=data/np.max(np.abs(data))
        p[0].plot(timi[ii],data+shm*k,color='k',linewidth=lw)

        tr=ps.stsv.select(id=ps.ids[ix])[0].copy()
        ixc=np.argmin(np.abs(ps.tsv-ps.tmxc))
        ixc2=np.argmin(np.abs(ps.tms-ps.tmxc))
        jj=np.arange(ps.ilmsv[ixc],ps.ilmsv[ixc+1])
        timi=(jj-ps.izero[0])*dtim
        timi=timi-ps.tshfs[ps.ids[ix]][ixc2]
        ii=np.logical_and(timi>-1.,timi<6)
        data=tr.data[jj[ii]]
        data=data/np.max(np.abs(data))
        p[0].plot(timi[ii],data+shm*k,color=cols[k],linestyle='-',
                  linewidth=lw)

        p[3].plot(ps.freq[isp],np.angle(fxc[isp,ix])*180/math.pi,
                  marker='x',linestyle='none',color=cols[k],
                  linewidth=lw)

    fs = 'medium'
    fs = 12

    ilbl=np.array([4,3,1,0,2])
    h = np.array(h)[ilbl]
    lg=p[2].legend(h,lbl,loc='upper center',fontsize=fs-1)

    p[4].legend(h2,['direct','inter-station'],fontsize=fs-1,
                loc='lower left')
    p[4].set_ylim([0.,1.2])
    p[4].set_ylim([-1.,2.2])

    p[3].set_yticks([-180,-90,0,90,180])
    p[3].set_ylim([-180,180])
    p[2].set_ylim([1.,4000])
    xlm = np.array([2.,15])
    for k in [2,3,4]:
        p[k].set_xscale('log')
        p[k].set_xlim(xlm)
        p[k].set_xlabel('frequency (Hz)',fontsize=fs)
        p[k].set_xticks([2,5,10])
        p[k].get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    for k in [0,1]:
        p[k].set_xlabel('time (s)',fontsize=fs)
        p[k].set_ylim([-1.5,shm*(len(iplt)-1)+1.5])
    p[0].set_yticks([])
    p[1].set_yticks([])
    p[1].set_ylabel('cross-correlations',fontsize=fs)
    p[3].set_ylabel('cross-correlation phases',fontsize=fs)
    p[4].set_ylabel('coherent energy fraction',fontsize=fs)
    p[2].set_ylabel('energy / template energy',fontsize=fs)
    p[2].set_yscale('log')
    p[1].set_xticks(np.arange(-4,4))
    p[1].set_xlim(ps.twin)

    for k in [2,3,4]:
        rax = p[k].twiny()
        rax.set_xlim(np.divide(1.,np.flipud(xlm))*1.3*3000.)
        rax.set_xscale('log')
        rax.set_xlabel('diameter (m)',fontsize=fs)
        rax.tick_params(axis='x',labelsize=fs)
        rax.invert_xaxis()
        rax.set_xticks([300,500,1000])
        rax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
        rax.xaxis.set_tick_params(labelsize=fs)
        rax.yaxis.set_tick_params(labelsize=fs)

    
    for ph in p:
        ph.xaxis.set_tick_params(labelsize=fs)
        ph.yaxis.set_tick_params(labelsize=fs)

    lbl=ps.tmxc[0].strftime('%Y%b%d_%H%M%S')
    lbl='PCcalcexamp_'+str(ps.fnum)+'_'+lbl
    graphical.printfigure(lbl,f)

    smpl=False
    if smpl:
    
        lbl=['direct coherent','inter-station coherent',
             'total','noise','LFE']
        h[0].remove()
        h[2].remove()
        h[3].remove()
        
        lbl=['inter-station coherent','LFE']
        lg=p[2].legend(h[np.array([1,4])],lbl,
                       loc='upper center',fontsize=fs-1)
        h2[0].remove()
        p[4].legend(h2,['inter-station'],fontsize=fs-1)    
        
        lbl=ps.tsv[0].strftime('%Y%b%d_%H%M%S')
        lbl='PCcalcexamp_'+str(ps.fnum)+'_'+lbl+'_simp'
        graphical.printfigure(lbl,f)

    return ps,tm

def aztostat(sta,eqloc):
    """
    :param     sta: waveforms with location information
    :param   eqloc: earthquake location
    :return    azm: azimuth dictionary by station
    """

    azm = dict.fromkeys([tr.stats.station for tr in sta])

    rlon = np.cos(np.pi*eqloc[1]/180)
    eqloc=np.atleast_1d(eqloc)
    
    for tr in sta:
        sloc=np.array([tr.stats.sac['stlo'],tr.stats.sac['stla'],0.])
        dr = sloc[0:2]-eqloc[0:2]
        dr[0] = dr[0]*rlon
        dr = dr[1]+1j*dr[0]
        dr = np.angle(dr)*180/np.pi

        azm[tr.stats.station]=dr % 360

    return azm
    


def plotind(ps=None,tm=None):
    
    if ps is None:
        ps = procsearch(37102)
        #ps.stackprep()
        ps.wgttype='bymax'
        ps.readstacks()
        ps.checksnr()

        ps.cmpuse='first_horizontal'
        ps.pickcomp()

        ps.readamps()
        ps.calcmedscale()
        ps.scaleamps()
        
        if tm is None:
            tm,xc = biglfetimes()
            tm,xc = tm[2],xc[2]

        ps.readmanylfes()
        ps.amplim = None
        ps.tshfumax = float('inf')
        ps.maxstd = float('inf')
        #ps.stsvn = ps.stsvn[0:1]

        ixc=np.argmin(np.abs(ps.tsv-tm))
        ixc=np.bincount([ixc],minlength=ps.tsv.size).astype(bool)
        ps.xcmanylfes(ixc=ixc,bystat=False)
    elif tm is not None:
        ixc=np.argmin(np.abs(ps.tsv-tm))
        ixc=np.bincount([ixc],minlength=ps.tsv.size).astype(bool)
        ps.xcmanylfes(ixc=ixc,bystat=True)
    else:
        tm = ps.tmxc[0]
        
        
    # timing
    dtim = ps.stam[0].stats.delta

    # Fourier coefficients
    fxc = ps.fxc[:,:,0,0]
    ampt = ps.ampt

    # un-normalize
    fxc = np.multiply(fxc,ampt)
    
    # back to time domain
    xc = np.fft.irfft(fxc,axis=0)
    iwin=np.round(np.atleast_1d(ps.twin)/dtim).astype(int)
    iwin=np.arange(iwin[0],iwin[1])
    iwin=iwin % xc.shape[0]
    xc = xc[iwin,:]

    # un-taper this
    tprx=ps.tprs[:,0]
    #tprx = scipy.signal.hann(iwin.size)
    #tprx = tprx / np.dot(tprx,tprx)**0.5
    xc = np.divide(xc,tprx.reshape([xc.shape[0],1]))
    xc[tprx==0.,:]=0.
    xc = ps.xcsave

    # times of x-c
    iwin=np.round(np.atleast_1d(ps.twin)/dtim).astype(int)
    tim = np.arange(iwin[0],iwin[1]).astype(float)*dtim

    # central portion
    #xc = xc[30:-30,:]
    #tim = tim[30:-30]

    # normalize
    xcm = np.max(np.abs(xc),axis=0)
    xc = np.divide(xc,xcm.reshape([1,xcm.size]))

    # which to plot
    iplt = np.arange(0,7,2)
    iplt,trash=np.where(np.sum(np.abs(ps.fxc[:,:,0]),axis=0)>0)
    iplt = np.random.choice(iplt,np.minimum(20,iplt.size),replace=False)
    iplt = np.unique(iplt)

    # which stations they belong to
    stns=np.array([idi.split('.')[1] for idi in ps.ids])
    stn,istat=np.unique(stns,return_index=True)
    azm = aztostat(ps.stam,ps.eqloc)
    azm = np.array([azm[vl] for vl in stn])
    azm = azm[iplt]
    ix=np.argsort(azm)
    azm,iplt=azm[ix],iplt[ix]
    #azm = np.arange(0,iplt.size).astype(float)
    azlm=general.minmax(azm,1.1)
    azlm=[-10,370] 
    azwd=np.diff(azlm)/15.

    idel,=np.where(np.diff(azm)<azwd/3.)
    while idel.size:
        iplt=np.delete(iplt,idel[0]+1)
        azm=np.delete(azm,idel[0]+1)
        idel,=np.where(np.diff(azm)<azwd/3.)

    tlm=[-4,5]
    rt=np.diff(ps.twin)[0]/np.diff(tlm)[0]
    
    plt.close()
    f = plt.figure(figsize=(8,10))
    gs=gridspec.GridSpec(1,1)
    gs.update(left=0.1,right=0.5,bottom=0.51,top=0.94)
    pseis=plt.subplot(gs[0])

    gs=gridspec.GridSpec(1,1)
    gs.update(left=0.1,right=0.44,bottom=0.06,top=0.49)
    pcorr=plt.subplot(gs[0])

    gs=gridspec.GridSpec(1,1)
    gs.update(left=0.54,right=0.97,bottom=0.51,top=0.94)
    pphas=plt.subplot(gs[0])

    gs=gridspec.GridSpec(1,1)
    gs.update(left=0.54,right=0.97,bottom=0.06,top=0.49)
    pener=plt.subplot(gs[0])

    p=np.array([pseis,pcorr,pener,pphas])

    lw = 1.5
    fsmth = 0.5
    dfreq = np.median(np.diff(ps.freq))
    nwl = int(np.round(fsmth*3*2/dfreq))
    nwl = nwl + 1 - (nwl%2)
    gwin=scipy.signal.gaussian(nwl,fsmth/dfreq)
    gwin=gwin/np.sum(gwin)
    gwin=gwin.reshape([gwin.size,1])
    cp=scipy.signal.convolve(ps.cp,gwin,mode='same').flatten()
    en=scipy.signal.convolve(ps.en,gwin,mode='same').flatten()
    ns = np.hstack(ps.enn)
    ns=scipy.signal.convolve(ns,gwin,mode='same')
    nsu = np.std(ns,axis=1)
    ns = np.mean(ns,axis=1)
    
    cp[cp<0]=float('nan')
    ns[ns<0]=float('nan')
    en[en<0]=float('nan')
    
    h=[]
    
    #p[2].plot(ps.freq,ns-2*nsu,color='gray',linestyle='--')
    #p[2].plot(ps.freq,ns+2*nsu,color='gray',linestyle='--')
    hh,=p[2].plot(ps.freq,ns,color='gray',linewidth=lw)
    h.append(hh)
    lbl=['$P_d$: direct coherent','$P_c$: inter-station coherent',
         '$P_t$: total (incl. noise)','$P_n$: noise','$P_l$: LFE']


    #p[2].plot(ps.freq,en-2*nsu,color='green',linestyle='--')
    #p[2].plot(ps.freq,en+2*nsu,color='green',linestyle='--')
    hh,=p[2].plot(ps.freq,en,color='green',linewidth=lw)
    h.append(hh)

    #p[2].plot(ps.freq,en-ns-2*nsu,color='blue',linestyle='--')
    #p[2].plot(ps.freq,en-ns+2*nsu,color='blue',linestyle='--')
    hh,=p[2].plot(ps.freq,en-ns,color='blue',linewidth=lw)
    h.append(hh)
    
    cpu = np.std(np.hstack(ps.cpn),axis=1)
    #p[2].plot(ps.freq,cp-2*cpu,color='r',linestyle='--')
    #p[2].plot(ps.freq,cp+2*cpu,color='r',linestyle='--')
    hh,=p[2].plot(ps.freq,cp,color='r',linewidth=lw)
    h.append(hh)

    fxc=ps.fxc[:,:,0,0]
    ten = np.real(fxc)
    ten=np.mean(ten,axis=1)
    ten=np.multiply(np.power(ten,2),np.sign(ten))
    hh,=p[2].plot(ps.freq,ten,color='darkgoldenrod',linewidth=lw)
    h.append(hh)

    isp = int(fsmth/dfreq)
    isp = np.arange(0,ps.freq.size,isp)
        
    cols = graphical.colors(4)
    cols = [cols[ix % len(cols)] for ix in range(0,iplt.size)]
    shm=2.5

    for k in range(0,len(iplt)):
        ix=iplt[k]
        
        p[1].plot(tim,-xc[:,ix]*azwd+azm[k],color=cols[k],linewidth=lw)
        
        #tr=ps.stam.select(id=ps.ids[ix])[0].copy()
        tr=ps.stam.select(station=stn[ix])[0].copy()
        #tr.filter('bandpass',freqmin=ps.flmget[0],freqmax=ps.flmget[1])
        timi=tr.times()-tr.stats.t3
        ii=np.logical_and(timi>=tlm[0],timi<=tlm[1])
        data=tr.data[ii]-np.median(tr.data[ii])
        data=data/np.max(np.abs(data))*azwd
        p[0].plot(timi[ii],-data+azm[k],color='k',linewidth=lw,zorder=2)

        #tr=ps.stsv.select(id=ps.ids[ix])[0].copy()
        tr=ps.stsv.select(station=stn[ix])[0].copy()
        ixc=np.argmin(np.abs(ps.tsv-ps.tmxc))
        ixc2=np.argmin(np.abs(ps.tms-ps.tmxc))
        jj=np.arange(ps.ilmsv[ixc],ps.ilmsv[ixc+1])
        timi=(jj-ps.izero[0])*dtim
        timi=timi-ps.tshfs[ps.ids[istat][ix]][ixc2]
        ii=np.logical_and(timi>=tlm[0],timi<=tlm[1])
        data=tr.data[jj[ii]]
        data=data-np.median(data)
        data=data/np.max(np.abs(data))*azwd
        p[0].plot(timi[ii],-data+azm[k],color=cols[k],linestyle='-',
                  linewidth=lw,zorder=1)

        # p[3].plot(ps.freq[isp],np.angle(fxc[isp,ix])*180/math.pi,
        #           marker='x',linestyle='none',color=cols[k],
        #           linewidth=lw)
        pmap=np.angle(fxc[isp,ix])*180/math.pi
        pmap=pmap.reshape([1,pmap.size])

        azmi=azm[k]+np.array([-1,1])*azwd/2.
        vls=p[3].pcolormesh(ps.freq[isp],azmi,pmap,
                            vmin=-180,vmax=180,cmap='hsv')

    fs = 'medium'
    fs = 12

    psc = p[3].get_position()
    psc = [psc.x0+psc.width*0.1,psc.y1-0.01,psc.width*0.8,0.02]
    #psc = [psc.x0-0.01,psc.height*0.2+psc.y0,0.02,psc.height*0.6]
    cbs = f.add_axes(psc,zorder=10)
    cb = f.colorbar(vls,cax=cbs,orientation='horizontal',
                    ticklocation='top',ticks=[-180,-90,0,90,180])
    cbs.tick_params(axis='x',labelsize=9,rotation=0)
    cb.set_label('$\hat{x}_k$ phase',fontsize=fs-1)

    p[0].set_xlim(tlm)
    p[1].set_xlim(ps.twin)


    ilbl=np.array([4,3,1,0,2])
    h = np.array(h)[ilbl]
    lg=p[2].legend(h,lbl,loc='upper left',fontsize=fs-1)

    #p[3].set_ylim([-180,180])
    for k in [0,1,3]:
        p[k].set_yticks([0,90,180,270,360])
        p[k].set_ylim(azlm)
        p[k].invert_yaxis()
    p[2].set_ylim([1.,40000])
    xlm = np.array([2.,15])
    xtk=np.array([2,3,4,5,6,7,8,9,10,20])
    xtk=np.arange(2,21)
    xtkl=['']*xtk.size
    ix,=np.where([xtki in [2,5,10,20] for xtki in xtk])
    for ixi in ix:
        xtkl[ixi]='{:d}'.format(xtk[ixi])
    for k in [2,3]:
        p[k].set_xscale('log')
        p[k].set_xlim(xlm)
        p[k].set_xlabel('frequency (Hz)',fontsize=fs)
        p[k].set_xticks(xtk)
        p[k].minorticks_off()
        p[k].get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
        p[k].set_xticklabels(xtkl)
    for k in [0,1]:
        p[k].set_xlabel('time (s)',fontsize=fs)
        #p[k].set_ylim([-1.5,shm*(len(iplt)-1)+1.5])
    p[0].xaxis.set_label_position('top')
    p[0].xaxis.tick_top()

    ix=np.array([0,1,3,2])
    graphical.cornerlabels(p[ix],'ul',fontsize=fs-2,xscl=0.025,yscl=0.02)

    p[0].axvspan(ps.txc[0],ps.txc[1],0,1,color='lightgray',
                 zorder=0)
    for k in [3]:
        p[k].set_xticklabels([])
        p[k].set_xlabel('')
    p[1].set_ylabel('cross-correlations',fontsize=fs)
    p[1].set_ylabel('station azimuth ($^{\circ}$)',fontsize=fs)
    p[0].set_ylabel('station azimuth ($^{\circ}$)',fontsize=fs)
    #p[3].set_ylabel('cross-correlation phases',fontsize=fs)
    p[3].set_yticklabels([])
    p[2].set_ylabel('power / template power',fontsize=fs)
    p[2].set_yscale('log')
    p[1].set_xticks(np.arange(-4,4))
    p[1].set_xlim(ps.twin)
    #p[1].set_xlim([-2.,2])

    xtk=np.arange(1,20)*100.
    xtkl=['']*xtk.size
    ix,=np.where([xtki in [300,500,1000] for xtki in xtk])
    for ixi in ix:
        xtkl[ixi]='{:0.0f}'.format(xtk[ixi])
    # for k in [2,3]:
    #     rax = p[k].twiny()
    #     rax.set_xlim(np.divide(1.,np.flipud(xlm))*1.3*3000.)
    #     rax.set_xscale('log')
    #     rax.minorticks_off()
    #     rax.tick_params(axis='x',labelsize=fs)
    #     rax.invert_xaxis()
    #     rax.set_xticks(xtk)
    #     rax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    #     rax.xaxis.set_tick_params(labelsize=fs)
    #     rax.yaxis.set_tick_params(labelsize=fs)
    #     if k==3:
    #         rax.set_xticklabels(xtkl)
    #         rax.set_xlabel('diameter (m)',fontsize=fs)
    #     else:
    #         rax.set_xticklabels([])
    #         rax.set_xlabel('',fontsize=fs)
    
    for ph in p:
        ph.xaxis.set_tick_params(labelsize=fs)
        ph.yaxis.set_tick_params(labelsize=fs)

    lbl=ps.tmxc[0].strftime('%Y%b%d_%H%M%S')
    lbl='PCplotind_'+str(ps.fnum)+'_'+lbl
    graphical.printfigure(lbl,f)

    smpl=False
    if smpl:
    
        lbl=['direct coherent','inter-station coherent',
             'total','noise','LFE']
        h[0].remove()
        h[2].remove()
        h[3].remove()
        
        lbl=['inter-station coherent','LFE']
        lg=p[2].legend(h[np.array([1,4])],lbl,
                       loc='upper center',fontsize=fs-1)
        
        lbl=ps.tsv[0].strftime('%Y%b%d_%H%M%S')
        lbl='PCcalcexamp_'+str(ps.fnum)+'_'+lbl+'_simp'
        graphical.printfigure(lbl,f)

    return ps,tm


def allfigures(ps=None):

    if ps is None:
        ps = 37140

    if isinstance(ps,int):
        ps = procsearch(ps)
        ps.wgttype='bymax_shifted'
        ps.readstacks()
        ps.checksnr()
        ps.cmpuse='both_horizontals'
        ps.pickcomp()
        
        ps.readamps()
        ps.calcmedscale()
        ps.scaleamps()

        ps.readmanylfes()
        ps.besttshf()

    txc = [np.array([-0.1,3.9]),np.array([-0.1,2.9])]
    #txc = [np.array([-0.1,2.9])]

    # amplitudes
    ps.meanamps()

    chtry = ['both_horizontals','first_horizontal','second_horizontal']
    chtry = ['both_horizontals']

    ps.rscl=True
    
    for chs in chtry:
        # read the data
        ps.readstacks()
        ps.checksnr()
        #ps.taperbydur()
        ps.cmpuse=chs
        ps.pickcomp()

        ps.calcmedscale()
        ps.scaleamps()

        # pick amplitudes
        ps.meanamps()
        amps = ps.meanamp.copy()
        amps = amps[amps<1e3]
        amps = amps[~amps.mask]
        amps = amps[~np.isnan(amps)]
        amps.sort()
        frc = np.linspace(0.,1.,4)
        frc = (frc*amps.size).astype(int)
        frc[-1]=frc[-1]-1
        frc = amps[frc]
        frc[0]=-20.
        frc[-1]=20.
        frc = np.round(frc,1)
        frc=np.array([-20.,20.])
        
        # sort into groups
        i1,i2=np.meshgrid(np.arange(0,len(frc)-1),
                          np.arange(len(frc)-1,len(frc)))
        iok=i2>i1
        i1,i2=i1[iok],i2[iok]

        eqloc = [0]

        
        for k in range(0,len(i1)):
            ps.amplim=np.array([frc[i1[k]],frc[i2[k]]])
            print(ps.amplim)
            ps.txc = np.array([-1.5,6])
            for txci in txc:
                ps.txc = txci
                print(ps.txc)
                
                for eqi in eqloc:
                    if eqi:
                        ps.readeqloc()
                    else:
                        ps.eqloc = None

                    ps.xcmanylfes()
                    try:
                        #ps.xcmanylfes()
                        #plotxccomb(ps,prt=True,mxdf=10)
                        plotxccomb(ps,prt=True,mxdf=5)
                        #plotxccomb(ps,prt=True,mxdf=3)
                    except:
                        print('Error with')
                        print(ps.amplim)

    if ps.fnum != 37140:
        ps = None
        #ps = None
        
    return ps

def allxzshf(ps=None):

    # grab the data
    #ps = initps(ps)

    # the reference
    ps2 = procsearch(ps=ps)
    #ps2.zerotshfs()
    ps2.besttshf()
    ps2.calcmedtshf()
    ps2.xcmanylfes()

    for shf in [0.1,0.25,0.5,0.75,1.]:
        ps.besttshf()
        ps.calcmedtshf()
        #ps.zerotshfs()
        ps.randtshfs(xshf=shf,addold=True)
        ps.usemshf=False
        ps.xcmanylfes()
        plotxccomb(ps,ps2=ps2,prt=True,mxdf=5)
    
    return

def liststats():

    fnums = famnum()

    fname=os.path.join(os.environ['WRITTEN'],'LFEArea','stationsummary')
    fl=open(fname,'w')

    stat=np.array([])
    for fnum in fnums:
        ps = procsearch(fnum)
        ps.wgttype='bymax_shifted'
        ps.readstacks()
        ps.checksnr()
        ps.cmpuse='both_horizontals'
        ps.pickcomp()
        stati=np.unique([tr.stats.station for tr in ps.stam])
        fl.write('{:d}'.format(ps.fnum)+' : ')
        fl.write('{:d}'.format(len(ps.stam))+' channels at '+
                 '{:d}'.format(len(stati))+' stations\n')
        stat=np.append(stat,[tr.stats.station for tr in ps.stam])
        stat=np.unique(stat)

        # takeoff angles and azimuths
        ps.calctakeang()
        ps.writetkang()

        plottemplates(ps)

    stat=np.unique(stat)
    fl.close()

    fname=os.path.join(os.environ['WRITTEN'],'LFEArea','stations')
    fl=open(fname,'w')
    for stati in stat:
        fl.write(stati+'\n')
    fl.close()
        

def initps(ps=None,juststack=False):

    if ps is None:
        ps = 37140

    if isinstance(ps,int):

        ps = procsearch(ps)
        ps.wgttype='bymax_shifted'
        ps.readstacks()
        ps.checksnr()
        # ps.taperbydur()
        ps.cmpuse='both_horizontals'
        ps.pickcomp()
        
        ps.readamps()
        ps.calcmedscale()
        ps.scaleamps()
        ps.meanamps()

        if not juststack:
            ps.readmanylfes()
            ps.besttshf()

        ps.txc = np.array([-0.1,2.9])
        ps.amplim=[-20.,20.]
        if not juststack:
            ps.eqloc = None
            ps.xcmanylfes()

    return ps


def cumlfe():
    i1=37140
    i2=37102

    tms1,xc1=lfetimes(i1)
    tms2,xc2=lfetimes(i2)

    c1=np.arange(0,tms1.size)
    c2=np.arange(0,tms2.size)

    dts1=np.array([matplotlib.dates.date2num(t.datetime) for t in tms1])
    dts2=np.array([matplotlib.dates.date2num(t.datetime) for t in tms2])

    gs,p=gridspec.GridSpec(1,1),[]
    gs.update(left=0.18,right=0.98)
    gs.update(bottom=0.06,top=0.98)
    gs.update(hspace=0.05,wspace=0.25)
    for k in range(0,1):
        p.append(plt.subplot(gs[k]))
    p=np.array(p)
    pm=p.reshape([p.size,1])
    p1=p[0]
    p2 = p1.twinx()

    ps=p1.get_position()
    #ps=[ps.x0+ps.width*0.6,ps.y0,ps.width*0.4,ps.height*0.4]
    ps=[ps.x0,ps.y0+ps.height*0.6,ps.width*0.4,ps.height*0.4]
    q1=plt.axes(ps)
    q2=q1.twinx()

    cols=['firebrick','navy']
    p1.plot_date(dts1,c1,marker='o',linestyle='none',color=cols[0])
    p2.plot_date(dts2,c2,marker='o',linestyle='none',color=cols[1])

    q1.plot_date(dts1,c1,marker='o',linestyle='none',color=cols[0])
    q2.plot_date(dts2,c2,marker='o',linestyle='none',color=cols[1])

    fs=12
    
    p1.spines['left'].set_color(cols[0])
    p2.spines['right'].set_color(cols[1])
    p1.tick_params(axis='y',colors=cols[0])
    p2.tick_params(axis='y',colors=cols[1])
    p1.set_ylabel('cumulative LFEs for family '+str(i1),fontsize=fs)
    p2.set_ylabel('cumulative LFEs for family '+str(i2),fontsize=fs)
    p1.yaxis.label.set_color(cols[0])
    p2.yaxis.label.set_color(cols[1])

    q1.spines['left'].set_color(cols[0])
    q2.spines['right'].set_color(cols[1])
    q1.tick_params(axis='y',colors=cols[0])
    q2.tick_params(axis='y',colors=cols[1])
    q1.set_ylabel('cumulative LFEs for family '+str(i1),fontsize=fs)
    q2.set_ylabel('cumulative LFEs for family '+str(i2),fontsize=fs)
    q1.yaxis.label.set_color(cols[0])
    q2.yaxis.label.set_color(cols[1])

def cumlfe2():
    i1=32192
    i2=37102

    tms1,xc1=lfetimes(i1)
    tms2,xc2=lfetimes(i2)

    c1=np.arange(0,tms1.size)
    c2=np.arange(0,tms2.size)

    tlmq=[obspy.UTCDateTime(2013,6,1),obspy.UTCDateTime(2015,6,1)]
    tlmqi=[matplotlib.dates.date2num(t.datetime) for t in tlmq]
    tlmp=[obspy.UTCDateTime(2013,10,7),obspy.UTCDateTime(2013,10,13)]
    tlmpi=[matplotlib.dates.date2num(t.datetime) for t in tlmp]
    dts1=np.array([matplotlib.dates.date2num(t.datetime) for t in tms1])
    dts2=np.array([matplotlib.dates.date2num(t.datetime) for t in tms2])

    fs=23
    plt.close()
    f=plt.figure(figsize=(8,4))
    gs,p=gridspec.GridSpec(1,1),[]
    gs.update(left=0.1,right=0.99)
    gs.update(bottom=0.1,top=0.98)
    gs.update(hspace=0.05,wspace=0.25)
    for k in range(0,1):
        p.append(plt.subplot(gs[k]))
    p=np.array(p)
    pm=p.reshape([p.size,1])
    p1=p[0]
    p2 = p1.twinx()

    ps=p1.get_position()
    ps=[ps.x0+ps.width*0.5,ps.y0+0.05,ps.width*0.5,ps.height*0.5]
    #ps=[ps.x0,ps.y0+ps.height*0.6,ps.width*0.4,ps.height*0.4]
    q1=plt.axes(ps)
    q1.xaxis.set_label_position('top')
    q1.xaxis.tick_top()
    q2=q1.twinx()

    lmp=np.interp(tlmpi,dts1,c1)
    cols=['firebrick','navy']
    p1.plot_date(dts1,c1-lmp[0],marker='o',linestyle='none',
                 color=cols[0],markersize=10)
    p2.plot_date(dts2,c2,marker='o',linestyle='none',color=cols[1])

    lmq=np.interp(tlmqi,dts1,c1)
    q1.plot_date(dts1,c1-lmq[0],marker='o',linestyle='none',color=cols[0])
    q2.plot_date(dts2,c2,marker='o',linestyle='none',color=cols[1])


    p1.yaxis.set_tick_params(labelsize=fs)
    p1.xaxis.set_tick_params(labelsize=fs)
    q1.yaxis.set_tick_params(labelsize=fs)
    q1.xaxis.set_tick_params(labelsize=fs)
    p1.set_yticks([0,250])

    q1.set_xlim(tlmqi)
    p1.set_xlim(tlmpi)
    p1.set_ylim(general.minmax(lmp-lmp[0],1.1))
    q1.set_ylim(lmq-lmq[0])

    
    p1.spines['left'].set_color(cols[0])
    p2.spines['right'].set_color(cols[1])
    p1.tick_params(axis='y',colors=cols[0])
    p2.tick_params(axis='y',colors=cols[1])
    p1.set_ylabel('cumulative LFEs for family '+str(i1),fontsize=fs)
    p2.set_ylabel('cumulative LFEs for family '+str(i2),fontsize=fs)
    p1.yaxis.label.set_color(cols[0])
    p2.yaxis.label.set_color(cols[1])

    q1.spines['left'].set_color(cols[0])
    q2.spines['right'].set_color(cols[1])
    q1.tick_params(axis='y',colors=cols[0])
    q2.tick_params(axis='y',colors=cols[1])
    q1.set_ylabel('cumulative LFEs for family '+str(i1),fontsize=fs)
    q2.set_ylabel('cumulative LFEs for family '+str(i2),fontsize=fs)
    q1.yaxis.label.set_color(cols[0])
    q2.yaxis.label.set_color(cols[1])

    p2.remove()
    q2.remove()

    q1.xaxis.set_major_locator(matplotlib.dates.YearLocator())
    yearsFmt = matplotlib.dates.DateFormatter('%Y')
    q1.xaxis.set_major_formatter(yearsFmt)
    q1.set_yticks([])
    q1.set_ylabel('')
    p1.yaxis.set_label_coords(-.05,0.5)
    # p1.set_ylabel('# of LFES',position=(0.05,0.5),
    #               transform=plt.gcf().transFigure)
    
    dayFmt = matplotlib.dates.DateFormatter('%d-%b')
    p1.xaxis.set_major_formatter(dayFmt)

    q1.axvspan(tlmpi[0],tlmpi[1],0,1,color='lightgray',
               zorder=0)
    p1.set_ylabel('# of LFEs',fontsize=fs)

    lbl = ['','8-Oct','','','','12-Oct','']
    p1.set_xticklabels(lbl)
    
    graphical.printfigure('VLcumlfe',f)
    
def summaryfig(ps=None):

    if ps is None:
        ps = 37140

    if isinstance(ps,int):

        ps = procsearch(ps)
        ps.readstacks()
        ps.checksnr()
        ps.cmpuse='both_horizontals'
        ps.pickcomp()
        
        ps.readamps()
        ps.calcmedscale()
        ps.scaleamps()
        ps.meanamps()

        ps.readmanylfes()

        ps.besttshf()
        
        ps.txc = np.array([-0.1,3.9])
        ps.amplim=[-20.,20.]
        ps.eqloc = None
        ps.xcmanylfes()
        
    # families to add
    fadd=np.array([37102, 70316,27270, 45688, 77401, 9707])
    fadd=np.array([37102,9707,77401,27270,45688,70316])
    # fadd=np.array([9707,77401,27270,37102,70316])
    Nf=fadd.size+1
    
    # plot the results
    f,p,rax,fname,nused=plotxccomb(ps,prt=False,mxdf=5,adjsum=True)
    f.set_size_inches(6,12)

    xvl=[0.12,0.98]
    wd=np.diff(xvl)[0]
    yvl=np.linspace(0.06,0.96,Nf+2)
    yvl=yvl-0.02
    hsp=0.01
    ht=np.diff(yvl)[0]-hsp
    yvl=np.flipud(yvl[0:-1])
    p[0].set_position([xvl[0],yvl[0],wd,ht+0.02])
    rax[0].set_position([xvl[0],yvl[0],wd,ht+0.02])
    p[1].set_position([xvl[0],yvl[1],wd,ht])
    rax[1].set_position([xvl[0],yvl[1],wd,ht])
    for k in range(0,Nf-1):
        p=np.append(p,plt.axes([xvl[0],yvl[k+2],wd,ht]))

    fs=10
    flm=[2.,20]
    xvl=np.exp(general.minmax(np.log(flm),0.87)[0])

    yvl=0.05
    lw=1.5
    for k in range(0,Nf-1):
        fnamei=fname.replace('{:0.0f}'.format(ps.fnum),
                             '{:0.0f}'.format(fadd[k]))
        freq,pc,pcb,pd,pdb,nlfe=readresults(fnamei)

        lbl = str(nlfe)+' LFEs from family '+str(fadd[k])
        # ht=p[k+2].text(xvl,yvl,lbl,fontsize=fs,horizontalalignment='left',
        #                verticalalignment='center',backgroundcolor='w',
        #                alpha=1.)


        hh,=p[k+2].plot(freq,pc,color='r',linewidth=lw)
        x = np.append(freq,np.flipud(freq))
        y = np.append(np.minimum(pcb[:,0],5),
                      np.flipud(np.maximum(pcb[:,1],-1)))
        y = np.minimum(np.maximum(y,-1),5)
        x,y=np.append(x,x[0]),np.append(y,y[0])
        ply = Polygon(np.vstack([x,y]).transpose())
        ply.set_edgecolor('none')
        ply.set_color('r')
        ply.set_alpha(0.3)
        ply.set_zorder(2)
        p[k+2].add_patch(ply)

        hh2,=p[k+2].plot(freq,pd,color='darkgoldenrod',linewidth=lw)
        x = np.append(freq,np.flipud(freq))
        y = np.append(np.minimum(pdb[:,0],5),
                      np.flipud(np.maximum(pdb[:,1],-1)))
        y = np.minimum(np.maximum(y,-1),5)
        x,y=np.append(x,x[0]),np.append(y,y[0])
        ply = Polygon(np.vstack([x,y]).transpose())
        ply.set_edgecolor('none')
        ply.set_color('darkgoldenrod')
        ply.set_alpha(0.3)
        ply.set_zorder(2)
        p[k+2].add_patch(ply)

        if fadd[k]==45688:
            lg=p[k+2].legend([hh2,hh],['$P_d$ / $ P_l$: directly coherent',
                                       '$P_c$ / $P_l$: inter-station coherent'],
                             fontsize=fs,loc='upper right')
            psi=p[k+1].get_position()
            lg.set_bbox_to_anchor((psi.x1+0.02,psi.y0+0.01),
                                  transform=plt.gcf().transFigure)
            lg.get_frame().set_alpha(0.95)

    for ph in p:
        ph.set_xscale('log')
        ph.set_xlim(flm)

    for ph in p[1:]:
        ph.set_ylim([-.1,1.2])
        ph.set_xticks([0,.5,1])
        ph.set_xticklabels(['0','','1'])
        ph.set_yticks([0.,0.6,1])
        # ph.plot(xlm,[0,0],color='dimgrey',linestyle=':',linewidth=0.5,zorder=1)
        # ph.plot(xlm,[.6,.6],color='dimgrey',linestyle=':',linewidth=0.5,zorder=1)
        # ph.plot(xlm,[1,1],color='dimgrey',linestyle=':',linewidth=0.5,zorder=1)

    xlm = np.array([flm[0],flm[1]*1.])
    p[0].set_yscale('log')
    xtk=np.arange(2,xlm[1]+0.0001)
    xtkl=['']*xtk.size
    ix,=np.where([xtki in [2,5,10,15,20] for xtki in xtk])
    for ixi in ix:
        xtkl[ixi]='{:0.0f}'.format(xtk[ixi])

    for ph in p:
        ph.set_xscale('log')
        ph.set_xlim(xlm)
        ph.set_ylabel('')
        #ph.set_yscale('log')
        ph.yaxis.set_tick_params(labelsize=fs)
        ph.xaxis.set_tick_params(labelsize=fs)
        ph.set_xticks(xtk)
        ph.minorticks_off()
        ph.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
        ph.set_xticklabels(xtkl)
        ph.grid(color='dimgrey',linestyle=':',linewidth=0.5)
    p[0].set_xticklabels([])
    

    p[1].set_ylim(np.array([-0.,1.2]))
    #p[0].set_xlabel('frequency (Hz)',fontsize=fs)
    imid=int(Nf/2)
    p[-1].set_xlabel('frequency (Hz)',fontsize=fs)
    p[imid+1].set_ylabel('coherent power / LFE power',fontsize=fs)
    p[0].set_ylabel('power / template',fontsize=fs)
    #p[0].set_ylim([1,1000])
    for ph in p[0:-1]:
        ph.set_xticklabels([])
        ph.set_xlabel('')
    
    xtk=np.arange(1,20)*100.
    xtkl=['']*xtk.size
    ix,=np.where([xtki in [300,500,1000] for xtki in xtk])
    for ixi in ix:
        xtkl[ixi]='{:0.0f}'.format(xtk[ixi])

    # add diameters to plot
    rax[0].xaxis.label.set_fontsize(fs)
    for raxi in rax:
        raxi.set_xlim(np.divide(1.,np.flipud(xlm))*1.3*3000.)
        raxi.set_xscale('log')
        raxi.tick_params(axis='x',labelsize=fs)
        raxi.invert_xaxis()
        raxi.set_xticks(xtk)
        raxi.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
        raxi.xaxis.set_tick_params(labelsize=fs)
        raxi.yaxis.set_tick_params(labelsize=fs)
    rax[0].set_xticklabels(xtkl)
    rax[1].set_xticklabels([])
    for raxi in rax:
        raxi.remove()
    # for ph in p[2:]:
    #     rax = ph.twiny()
    #     rax.set_xlim(np.divide(1.,np.flipud(xlm))*1.3*3000.)
    #     rax.set_xscale('log')
    #     rax.tick_params(axis='x',labelsize=fs)
    #     rax.invert_xaxis()
    #     rax.set_xticks(xtk)
    #     rax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    #     rax.xaxis.set_tick_params(labelsize=fs)
    #     rax.yaxis.set_tick_params(labelsize=fs)
    #     rax.set_xticklabels([])

    ht=graphical.cornerlabels(p[0:1],'ll',fontsize=fs-3,
                              xscl=0.02,yscl=0.07)
    ht=graphical.cornerlabels(p[1:],'ll',fontsize=fs-3,
                              xscl=0.02,yscl=0.07,lskip=1)

    lbl = str(nused)+' LFEs from family '+str(ps.fnum)
    ht=p[1].text(xvl,yvl,lbl,fontsize=fs,horizontalalignment='left',
                 verticalalignment='center',backgroundcolor='w',
                 alpha=1)
    ht.set_bbox({'facecolor':'w','alpha':0.5,'edgecolor':'none'})
    for k in range(0,Nf-1):
        fnamei=fname.replace('{:0.0f}'.format(ps.fnum),
                             '{:0.0f}'.format(fadd[k]))
        freq,pc,pcb,pd,pdb,nlfe=readresults(fnamei)

        lbl = str(nlfe)+' LFEs from family '+str(fadd[k])
        ht=p[k+2].text(xvl,yvl,lbl,fontsize=fs,horizontalalignment='left',
                       verticalalignment='center',backgroundcolor='w',
                       alpha=1)
        ht.set_bbox({'facecolor':'w','alpha':0.5,'edgecolor':'none'})

    lg1=p[0].legend(loc='center left',fontsize=fs-1)
    psi=p[0].get_position()
    lg1.set_bbox_to_anchor((psi.x0,psi.y1),transform=plt.gcf().transFigure)
    lg1.get_frame().set_alpha(0.95)
    
    graphical.printfigure('PCsummaryfig',f)
    graphical.printfigure('PCsummaryfig_'+fname[9:],f)


def readresults(fname):
    """
    :param       fname: file name in results directory
    :return       freq: frequencies
    :return         pc: inter-station coherent ratio
    :return        pcb: inter-station coherent ratio, bootstrapped
    :return         pd: direct coherent ratio
    :return        pdb: direct coherent ratio, bootstrapped
    :return       nlfe: number of LFEs used
    """
    
    fname=os.path.join(os.environ['DATA'],'TREMORAREA','Results',fname)
    #vls=np.loadtxt(fname,dtype=float,delimiter=',')

    fl = open(fname,'r')
    nlfe=int(fl.readline())

    # frequencies
    freq=np.array(fl.readline().split(',')).astype(float)
    flm=[2.,20.]
    #ii = np.logical_and(freq>=flm[0],freq<=flm[1])
    #freq = freq[ii]

    # read the inter-station coherence    
    pc=np.array(fl.readline().split(',')).astype(float)
    pcb1=np.array(fl.readline().split(',')).astype(float)
    pcb2=np.array(fl.readline().split(',')).astype(float)
    pcb=np.vstack([pcb1,pcb2]).T

    # read the direct coherence    
    pd=np.array(fl.readline().split(',')).astype(float)
    pdb1=np.array(fl.readline().split(',')).astype(float)
    pdb2=np.array(fl.readline().split(',')).astype(float)
    pdb=np.vstack([pdb1,pdb2]).T

    fl.close()
        
    return freq,pc,pcb,pd,pdb,nlfe
        
def famnum():

    fnums=np.array([37102, 70316, 27270, 45688, 77401, 9707, 37140])

    return fnums

def ffscale(vrvs=None,isrep=False,nlfe=1):
    """
    :param     vrvs: rupture velocity / wave propagation speed
    :param    isrep: is a repetitive rupture
    :param     nlfe: number of LFEs in composite
    :return     ffs: falloff frequency / (V_s/D)
    :return    ffsf: range of falloff frequency / (V_s/D)
    :return    vrvs: rupture velocity / wave propagation speed
    """

    if vrvs is None:
        vrvs=np.exp(np.linspace(np.log(0.02),np.log(1.2),500))
    vrvs=np.atleast_1d(vrvs)

    f1,f2,f3=0.05,0.6,1
    s1,s2,s3=0.5,1.2,1.1

    if nlfe==1:
        if isrep:
            f1,f2,f3=0.05,0.5,1
            s1,s2,s3=0.91,0.91,8
            s2,trash,trash=ffscale(vrvs=f2,isrep=False,nlfe=1)
            s2=s2*1.03
        else:
            f1,f2,f3=0.05,0.4,1
            s1,s2,s3=0.8,0.8,1.6

            f1,f2,f3=0.05,0.1,0.9
            s1,s2,s3=0.6,0.6,2.2

            f1,f2,f3=0.05,0.1,0.8
            s1,s2,s3=0.6,0.6,2.7

            f1,f2,f3=0.05,0.1,0.7
            s1,s2,s3=0.7,0.7,2.2

    elif nlfe>1:
        f1,f2,f3=0.05,0.5,0.9
        s1,s2,s3=0.59,0.59,2.0
        s2,trash,trash=ffscale(vrvs=f2,isrep=False,nlfe=1)
        s2=s2/1.03


    lf1,lf2,lf3=np.log(f1),np.log(f2),np.log(f3)
    ls1,ls2,ls3=np.log(s1),np.log(s2),np.log(s3)

    ffs=np.ndarray(vrvs.size,dtype=float)
    ffs[vrvs<=f1]=s1
    ffs[vrvs>=f3]=s3

    ii=np.logical_and(vrvs>=f1,vrvs<f2)
    scl=np.log(vrvs[ii]/f1)/np.log(f2/f1)
    scl=np.exp(ls1+scl*(ls2-ls1))
    ffs[ii]=scl

    ii=np.logical_and(vrvs>=f2,vrvs<f3)
    scl=np.log(vrvs[ii]/f2)/np.log(f3/f2)
    scl=np.exp(ls2+scl*(ls3-ls2))
    ffs[ii]=scl

    # fill in other values for repeaters
    if isrep:
        vls,trash,trash=ffscale(vrvs=vrvs,isrep=False,nlfe=1)
        ffs[vrvs<f2]=vls[vrvs<f2]*1.03
    elif nlfe>1:
        vls,trash,trash=ffscale(vrvs=vrvs,isrep=False,nlfe=1)
        ffs[vrvs<f2]=vls[vrvs<f2]/1.03
        
    ffsf=np.ndarray([vrvs.size,2],dtype=float)
    ffsf[:,0]=ffs*0.7
    if not isrep:
        ffsf[:,0]=0.9*0.6
    ffsf[:,1]=ffs*1.2

    return ffs,ffsf,vrvs

def durscale(nlfe=1,cntr=True,use70=True):
    """
    :param    nlfe: number of LFEs in composite
    :return     ts: scaling of duration with D/Vr
    :return    tsf: range of scaling of duration with D/Vr
    """

    cntr=int(cntr)
    
    if nlfe==1:
        if cntr==0:
            if use70:
                ts=0.30
                tsf=np.array([0.29,0.31])
            else:
                ts=0.5
                tsf=np.array([0.49,0.54])
        elif cntr==2:
            if use70:
                ts=0.30
                tsf=np.array([0.35,0.37])
            else:
                ts=0.65
                tsf=np.array([0.60,0.66])
        elif cntr==1:
            if use70:
                ts=0.26
                tsf=np.array([0.25,0.28])
            else:
                ts=0.41
                tsf=np.array([0.39,0.47])
    elif nlfe==2:
        if use70:
            ts=1.8
            tsf=np.array([1.75,1.9])
        else:
            ts=3
            tsf=np.array([2.5,3.5])

    ts=ts
    tsf=tsf
        
    return ts,tsf


def ffdscale(nlfe=1,isrep=False,vrvs=0.3):
    """
    :param    nlfe: number of LFEs in composite
    :return     fs: scaling of duration with Vr/D
    :return    fsf: range of scaling of duration with Vr/D
    """

    vrvsi=np.atleast_1d(vrvs)

    ons=np.ones(vrvsi.size)
    ons2=np.ones(vrvsi.size).reshape([vrvsi.size,1])

    fsl=2.8
    fslf=np.array([0.8,1.2])*fsl
    
    if nlfe==1:
        if not isrep:
            fs=fsl*ons

            x1=0.55
            a2=fsl*x1
            ii=vrvsi>x1
            fs[ii]=np.divide(a2,vrvsi[ii])

            # use inter-station coherence at high values
            fs,trash,trash=ffscale(vrvsi,nlfe=1,isrep=isrep)
            fs=np.divide(fs,vrvsi)/1.6

            ii=fs>fsl
            fs[ii]=fsl
            
            #fsf=np.multiply(ons2,np.array([1.3,1.7]).reshape([1,2]))
            fsf=np.vstack([fs*0.8,fs*1.2]).T
        else:
            # x1,x2=0.3,1
            # a1,a2=fsl*1.1,4
            
            # scl=np.log(vrvsi/x1)/np.log(x2/x1)
            # fs=np.exp(np.log(a1)+np.log(a2/a1)*scl)
            # fs[vrvsi>x2]=a2
            # fsf=np.vstack([fs*0.8,fs*1.2]).T

            # start with scaled nonrepeater
            fs,fsf=ffdscale(nlfe=1,isrep=False,vrvs=vrvsi)
            fs,fsf=fs*1.05,fsf*1.05
            
            # use inter-station coherence at high values
            fs2,trash,trash=ffscale(vrvsi,nlfe=1,isrep=isrep)
            fs2=np.minimum(fs2/1.5,2.5)
            fs2=np.divide(fs2,vrvsi)

            # pick the minimum
            ii=np.logical_and(fs2<fs,vrvsi>0.2)
            fs[ii]=fs2[ii]

            # but tend to high velocity value
            vmax = np.divide(1.6,vrvsi)
            x1,x2=0.4,0.7
            ii=vrvsi>=x2
            fs[ii]=vmax[ii]

            ii=np.logical_and(vrvsi<=x2,vrvsi>x1)
            scl=np.log(vrvsi[ii]/x1)/np.log(x2/x1)
            fs[ii]=np.multiply(1-scl,fs[ii])+np.multiply(vmax[ii],scl)

            fsf[:,0]=fs*0.7
            fsf[:,1]=fs*1.3



#            fsf=np.vstack([fs*0.8,fs*1.2]).T
    elif nlfe>=2:
        fs=0.25*ons
        fsf=np.array([0.8,1.2])*fs[0]
        fsf=np.multiply(ons2,fsf.reshape([1,2]))

        if isinstance(vrvs,float) or isinstance(vrvs,int):
            fs=fs[0]
            fsf=fsf[0,:]
        
    return fs,fsf

def ffdscaletr(nlfe=1,isrep=False,vrvs=0.3,trvod=None):
    """
    :param    nlfe: number of LFEs in composite
    :return     fs: scaling of duration with Vr/D
    :return    fsf: range of scaling of duration with Vr/D
    """

    if trvod==None:
        trvod=np.exp(np.linspace(np.log(0.1),np.log(10),300))
        
    # array of normalized rise times
    trvod=np.atleast_1d(trvod)
    vrvs=np.atleast_1d(vrvs)
    
    ons=np.ones(trvod.size)
    ons2=np.ones(trvod.size).reshape([trvod.size,1])

    # first get the short tr value
    fs,fsf=ffdscale(nlfe=nlfe,isrep=isrep,vrvs=vrvs)
    fs=fs[0]
    
    if nlfe==1 and not isrep:
        f1,f2,f3=0.05,0.7,11.
        s1,s2,s3=1.,1.,0.35

    elif nlfe==1 and isrep:
        f1,f2,f3=0.05,0.7,11
        s1,s2,s3=1.,1.,0.35

    elif nlfe>=2:
        fs=ons*fs
        fsf=np.array([0.8,1.2])*fs[0]
        fsf=np.multiply(ons2,fsf.reshape([1,2]))

    if nlfe==1:
        lf1,lf2,lf3=np.log(f1),np.log(f2),np.log(f3)
        ls1,ls2,ls3=np.log(s1),np.log(s2),np.log(s3)
        
        ffs=np.ndarray(trvod.size,dtype=float)
        ffs[trvod<=f1]=s1
        ffs[trvod>=f3]=s3
        
        ii=np.logical_and(trvod>=f1,trvod<f2)
        scl=np.log(trvod[ii]/f1)/np.log(f2/f1)
        scl=np.exp(ls1+scl*(ls2-ls1))
        ffs[ii]=scl
        
        ii=np.logical_and(trvod>=f2,trvod<f3)
        scl=np.log(trvod[ii]/f2)/np.log(f3/f2)
        scl=np.exp(ls2+scl*(ls3-ls2))
        ffs[ii]=scl
        
        ffsf=np.ndarray([trvod.size,2],dtype=float)
        ffsf[:,0]=ffs*0.7

        ffsf=ffsf*fs
        fs=ffs*fs

        
    return fs,fsf,trvod

def ffscaletr(nlfe=1,isrep=False,vrvs=0.3,trvod=None):
    """
    :param    nlfe: number of LFEs in composite
    :return     fs: scaling of duration with Vr/D
    :return    fsf: range of scaling of duration with Vr/D
    """

    if trvod==None:
        trvod=np.exp(np.linspace(np.log(0.1),np.log(10),300))
        
    # array of normalized rise times
    trvod=np.atleast_1d(trvod)
    vrvs=np.atleast_1d(vrvs)
    
    ons=np.ones(trvod.size)
    ons2=np.ones(trvod.size).reshape([trvod.size,1])

    # first get the short tr value
    fs,fsf,trash=ffscale(nlfe=nlfe,isrep=isrep,vrvs=vrvs)
    fs=fs[0]

    if nlfe==1 and not isrep:
        f1,f2,f3=0.05,2.,11
        s1,s2,s3=1.,1.,0.6

    elif nlfe==1 and isrep:
        f1,f2,f3=0.05,1.5,11
        s1,s2,s3=1.,1.,0.25

    elif nlfe>=2:
        f1,f2,f3=0.05,2.,11
        s1,s2,s3=1.,1.,0.6

    if nlfe>=1:
        lf1,lf2,lf3=np.log(f1),np.log(f2),np.log(f3)
        ls1,ls2,ls3=np.log(s1),np.log(s2),np.log(s3)
        
        ffs=np.ndarray(trvod.size,dtype=float)
        ffs[trvod<=f1]=s1
        ffs[trvod>=f3]=s3
        
        ii=np.logical_and(trvod>=f1,trvod<f2)
        scl=np.log(trvod[ii]/f1)/np.log(f2/f1)
        scl=np.exp(ls1+scl*(ls2-ls1))
        ffs[ii]=scl
        
        ii=np.logical_and(trvod>=f2,trvod<f3)
        scl=np.log(trvod[ii]/f2)/np.log(f3/f2)
        scl=np.exp(ls2+scl*(ls3-ls2))
        ffs[ii]=scl
        
        ffsf=np.ndarray([trvod.size,2],dtype=float)
        ffsf[:,0]=ffs*0.7

        ffsf=ffsf*fs
        fs=ffs*fs

        
    return fs,fsf,trvod

def plotmatch():

    plt.close()
    f = plt.figure(figsize=(6,15))
    gs,p=gridspec.GridSpec(4,1),[]
    gs.update(left=0.18,right=0.98)
    gs.update(bottom=0.06,top=0.98)
    gs.update(hspace=0.05,wspace=0.25)
    for k in range(0,4):
        p.append(plt.subplot(gs[k]))
    p=np.array(p)
    pm=p.reshape([p.size,1])

    fsi = 15
    trvod=5.
    
    # diameters to consider
    dlm=np.array([50,1500])

    # rupture velocities to allow
    vrvs=np.exp(np.linspace(np.log(0.05),np.log(1.2),500))
    vs=4000.

    #---------------FOR PC------------------------
    # the assumed rupture velocity and ff
    ff=9.
    ffmax=16.
    plotmax=True

    # compute min diameters for 1 LFE
    ffs,ffsf,vrvs=ffscale(vrvs=vrvs,nlfe=1)
    d = ffsf[:,1]*vs/ff
    d = ffs*vs/ff
    x = np.append(vrvs,np.flipud(vrvs))
    y = np.append(d,np.ones(vrvs.size)*dlm[0])
    y2 = np.append(d*ff/ffmax,np.ones(vrvs.size)*dlm[0])

    hpc=[]
    for k in [0]:
        ply = Polygon(np.vstack([x,y]).transpose())
        ply.set_edgecolor('none')
        ply.set_color('blue')
        ply.set_alpha(1)
        ply.set_hatch('\\')
        ply.set_fill(False)
        hpci=p[k].add_patch(ply)
        hpc.append(hpci)
        p[k].plot(vrvs,ffs*vs/ff,color='blue',linewidth=1.5)
        if plotmax:
            p[k].plot(vrvs,ffs*vs/ffmax,color='blue',linewidth=1.5)

            ply = Polygon(np.vstack([x,y2]).transpose())
            ply.set_edgecolor('none')
            ply.set_color('lightblue')
            ply.set_alpha(0.7)
            #ply.set_hatch('\\')
            ply.set_fill(True)
            hpc2=p[k].add_patch(ply)

    hpc=hpc[0]

    # for long rise time
    fs = []
    for vrvsi in vrvs:
        fsj,fsf,trash=ffscaletr(nlfe=1,vrvs=vrvsi,trvod=trvod)
        fs.append(fsj[0])
    fs=np.array(fs)

    d = fs*vs/ff
    x = np.append(vrvs,np.flipud(vrvs))
    y = np.append(d,np.ones(vrvs.size)*dlm[0])
    y2 = np.append(d*ff/ffmax,np.ones(vrvs.size)*dlm[0])

    ply = Polygon(np.vstack([x,y]).transpose())
    ply.set_edgecolor('none')
    ply.set_color('blue')
    ply.set_alpha(1)
    ply.set_hatch('\\')
    ply.set_fill(False)
    p[2].add_patch(ply)
    p[2].plot(vrvs,fs*vs/ff,color='blue',linewidth=1.5)
    if plotmax:
        p[2].plot(vrvs,fs*vs/ffmax,color='blue',linewidth=1.5)
        ply = Polygon(np.vstack([x,y2]).transpose())
        ply.set_edgecolor('none')
        ply.set_color('lightblue')
        ply.set_alpha(0.7)
        ply.set_fill(True)
        p[2].add_patch(ply)

    
    # compute min diameters for 2 LFEs
    ffs,ffsf,vrvs=ffscale(vrvs=vrvs,nlfe=2)
    p[1].plot(vrvs,ffs*vs/ff,color='blue',linewidth=1.5)
    if plotmax:
        p[1].plot(vrvs,ffs*vs/ffmax,color='blue',linewidth=1.5)


    d = ffsf[:,1]*vs/ff
    d = ffs*vs/ff
    x = np.append(vrvs,np.flipud(vrvs))
    y = np.append(d,np.ones(vrvs.size)*dlm[0])
    y2 = np.append(d*ff/ffmax,np.ones(vrvs.size)*dlm[0])

    ply = Polygon(np.vstack([x,y]).transpose())
    ply.set_edgecolor('none')
    ply.set_color('blue')
    ply.set_alpha(1)
    ply.set_hatch('\\')
    ply.set_fill(False)
    p[1].add_patch(ply)

    if plotmax:
        ply = Polygon(np.vstack([x,y2]).transpose())
        ply.set_edgecolor('none')
        ply.set_color('lightblue')
        ply.set_alpha(0.7)
        ply.set_fill(True)
        p[1].add_patch(ply)


    # compute min diameters for repetitive LFE
    ffs,ffsf,vrvs=ffscale(vrvs=vrvs,isrep=True,nlfe=1)
    d = ffsf[:,1]*vs/ff
    d = ffs*vs/ff
    x = np.append(vrvs,np.flipud(vrvs))
    y = np.append(d,np.ones(vrvs.size)*dlm[0])
    y2 = np.append(d*ff/ffmax,np.ones(vrvs.size)*dlm[0])

    p[3].plot(vrvs,ffs*vs/ff,color='blue',linewidth=1.5)
    if plotmax:
        p[3].plot(vrvs,ffs*vs/ffmax,color='blue',linewidth=1.5)
    ply = Polygon(np.vstack([x,y]).transpose())
    ply.set_edgecolor('none')
    ply.set_color('blue')
    ply.set_alpha(0.7)
    ply.set_hatch('\\')
    ply.set_fill(False)
    hpci=p[3].add_patch(ply)

    if plotmax:
        ply = Polygon(np.vstack([x,y2]).transpose())
        ply.set_edgecolor('none')
        ply.set_color('lightblue')
        ply.set_alpha(0.7)
        ply.set_fill(True)
        hpci=p[3].add_patch(ply)

    for ph in p.flatten():
        ph.set_xlim(general.minmax(vrvs))
        ph.set_xscale('log')
        ph.set_ylim(dlm)
        ph.set_yscale('log')

    #---------------FOR durations-------------------
    # range of durations
    drng=np.array([0.15,0.3])

    # to scale 90% of signal
    vl = np.hanning(1000)
    vl = np.cumsum(vl)
    vl = vl/vl[-1]
    twin = np.interp([0.15,0.85],vl,np.arange(0,vl.size))
    twin = np.diff(twin)[0]/vl.size
    drng=drng*twin
    tbest=np.array([0.19,0.22])*twin

    drng=drng+np.array([-1,1])*0.04*twin*0
    print(twin)
    print(drng)
    
    # compute diameters for 1 LFE
    ts,tsf=durscale(1,cntr=0)
    dbest=tbest/np.mean(tsf)*vs
    ts2,tsf2=durscale(1,cntr=0)
    tsf=general.minmax(np.append(tsf,tsf2))
    dscl=np.divide(drng,np.flipud(tsf))*vs
    x = np.append(vrvs,np.flipud(vrvs))
    y = np.append(dscl[0]*vrvs,np.flipud(vrvs)*dscl[1])

    ply = Polygon(np.vstack([x,y]).transpose())
    ply.set_edgecolor('none')
    ply.set_color('red')
    ply.set_alpha(0.3)
    #ply.set_hatch('|')
    ply.set_fill(True)
    hdur=p[0].add_patch(ply)
    p[0].plot(vrvs,dbest[0]*vrvs,color='r',linewidth=2.5)
    p[0].plot(vrvs,dbest[1]*vrvs,color='r',linewidth=2.5)


    ts,tsf=durscale(1,cntr=1)
    dbest=tbest/np.mean(tsf)*vs
    ts2,tsf2=durscale(1,cntr=1)
    tsf=general.minmax(np.append(tsf,tsf2))
    dscl=np.divide(drng,np.flipud(tsf))*vs
    x = np.append(vrvs,np.flipud(vrvs))
    y = np.append(dscl[0]*vrvs,np.flipud(vrvs)*dscl[1])


    ply = Polygon(np.vstack([x,y]).transpose())
    ply.set_edgecolor('none')
    ply.set_color('red')
    ply.set_alpha(0.3)
    #ply.set_hatch('|')
    ply.set_fill(True)
    p[3].add_patch(ply)

    p[3].plot(vrvs,dbest[0]*vrvs,color='r',linewidth=2.5)
    p[3].plot(vrvs,dbest[1]*vrvs,color='r',linewidth=2.5)


    # compute diameters for 2 LFEs
    ts,tsf=durscale(2)
    dbest=tbest/np.mean(tsf)*vs
    dscl=np.divide(drng,np.flipud(tsf))*vs
    x = np.append(vrvs,np.flipud(vrvs))
    y = np.append(dscl[0]*vrvs,np.flipud(vrvs)*dscl[1])

    ply = Polygon(np.vstack([x,y]).transpose())
    ply.set_edgecolor('none')
    ply.set_color('red')
    ply.set_alpha(0.3)
    #ply.set_hatch('|')
    ply.set_fill(True)
    p[1].add_patch(ply)

    p[1].plot(vrvs,dbest[0]*vrvs,color='r',linewidth=2.5)
    p[1].plot(vrvs,dbest[1]*vrvs,color='r',linewidth=2.5)


    # and for long rise time
    ts,tsf=durscale(1,cntr=0)
    ts2,tsf2=durscale(1,cntr=2)
    tsf=general.minmax(np.append(tsf,tsf2))
    tsf=np.array([0.4,0.5])*trvod
    dscl=np.divide(drng,np.flipud(tsf))*vs
    dbest=tbest/np.mean(tsf)*vs

    x = np.append(vrvs,np.flipud(vrvs))
    y = np.append(np.ones(vrvs.size)*dlm[0],
                  np.flipud(dscl[1]*vrvs))
    y = np.append(dscl[0]*vrvs,
                  np.flipud(dscl[1]*vrvs))

    
    ply = Polygon(np.vstack([x,y]).transpose())
    ply.set_edgecolor('none')
    ply.set_color('red')
    ply.set_alpha(0.3)
    #ply.set_hatch('|')
    ply.set_fill(True)
    p[2].add_patch(ply)

    p[2].plot(vrvs,dbest[0]*vrvs,color='r',linewidth=2.5)
    p[2].plot(vrvs,dbest[1]*vrvs,color='r',linewidth=2.5)

    #--------FOR PD----------------------
    ffd = 5.
    col='orange'

    # for simple LFE
    fs,fsf=ffdscale(nlfe=1,vrvs=vrvs)
    fsf=fsf[:,1]/ffd
    fsf=fs/ffd
    x = np.append(vrvs,np.flipud(vrvs))
    y = np.append(np.multiply(vrvs,fsf)*vs,np.ones(vrvs.size)*dlm[0])

    matplotlib.rcParams['hatch.linewidth']=1.5

    ply = Polygon(np.vstack([x,y]).transpose())
    ply.set_edgecolor('none')
    ply.set_color(col)
    ply.set_linewidth(1.5)
    ply.set_alpha(1)
    ply.set_hatch('//')
    ply.set_fill(False)
    hpd=p[0].add_patch(ply)

    # for simple repetitive LFE
    fs,fsf=ffdscale(nlfe=1,isrep=True,vrvs=vrvs)
    fsf=fsf[:,1]/ffd
    fsf=fs/ffd
    x = np.append(vrvs,np.flipud(vrvs))
    y = np.append(np.multiply(vrvs,fsf)*vs,np.ones(vrvs.size)*dlm[0])

    ply = Polygon(np.vstack([x,y]).transpose())
    ply.set_edgecolor('none')
    ply.set_color(col)
    ply.set_alpha(1)
    ply.set_hatch('//')
    ply.set_fill(False)
    p[3].add_patch(ply)

    # for double LFE
    fs,fsf=ffdscale(nlfe=2,vrvs=vrvs)
    fsf=fsf[:,1]/ffd
    fsf=fs/ffd
    x = np.append(vrvs,np.flipud(vrvs))
    y = np.append(np.multiply(vrvs,fsf)*vs,np.ones(vrvs.size)*dlm[0])

    ply = Polygon(np.vstack([x,y]).transpose())
    ply.set_edgecolor('none')
    ply.set_color(col)
    ply.set_alpha(1)
    ply.set_hatch('//')
    ply.set_fill(False)
    p[1].add_patch(ply)

    # for simple LFE with long rise time
    fs = []
    for vrvsi in vrvs:
        fsj,fsf,trash=ffdscaletr(nlfe=1,vrvs=vrvsi,trvod=trvod)
        fs.append(fsj[0])
    fs=np.array(fs)
    fsf=fs/ffd
    x = np.append(vrvs,np.flipud(vrvs))
    y = np.append(np.multiply(vrvs,fsf)*vs,np.ones(vrvs.size)*dlm[0])

    ply = Polygon(np.vstack([x,y]).transpose())
    ply.set_edgecolor('none')
    ply.set_color(col)
    ply.set_alpha(1)
    ply.set_hatch('//')
    ply.set_fill(False)
    p[2].add_patch(ply)



    xtk=np.hstack([np.arange(vrvs[0],0.1,0.01),
                   np.arange(0.1,1,0.1),
                   np.arange(1.,vrvs[-1],1)])
    xtkl=['']*xtk.size
    ix,=np.where([xtki in [0.1,0.2,0.5,1] for xtki in xtk])
    for ixi in ix:
        xtkl[ixi]='{:g}'.format(xtk[ixi])

    ytk=np.hstack([np.arange(dlm[0],100,10),
                   np.arange(100,1000,100),
                   np.arange(1000,dlm[-1],1000)])
    ytkl=['']*ytk.size
    ix,=np.where([ytki in [100,200,500,1000] for ytki in ytk])
    for ixi in ix:
        ytkl[ixi]='{:g}'.format(ytk[ixi])

    xpl=np.array([0.1,0.2,0.5,1])
    for ph in p:
        for vl in xpl:
            ph.plot(np.ones(2)*vl,dlm,color='k',linestyle='--',
                    zorder=0,linewidth=0.5)
        
    for ph in p:
        ph.set_xscale('log')
        ph.set_xlim(general.minmax(vrvs))
        ph.set_yscale('log')
        ph.set_ylabel('LFE diameter (m)',fontsize=fsi)
        ph.get_yaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
        ph.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
        ph.yaxis.set_tick_params(labelsize=fsi)
        ph.xaxis.set_tick_params(labelsize=fsi)
        ph.minorticks_off()
        ph.set_xticks(xtk)
        ph.set_yticks(ytk)
        ph.set_yticklabels(ytkl)
        #ph.xaxis.grid(which='both',linestyle=':',color='k')
        #ph.yaxis.grid(which='both',linestyle=':',color='k')

    p[-1].set_xticklabels(xtkl)
    p[-1].set_xlabel('$V_r$ / $V_s$',fontsize=fsi)

    tlbl=['simple LFEs','composite LFEs',
          'simple LFEs with long rise time',
          'repetitive simple LFEs']
    xvl=np.exp(general.minmax(np.log(vrvs),0.85)[0])
    yvl=np.exp(general.minmax(np.log(dlm),0.87)[1])
    for k in range(0,4):
        ht=p[k].text(xvl,yvl,tlbl[k],horizontalalignment='left',
                     verticalalignment='center',fontsize=fsi,
                     backgroundcolor='w')
    graphical.delticklabels(pm)

    plt.sca(p[1])
    fortalk=False
    if fortalk:
        hpd.remove()
        lbl=['match inter-station $f_{fc}$',
             'match duration']
        hpc.set_fill(True)
        lg=p[0].legend([hpc,hdur,],lbl,loc='lower right',
                       fontsize=fsi)
    else:
        lbl=['inter-station\n$f_{fc}$, 37102',
             'inter-station\n$f_{fc}$, 37140',
             'duration','direct $f_{fd}$']
        lg=p[1].legend([hpc,hpc2,hdur,hpd],lbl,loc='lower right',
                       fontsize=fsi,bbox_to_anchor=(1,0.85))
        lg.set_title('to match')
        plt.setp(lg.get_title(),fontsize=fsi)
        ht=graphical.cornerlabels(p,'ul',fontsize=fsi-1,
                                  xscl=0.03,yscl=0.04,lskip=0)



    graphical.printfigure('PCplotmatch',f)
    
def plotmatchold():

    plt.close()
    f = plt.figure(figsize=(6,12))
    gs,p=gridspec.GridSpec(2,1),[]
    gs.update(left=0.1,right=0.97)
    gs.update(bottom=0.07,top=0.98)
    gs.update(hspace=0.07,wspace=0.25)
    for k in range(0,2):
        p.append(plt.subplot(gs[k]))
    p=np.array(p)
    pm=p.reshape([p.size,1])

    # compute scaling
    vrvs=np.exp(np.linspace(np.log(0.05),np.log(1.2),500))
    ffs,ffsf,vrvs=ffscale(vrvs=vrvs)

    # the assumed rupture velocity and ff
    ff=10.
    vs=4000.

    #-----------FOR 1 LFE------------------------------------

    # compute diameter
    d = ffs*vs/ff
    x = np.append(vrvs,np.flipud(vrvs))
    yd = np.append(ffsf[:,0]*vs/ff,np.flipud(ffsf[:,1]*vs/ff))

    # compute duration
    ts,tsf=durscale(nlfe=1)
    T = ts*np.divide(d,vs*vrvs)
    yT = np.append(tsf[0]*np.divide(d,vs*vrvs),
                   np.flipud(tsf[1]*np.divide(d,vs*vrvs)))
    vok=np.logical_and(yT[0:d.size]<=0.3,np.flipud(yT[d.size:]>=0.2))
    vok1=np.flipud(yT[d.size:]>=0.2)
    
    vokr=np.append(vok1,np.flipud(vok1))
    ydr=np.append(d,np.flipud(d))

    ply = Polygon(np.vstack([x,yd]).transpose())
    ply.set_edgecolor('none')
    ply.set_color('blue')
    ply.set_alpha(0.3)
    p[0].add_patch(ply)

    yd[~vokr]=ydr[~vokr]
    ply = Polygon(np.vstack([x,yd]).transpose())
    ply.set_edgecolor('none')
    ply.set_color('red')
    ply.set_alpha(0.3)
    p[0].add_patch(ply)

    ply = Polygon(np.vstack([x,yT]).transpose())
    ply.set_edgecolor('none')
    ply.set_color('blue')
    ply.set_alpha(0.3)
    p[1].add_patch(ply)

    p[0].plot(vrvs,d,color='navy')
    p[1].plot(vrvs,T,color='navy')


    #-----------FOR 2 LFEs------------------------------------

    # compute diameter
    d = ffs*vs/ff
    x = np.append(vrvs,np.flipud(vrvs))
    yd = np.append(ffsf[:,0]*vs/ff,np.flipud(ffsf[:,1]*vs/ff))

    # compute duration
    ts,tsf=durscale(nlfe=2)
    T = ts*np.divide(d,vs*vrvs)
    yT = np.append(tsf[0]*np.divide(d,vs*vrvs),
                   np.flipud(tsf[1]*np.divide(d,vs*vrvs)))
    vok=np.logical_and(yT[0:d.size]<=0.3,np.flipud(yT[d.size:]>=0.2))
    vok2=np.flipud(yT[d.size:]>=0.2)
    vok2=np.logical_and(vok2,~vok1)
    
    vokr=np.append(vok2,np.flipud(vok2))
    ydr=np.append(d,np.flipud(d))

    ply = Polygon(np.vstack([x,yd]).transpose())
    ply.set_edgecolor('none')
    ply.set_color('blue')
    ply.set_alpha(0.3)
    p[0].add_patch(ply)

    yd[~vokr]=ydr[~vokr]
    ply = Polygon(np.vstack([x,yd]).transpose())
    ply.set_edgecolor('none')
    ply.set_color('red')
    ply.set_alpha(0.3)
    p[0].add_patch(ply)

    ply = Polygon(np.vstack([x,yT]).transpose())
    ply.set_edgecolor('none')
    ply.set_color('blue')
    ply.set_alpha(0.3)
    p[1].add_patch(ply)

    p[0].plot(vrvs,d,color='navy')
    p[1].plot(vrvs,T,color='navy')
    
    fs=12
    p[0].set_ylabel('inferred max LFE diameter (m)',fontsize=fs)
    p[1].set_ylabel('inferred max LFE duration (s)',fontsize=fs)
    p[1].axhspan(0.2,0.3,0,1,color='red',alpha=0.2)
    for ph in p:
        ph.set_xscale('log')
        ph.set_xlim(general.minmax(vrvs))
        ph.set_yscale('log')
        ph.yaxis.set_tick_params(labelsize=fsi)
        ph.xaxis.set_tick_params(labelsize=fsi)
        ph.set_xlabel('$V_r$ / $V_s$',fontsize=fs)
        ph.get_yaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
        ph.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())

    p[0].set_ylim([30,800])
    xtk=np.append(np.arange(30,100,10),np.arange(100,801,100))
    xtkl=['']*xtk.size
    ix,=np.where([xtki in [50,100,200,400,800] for xtki in xtk])
    for ixi in ix:
        xtkl[ixi]='{:0.0f}'.format(xtk[ixi])
    p[0].set_yticks(xtk)
    p[0].set_yticklabels(xtkl)

    p[1].set_ylim([0.03,0.4])
    xtk=np.append(np.arange(0.03,0.1,0.01),np.arange(0.1,0.4,0.1))
    xtkl=['']*xtk.size
    ix,=np.where([xtki in [0.05,0.1,0.2,0.4] for xtki in xtk])
    for ixi in ix:
        xtkl[ixi]='{:g}'.format(xtk[ixi])
    p[1].set_yticks(xtk)
    p[1].set_yticklabels(xtkl)

    xtk=p[1].get_xlim()
    xtk=np.append(np.arange(xtk[0],0.1,0.01),np.arange(0.1,xtk[1],0.1))
    xtkl=['']*xtk.size
    ix,=np.where([xtki in [0.1,0.2,0.5,1] for xtki in xtk])
    for ixi in ix:
        xtkl[ixi]='{:g}'.format(xtk[ixi])
    p[1].set_xticks(xtk)
    p[0].set_xticks(xtk)
    p[1].set_xticklabels(xtkl)
    

    graphical.delticklabels(pm)
    
    graphical.printfigure('PCplotmatch',f)


def checkphscoh(ps,tm=None,st=None):

    # the time range
    if tm is None:
        tm,xc = biglfetimes()
        tm=tm[2]
        tlm=np.array([-10,-9])*3600.
        tm=[tm+tlm[0],tm+tlm[1]]

    # the templates
    sta = ps.stam.copy()
    ids = [tr.get_id() for tr in sta]

    if st is None:
        st = readdata(tm,ids)

def addsynth(st,sta,nadd=3,tbuf=10.,amp=1.):
    """
    :param      st: noise
    :param     sta: templates to add
    :param    nadd: number to add
    :param    tbuf: time buffer to allow on the edges
    """

    tref=st[0].stats.starttime
    tst=tref+np.median([tr.stats.starttime-tref for tr in st])
    tnd=tref+np.median([tr.stats.endtime-tref for tr in st])
    
    # times to add
    trng=[tbuf,tnd-tst-tbuf*2]
    trng=np.random.rand(nadd)*np.diff(trng)+trng[0]
    trng=np.array([tst+tmi for tmi in trng])

    # range to add
    tlm=[-7,9]
    pk='t3'
    pkave=tref+np.median([tr.stats.starttime+tr.stats.t3-tref for tr in sta])
    
    # added synthetics
    sts=obspy.Stream()
    for tr in sta:
        tri=st.select(id=tr.get_id())
        if tri:
            # trace to add to 
            tri=tri[0].copy()

            # random shift
            tri.data=np.roll(tri.data,np.random.choice(500))

            # grab part of the template
            tref=tr.stats.starttime+tr.stats.t3
            tra=tr.copy().trim(starttime=tref+tlm[0],endtime=tref+tlm[1])
            tra.taper(side='both',max_percentage=0.25,max_length=1,type='hann')
            tms=tra.times()+tlm[0]

            # add to the data
            tshf=tref-pkave
            for k in range(0,nadd):
                tmss=tri.times() + (tri.stats.starttime-trng[k])
                tri.data=tri.data+np.interp(tmss-tshf,tms,tra.data*amp)
            tri.stats.t3=tst-tri.stats.starttime+tshf
                
            # add to set
            sts.append(tri)

    return sts

def checkcohrand(sts,sta):
    """
    :param      sts: data
    :param      sta: templates
    """

    Cp=[]
    stsi=sts.copy()
    for k in range(0,100):
        # generate random noise
        for tr in stsi:
            tr.data=np.random.randn(tr.data.size)
        stsi.filter('bandpass',freqmin=2,freqmax=30.)

        # coherence
        vl = checkcoh(stsi,sta)
        Cp.append(vl['Cpstat'][0])

    Cp = np.array(Cp)
    perr=vl['stds'][0]

    return Cp,perr

def checkcohsev(st,sta,nadd=100,amp=0.5,stau=None):
    """
    :param      sts: data
    :param      sta: templates
    """

    Cp=[]

    for k in range(0,100):
        # data
        sts = addsynth(st,sta,amp=amp,nadd=nadd)

        # coherence
        vl = checkcoh(sts,stau)
        Cp.append(vl['Cpstat'][0])

    Cp = np.array(Cp)
    perr=vl['stds'][0]

    print(np.median(Cp))

    return Cp,perr

def checkcoh(sts,sta):
    """
    :param      sts: data
    :param      sta: templates
    """

    vl=phasecoh.phasecoh(sta,reftemp=None,shtemp='t3',
                         wintemp=[-.2,3.8],buftemp=0.5,
                         st2=sts,reflook=None,shlook='t3',
                         wlenlook=80.,tlook=np.array([50.]),
                         blim=[2.,10],shtry=None,shgrid=None)

    return vl

def readdata(tm,ids):
    """
    :param    tm: tim limit
    :param   ids: waveform ids
    """
    
    # time limits
    tbuf=20.
    t1g,t2g=(tm[0]-tbuf).timestamp,(tm[1]+tbuf).timestamp

    # open the database
    session = waveformdb.opendatabase('pksdrop')

    # identify the files of interest
    flsa=session.query(Waveform).yield_per(1000)
    fls=flsa.filter(or_(Waveform.net=='BP',Waveform.net=='PB'),
                    Waveform.starttime<=t2g,
                    Waveform.endtime>=t1g)
    session.close()

    # read them
    st = obspy.Stream()
    for fl in fls:
        print(fl)
        if fl.get_id() in ids:
            st=st+fl.waveform(t1=tm[0]-tbuf,t2=tm[1]+tbuf)
    st.detrend()
    st = st.merge()

    return st

def junk():

    # response corrections
    pre_filt = [0.2,0.4,30.,40.]
    for tr in st:
        try:
            responsecorr.removeresponsepz(tr,water_level=100,
                                          pre_filt=pre_filt)
        except:
            st.remove(tr)

    # trim
    st.trim(starttime=tm[0],endtime=tm[1])

    st.filter('bandpass',freqmin=2,freqmax=30)
    st.resample(sampling_rate=100)

        
    return st
