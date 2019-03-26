#!/usr/bin/env python3.6

import numpy as np
import scipy
import datetime
import obspy
import glob
import sys
import os

from os.path import join,expanduser

# Internal
from CascadiaUtils import *

# -------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------
def readstraindata(stn,channel='L',years='all'):
    '''
    Read processed strainmeter data
    Args:
            * stn     : station name
            * channel : 'L' (default) or 'R'
            * year    : list of years to read. 'all' (default), int, or list
    '''

    # Which years to read?
    if years == 'all':
        years = [2005,2006,2007,2008,2009,2010,\
                 2011,2012,2012,2014,2015,2016]

    elif type(years) is int:
        years = [years]

    # Get path to strain data
    paf = join(os.environ['STRAINDATA'],'PROCESSED2')

    # Create empty stream 
    S = obspy.Stream()

    # Loop in years to read data
    for yr in years:
        for tr in obspy.read(join(paf,'{}.*.{}.SAC'.format(stn,yr))):
            S.append(tr)

    return S
