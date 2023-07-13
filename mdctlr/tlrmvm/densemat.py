##################################################################
# @copyright (c) 2021- King Abdullah University of Science and
#                      Technology (KAUST). All rights reserved.
#
# Author: Yuxi Hong
# Description: TLRMVM python simluation class.
##################################################################
from fileinput import filename
import time
import pickle 
import os
from scipy.io import loadmat
from os.path import join,exists
import numpy as np
from enum import IntEnum
from tlrmvm.generatedataset import ApplyReordering


class DenseMat:
    def __init__(self, rankmode, order, freqid, datafolder, nb, acc, 
                 ntg, mtg, fake_value=None, order="hilbert", 
                 suffix=None) -> None:
        suffix = "" if suffix is None else suffix
        self.ntg = ntg
        self.mtg = mtg
        self.nb = nb
        self.datafolder = datafolder
        self.rankmode = rankmode
        self.order = order
        self.freqid = freqid
        self.acc = acc
        filenameprefix = 'Mode{}_Order{}_{}{}'.format(rankmode, order, suffix, freqid)
        self.rankfile = join(datafolder,
            '{}_Rmat_nb{}_acc{}.bin'.format(filenameprefix, nb, acc))
        self.ubasefile = join(datafolder,
            '{}_Ubases_nb{}_acc{}.bin'.format(filenameprefix, nb, acc))
        self.vbasefile = join(datafolder,
            '{}_Vbases_nb{}_acc{}.bin'.format(filenameprefix, nb, acc))

    def rank(self):
        rank = np.fromfile(self.rankfile, dtype=np.int32).reshape(self.ntg, self.mtg).T
        return rank

    def Ubases(self):
        u = np.fromfile(self.ubasefile, dtype=np.csingle)
        return u

    def Vbases(self):
        v = np.fromfile(self.vbasefile, dtype=np.csingle)
        return v

    def origin(self):
        self.originfile = join(self.datafolder, 'Mck_freqslices',
            'Mck_freqslice{}_sub1.mat'.format(self.freqid))
        A = loadmat(self.originfile)['Rfreq']
        A = ApplyReordering(A, self.order, self.nb)
        return A
    
