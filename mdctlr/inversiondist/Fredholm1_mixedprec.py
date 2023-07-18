import numpy as np
import cupy as cp
from pylops import LinearOperator
from mpi4py import MPI
import time


class Fredholm1mixed(LinearOperator):
    r"""Fredholm integral of first kind with mixed precision TLR.

    Implement a multi-dimensional Fredholm integral of first kind using a kernel 
    stored in mixed precision TLR format. The entire MVM operation is shipped
    to the tlr-mvm C++/CUDA library via PyBind11.

    Parameters
    ----------
    TLRop : :obj:`tlrmvm.tilematrix.TilematrixGPU_Ove3D`
        TLR-MVM operator
    nb : :obj:`int`, optional
        Tile size of TLR compression
    acc : :obj:`str`, optional
        Accuracy of TLR compression
    nfreq : :obj:`int`
        Number of frequencies
    n : :obj:`int`
        Number of rows of kernel matrices
    m : :obj:`int`
        Number of columns of kernel matrices
    datafolder : :obj:`str`, optional
        Path of folder containing U and V bases for TLR-MVM
    conj : :obj:`str`, optional
        Perform Fredholm integral computation with complex conjugate of ``G``
    scaling : :obj:`float`, optional
        Scaling to apply to output (if None no scaling is applied)
    dtype : :obj:`str`, optional
        Dtype of operator

    Attributes
    ----------
    shape : :obj:`tuple`
        Operator shape
    explicit : :obj:`bool`
        Operator contains a matrix that can be solved
        explicitly (``True``) or not (``False``)

    """
    def __init__(self, TLRop, nb, acc, nfreq, n, m, datafolder,
                 conj=False, scaling=None, dtype='float64'):
        self.nb = nb
        self.acc = acc
        self.nfreq = nfreq
        self.n, self.m = n, m
        self.datafolder = datafolder
        self.conj = conj
        self.scaling = scaling
        self.tlrmat = TLRop
        self.shape = (self.nfreq * self.n, self.nfreq * self.m)
        self.dtype = np.dtype(dtype)
        self.explicit = False
        self.Ownfreqlist = self.tlrmat.Ownfreqlist
        self.Splitfreqlist = self.tlrmat.Splitfreqlist
        self.comm = MPI.COMM_WORLD
        self.mpirank = self.comm.Get_rank()
        self.mpisize = self.comm.Get_size()
        self.opcount = 0
        self.debug = False

    def _matvec(self, Invector):
        t0 = time.time()
        self.opcount += 1
        if self.conj:
            Invector = cp.conj(Invector)
        # Split input over frequencies across ranks
        spx = cp.split(Invector, self.nfreq)
        xlist = cp.concatenate([spx[i] for i in self.Ownfreqlist])
        # Run distributed tlrmvm
        ydev = self.tlrmat.tlrmvmgpuinput(xlist, transpose=False)
        ylist = np.split(ydev.get(), len(self.Ownfreqlist))
        # Reconstruct total output in each rank
        eachyfinal = np.zeros(self.nfreq * self.n).astype(np.csingle)
        for idx, ownfreq in enumerate(self.Ownfreqlist):
            eachyfinal[ownfreq * self.n : (ownfreq+1) * self.n] = ylist[idx]
        yfinal = np.zeros_like(eachyfinal).astype(np.csingle)
        self.comm.Allreduce(eachyfinal, yfinal)
        t1 = time.time()
        yfinal = cp.asarray(yfinal)
        if self.conj:
            yfinal = cp.conj(yfinal)  
        return yfinal

    def _rmatvec(self, Invector):
        t0 = time.time()
        self.opcount += 1
        if not self.conj:
            Invector = cp.conj(Invector)
        # Split input over frequencies across ranks
        spx = cp.split(Invector, self.nfreq)
        xlist = cp.concatenate([spx[i] for i in self.Ownfreqlist])
        # Run distributed tlrmvm
        ydev = self.tlrmat.tlrmvmgpuinput(xlist,transpose=True)
        ylist = np.split(ydev.get(),len(self.Ownfreqlist))
        # Reconstruct total output in each rank
        eachyfinal = np.zeros(self.nfreq * self.m).astype(np.csingle)
        for idx, ownfreq in enumerate(self.Ownfreqlist):
            eachyfinal[ownfreq * self.m : (ownfreq+1) * self.m] = ylist[idx]
        yfinal = np.zeros_like(eachyfinal).astype(np.csingle)
        self.comm.Allreduce(eachyfinal, yfinal)
        t1 = time.time()
        yfinal = cp.asarray(yfinal)
        if not self.conj:
            yfinal = cp.conj(yfinal)  
        return yfinal