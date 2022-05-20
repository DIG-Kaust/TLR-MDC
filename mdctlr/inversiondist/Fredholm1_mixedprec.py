import numpy as np
import cupy as cp
from pylops import LinearOperator
from mpi4py import MPI
import time


class Fredholm1mixed(LinearOperator):
    r"""Fredholm integral of first kind with mixed precision TLR.

    Implement a multi-dimensional Fredholm integral of first kind. Note that if
    the integral is two dimensional, this can be directly implemented using
    :class:`pylops.basicoperators.MatrixMult`. A multi-dimensional
    Fredholm integral can be performed as a :class:`pylops.basicoperators.BlockDiag`
    operator of a series of :class:`pylops.basicoperators.MatrixMult`. However,
    here we take advantage of the structure of the kernel and perform it in a
    more efficient manner.

    Parameters
    ----------
    U : :obj:`numpy.ndarray`
        Multi-dimensional convolution U basis of kernel of size
        :math:`[n_{slice} \times n_x \times n_y]`
    V : :obj:`numpy.ndarray`
        Multi-dimensional convolution U basis of kernel of size
        :math:`[n_{slice} \times n_x \times n_y]`
    nz : :obj:`numpy.ndarray`, optional
        Additional dimension of model
    saveGt : :obj:`bool`, optional
        Save ``G`` and ``G^H`` to speed up the computation of adjoint
        (``True``) or create ``G^H`` on-the-fly (``False``)
        Note that ``saveGt=True`` will double the amount of required memory
    usematmul : :obj:`bool`, optional
        Use :func:`numpy.matmul` (``True``) or for-loop with :func:`numpy.dot`
        (``False``). As it is not possible to define which approach is more
        performant (this is highly dependent on the size of ``G`` and input
        arrays as well as the hardware used in the compution), we advise users
        to time both methods for their specific problem prior to making a
        choice.
    dtype : :obj:`str`, optional
        Type of elements in input array.

    Attributes
    ----------
    shape : :obj:`tuple`
        Operator shape
    explicit : :obj:`bool`
        Operator contains a matrix that can be solved
        explicitly (``True``) or not (``False``)

    Notes
    -----
    A multi-dimensional Fredholm integral of first kind can be expressed as

    .. math::

        d(sl, x, z) = \int{G(sl, x, y) m(sl, y, z) dy}
        \quad \forall sl=1,n_{slice}

    on the other hand its adjoin is expressed as

    .. math::

        m(sl, y, z) = \int{G^*(sl, y, x) d(sl, x, z) dx}
        \quad \forall sl=1,n_{slice}

    In discrete form, this operator can be seen as a block-diagonal
    matrix multiplication:

    .. math::
        \begin{bmatrix}
            \mathbf{G}_{sl1}  & \mathbf{0}       &  ... & \mathbf{0} \\
            \mathbf{0}        & \mathbf{G}_{sl2} &  ... & \mathbf{0} \\
            ...               & ...              &  ... & ...        \\
            \mathbf{0}        & \mathbf{0}       &  ... & \mathbf{G}_{slN}
        \end{bmatrix}
        \begin{bmatrix}
            \mathbf{m}_{sl1}  \\
            \mathbf{m}_{sl2}  \\
            ...     \\
            \mathbf{m}_{slN}
        \end{bmatrix}

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
        self.shape = (self.nfreq * self.n,self.nfreq * self.m)
        self.dtype = np.dtype(dtype)
        self.explicit = False
        self.Ownfreqlist = self.tlrmat.Ownfreqlist
        self.Splitfreqlist = self.tlrmat.Splitfreqlist
        self.comm = MPI.COMM_WORLD
        self.mpirank = self.comm.Get_rank()
        self.mpisize = self.comm.Get_size()
        self.freqmap = {}
        self.opcount = 0
        for x in self.Splitfreqlist:
            for i in range(len(x)):
                self.freqmap[x[i]] = i
        self.debug = False
    def _matvec(self, Invector):
        t0 = time.time()
        Invector = cp.asnumpy(Invector)
        self.opcount += 1
        invecmax = np.max(np.abs(Invector))
        if self.debug:
            if self.mpirank == 0:
                print("matvec transpose and conjugate", False, self.conj, " for ", self.opcount)
        scalex = False
        if invecmax > 1e-12:
            scalex = True
            Invector /= invecmax
        self.tlrmat.SetTransposeConjugate(transpose=False, conjugate=self.conj)
        spx = np.split(Invector, self.nfreq)
        xlist = np.hstack([spx[i] for i in self.Ownfreqlist])
        y = self.tlrmat.MVM(xlist)
        if self.scaling is not None:
            y *= self.scaling
        if scalex:
            y *= invecmax
        sy = np.split(y, len(self.Ownfreqlist))
        eachyfinal = np.zeros_like(Invector).astype(np.csingle)
        for idx, ownfreq in enumerate(self.Ownfreqlist):
            eachyfinal[ownfreq * 9801 : (ownfreq+1)*9801] = sy[idx]
        yfinal = np.zeros_like(Invector).astype(np.csingle)
        self.comm.Allreduce(eachyfinal, yfinal)
        t1 = time.time()
        yfinal = cp.asarray(yfinal)
        if self.mpirank == 0:
            print("Fredholm matvec time: {:.6f} s.".format(t1-t0))
        return yfinal

    def _rmatvec(self, Invector):
        t0 = time.time()
        Invector = cp.asnumpy(Invector)
        self.opcount += 1
        invecmax = np.max(np.abs(Invector))
        scalex = False
        if invecmax > 1e-12:
            scalex = True
            Invector /= invecmax
        self.tlrmat.SetTransposeConjugate(transpose=True, conjugate=not self.conj)        
        spx = np.split(Invector, self.nfreq)
        xlist = np.hstack([spx[i] for i in self.Ownfreqlist])
        y = self.tlrmat.MVM(xlist)
        if self.scaling is not None:
            y *= self.scaling
        if scalex:
            y *= invecmax
        sy = np.split(y, len(self.Ownfreqlist))
        eachyfinal = np.zeros_like(Invector).astype(np.csingle)
        for idx, ownfreq in enumerate(self.Ownfreqlist):
            eachyfinal[ownfreq * 9801 : (ownfreq+1)*9801] = sy[idx]
        yfinal = np.zeros_like(Invector).astype(np.csingle)
        self.comm.Allreduce(eachyfinal, yfinal)
        t1 = time.time()
        yfinal = cp.asarray(yfinal)
        if self.mpirank == 0:
            print("Fredholm rmatvec time: {:.6f} s.".format(t1-t0))
        return yfinal