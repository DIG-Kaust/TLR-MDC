import numpy as np

from pylops import LinearOperator
from pylops.utils.backend import get_array_module


class Fredholm1(LinearOperator):
    r"""Fredholm integral of first kind.

    Implement a multi-dimensional Fredholm integral of first kind. Note that if
    the integral is two dimensional, this can be directly implemented using
    :class:`pylops.basicoperators.MatrixMult`. A multi-dimensional
    Fredholm integral can be performed as a :class:`pylops.basicoperators.BlockDiag`
    operator of a series of :class:`pylops.basicoperators.MatrixMult`. However,
    here we take advantage of the structure of the kernel and perform it in a
    more efficient manner.

    Parameters
    ----------
    G : :obj:`numpy.ndarray` or :obj:`tuple`
        Multi-dimensional convolution kernel of size
        :math:`[n_{slice} \times n_x \times n_y]` or tuple of two lists each
        containing kernels of sizes
        :math:`[n_{slice} \times n_x \times k_{sl}]` and
        :math:`[n_{slice} \times k_{sl} \times n_y]` obtained via
        low-rank approximation (with possibly varying math:`k_{sl}` sizes)
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
        choice. Note that when ``G`` is a tuple this flag will be ignored.
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
    def __init__(self, G, nz=1, saveGt=True, usematmul=True, dtype='float64'):

        # Store G and create flag based on its type
        self.G = G
        self.Gpair = True if isinstance(G, tuple) else False
        if saveGt and not self.Gpair:
            self.GT = G.transpose((0, 2, 1)).conj()
        if saveGt and self.Gpair:
            self.GT = ([G[0][i].T.conj() for i in range(len(G[0]))],
                       [G[1][i].T.conj() for i in range(len(G[1]))])
        self.nz = nz
        if self.Gpair:
            self.nsl, self.nx, self.ny = \
                len(G[0]), G[0][0].shape[0], G[1][0].shape[1]
        else:
            self.nsl, self.nx, self.ny = G.shape
        self.usematmul = False if self.Gpair else usematmul
        self.shape = (self.nsl * self.nx * self.nz,
                      self.nsl * self.ny * self.nz)
        self.dtype = np.dtype(dtype)
        self.explicit = False

    def _matvec(self, x):
        ncp = get_array_module(x)
        x = ncp.squeeze(x.reshape(self.nsl, self.ny, self.nz))
        if self.usematmul:
            if self.nz == 1:
                x = x[..., ncp.newaxis]
            y = ncp.matmul(self.G, x)
        else:
            y = ncp.squeeze(ncp.zeros((self.nsl, self.nx, self.nz),
                                      dtype=self.dtype))
            if not self.Gpair:
                for isl in range(self.nsl):
                    y[isl] = ncp.dot(self.G[isl], x[isl])
            else:
                for isl in range(self.nsl):
                    y[isl] = \
                        ncp.dot(self.G[0][isl], ncp.dot(self.G[1][isl], x[isl]))
        return y.ravel()

    def _rmatvec(self, x):
        ncp = get_array_module(x)
        x = ncp.squeeze(x.reshape(self.nsl, self.nx, self.nz))
        if self.usematmul:
            if self.nz == 1:
                x = x[..., ncp.newaxis]
            if hasattr(self, 'GT'):
                y = ncp.matmul(self.GT, x)
            else:
                y = ncp.matmul(self.G.transpose((0, 2, 1)).conj(), x)
        else:
            y = ncp.squeeze(ncp.zeros((self.nsl, self.ny, self.nz),
                                      dtype=self.dtype))
            if hasattr(self, 'GT'):
                if not self.Gpair:
                    for isl in range(self.nsl):
                        y[isl] = ncp.dot(self.GT[isl], x[isl])
                else:
                    for isl in range(self.nsl):
                        y[isl] = ncp.dot(self.GT[1][isl],
                                         ncp.dot(self.GT[0][isl], x[isl]))
            else:
                if not self.Gpair:
                    for isl in range(self.nsl):
                        y[isl] = ncp.dot(self.G[isl].conj().T, x[isl])
                else:
                    for isl in range(self.nsl):
                        y[isl] = ncp.dot(self.G[1][isl].conj().T,
                                         ncp.dot(self.G[0][isl].conj().T, x[isl]))
        return y.ravel()
