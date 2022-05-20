import logging
import numpy as np

from pylops import Identity, Transpose
from pylops.signalprocessing import FFT
from .Fredholm1_mixedprec import Fredholm1mixed

def _MDC(nt, nv, dt=1., dr=1., twosided=True, fast=None, dtype=None,
         transpose=True, saveGt=True, conj=False, prescaled=False,
         _Identity=Identity, _Transpose=Transpose, _FFT=FFT,
         _Fredholm1=Fredholm1mixed, args_Identity={}, args_Transpose={},
         args_FFT={}, args_Identity1={}, args_Transpose1={},
         args_FFT1={}, args_Fredholm1={}):
    r"""Multi-dimensional convolution.

    Used to be able to provide operators from different libraries to
    MDC. It operates in the same way as public method
    (PoststackLinearModelling) but has additional input parameters allowing
    passing a different operator and additional arguments to be passed to such
    operator.

    """
    # warnings.warn('A new implementation of MDC is provided in v1.5.0. This '
    #               'currently affects only the inner working of the operator, '
    #               'end-users can continue using the operator in the same way. '
    #               'Nevertheless, it is now recommended to start using the '
    #               'operator with transpose=True, as this behaviour will '
    #               'become default in version v2.0.0 and the behaviour with '
    #               'transpose=False will be deprecated.', FutureWarning)

    if twosided and nt % 2 == 0:
        raise ValueError('nt must be odd number')

    # find out dtype of G
    dtype = args_Fredholm1['dtype']
    rdtype = np.real(np.ones(1, dtype=dtype)).dtype

    # create Fredholm operator
    if prescaled:
        args_Fredholm1['scaling'] = 1.
    else:
        args_Fredholm1['scaling'] = (dr * dt * np.sqrt(nt))
    args_Fredholm1['conj'] = conj
    Frop = _Fredholm1(**args_Fredholm1)

    # create FFT operators
    nfmax, ns, nr = args_Fredholm1['nfreq'], args_Fredholm1['n'], args_Fredholm1['m']
    # ensure that nfmax is not bigger than allowed
    nfft = int(np.ceil((nt + 1) / 2))
    if nfmax > nfft:
        nfmax = nfft
        logging.warning('nfmax set equal to ceil[(nt+1)/2=%d]' % nfmax)

    Fop = _FFT(dims=(nt, nr, nv), dir=0, real=True,
               fftshift=twosided, dtype=rdtype, **args_FFT)
    F1op = _FFT(dims=(nt, ns, nv), dir=0, real=True,
                fftshift=False, dtype=rdtype, **args_FFT1)

    # create Identity operator to extract only relevant frequencies
    Iop = _Identity(N=nfmax * nr * nv, M=nfft * nr * nv,
                    inplace=True, dtype=dtype, **args_Identity)
    I1op = _Identity(N=nfmax * ns * nv, M=nfft * ns * nv,
                     inplace=True, dtype=dtype, **args_Identity1)
    F1opH = F1op.H
    I1opH = I1op.H

    # create transpose operator
    if transpose:
        dims = [nr, nt] if nv == 1 else [nr, nv, nt]
        axes = (1, 0) if nv == 1 else (2, 0, 1)
        Top = _Transpose(dims, axes, dtype=dtype, **args_Transpose)

        dims = [nt, ns] if nv == 1 else [nt, ns, nv]
        axes = (1, 0) if nv == 1 else (1, 2, 0)
        TopH = _Transpose(dims, axes, dtype=dtype, **args_Transpose1)

    # create MDC operator
    MDCop = F1opH * I1opH * Frop * Iop * Fop
    if transpose:
        MDCop = TopH * MDCop * Top

    # force dtype to be real (as FFT operators assume real inputs and outputs)
    MDCop.dtype = rdtype

    return MDCop


def MDCmixed(TLRop, ns, nr, nt, nfreq, nv, dt=1., dr=1., twosided=True, fast=None,
             dtype='complex64', fftengine='numpy', transpose=True,
             saveGt=True, conj=False, prescaled=False,
             nb=128, acc='0.001', datafolder='.'):
    r"""Multi-dimensional convolution with mixed-precision TLR.

    Apply multi-dimensional convolution between two datasets. If
    ``transpose=True``, model and data should be provided after flattening
    2- or 3-dimensional arrays of size :math:`[n_r (\times n_{vs}) \times n_t]`
    and :math:`[n_s (\times n_{vs}) \times n_t]` (or :math:`2*n_t-1` for
    ``twosided=True``), respectively. If ``transpose=False``, model and data
    should be provided after flattening 2- or 3-dimensional arrays of size
    :math:`[n_t \times n_r (\times n_{vs})]` and
    :math:`[n_t \times n_s (\times n_{vs})]` (or :math:`2*n_t-1` for
    ``twosided=True``), respectively.

    .. warning:: A new implementation of MDC is provided in v1.5.0. This
      currently affects only the inner working of the operator and end-users
      can use the operator in the same way as they used to do with the previous
      one. Nevertheless, it is now reccomended to use the operator with
      ``transpose=False``, as this behaviour will become default in version
      v2.0.0 and the behaviour with ``transpose=True`` will be deprecated.

    Parameters
    ----------
    U : :obj:`numpy.ndarray`
        Multi-dimensional convolution kernel U basis in frequency domain of size
        :math:`[n_s \times n_r \times n_{fmax}]` if ``transpose=True``
        or size :math:`[n_{fmax} \times n_s \times n_r]` if ``transpose=False``
    V : :obj:`numpy.ndarray`
        Multi-dimensional convolution kernel V basis in frequency domain of size
        :math:`[n_s \times n_r \times n_{fmax}]` if ``transpose=True``
        or size :math:`[n_{fmax} \times n_s \times n_r]` if ``transpose=False``
    nt : :obj:`int`
        Number of samples along time axis for model and data (note that this
        must be equal to ``2*n_t-1`` when working with ``twosided=True``.
    nv : :obj:`int`
        Number of samples along virtual source axis
    dt : :obj:`float`, optional
        Sampling of time integration axis
    dr : :obj:`float`, optional
        Sampling of receiver integration axis
    twosided : :obj:`bool`, optional
        MDC operator has both negative and positive time (``True``) or
        only positive (``False``)
    fast : :obj:`bool`, optional
        *Deprecated*, will be removed in v2.0.0
    dtype : :obj:`str`, optional
        *Deprecated*, will be removed in v2.0.0
    fftengine : :obj:`str`, optional
        Engine used for fft computation (``numpy`` or ``fftw``)
    transpose : :obj:`bool`, optional
        Transpose ``G`` and inputs such that time/frequency is placed in first
        dimension. This allows back-compatibility with v1.4.0 and older but
        will be removed in v2.0.0 where time/frequency axis will be required
        to be in first dimension for efficiency reasons.
    saveGt : :obj:`bool`, optional
        Save ``G`` and ``G^H`` to speed up the computation of adjoint of
        :class:`pylops.signalprocessing.Fredholm1` (``True``) or create
        ``G^H`` on-the-fly (``False``) Note that ``saveGt=True`` will be
        faster but double the amount of required memory
    conj : :obj:`str`, optional
        Perform Fredholm integral computation with complex conjugate of ``G``
    usematmul : :obj:`bool`, optional
        Use :func:`numpy.matmul` (``True``) or for-loop with :func:`numpy.dot`
        (``False``) in :py:class:`pylops.signalprocessing.Fredholm1` operator.
        Refer to Fredholm1 documentation for details.
    prescaled : :obj:`bool`, optional
        Apply scaling to kernel (``False``) or not (``False``) when performing
        spatial and temporal summations. In case ``prescaled=True``, the
        kernel is assumed to have been pre-scaled when passed to the MDC
        routine.

    Raises
    ------
    ValueError
        If ``nt`` is even and ``twosided=True``

    See Also
    --------
    MDD : Multi-dimensional deconvolution

    Notes
    -----
    The so-called multi-dimensional convolution (MDC) is a chained
    operator [1]_. It is composed of a forward Fourier transform,
    a multi-dimensional integration, and an inverse Fourier transform:

    .. math::
        y(t, s, v) = \mathscr{F}^{-1} \Big( \int_S G(f, s, r)
        \mathscr{F}(x(t, r, v)) dr \Big)

    which is discretized as follows:

    .. math::
        y(t, s, v) = \mathscr{F}^{-1} \Big( \sum_{i_r=0}^{n_r}
        (\sqrt{n_t} * d_t * d_r) G(f, s, i_r) \mathscr{F}(x(t, i_r, v)) \Big)

    where :math:`(\sqrt{n_t} * d_t * d_r)` is not applied if ``prescaled=True``.

    This operation can be discretized and performed by means of a
    linear operator

    .. math::
        \mathbf{D}= \mathbf{F}^H  \mathbf{G} \mathbf{F}

    where :math:`\mathbf{F}` is the Fourier transform applied along
    the time axis and :math:`\mathbf{G}` is the multi-dimensional
    convolution kernel.

    .. [1] Wapenaar, K., van der Neut, J., Ruigrok, E., Draganov, D., Hunziker,
       J., Slob, E., Thorbecke, J., and Snieder, R., "Seismic interferometry
       by crosscorrelation and by multi-dimensional deconvolution: a
       systematic comparison", Geophysical Journal International, vol. 185,
       pp. 1335-1364. 2011.

    """
    
    return _MDC(nt, nv, dt=dt, dr=dr, twosided=twosided, fast=fast,
                dtype=dtype, transpose=transpose, saveGt=saveGt,
                conj=conj, prescaled=prescaled,
                args_FFT={'engine': fftengine},
                args_Fredholm1={'TLRop':TLRop, 'nb':nb, 'acc':acc, 'nfreq':nfreq,
                                'n':ns, 'm':nr,
                                'datafolder':datafolder, 'dtype':dtype})
