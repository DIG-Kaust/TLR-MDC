import logging
import numpy as np

from pylops import Identity
from pylops.signalprocessing import FFT
from .Fredholm1_mixedprec import Fredholm1mixed


def _MDC(nt, nv, dt=1., dr=1., twosided=True, dtype=None,
         conj=False, prescaled=False,
         _Identity=Identity, _FFT=FFT,
         _Fredholm1=Fredholm1mixed, args_Identity={},
         args_FFT={}, args_Identity1={}, 
         args_FFT1={}, args_Fredholm1={}):
    r"""Multi-dimensional convolution.

    Used to be able to provide operators from different libraries to
    MDC. It operates in the same way as public method
    (PoststackLinearModelling) but has additional input parameters allowing
    passing a different operator and additional arguments to be passed to such
    operator.

    """
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

    # create MDC operator
    MDCop = F1opH * I1opH * Frop * Iop * Fop
    
    # force dtype to be real (as FFT operators assume real inputs and outputs)
    MDCop.dtype = rdtype

    return MDCop


def MDCmixed(TLRop, ns, nr, nt, nfreq, nv, dt=1., dr=1., twosided=True,
             dtype='complex64', fftengine='numpy',
             conj=False, prescaled=False,
             nb=128, acc='0.001', datafolder='.'):
    r"""Multi-dimensional convolution with mixed-precision TLR-MVM kernel.

    Apply multi-dimensional convolution between two datasets with mixed-precision 
    TLR-MVM kernel. Model and data should be provided after flattening 2- or 
    3-dimensional arrays of size :math:`[n_t \times n_r (\times n_{vs})]` and
    :math:`[n_t \times n_s (\times n_{vs})]` (or :math:`2*n_t-1` for
    ``twosided=True``), respectively.

    Parameters
    ----------
    TLRop : :obj:`tlrmvm.tilematrix.TilematrixGPU_Ove3D`
        TLR-MVM operator
    ns : :obj:`int`
        Number of samples along source axis
    nr : :obj:`int`
        Number of receiver along source axis
    nt : :obj:`int`
        Number of samples along time axis for model and data (note that this
        must be equal to ``2*n_t-1`` when working with ``twosided=True``.
    nfreq : :obj:`int`
        Number of frequencies
    nv : :obj:`int`
        Number of samples along virtual source axis
    dt : :obj:`float`, optional
        Sampling of time integration axis
    dr : :obj:`float`, optional
        Sampling of receiver integration axis
    twosided : :obj:`bool`, optional
        MDC operator has both negative and positive time (``True``) or
        only positive (``False``)
    dtype : :obj:`str`, optional
        *Deprecated*, will be removed in v2.0.0
    fftengine : :obj:`str`, optional
        Engine used for fft computation (``numpy`` or ``fftw``)
    conj : :obj:`str`, optional
        Perform Fredholm integral computation with complex conjugate of ``G``
    prescaled : :obj:`bool`, optional
        Apply scaling to kernel (``False``) or not (``False``) when performing
        spatial and temporal summations. In case ``prescaled=True``, the
        kernel is assumed to have been pre-scaled when passed to the MDC
        routine.
    nb : :obj:`int`, optional
        Tile size of TLR compression
    acc : :obj:`str`, optional
        Accuracy of TLR compression
    datafolder : :obj:`str`, optional
        Path of folder containing U and V bases for TLR-MVM


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
    
    return _MDC(nt, nv, dt=dt, dr=dr, twosided=twosided,
                dtype=dtype,
                conj=conj, prescaled=prescaled,
                args_FFT={'engine': fftengine},
                args_Fredholm1={'TLRop':TLRop, 'nb':nb, 'acc':acc, 'nfreq':nfreq,
                                'n':ns, 'm':nr,
                                'datafolder':datafolder, 'dtype':dtype})
