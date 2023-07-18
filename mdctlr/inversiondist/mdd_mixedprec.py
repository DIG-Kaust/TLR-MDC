import logging
import numpy as np

from pylops import Identity
from pylops.signalprocessing import FFT
from .Fredholm1_mixedprec import Fredholm1mixed


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

    """
    if twosided and nt % 2 == 0:
        raise ValueError('nt must be odd number')

    # find out dtype of G
    rdtype = np.real(np.ones(1, dtype=dtype)).dtype

    # create Fredholm operator
    if prescaled:
        scaling = 1.
    else:
        scaling = (dr * dt * np.sqrt(nt))
    Frop = Fredholm1mixed(TLRop=TLRop, nb=nb, acc=acc, nfreq=nfreq,
                          scaling=scaling, conj=conj,
                          n=ns, m=nr, datafolder=datafolder, dtype=dtype)

    # create FFT operators
    nfmax = nfreq
    # ensure that nfmax is not bigger than allowed
    nfft = int(np.ceil((nt + 1) / 2))
    if nfmax > nfft:
        nfmax = nfft
        logging.warning('nfmax set equal to ceil[(nt+1)/2=%d]' % nfmax)

    Fop = FFT(dims=(nt, nr, nv), dir=0, real=True,
              fftshift=twosided, dtype=rdtype, engine=fftengine)
    F1op = FFT(dims=(nt, ns, nv), dir=0, real=True,
               fftshift=False, dtype=rdtype, engine=fftengine)

    # create Identity operator to extract only relevant frequencies
    Iop = Identity(N=nfmax * nr * nv, M=nfft * nr * nv,
                   inplace=True, dtype=dtype)
    I1op = Identity(N=nfmax * ns * nv, M=nfft * ns * nv,
                    inplace=True, dtype=dtype)
    F1opH = F1op.H
    I1opH = I1op.H

    # create MDC operator
    MDCop = F1opH * I1opH * Frop * Iop * Fop
    
    # force dtype to be real (as FFT operators assume real inputs and outputs)
    MDCop.dtype = rdtype

    return MDCop