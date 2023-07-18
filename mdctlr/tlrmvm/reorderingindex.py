import numpy as np
from hilbertcurve.hilbertcurve import HilbertCurve


def gethilbertindex(ny, nx, dy, dx, osx, dsx, osy, dsy, orx, drx, ory, dry, psrc, prec):
    # Model size
    y, x = np.arange(ny)*dy, np.arange(nx)*dx

    # Sources
    srcx = np.arange(osx, x[-1]-osx, dsx)
    srcy = np.arange(osy, y[-1]-osy, dsy)
    SRCY, SRCX = np.meshgrid(srcy, srcx, indexing='ij')
    SRCX, SRCY = SRCX.ravel(), SRCY.ravel()
    #zipxy = [(x[0]-SRCX[0],x[1]-SRCY[0]) for x in zip(SRCX,SRCY)]
    # shift to original point and scale down
    SRCPoints = [(int(x[0]/dsx),int(x[1]/dsy)) for x in zip(SRCX-osx, SRCY-osy)]
    hilbert_curve_src = HilbertCurve(psrc, 2, -1)
    hilbertcodes_src = hilbert_curve_src.distances_from_points(SRCPoints)
    idx_src = np.argsort(hilbertcodes_src).astype(np.int32)
    
    # Receivers
    recx = np.arange(orx, x[-1]-orx, drx)
    recy = np.arange(ory, y[-1]-ory, dry)
    RECY, RECX = np.meshgrid(recy, recx, indexing='ij')
    RECX, RECY = RECX.ravel(), RECY.ravel()
    #zipxy = [(x[0]-RECX[0],x[1]-RECY[0]) for x in zip(RECX,RECY)]
    # shift to original point and scale down
    RECPoints = [(int(x[0]/drx),int(x[1]/dry)) for x in zip(RECX-orx,RECY-ory)]
    hilbert_curve_rec = HilbertCurve(prec, 2, -1)
    hilbertcodes_rec = hilbert_curve_rec.distances_from_points(RECPoints)
    idx_rec = np.argsort(hilbertcodes_rec).astype(np.int32)
    
    return idx_src, idx_rec
