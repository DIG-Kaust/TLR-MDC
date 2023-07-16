##################################################################
# @copyright (c) 2021- King Abdullah University of Science and
#                      Technology (KAUST). All rights reserved.
#
# Authors: Matteo, Ravasi, Yuxi Hong
# Description: Multidimensional convolution
##################################################################

import os
import time
import argparse
import cupy as cp
import matplotlib.pyplot as plt

from os.path import join, exists
from time import sleep
from dotenv import load_dotenv
from scipy.signal import convolve
from mpi4py import MPI

from pylops.utils.wavelets import *
from pylops.utils.tapers import *
from mdctlr.inversiondist import MDCmixed
from mdctlr.utils import voronoi_volumes
from mdctlr.tlrmvm.tilematrix import TilematrixGPU_Ove3D


def main(parser):

    ######### INPUT PARAMS #########
    parser.add_argument("--AuxFile", type=str, default="AuxFile.npz", help="File with Auxiliary information for MDC")
    parser.add_argument("--MVMType", type=str, default="Dense", help="Type of MVM: Dense, TLR")
    parser.add_argument("--TLRType", type=str, default="fp32", help="TLR Precision: fp32, fp16, fp16int8, int8")
    parser.add_argument("--bandlen", type=int, default=10, help="TLR Band length")
    parser.add_argument("--nfmax", type=int, default=150, help="TLR Number of frequencies")
    parser.add_argument("--wavfreq", type=float, default=15, help="Ricker wavelet peak frequency used to convolve the input")
    parser.add_argument("--OrderType", type=str, default="normal", help="Matrix reordering method: normal, l1, hilbert")
    parser.add_argument("--ModeValue", type=int, default=8, help="Rank mode")
    parser.add_argument("--M", type=int, default=9801, help="Number of sources/rows in seismic frequency data")
    parser.add_argument("--N", type=int, default=9801, help="Number of receivers/columns in seismic frequency data")
    parser.add_argument("--nb", type=int, default=256, help="TLR Tile size")
    parser.add_argument("--threshold", type=str, default="0.001", help="TLR Error threshold")
    parser.add_argument("--repeat", type=int, default=1, help="Number of repeated MDC computation for statistics")

    parser.add_argument('--debug', default=True, action='store_true', help='Debug')

    args = parser.parse_args()

    ######### SETUP MPI #########
    comm = MPI.COMM_WORLD
    mpirank = comm.Get_rank()
    mpisize = comm.Get_size()
    t0all = time.time()

    ######### SETUP GPUs #########
    if mpirank == 0:
        print('Cuda count', cp.cuda.runtime.getDeviceCount())
        for idev in range(cp.cuda.runtime.getDeviceCount()):
            print(cp.cuda.runtime.getDeviceProperties(idev)['name'])

    cp.cuda.Device(device=mpirank).use()

    ######### PROBLEM PARAMETERS (should be lifted out into a config file #########
    bandlen = args.bandlen   # TLR bandlenght
    nfmax = args.nfmax       # max frequency for MDC (#samples)
    wavfreq = args.wavfreq   # Ricker wavelet peak frequency

    ######### DEFINE DATA AND FIGURES DIRECTORIES #########
    STORE_PATH=os.environ["STORE_PATH"]
    FIG_PATH=os.environ["FIG_PATH"]
    if args.MVMType != "Dense":
        if args.TLRType != 'fp16int8':
            args.MVMType = "TLR" + args.TLRType
            TARGET_FIG_PATH = join(FIG_PATH, f"MDC_MVMType{args.MVMType}_OrderType{args.OrderType}_Mode{args.ModeValue}")
        else:
            args.MVMType = "TLR" + args.TLRType + "_bandlen{bandlen}"
            TARGET_FIG_PATH = join(FIG_PATH, f"MDC_MVMType{args.MVMType}_OrderType{args.OrderType}_Mode{args.ModeValue}")
    else:
        TARGET_FIG_PATH = join(FIG_PATH, f"MDC_MVMType{args.MVMType}")

    # create figure folder is not available
    if mpirank == 0:
        if not exists(TARGET_FIG_PATH):
            os.mkdir(TARGET_FIG_PATH)
    comm.Barrier()

    if mpirank == 0:
        print("-" * 80)
        print("MDC APP")
        print("-" * 80)
        options = vars(args)
        for key, value in options.items():
            print(f'{key} = {value}')
        print("-" * 80)
        print("STORE_PATH", STORE_PATH)
        print("FIG_PATH", TARGET_FIG_PATH)
        print("-" * 80)

    ######### DEFINE FREQUENCIES TO ASSIGN TO EACH MPI PROCESS #########
    Totalfreqlist = [x for x in range(nfmax)]
    splitfreqlist = []
    cnt = 0
    reverse = False
    while cnt < nfmax:
        tmp = []
        idx = 0
        while idx < mpisize:
            tmp.append(cnt)
            cnt += 1
            if cnt >= nfmax:
                break
            idx += 1
        if reverse:
            splitfreqlist.append([x for x in tmp[::-1]])
        else:
            splitfreqlist.append([x for x in tmp])
        reverse = ~reverse
    Ownfreqlist = []
    for x in splitfreqlist:
        if len(x) > mpirank:
            Ownfreqlist.append(x[mpirank])
    sleep(mpirank * 0.1)
    if mpirank == 0:
        print('Frequencies allocation:')
    print(f"Rank {mpirank}: {Ownfreqlist}")
    print("-" * 80)

    ######### LOAD AUXILIARY INPUTS (GEOMETRY, SUBSURFACE WAVEFIELDS, WAVELET) AND PREPARE FOR MCK #########
    inputfile_aux = join(STORE_PATH, args.AuxFile)
    inputdata_aux = np.load(inputfile_aux)
    # Receivers
    r = inputdata_aux['recs'].T
    nr = r.shape[1]
    # Sources
    s = inputdata_aux['srcs'].T
    ns = s.shape[1]
    # Virtual points
    vs = inputdata_aux['vs']
    # Time axis
    #t = inputdata_aux['t']
    #ot, dt, nt = t[0], t[1], len(t)
    # Time axis
    ot, dt, nt = 0, 2.5e-3, 601
    t = np.arange(nt) * dt

    # Find area of each volume - note that areas at the edges and on vertex are unbounded,
    # we will assume that they are and use the minimum are for all points in this example
    vertex, vols = voronoi_volumes(r[:2].T)
    darea = np.min(np.unique(vols))
    #if mpirank == 0:
    #    print('Integration area %f' % darea)

    # Load subsurface wavefields
    G0sub = inputdata_aux['G0']
    wav = ricker(t[:51], wavfreq)[0]
    wav_c = np.argmax(wav)

    # Convolve with wavelet
    G0sub = np.apply_along_axis(convolve, 0, G0sub, wav, mode='full')
    G0sub = G0sub[wav_c:][:nt]

    # Rearrange inputs according to matrix re-arrangement
    if args.OrderType != 'normal':
        idx = np.load(join(STORE_PATH, 'Mck_rearrangement.npy'))
        G0sub = G0sub[:, idx]

    ######### CREATE TLR-MVM OPERATOR #########
    if mpirank == 0:
        print('Loading Kernel of MDC Operator...')
        print("-" * 80)
    comm.Barrier()

    if args.MVMType == "Dense":
        # Load dense kernel (need to check it...)
        pass
        # dev = cp.cuda.Device(mpirank)
        # dev.use()
        # t0 = time.time()
        # mvmops = DenseGPU(Ownfreqlist, Totalfreqlist, splitfreqlist, args.nfmax, STORE_PATH)
        # t1 = time.time()
        # if mpirank == 0:
        #     print("Init dense GPU Time is ", t1-t0)
    else:
        # Load TLR kernel
        mvmops = TilematrixGPU_Ove3D(args.M, args.N, args.nb, 
                                     synthetic=False, datafolder=join(STORE_PATH,'compresseddata'), 
                                     order=args.OrderType, acc=args.threshold, 
                                     freqlist=Ownfreqlist, suffix="Mck_freqslice_")
        mvmops.estimategpumemory()
        mvmops.loaduvbuffer()
        mvmops.setcolB(1) # just 1 point
        
        mvmops.Ownfreqlist = Ownfreqlist
        mvmops.Splitfreqlist = splitfreqlist
        print("-" * 80)

    ######### CREATE MDC OPERATOR #########
    if mpirank == 0:
        print('Creating MDC Operator...')
        print("-" * 80)
    comm.Barrier()

    dRop = MDCmixed(mvmops, ns, nr, nt=2*nt-1, nfreq=nfmax, nv=1, dt=dt, dr=2*darea, nb=args.nb,  acc=args.threshold,
                    prescaled=True, datafolder=join(STORE_PATH, 'compresseddata'), conj=False)

    ######### CREATE DATA FOR MDC #########
    if mpirank == 0:
        print('Creating Input data...')
        print("-" * 80)
    comm.Barrier()

    # Input wavefield for MDC (chosen as direct focusing function for Mck)
    dfd_plus = np.concatenate((np.fliplr(G0sub.T).T, np.zeros((nt-1, nr), dtype=np.float32)))
    dfd_plus /= dfd_plus.max()
    dfd_plus = cp.asarray(dfd_plus) # move to gpu
    if mpirank == 0:
        print('dfd_plus', dfd_plus.min(), dfd_plus.max())
        print(dfd_plus.shape, dfd_plus.size, dRop)
    
    # Compute forward
    ctimes = []
    for _ in range(args.repeat):
        t0 = time.time()
        dforward = dRop.matvec(dfd_plus.ravel())
        if mpirank == 0:
            t1 = time.time()
            ctimes.append(t1 - t0)
            print(f"MDC : {t1 - t0} s.")
    dforward = cp.asnumpy(dforward.reshape(2 * nt - 1, nr)) # move to back to cpu and reshape
    if mpirank == 0:
        print('dforward', dforward.min(), dforward.max())
    
    # Report statistics
    if mpirank == 0:
        ctimes = np.array(ctimes)
        print(f"MDC timing: {ctimes.mean()} +- {ctimes.std()} s.")

    # Rearrange output according to matrix re-arrangement
    if args.OrderType != 'normal':
        dforward_reshuffled = np.zeros_like(dforward)
        dforward_reshuffled[:, idx] = dforward
    else:
        dforward_reshuffled = dforward

    if mpirank == 0 and args.debug:
        fig, ax = plt.subplots(1, 1, sharey=True, figsize=(16, 7))
        ax.imshow(dforward_reshuffled, cmap='gray', extent=(0, nr, t[-1], -t[-1]))
        ax.set_title(r'$f_d^+$'), ax.set_xlabel(r'$x_R$'), ax.set_ylabel(r'$t$')
        ax.axis('tight')
        plt.savefig(join(TARGET_FIG_PATH, 'dfplus.png'), bbox_inches='tight')

    if mpirank == 0:
        t1all = time.time()
        print(f"Done! Total time : {t1all - t0all} s.")
        print("-" * 80)
    comm.Barrier()

if __name__ == "__main__":
    description = '3D Multi-Dimensional Convolution with TLR-MDC and matrix reordering'
    main(argparse.ArgumentParser(description=description))


# TO DO:
# - Dense
# - TLR gives NaN!