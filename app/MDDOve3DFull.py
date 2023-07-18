###################################################################
# @copyright (c) 2021- King Abdullah University of Science and
#                      Technology (KAUST). All rights reserved.
#
# Authors: Matteo Ravasi, Yuxi Hong
# Description: Multidimensional deconvolution of 3D Overthrust data
###################################################################

import os
import time
import argparse
import numpy as np
import cupy as cp
import matplotlib.pyplot as plt
#import zarr

from os.path import join, exists
from time import sleep
from scipy.signal import convolve
from mpi4py import MPI

from hilbertcurve.hilbertcurve import HilbertCurve
from pylops.utils.wavelets import *
from pylops.utils.tapers import *
from mdctlr.inversiondist import MDCmixed
from mdctlr.lsqr import lsqr
from mdctlr.tlrmvm.tilematrix import TilematrixGPU
from mdctlr.tlrmvm.reorderingindex import gethilbertindex


def main(parser):

    ######### INPUT PARAMS #########
    parser.add_argument("--AuxFile", type=str, default="AuxFile.npz", help="File with Auxiliary information for MDD")
    parser.add_argument("--DataFolder", type=str, default="compresseddata_full", help="Folder containing compressed data")
    parser.add_argument("--PupFolder", type=str, default="pupdata", help="Folder containing pup data")
    parser.add_argument("--FigFolder", type=str, default="MDDOve3D", help="Prefix of folder containing figures")
    parser.add_argument("--MVMType", type=str, default="Dense", help="Type of MVM: Dense, TLR")
    parser.add_argument("--TLRType", type=str, default="fp32", help="TLR Precision: fp32, fp16, fp16int8, int8")
    parser.add_argument("--bandlen", type=int, default=10, help="TLR Band length")
    parser.add_argument("--nfmax", type=int, default=150, help="TLR Number of frequencies")
    parser.add_argument("--OrderType", type=str, default="normal", help="Matrix reordering method: normal, l1, hilbert")
    parser.add_argument("--PHilbertSrc", type=int, default=12, help="Hilbert size of source axis (2^p)")
    parser.add_argument("--PHilbertRec", type=int, default=12, help="Hilbert size of receiver axis (2^p)")
    parser.add_argument("--ModeValue", type=int, default=8, help="Rank mode")
    parser.add_argument("--M", type=int, default=9801, help="Number of sources/rows in seismic frequency data")
    parser.add_argument("--N", type=int, default=9801, help="Number of receivers/columns in seismic frequency data")
    parser.add_argument("--nb", type=int, default=256, help="TLR Tile size")
    parser.add_argument("--threshold", type=str, default="0.001", help="TLR Error threshold")
    parser.add_argument("--vs", type=str, default=9115, help="Virtual source")
    parser.add_argument("--niter", type=int, default=30, help="Number of iterations of MDD")
    parser.add_argument("--damp", type=float, default=0.2, help="Damping of MDD")

    parser.add_argument('--debug', default=True, action='store_true', help='Debug')

    args = parser.parse_args()

    ######### SETUP MPI #########
    comm = MPI.COMM_WORLD
    mpirank = comm.Get_rank()
    mpisize = comm.Get_size()
    t0all = time.time()

    ######### SETUP GPUs #########
    if mpirank == 0 and args.debug:
        print('Cuda count', cp.cuda.runtime.getDeviceCount())
        for idev in range(cp.cuda.runtime.getDeviceCount()):
            print(f'Rank{mpirank} using GPU:', cp.cuda.runtime.getDeviceProperties(idev)['name'])

    cp.cuda.Device(device=mpirank).use()
    mempool = cp.get_default_memory_pool()

    ######### PROBLEM PARAMETERS (should be lifted out into a config file #########
    nfmax = args.nfmax  # max frequency for MDC (#samples)
    n_iter = args.niter # number of iterations
    damp = args.damp    # damping

    ######### DEFINE DATA AND FIGURES DIRECTORIES #########
    STORE_PATH=os.environ["STORE_PATH"]
    FIG_PATH=os.environ["FIG_PATH"]
    if args.MVMType != "Dense":
        if args.TLRType != 'fp16int8':
            args.MVMType = "TLR" + args.TLRType
            TARGET_FIG_PATH = join(FIG_PATH, f"{args.FigFolder}_OrderType{args.OrderType}_nb{args.nb}_acc{args.threshold}")
        else:
            args.MVMType = "TLR" + args.TLRType + "_bandlen{bandlen}"
            TARGET_FIG_PATH = join(FIG_PATH, f"{args.FigFolder}_OrderType{args.OrderType}_nb{args.nb}_acc{args.threshold}")
    else:
        TARGET_FIG_PATH = join(FIG_PATH, f"{args.FigFolder}_MVMType{args.MVMType}")

    # create figure folder is not available
    if mpirank == 0:
        if not exists(TARGET_FIG_PATH):
            os.mkdir(TARGET_FIG_PATH)
    comm.Barrier()

    if mpirank == 0:
        print("-" * 80)
        print("MDD APP")
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

    ######### DEFINE GEOMETRY AND OTHER INPUTS #########
    inputfile_aux = join(STORE_PATH, args.AuxFile)
    inputdata_aux = np.load(inputfile_aux)
    
    # Virtual sources grid
    nrx = inputdata_aux['nrx']
    drx = inputdata_aux['drx']
    nry = inputdata_aux['nry']
    dry = inputdata_aux['dry']
    nr = nrx * nry
    ivs = args.vs

    # Time axis
    t = inputdata_aux['t']
    ot, dt, nt = t[0], t[1], len(t)


    ######### DEFINE SORTING (OPTIONAL) #########
    srcidx, recidx = None, None
    if args.OrderType == "hilbert":
        srcidx, recidx = gethilbertindex(inputdata_aux['ny'], inputdata_aux['nx'], inputdata_aux['dy'], inputdata_aux['dx'], 
                                         inputdata_aux['osx'], inputdata_aux['dsx'], inputdata_aux['osy'], inputdata_aux['dsy'], 
                                         inputdata_aux['orx'], inputdata_aux['drx'], inputdata_aux['ory'], inputdata_aux['dry'],
                                         args.PHilbertSrc, args.PHilbertRec)
        
    ######### CREATE TLR-MVM OPERATOR #########
    if mpirank == 0:
        print('Loading Kernel of MDC Operator...')
        print("-" * 80)
    comm.Barrier()

    if args.MVMType == "Dense":
        pass
    else:
        # Load TLR kernel
        problems = [f'Mode{args.ModeValue}_Order{args.OrderType}_{i}' for i in Ownfreqlist]
        mvmops = TilematrixGPU(args.M, args.N, args.nb, 
                               synthetic=False, datafolder=join(STORE_PATH, args.DataFolder), 
                               acc=args.threshold, freqlist=Ownfreqlist,
                               order=args.OrderType, srcidx=srcidx, recidx=recidx)
        mvmops.estimategpumemory()
        mvmops.loaduvbuffer()
        mvmops.setcolB(1) # just 1 point
        
        mvmops.Ownfreqlist = Ownfreqlist
        mvmops.Splitfreqlist = splitfreqlist

    ######### CREATE MDC OPERATOR #########
    if mpirank == 0:
        print('Creating MDC Operator...')
        print("-" * 80)
    comm.Barrier()

    dRop = MDCmixed(mvmops, args.M, nr, nt=nt, nfreq=nfmax, nv=1, dt=dt, dr=drx * dry, twosided=False,
                    conj=False)

    ######### CREATE DATA FOR MDD #########
    if mpirank == 0:
        print('Loading data...')
        print("-" * 80)
    comm.Barrier()

    gminus_filename = f'pup{ivs}.npy'
    gminus_filepath = join(STORE_PATH, args.PupFolder, gminus_filename)
    Gminus_vs = np.load(gminus_filepath).astype(np.float32)
    if args.OrderType == "hilbert":
        Gminus_vs_reshuffled = Gminus_vs[:, srcidx]
    else:
        Gminus_vs_reshuffled = Gminus_vs
    Gminus_vs_reshuffled = cp.asarray(Gminus_vs_reshuffled) # move to gpu
    
    # Adjoint
    if mpirank == 0:
        print('Perform adjoint...')
        print("-" * 80)
    comm.Barrier()

    t0 = time.time()
    radj = dRop.rmatvec(Gminus_vs_reshuffled.ravel())
    if mpirank == 0:
        t1 = time.time()
        print(f"MDC : {t1 - t0} s.")
    radj = cp.asnumpy(radj.reshape(nt, nr)) # move to back to cpu and reshape
    if args.OrderType == "hilbert":
        radj_reshuffled = np.zeros_like(radj)
        radj_reshuffled[:, recidx] = radj
    else:
        radj_reshuffled = radj
    
    # Inversion
    if mpirank == 0:
        print('Perform inversion...')
        print("-" * 80)
    comm.Barrier()
    t0 = time.time()
    if mpirank == 0:
        rinv = lsqr(dRop, Gminus_vs_reshuffled.ravel(), x0=cp.zeros(nt * nr, dtype=np.float32),
                    damp=damp, iter_lim=n_iter, atol=0, btol=0, show=True)[0]
    else:
        rinv = lsqr(dRop, Gminus_vs_reshuffled.ravel(), x0=cp.zeros(nt * nr, dtype=np.float32),
                    damp=damp, iter_lim=n_iter, atol=0, btol=0, show=False)[0]
    rinv = cp.asnumpy(rinv.reshape(nt, nr))  # move to back to cpu and reshape
    if args.OrderType == "hilbert":
        rinv_reshuffled = np.zeros_like(rinv)
        rinv_reshuffled[:, recidx] = rinv
    else:
        rinv_reshuffled = rinv
    t1 = time.time()
    if mpirank == 0:
        print(f"Total lsqr time : {t1 - t0} s.")
    comm.Barrier()

    # Save results
    if mpirank == 0:
        np.savez(join(TARGET_FIG_PATH, f"r_inv{ivs}"), radj=radj_reshuffled, rinv=rinv_reshuffled)
    comm.Barrier()

    # Display results
    if mpirank == 0 and args.debug:
        clip_adj = 0.05
        fig, ax = plt.subplots(1, 1, sharey=True, figsize=(16, 7))
        ax.imshow(radj_reshuffled, vmin=-clip_adj*radj.max(), vmax=clip_adj*radj.max(),
                  cmap='gray', extent=(0, nr, t[-1], -t[-1]))
        ax.set_title(r'$R_{adj}$'), ax.set_xlabel(r'$x_R$'), ax.set_ylabel(r'$t$')
        ax.axis('tight')
        plt.savefig(join(TARGET_FIG_PATH, f'madj{ivs}.png'), bbox_inches='tight')

        clip_inv = 0.05
        fig, ax = plt.subplots(1, 1, sharey=True, figsize=(16, 7))
        ax.imshow(rinv_reshuffled, vmin=-clip_inv * rinv.max(), vmax=clip_inv * rinv.max(),
                  cmap='gray', extent=(0, nr, t[-1], -t[-1]))
        ax.set_title(r'$R_{inv}$'), ax.set_xlabel(r'$x_R$'), ax.set_ylabel(r'$t$')
        ax.axis('tight')
        plt.savefig(join(TARGET_FIG_PATH, f'minv{ivs}.png'), bbox_inches='tight')

        fig, (ax0, ax1) = plt.subplots(2, 1, sharex=True, figsize=(15, 10))
        ax0.imshow(radj_reshuffled.reshape(nt, nry, nrx)[:, [20, 40, 60, 80]].reshape(nt, nrx * 4),
                   cmap='gray',
                   vmin=-clip_adj * radj.max(), vmax=clip_adj * radj.max(),
                   extent=(0, nrx * nry, t.max(), 0))
        ax0.axis('tight')
        ax0.set_ylabel(r'$t(s)$')
        ax0.set_title(r'$\mathbf{R^{Mck}_{adj}}$')
        ax0.set_ylim(2.5, 0.)
        ax1.imshow(rinv_reshuffled.reshape(nt, nry, nrx)[:, [20, 40, 60, 80]].reshape(nt, nrx * 4),
                   cmap='gray',
                   vmin=-clip_inv * rinv.max(), vmax=clip_inv * rinv.max(),
                   extent=(0, nrx * nry, t.max(), 0))
        ax1.axis('tight')
        ax1.set_ylabel(r'$t(s)$')
        ax1.set_title(r'$\mathbf{R^{Mck}_{inv}}$')
        ax1.set_ylim(2.5, 0.)
        plt.savefig(join(TARGET_FIG_PATH, f'radj_inv{ivs}.png'), bbox_inches='tight')

    if mpirank == 0:
        t1all = time.time()
        print(f"Done! Total time : {t1all - t0all} s.")
        print("-" * 80)
    
    print('Used memory', mempool.used_bytes(), mempool.total_bytes())


if __name__ == "__main__":
    description = '3D Multi-Dimensional Deconvolution with TLR-MDC and matrix reordering'
    main(argparse.ArgumentParser(description=description))
