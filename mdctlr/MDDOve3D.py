##################################################################
# @copyright (c) 2021- King Abdullah University of Science and
#                      Technology (KAUST). All rights reserved.
#
# Author: Matteo Ravasi
# Description: Multidimensional deconvolution
##################################################################

import os
import time
import argparse
import cupy as cp
import zarr
import matplotlib.pyplot as plt

from os.path import join, exists
from time import sleep
from dotenv import load_dotenv
from scipy.signal import convolve
from mpi4py import MPI

from pylops.utils.wavelets import *
from pylops.utils.tapers import *
from pytlrmvm import BatchedTlrmvm
from mdctlr import DenseGPU
from mdctlr.inversiondist import MDCmixed
from mdctlr.lsqr import lsqr

load_dotenv()


def main(parser):

    ######### INPUT PARAMS #########
    parser.add_argument("--AuxFile", type=str, default="AuxFile.npz", help="File with Auxiliar information for Mck redatuming")
    parser.add_argument("--MVMType", type=str, default="Dense", help="Type of MVM: Dense, TLR")
    parser.add_argument("--TLRType", type=str, default="fp32", help="TLR Precision: fp32, fp16, fp16int8, int8")
    parser.add_argument("--bandlen", type=int, default=10, help="TLR Band length")
    parser.add_argument("--nfmax", type=int, default=150, help="TLR Number of frequencies")
    parser.add_argument("--OrderType", type=str, default="normal", help="Matrix reordering method: normal, l1, hilbert")
    parser.add_argument("--ModeValue", type=int, default=8, help="Rank mode")
    parser.add_argument("--M", type=int, default=9801, help="Number of sources/rows in seismic frequency data")
    parser.add_argument("--N", type=int, default=9801, help="Number of receivers/columns in seismic frequency data")
    parser.add_argument("--nb", type=int, default=256, help="TLR Tile size")
    parser.add_argument("--threshold", type=str, default="0.001", help="TLR Error threshold")

    parser.add_argument('--debug', default=True, action='store_true', help='Debug')

    args = parser.parse_args()

    ######### SETUP MPI #########
    comm = MPI.COMM_WORLD
    mpirank = comm.Get_rank()
    mpisize = comm.Get_size()
    t0all = time.time()

    ######### PROBLEM PARAMETERS (should be lifted out into a config file #########
    nfmax = args.nfmax  # max frequency for MDC (#samples)
    n_iter = 30         # iterations
    damp = 2e-1         # damping

    ######### DEFINE DATA AND FIGURES DIRECTORIES #########
    STORE_PATH=os.environ["STORE_PATH"]
    FIG_PATH=os.environ["FIG_PATH"]
    if args.MVMType != "Dense":
        if args.TLRType != 'fp16int8':
            args.MVMType = "TLR" + args.TLRType
            TARGET_FIG_PATH = join(FIG_PATH, f"MDDOve3D_MVMType{args.MVMType}_OrderType{args.OrderType}_Mode{args.ModeValue}")
        else:
            args.MVMType = "TLR" + args.TLRType + "_bandlen{bandlen}"
            TARGET_FIG_PATH = join(FIG_PATH, f"MDDOve3D_MVMType{args.MVMType}_OrderType{args.OrderType}_Mode{args.ModeValue}")
    else:
        TARGET_FIG_PATH = join(FIG_PATH, f"MDDOve3D_MVMType{args.MVMType}")

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

    ######### LOAD AUXILIARY INPUTS (GEOMETRY, SUBSURFACE WAVEFIELDS, WAVELET) AND PREPARE FOR MCK #########
    #inputfile_aux = join(STORE_PATH, args.AuxFile)
    #inputdata_aux = np.load(inputfile_aux)

    # Sources
    #s = inputdata_aux['srcs'].T
    ns = 6510

    # Virtual sources grid
    nrx = 177
    drx = 20
    nry = 90
    dry = 20
    nr = nrx * nry
    irplot = 9115

    # Time axis
    ot, dt, nt = 0, 0.004, 1126
    t = np.arange(nt) * dt

    ######### CREATE TLR-MVM OPERATOR #########
    if mpirank == 0:
        print('Loading Kernel of MDC Operator...')
        print("-" * 80)
    comm.Barrier()

    if args.MVMType == "Dense":
        pass
    else:
        # Load TLR kernel
        problems = [f'PDOWN_Mode{args.ModeValue}_Order{args.OrderType}_PDOWN_{i}' for i in Ownfreqlist]
        mvmops = BatchedTlrmvm(join(STORE_PATH, 'compresseddata'), problems, args.threshold, args.M, args.N, args.nb, 'bf16')
        mvmops.Ownfreqlist = Ownfreqlist
        mvmops.Splitfreqlist = splitfreqlist

    ######### CREATE MDC OPERATOR #########
    if mpirank == 0:
        print('Creating MDC Operator...')
        print("-" * 80)
    comm.Barrier()

    dRop = MDCmixed(mvmops, ns, nr, nt=nt, nfreq=nfmax, nv=1, dt=dt, dr=drx * dry, twosided=False,
                    nb=args.nb, acc=args.threshold, prescaled=True, datafolder=join(STORE_PATH, 'compresseddata'),
                    transpose=False, conj=False)

    ######### CREATE DATA FOR MDD #########
    if mpirank == 0:
        print('Loading data...')
        print("-" * 80)
    comm.Barrier()

    # Input upgoing wavefield
    #gminus_filename = 'pup.zarr'
    #gminus_filepath = '/home/ravasim/Documents/Data/Overtrust3D/Data/' + gminus_filename
    #Gminus = zarr.open(gminus_filepath, mode='r')
    #print(Gminus.shape, dRop)
    #Gminus_vs = Gminus[:, :, irplot].astype(np.float32)

    gminus_filename = f'pup{irplot}.npy'
    gminus_filepath = '/home/ravasim/Documents/Data/Overtrust3D/Data/' + gminus_filename
    Gminus_vs = np.load(gminus_filepath).astype(np.float32)
    print(Gminus_vs.shape, dRop)
    Gminus_vs = cp.asarray(Gminus_vs) # move to gpu

    # Adjoint
    if mpirank == 0:
        print('Perform adjoint...')
        print("-" * 80)
    comm.Barrier()

    t0 = time.time()
    radj = dRop.rmatvec(Gminus_vs.ravel())
    if mpirank == 0:
        t1 = time.time()
        print(f"MDC : {t1 - t0} s.")
    radj = cp.asnumpy(radj.reshape(nt, nr)) # move to back to cpu and reshape

    # Inversion
    if mpirank == 0:
        print('Perform inversion...')
        print("-" * 80)
    comm.Barrier()
    t0 = time.time()
    if mpirank == 0:
        rinv = lsqr(dRop, Gminus_vs.ravel(), x0=cp.zeros(nt * nr, dtype=np.float32),
                    damp=damp, iter_lim=n_iter, atol=0, btol=0, show=True)[0]
    else:
        rinv = lsqr(dRop, Gminus_vs.ravel(), x0=cp.zeros(nt * nr, dtype=np.float32),
                    damp=damp, iter_lim=n_iter, atol=0, btol=0, show=False)[0]
    rinv = cp.asnumpy(rinv.reshape(nt, nr))  # move to back to cpu and reshape
    t1 = time.time()
    if mpirank == 0:
        print(f"Total lsqr time : {t1 - t0} s.")
    comm.Barrier()

    # Save results
    if mpirank == 0:
        np.savez(join(TARGET_FIG_PATH, f"r_inv{irplot}"), radj=radj, rinv=rinv)
    comm.Barrier()

    # Display results
    if mpirank == 0 and args.debug:
        clip_adj = 0.05
        fig, ax = plt.subplots(1, 1, sharey=True, figsize=(16, 7))
        ax.imshow(radj, vmin=-clip_adj*radj.max(), vmax=clip_adj*radj.max(),
                  cmap='gray', extent=(0, nr, t[-1], -t[-1]))
        ax.set_title(r'$R_{adj}$'), ax.set_xlabel(r'$x_R$'), ax.set_ylabel(r'$t$')
        ax.axis('tight')
        plt.savefig(join(TARGET_FIG_PATH, 'madj.png'), bbox_inches='tight')

        clip_inv = 0.05
        fig, ax = plt.subplots(1, 1, sharey=True, figsize=(16, 7))
        ax.imshow(rinv, vmin=-clip_inv * rinv.max(), vmax=clip_inv * rinv.max(),
                  cmap='gray', extent=(0, nr, t[-1], -t[-1]))
        ax.set_title(r'$R_{inv}$'), ax.set_xlabel(r'$x_R$'), ax.set_ylabel(r'$t$')
        ax.axis('tight')
        plt.savefig(join(TARGET_FIG_PATH, 'minv.png'), bbox_inches='tight')

        fig, (ax0, ax1) = plt.subplots(2, 1, sharex=True, figsize=(15, 10))
        ax0.imshow(radj.reshape(nt, nry, nrx)[:, [20, 40, 60, 80]].reshape(nt, nrx * 4),
                   cmap='gray',
                   vmin=-clip_adj * radj.max(), vmax=clip_adj * radj.max(),
                   extent=(0, nrx * nry, t.max(), 0))
        ax0.axis('tight')
        ax0.set_ylabel(r'$t(s)$')
        ax0.set_title(r'$\mathbf{R^{Mck}_{adj}}$')
        ax0.set_ylim(2.5, 0.)
        ax1.imshow(rinv.reshape(nt, nry, nrx)[:, [20, 40, 60, 80]].reshape(nt, nrx * 4),
                   cmap='gray',
                   vmin=-clip_inv * rinv.max(), vmax=clip_inv * rinv.max(),
                   extent=(0, nrx * nry, t.max(), 0))
        ax1.axis('tight')
        ax1.set_ylabel(r'$t(s)$')
        ax1.set_title(r'$\mathbf{R^{Mck}_{inv}}$')
        ax1.set_ylim(2.5, 0.)
        plt.savefig(join(TARGET_FIG_PATH, 'radj_inv.png'), bbox_inches='tight')

    if mpirank == 0:
        t1all = time.time()
        print(f"Done! Total time : {t1all - t0all} s.")
        print("-" * 80)


if __name__ == "__main__":
    description = '3D Multi-Dimensional Deconvolution with TLR-MDC and matrix reordering'
    main(argparse.ArgumentParser(description=description))

