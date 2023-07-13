##################################################################
# @copyright (c) 2021- King Abdullah University of Science and
#                      Technology (KAUST). All rights reserved.
#
# Authors: Matteo, Ravasi, Yuxi Hong
# Description: Multidimensional deconvolution
##################################################################

import os
import time
import argparse
import cupy as cp
import matplotlib.pyplot as plt
import zarr

from os.path import join, exists
from time import sleep
from dotenv import load_dotenv
from scipy.signal import convolve
from mpi4py import MPI

from pylops.utils.wavelets import *
from pylops.utils.tapers import *
from mdctlr.inversiondist import MDCmixed
from mdctlr.lsqr import lsqr
from tlrmvm.tilematrix import TilematrixGPU_Ove3D


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

    ######### SETUP GPUs #########
    cp.cuda.Device(device=mpirank).use()

    ######### PROBLEM PARAMETERS (should be lifted out into a config file #########
    n_iter = 10              # iterations
    nfmax = args.nfmax       # max frequency for MDC (#samples)

    ######### DEFINE DATA AND FIGURES DIRECTORIES #########
    STORE_PATH=os.environ["STORE_PATH"]
    FIG_PATH=os.environ["FIG_PATH"]
    if args.MVMType != "Dense":
        if args.TLRType != 'fp16int8':
            args.MVMType = "TLR" + args.TLRType
            TARGET_FIG_PATH = join(FIG_PATH, f"MDD_MVMType{args.MVMType}_OrderType{args.OrderType}_Mode{args.ModeValue}")
        else:
            args.MVMType = "TLR" + args.TLRType + "_bandlen{bandlen}"
            TARGET_FIG_PATH = join(FIG_PATH, f"MDD_MVMType{args.MVMType}_OrderType{args.OrderType}_Mode{args.ModeValue}")
    else:
        TARGET_FIG_PATH = join(FIG_PATH, f"MDD_MVMType{args.MVMType}")

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
    comm.Barrier()

    print(f"MPIRANK{mpirank}: Gpu{cp.cuda.get_device_id()}")
    if mpirank == 0:
        print("-" * 80)
    comm.Barrier()

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

    # Sources
    s = inputdata_aux['srcs'].T
    ns = s.shape[1]

    # Virtual sources grid
    vsz = 650
    nvsx = 71
    dvsx = 20
    ovsx = 200
    nvsy = 41
    dvsy = 20
    ovsy = 200
    nvs = nvsx * nvsy
    ivsx, ivsy = 21, 19
    ivsinv = ivsx * nvsy + ivsy

    # Time axis
    t = inputdata_aux['t']
    ot, dt, nt = t[0], t[1], len(t)


    ######### CREATE TLR-MVM OPERATOR #########
    if mpirank == 0:
        print('Loading Kernel of MDC Operator...')
        print("-" * 80)
    comm.Barrier()

    if args.MVMType == "Dense":
        # Load dense kernel
        pass
        # dev = cp.cuda.Device(mpirank)
        # dev.use()
        # t0 = time.time()
        # mvmops = DenseGPU(Ownfreqlist, Totalfreqlist, splitfreqlist, args.nfmax, STORE_PATH,
        #                  'Gplus_freqslices', 'Gplus_freqslice', matname='Gplusfreq')
        # t1 = time.time()
        # if mpirank == 0:
        #     print("Init dense GPU Time is ", t1-t0)
    else:
        # Load TLR kernel
        mvmops = TilematrixGPU_Ove3D(args.M, args.N, args.nb, 
                                     synthetic=False, datafolder=join(STORE_PATH,'compresseddata'), 
                                     acc=args.threshold, freqlist=Ownfreqlist, order=args.OrderType,
                                     mode=4, prefix="Gplus_freqslice_", suffix="Gplus_freqslice_")
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

    dRop = MDCmixed(mvmops, ns, nvs, nt=2 * nt - 1, nfreq=nfmax, nv=1, dt=dt, dr=dvsx * dvsy, twosided=True,
                    nb=args.nb, acc=args.threshold, prescaled=True, datafolder=join(STORE_PATH, 'compresseddata'),
                    transpose=False, conj=False)

    ######### CREATE DATA FOR MDD #########
    if mpirank == 0:
        print('Loading Data...')
        print("-" * 80)
    comm.Barrier()

    # Input upgoing wavefield (single virtual source)
    gminus_filename = 'Gminus_sub1.zarr'
    gminus_filepath = join(STORE_PATH, gminus_filename) 

    Gminus = zarr.open(gminus_filepath, mode='r')
    Gminus_vs = Gminus[:, :, ivsinv].astype(np.float32)
    Gminus_vs = np.concatenate((np.zeros((nt - 1, ns), dtype=np.float32), Gminus_vs), axis=0)
    Gminus_vs = cp.asarray(Gminus_vs) # move to gpu

    # Adjoint
    t0 = time.time()
    radj = dRop.rmatvec(Gminus_vs.ravel())
    if mpirank == 0:
        t1 = time.time()
        print(f"MDC : {t1 - t0} s.")
    radj = cp.asnumpy(radj.reshape(2 * nt - 1, nvs)) # move to back to cpu and reshape

    # Inversion
    t0 = time.time()
    if mpirank == 0:
        rinv = lsqr(dRop, Gminus_vs.ravel(), x0=cp.zeros((2 * nt - 1) * nvs, dtype=np.float32),
                    iter_lim=n_iter, atol=0, btol=0, show=True)[0]
    else:
        rinv = lsqr(dRop, Gminus_vs.ravel(), x0=cp.zeros((2 * nt - 1) * nvs, dtype=np.float32),
                    iter_lim=n_iter, atol=0, btol=0, show=False)[0]
    rinv = cp.asnumpy(rinv.reshape(2 * nt - 1, nvs))  # move to back to cpu and reshape
    t1 = time.time()
    if mpirank == 0:
        print(f"Total lsqr time : {t1 - t0} s.")
    comm.Barrier()

    # Save results
    if mpirank == 0:
        np.savez(join(TARGET_FIG_PATH, f"r_inv{ivsinv}"), radj=radj, rinv=rinv)
    comm.Barrier()

    # Display results
    if mpirank == 0 and args.debug:
        clip_adj = 1e-1
        fig, ax = plt.subplots(1, 1, sharey=True, figsize=(16, 7))
        ax.imshow(radj, vmin=-clip_adj*radj.max(), vmax=clip_adj*radj.max(),
                  cmap='gray', extent=(0, nvs, t[-1], -t[-1]))
        ax.set_title(r'$R_{adj}$'), ax.set_xlabel(r'$x_R$'), ax.set_ylabel(r'$t$')
        ax.axis('tight')
        plt.savefig(join(TARGET_FIG_PATH, 'madj.png'), bbox_inches='tight')

        clip_inv = 1e-1
        fig, ax = plt.subplots(1, 1, sharey=True, figsize=(16, 7))
        ax.imshow(rinv, vmin=-clip_inv * rinv.max(), vmax=clip_inv * rinv.max(),
                  cmap='gray', extent=(0, nvs, t[-1], -t[-1]))
        ax.set_title(r'$R_{inv}$'), ax.set_xlabel(r'$x_R$'), ax.set_ylabel(r'$t$')
        ax.axis('tight')
        plt.savefig(join(TARGET_FIG_PATH, 'minv.png'), bbox_inches='tight')

        fig, (ax0, ax1) = plt.subplots(2, 1, sharex=True, figsize=(15, 10))
        ax0.imshow(radj[nt - 1:].reshape(nt, nvsx, nvsy)[:, 1::10].reshape(nt, nvsy * 7),
                   cmap='gray',
                   vmin=-clip_adj * radj.max(), vmax=clip_adj * radj.max(),
                   extent=(0, nvsx * nvsy, t.max(), 0))
        ax0.axis('tight')
        ax0.set_ylabel(r'$t(s)$')
        ax0.set_title(r'$\mathbf{R^{Mck}_{adj}}$')
        ax0.set_ylim(0.5, 0.)

        ax1.imshow(rinv[nt - 1:].reshape(nt, nvsx, nvsy)[:, 1::10].reshape(nt, nvsy * 7),
                   cmap='gray',
                   vmin=-clip_inv * rinv.max(), vmax=clip_inv * rinv.max(),
                   extent=(0, nvsx * nvsy, t.max(), 0))
        ax1.axis('tight')
        ax1.set_ylabel(r'$t(s)$')
        ax1.set_title(r'$\mathbf{R^{Mck}_{inv}}$')
        ax1.set_ylim(0.5, 0.)
        plt.savefig(join(TARGET_FIG_PATH, 'radj_inv.png'), bbox_inches='tight')

    if mpirank == 0:
        t1all = time.time()
        print(f"Done! Total time : {t1all - t0all} s.")
        print("-" * 80)
    comm.Barrier()

if __name__ == "__main__":
    description = '3D Multi-Dimensional Deconvolution with TLR-MDC and matrix reordering'
    main(argparse.ArgumentParser(description=description))

