##################################################################
# @copyright (c) 2021- King Abdullah University of Science and
#                      Technology (KAUST). All rights reserved.
#
# Author: Matteo, Ravasi, Yuxi Hong
# Description: Marchenko redatuming with TLR-MDC and
# geometric-reordering
##################################################################

import os
import time
import argparse
import cupy as cp
import matplotlib.pyplot as plt

from os.path import join, exists
from time import sleep
from dotenv import load_dotenv
from scipy.signal import convolve, filtfilt
from mpi4py import MPI

from pylops.basicoperators import *
from pylops.utils.wavelets import *
from pylops.utils.tapers import *
from pylops.waveeqprocessing.marchenko import directwave
from pylops.basicoperators import Diagonal
from pylops.basicoperators import Identity
from pylops.basicoperators import Roll
from mdctlr.inversiondist import MDCmixed
from mdctlr.lsqr import lsqr
from mdctlr.utils import voronoi_volumes
from tlrmvm.tilematrix import TilematrixGPU_Ove3D


def main(parser):

    ######### INPUT PARAMS #########
    parser.add_argument("--AuxFile", type=str, default="AuxFile.npz", help="File with Auxiliar information for Mck redatuming")
    parser.add_argument("--MVMType", type=str, default="Dense", help="Type of MVM: Dense, TLR")
    parser.add_argument("--TLRType", type=str, default="fp32", help="TLR Precision: fp32, fp16, int8")
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
    if mpirank == 0:
        print('Cuda count', cp.cuda.runtime.getDeviceCount())
        for idev in range(cp.cuda.runtime.getDeviceCount()):
            print(cp.cuda.runtime.getDeviceProperties(idev)['name'])

    cp.cuda.Device(device=mpirank).use()

    ######### PROBLEM PARAMETERS (should be lifted out into a config file #########
    vel = 2400.0             # velocity
    toff = 0.045             # direct arrival time shift
    nsmooth = 10             # time window smoothing
    n_iter = 10              # iterations
    bandlen = args.bandlen   # TLR bandlenght
    nfmax = args.nfmax       # max frequency for MDC (#samples)
    wavfreq = 15             # Ricker wavelet peak frequency

    ######### DEFINE DATA AND FIGURES DIRECTORIES #########
    STORE_PATH=os.environ["STORE_PATH"]
    FIG_PATH=os.environ["FIG_PATH"]
    if args.MVMType != "Dense":
        if args.TLRType != 'fp16int8':
            args.MVMType = "TLR" + args.TLRType
            TARGET_FIG_PATH = join(FIG_PATH, f"Mck_MVMType{args.MVMType}_OrderType{args.OrderType}_Mode{args.ModeValue}")
        else:
            args.MVMType = "TLR" + args.TLRType + "_bandlen{bandlen}"
            TARGET_FIG_PATH = join(FIG_PATH, f"Mck_MVMType{args.MVMType}_OrderType{args.OrderType}_Mode{args.ModeValue}")
    else:
        TARGET_FIG_PATH = join(FIG_PATH, f"Mck_MVMType{args.MVMType}")

    # create figure folder is not available
    if mpirank == 0:
        if not exists(TARGET_FIG_PATH):
            os.mkdir(TARGET_FIG_PATH)
    comm.Barrier()

    if mpirank == 0:
        print("-" * 80)
        print("MARCHENKO REDATUMING APP")
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
    # Receivers
    r = inputdata_aux['recs'].T
    nr = r.shape[1]
    # Sources
    s = inputdata_aux['srcs'].T
    ns = s.shape[1]
    # Virtual points
    vs = inputdata_aux['vs']
    # Time axis
    t = inputdata_aux['t']
    ot, dt, nt = t[0], t[1], len(t)

    # Find area of each volume - note that areas at the edges and on vertex are unbounded,
    # we will assume that they are and use the minimum are for all points in this example
    vertex, vols = voronoi_volumes(r[:2].T)
    darea = np.min(np.unique(vols))
    #if mpirank == 0:
    #    print('Integration area %f' % darea)

    # Load subsurface wavefields
    Gsub = inputdata_aux['G']
    G0sub = inputdata_aux['G0']
    wav = ricker(t[:51], wavfreq)[0]
    wav_c = np.argmax(wav)

    # Convolve with wavelet
    Gsub = np.apply_along_axis(convolve, 0, Gsub, wav, mode='full')
    Gsub = Gsub[wav_c:][:nt]
    G0sub = np.apply_along_axis(convolve, 0, G0sub, wav, mode='full')
    G0sub = G0sub[wav_c:][:nt]

    # Direct arrival window - traveltime
    distVS = np.sqrt((vs[0]-r[0])**2 +(vs[1]-r[1])**2 +(vs[2]-r[2])**2)
    directVS = distVS / vel
    directVS_off = directVS - toff

    # Window
    idirectVS_off = np.round(directVS_off/dt).astype(np.int32)
    w = np.zeros((nr, nt))
    for ir in range(nr):
        w[ir, :idirectVS_off[ir]]=1
    w = np.hstack((np.fliplr(w), w[:, 1:]))
    if nsmooth > 0:
        smooth=np.ones(nsmooth)/nsmooth
        w = filtfilt(smooth, 1, w)

    # Create analytical direct wave
    G0sub_ana = directwave(wav, directVS, nt, dt, nfft=2 ** 11, dist=distVS, kind='3d')
    G0sub_ana = G0sub_ana * (G0sub.max() / G0sub_ana.max())
    G0sub_ana = G0sub_ana.astype(np.float32)
    w = w.astype(np.float32)

    if mpirank == 0 and args.debug:
        fig, ax = plt.subplots(1, 1,  sharey=True, figsize=(20, 5))
        im = ax.imshow(w.T, cmap='gray', extent=(0, nr, t[-1], -t[-1]))
        ax.plot(np.arange(0, nr), directVS_off, '--r')
        ax.plot(np.arange(0, nr), -directVS_off, '--r')
        ax.set_title('Window')
        ax.set_xlabel(r'$x_R$')
        ax.set_ylabel(r'$t$')
        ax.axis('tight')
        ax.set_xlim(800, 1000)
        fig.colorbar(im, ax=ax)
        plt.savefig(join(TARGET_FIG_PATH,"window.jpg"))

        plt.figure(figsize=(15,5))
        ax1 = plt.subplot2grid((1, 5), (0, 0), colspan=2)
        ax2 = plt.subplot2grid((1, 5), (0, 2), colspan=2)
        ax3 = plt.subplot2grid((1, 5), (0, 4))
        ax1.imshow(G0sub / G0sub.max(), cmap='gray', vmin=-1, vmax=1, extent=(0, nr, t[-1], t[0]))
        ax1.set_title(r'$G0_{FD}$'), ax1.set_xlabel(r'$x_R$'), ax1.set_ylabel(r'$t$')
        ax1.plot(np.arange(0, nr), directVS, '--r')
        ax1.axis('tight')
        ax1.set_ylim(.6, 0)
        ax1.set_xlim(800, 1200)
        ax2.imshow(G0sub_ana / G0sub_ana.max(), cmap='gray', vmin=-1, vmax=1, extent=(0, nr, t[-1], t[0]))
        ax2.set_title(r'$G0_{Ana}$'), ax2.set_xlabel(r'$x_R$'), ax2.set_ylabel(r'$t$')
        ax2.plot(np.arange(0, nr), directVS, '--r')
        ax2.axis('tight')
        ax2.set_ylim(.6, 0)
        ax2.set_xlim(800, 1200)
        ax3.plot(G0sub[:, nr//2-10]/G0sub.max(), t, 'r', lw=5)
        ax3.plot(G0sub_ana[:, nr//2-10]/G0sub_ana.max(), t, 'k', lw=3)
        ax3.set_ylim(1., 0)
        plt.savefig(join(TARGET_FIG_PATH, "analyticaldirectwave.jpg"))

    # Rearrange inputs according to matrix re-arrangement
    if args.OrderType != 'normal':
        idx = np.load(join(STORE_PATH,'Mck_rearrangement.npy'))
        G0sub_ana_shuffled = G0sub_ana[:, idx]
        w_shuffled = w[idx]
    else:
        G0sub_ana_shuffled = G0sub_ana
        w_shuffled = w

    if mpirank == 0 and args.debug:
        fig, ax = plt.subplots(1, 1, sharey=True, figsize=(20, 5))
        ax.imshow(G0sub_ana_shuffled, cmap='gray', extent=(0, nr, t[-1], -t[-1]))
        ax.set_title('G0')
        ax.set_xlabel(r'$x_R$')
        ax.set_ylabel(r'$t$')
        ax.axis('tight')
        plt.savefig(join(TARGET_FIG_PATH, 'g0_shuffled.png'), bbox_inches='tight')

        fig, ax = plt.subplots(1, 1,  sharey=True, figsize=(20, 5))
        ax.imshow(G0sub_ana, cmap='gray', extent=(0, nr, t[-1], -t[-1]))
        ax.set_title('G0')
        ax.set_xlabel(r'$x_R$')
        ax.set_ylabel(r'$t$')
        ax.axis('tight')
        plt.savefig(join(TARGET_FIG_PATH, 'g0.png'), bbox_inches='tight')

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
                                     order=args.OrderType, acc=args.threshold, freqlist=Ownfreqlist, 
                                     suffix="Mck_freqslice_")
        mvmops.estimategpumemory()
        mvmops.loaduvbuffer()
        mvmops.setcolB(1) # just 1 point
        
        mvmops.Ownfreqlist = Ownfreqlist
        mvmops.Splitfreqlist = splitfreqlist
        print("-" * 80)

    ######### CREATE MARCHENKO OPERATOR #########
    if mpirank == 0:
        print('Creating Marchenko Operator...')
        print("-" * 80)
    comm.Barrier()

    dRop = MDCmixed(mvmops, ns, nr, nt=2*nt-1, nfreq=nfmax, nv=1, dt=dt, dr=2*darea, nb=args.nb,  acc=args.threshold,
                    datafolder=join(STORE_PATH, 'compresseddata'), transpose=False, conj=False)
    dR1op = MDCmixed(mvmops, ns, nr, nt=2*nt-1, nfreq=nfmax, nv=1, dt=dt, dr=2*darea, nb=args.nb,  acc=args.threshold,
                     datafolder=join(STORE_PATH, 'compresseddata'), transpose=False, conj=True)
    dRollop = Roll((2*nt-1) * nr,dims=(2*nt-1, nr),dir=0, shift=-1)

    dWop = Diagonal(w_shuffled.T.flatten())
    dIop = Identity(nr*(2*nt-1))

    dMop = VStack([HStack([dIop, -1*dWop*dRop]),
                    HStack([-1*dWop*dRollop*dR1op, dIop])])*BlockDiag([dWop, dWop])
    dGop = VStack([HStack([dIop, -1*dRop]),
                    HStack([-1*dRollop*dR1op, dIop])])

    ######### CREATE DATA FOR MARCHENKO REDATUMING #########
    if mpirank == 0:
        print('Creating Input data...')
        print("-" * 80)
    comm.Barrier()

    # Input focusing function
    dfd_plus = np.concatenate((np.fliplr(G0sub_ana_shuffled.T).T, np.zeros((nt-1, nr), dtype=np.float32)))
    dfd_plus = cp.asarray(dfd_plus) # move to gpu

    if mpirank == 0 and args.debug:
        fig, ax = plt.subplots(1, 1, sharey=True, figsize=(16, 7))
        ax.imshow(cp.asnumpy(dfd_plus), cmap='gray', extent=(0, nr, t[-1], -t[-1]))
        ax.set_title(r'$f_d^+$'), ax.set_xlabel(r'$x_R$'), ax.set_ylabel(r'$t$')
        ax.axis('tight');
        plt.savefig(join(TARGET_FIG_PATH, 'dfplus_shuffled.png'), bbox_inches='tight')

    # Create data
    dd = dWop*dRop*dfd_plus.flatten()
    dd = cp.concatenate((dd.reshape(2*nt-1, nr), cp.zeros((2*nt-1, nr), dtype=np.float32)))

    ######### RUN INVERSION #########
    if mpirank == 0:
        print('Running Marchenko Inversion...')
        print("-" * 80)
    comm.Barrier()

    # Invert
    t0 = time.time()
    if mpirank == 0:
        df1_inv = lsqr(dMop, dd.ravel(), x0=cp.zeros(2*(2*nt-1)*nr, dtype=np.float32),
                       iter_lim=n_iter, atol=0, btol=0, show=True)[0]
    else:
        df1_inv = lsqr(dMop, dd.ravel(), x0=cp.zeros(2*(2*nt-1)*nr, dtype=np.float32),
                       iter_lim=n_iter, atol=0, btol=0, show=False)[0]
    t1 = time.time()
    if mpirank == 0:
        print(f"Total lsqr time : {t1-t0} s.")
    comm.Barrier()

    # Rearrange solution
    df1_inv = df1_inv.reshape(2*(2*nt-1), nr)
    df1_inv_tot = df1_inv + cp.concatenate((cp.zeros((2*nt-1, nr), dtype=np.float32),dfd_plus))

    # Estimate Green's functions
    dg_inv = dGop * df1_inv_tot.flatten()
    dg_inv = dg_inv.reshape(2*(2*nt-1), nr)

    if mpirank == 0:
        print("-" * 80)
    comm.Barrier()

    # Extract up and down focusing and Green's functions from model vectors and create total Green's function
    dg_inv = np.real(dg_inv)
    dg_inv_minus, dg_inv_plus = -dg_inv[:(2*nt-1)].T, np.fliplr(dg_inv[(2*nt-1):].T)
    dg_inv_tot = dg_inv_minus + dg_inv_plus

    # Move to numpy
    dg_inv_tot = cp.asnumpy(dg_inv_tot)

    if args.OrderType != 'normal':
        # Rearrange outputs according to matrix re-arrangement
        dg_inv_tot_reshuffled = np.zeros_like(dg_inv_tot)
        dg_inv_tot_reshuffled[idx] = dg_inv_tot
    else:
        dg_inv_tot_reshuffled = dg_inv_tot

    # Save results
    if mpirank == 0:
        np.save(join(TARGET_FIG_PATH, "dg_inv_tot"), dg_inv_tot_reshuffled)
    comm.Barrier()

    # Display results
    if mpirank == 0 and args.debug:
        plt.figure(figsize=(15, 7))
        ax1 = plt.subplot2grid((4, 5), (0, 0), rowspan=2, colspan=4)
        ax2 = plt.subplot2grid((4, 5), (2, 0), rowspan=2, colspan=4)
        ax3 = plt.subplot2grid((4, 5), (0, 4), rowspan=4)

        ax1.imshow(Gsub, cmap='gray', vmin=-5e3, vmax=5e3, extent=(0, nr, t[-1], t[0]), interpolation='sinc')
        ax1.set_title(r'$G_{true}$'), ax1.set_xlabel(r'$x_R$'), ax1.set_ylabel(r'$t$')
        ax1.axis('tight')
        ax1.set_ylim(1., 0)
        ax1.set_xlim(nr//2-100,nr//2+100)
        ax2.imshow(dg_inv_tot_reshuffled.T, cmap='gray', vmin=-5e3, vmax=5e3, extent=(0, nr, t[-1], -t[-1]), interpolation='sinc')
        ax2.set_title(r'$G_{est}$'), ax2.set_xlabel(r'$x_R$'), ax2.set_ylabel(r'$t$')
        ax2.axis('tight')
        ax2.set_ylim(1., 0)
        ax2.set_xlim(nr//2-100,nr//2+100)

        ax3.plot(np.exp(4*t) * Gsub[:, nr//2]/Gsub.max(), t, 'r', lw=5)
        ax3.plot(np.exp(4*t) * dg_inv_tot_reshuffled[nr//2, nt-1:]/dg_inv_tot_reshuffled.max(), t, 'k', lw=3)
        ax3.set_ylim(1., 0)
        plt.savefig(join(TARGET_FIG_PATH,"finalresults.png"))

    if mpirank == 0:
        t1all = time.time()
        print(f"Done! Total time : {t1all - t0all} s.")
        print("-" * 80)
    comm.Barrier()

if __name__ == "__main__":
    description = '3D Marchenko Redatuming with TLR-MDC and matrix reordering'
    main(argparse.ArgumentParser(description=description))

