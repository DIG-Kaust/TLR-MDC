import os
import numpy as np 
import sys
import argparse
import time
import zarr
from os.path import exists,join
from scipy.io import loadmat
from dotenv import load_dotenv
# load tlrmvm python library
from mdctlr.geometrysorting import GeometryArrangement
from mdctlr.tlrmvm.tlrmvmtools import TLRMVM_Util

load_dotenv()
STORE_PATH = os.environ['STORE_PATH']

def ApplyReordering(A, ordertype, nx, ny, nb, p=7):
    """apply different ordering type to input frequency matrix."""
    if ordertype == 'normal':
        return A
    if ordertype in ['bb','l1','l2','morton','hilbert']:
        x, y = np.arange(nx), np.arange(ny)
        X, Y = np.meshgrid(x, y, indexing='ij')
        X, Y = X.ravel(), Y.ravel()
        geomsorting = GeometryArrangement(X, Y)
        idx_hilbert, _ = geomsorting.rearrange(nb, p=p, kind=ordertype)
        Mck_freqslice_rearr = A.copy()
        Mck_freqslice_rearr = Mck_freqslice_rearr[:, idx_hilbert]
        Mck_freqslice_rearr = Mck_freqslice_rearr[idx_hilbert, :]
        return Mck_freqslice_rearr

if __name__ == "__main__":
    ## test generate reordering dataset.
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    rank = comm.rank 
    size = comm.size

    parser = argparse.ArgumentParser()
    parser.add_argument('--nb', type=int, default=256, help='nb')
    parser.add_argument('--error_threshold', type=str, default='0.001', 
        help='error threshold')
    parser.add_argument('--reordering', type=str,default='normal', 
        help='geometry reordering type: hilbert, normal')
    parser.add_argument('--freqlist', type=str, default='100',
        help='processing freqlist')
    parser.add_argument('--rankmodule', type=int, default=1, 
        help='all rank in the matrix dividable by certain value')
    parser.add_argument('--nrx', type=int, default=81,
                        help='number of receivers along the x axis')
    parser.add_argument('--nry', type=int, default=121,
                        help='number of receivers along the y axis')
    parser.add_argument('--foldername', type=str, default='Mck_freqslices',
                        help='foldername where input data are stored')
    parser.add_argument('--prefix', type=str, default='Mck_freqslice',
                        help='prefix of filenames')
    parser.add_argument('--suffix', type=str, default='_sub1',
                        help='suffix of filenames')
    parser.add_argument('--format', type=str, default='mat',
                        help='format of file (mat or zarr)')
    parser.add_argument('--matname', type=str, default='Rfreq',
                        help='name of variable in matfile')

    args = parser.parse_args()

    print("Your data path: ", STORE_PATH)
    freqlist=[int(x) for x in args.freqlist.split(',')]
    ownfreqlist = [f for f in freqlist if f % size == rank]
    print("rank ", rank, " my freqlist ", ownfreqlist)
    def run(freqid):
        if args.format == 'mat':
            Afilename = join(STORE_PATH, args.foldername, '{}{}{}.mat'.format(args.prefix, freqid, args.suffix))
            print('Afilename', Afilename)
            A = loadmat(Afilename)[args.matname]
        elif args.format == 'zarr':
            Afilename = join(STORE_PATH, args.foldername, '{}{}.zarr'.format(args.prefix, args.suffix))
            print(Afilename)
            Afile = zarr.open(Afilename, mode='r')
            A = Afile[freqid]
        else:
            raise ValueError('format must be mat or zarr, {} not recognized'.format(args.format))
        A = ApplyReordering(A, args.reordering, args.nrx, args.nry, args.nb)
        datasetname = '%s_Mode%d_Order%s_%s_%d' % (args.prefix, args.rankmodule, args.reordering, args.prefix, freqid)
        print('Datasetname', datasetname)
        tlrmvmutil = TLRMVM_Util(A, args.nb, STORE_PATH, args.error_threshold, datasetname, args.rankmodule)
        tlrmvmutil.computesvd()
        tlrmvmutil.saveUV()
        tlrmvmutil.printdatainfo()

    for freqid in ownfreqlist:
        run(freqid)

