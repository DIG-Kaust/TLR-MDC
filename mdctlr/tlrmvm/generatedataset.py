import os
import numpy as np 
import sys
import argparse
import time
from os.path import exists,join
from scipy.io import loadmat
from dotenv import load_dotenv
# load tlrmvm python library
from mdctlr.geometrysorting import GeometryArrangement
from mdctlr.tlrmvm.tlrmvmtools import TLRMVM_Util

load_dotenv()
STORE_PATH=os.environ['STORE_PATH']

def ApplyReordering(A, ordertype, nb, p=7):
    """apply different ordering type to input frequency matrix."""
    if ordertype == 'normal':
        return A
    if ordertype in ['bb','l1','l2','morton','hilbert']:
        nx, ny = 81, 121 # grid size
        n = nx * ny
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
    args = parser.parse_args()
    print("Your data path: ", STORE_PATH)
    freqlist=[int(x) for x in args.freqlist.split(',')]
    ownfreqlist = [f for f in freqlist if f % size == rank]
    print("rank ", rank, " my freqlist ", ownfreqlist)
    def run(freqid):
        Afilename = join(STORE_PATH, 'Mck_freqslices', 'Mck_freqslice{}_sub1.mat'.format(freqid))
        A = loadmat(Afilename)['Rfreq']
        A = ApplyReordering(A, args.reordering, args.nb)
        datasetname = 'Mode%d_Order%s_Mck_freqslice_%d' % (args.rankmodule,args.reordering,freqid)
        print('Datasetname', datasetname)
        tlrmvmutil = TLRMVM_Util(A, args.nb, STORE_PATH, args.error_threshold, 
            datasetname, args.rankmodule)
        tlrmvmutil.computesvd()
        tlrmvmutil.saveUV()
        tlrmvmutil.printdatainfo()

    for freqid in ownfreqlist:
        run(freqid)

