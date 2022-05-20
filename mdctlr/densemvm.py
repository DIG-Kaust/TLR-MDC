import cupy as cp
import numpy as np
from scipy.io import loadmat as loadmat
from os.path import join, exists
__all__ = ['DenseGPU']

class DenseGPU:
    def __init__(self, Ownfreqlist, Totalfreqlist, Splitfreqlist, nfmax, datasetprefix):
        self.nfreq = nfmax
        self.Ownfreqlist = Ownfreqlist
        self.Totalfreqlist = Totalfreqlist
        self.Splitfreqlist = Splitfreqlist
        self.cupyarray = []
        for freq in self.Ownfreqlist:
            problem = join(datasetprefix, 'Mck_freqslices', 'Mck_freqslice{}_sub1.mat'.format(freq))

            if not exists(problem):
                problem = join(datasetprefix, 'Mck_freqslices', 'Mck_freqslice{}_sub1.npy'.format(freq))
                A = np.load(problem)
            else:
                A = loadmat(problem)['Rfreq']

            self.cupyarray.append(cp.array(A))
        
    def SetTransposeConjugate(self,transpose, conjugate):
        self.transpose = transpose
        self.conjugate = conjugate

    def MVM(self,xlist):
        spx = np.split(xlist, len(self.Ownfreqlist))
        yfinal = [None for _ in self.Ownfreqlist]
        for idx, xvec in enumerate(spx):
            if self.transpose and self.conjugate:
                yfinal[idx] = np.conjugate(cp.asnumpy(self.cupyarray[idx].T @ cp.array(np.conjugate(xvec))))
            elif not self.transpose and self.conjugate:
                yfinal[idx] = np.conjugate(cp.asnumpy(self.cupyarray[idx] @ np.conjugate(cp.array(xvec))))
            elif self.transpose and not self.conjugate:
                yfinal[idx] = cp.asnumpy(self.cupyarray[idx].T @ cp.array(xvec))
            else:
                yfinal[idx] = cp.asnumpy(self.cupyarray[idx] @ cp.array(xvec))
        return np.hstack(yfinal)

