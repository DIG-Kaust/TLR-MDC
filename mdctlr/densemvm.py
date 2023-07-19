import cupy as cp
import numpy as np
from scipy.io import loadmat as loadmat
from os.path import join, exists
__all__ = ['DenseGPU']

# TEMP NAME: need to have it like this to be the same as TLR codes for Fredholm
class YObj:

    def __init__(self, yfinal):
        self.yfinal = yfinal

    def get(self): 
        return self.yfinal


class DenseGPU:
    def __init__(self, Ownfreqlist, Totalfreqlist, Splitfreqlist, nfmax, datasetprefix,
                 foldername='Mck_freqslices', fileprefix='Mck_freqslice', filesuffix='_sub1', 
                 matname='Rfreq'):
        self.nfreq = nfmax
        self.Ownfreqlist = Ownfreqlist
        self.Totalfreqlist = Totalfreqlist
        self.Splitfreqlist = Splitfreqlist
        self.cupyarray = []
        for freq in self.Ownfreqlist:
            problem = join(datasetprefix, foldername, '{}{}{}.mat'.format(fileprefix, freq, filesuffix))

            if not exists(problem):
                problem = join(datasetprefix, foldername, '{}{}{}.npy'.format(fileprefix, freq, filesuffix))
                A = np.load(problem)
            else:
                A = loadmat(problem)[matname]

            self.cupyarray.append(cp.array(A))
        
    #def SetTransposeConjugate(self,transpose, conjugate):
    #    self.transpose = transpose
    #    self.conjugate = conjugate
    
    
    def tlrmvmgpuinput(self, xlist, transpose=None): # TEMP NAME: need to call it like this to be the same as TLR codes for Fredholm
        conjugate = False # always handled outside
        spx = np.split(xlist, len(self.Ownfreqlist))
        yfinal = [None for _ in self.Ownfreqlist]
        for idx, xvec in enumerate(spx):
            if transpose and conjugate:
                yfinal[idx] = np.conjugate(cp.asnumpy(self.cupyarray[idx].T @ cp.array(np.conjugate(xvec))))
            elif not transpose and conjugate:
                yfinal[idx] = np.conjugate(cp.asnumpy(self.cupyarray[idx] @ np.conjugate(cp.array(xvec))))
            elif transpose and not conjugate:
                yfinal[idx] = cp.asnumpy(self.cupyarray[idx].T @ cp.array(xvec))
            else:
                yfinal[idx] = cp.asnumpy(self.cupyarray[idx] @ cp.array(xvec))
        return YObj(np.hstack(yfinal))
