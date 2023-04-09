##################################################################
# @copyright (c) 2021- King Abdullah University of Science and
#                      Technology (KAUST). All rights reserved.
#
# Author: Yuxi Hong
# Description: Tile Matrix Class on GPU
##################################################################

import numpy as np
import cupy as cp
from pytlr import *
import time
from hilbertcurve.hilbertcurve import HilbertCurve

def randomarray(size,dtype):
    return (np.random.rand(size) + 1j * np.random.rand(size)).astype(dtype)

def randomarray2d(d1,d2,dtype):
    return (np.random.rand(d1,d2) + 1j * np.random.rand(d1,d2)).astype(dtype)

class Tilematrix_Ove3D:
    def __init__(self, m, n, nb, synthetic=False, acc=None, freqlist=None, 
                 datafolder=None):
        self.m = m
        self.n = n
        self.nb = nb
        self.mt = (m + nb -1) // nb
        self.nt = (n + nb -1) // nb
        self.acc = acc
        self.datafolder = datafolder
        # load rank matrix
        self.rankfilelist =["{}/Mode8_Orderhilbert_{}_Rmat_nb{}_acc{}.bin"
                .format(datafolder,freq,nb,acc) for freq in freqlist]
        self.ufilelist =["{}/Mode8_Orderhilbert_{}_Ubases_nb{}_acc{}.bin"
                .format(datafolder,freq,nb,acc) for freq in freqlist]
        self.vfilelist =["{}/Mode8_Orderhilbert_{}_Vbases_nb{}_acc{}.bin"
                .format(datafolder,freq,nb,acc) for freq in freqlist]
        self.rankmatlist = [np.fromfile(rankfile,dtype=np.int32).reshape((self.nt,self.mt)).T
            for rankfile in self.rankfilelist]

    def converttotlrmvminput(self):
        ## generate U buffer
        ubuflist = []
        for i in range(self.mt):
            curbuffer = np.hstack(self.ulist[i])
            ubuflist.append(curbuffer)
        ubuffer = np.hstack(ubuflist).flatten(order='F')
        ## generate V buffer
        vbuflist = []
        for i in range(self.nt):
            curbuffer = []
            for j in range(self.mt):
                curbuffer.append(self.vlist[j][i])
            vbuflist.append(np.vstack(curbuffer).flatten(order='F'))
        vbuffer = np.concatenate(vbuflist)
        return ubuffer,vbuffer

    def converttodense(self):
        # convert tile matrix to a large dense matrix
        denseA = np.zeros((self.mt*self.nb,self.nt*self.nb),dtype=np.csingle)
        for i in range(self.mt):
            for j in range(self.nt):
                denseA[i*self.nb:(i+1)*self.nb,j*self.nb:(j+1)*self.nb] = self.ulist[i][j] @ self.vlist[i][j]
        return denseA[:self.m, :self.n]

    def loaduvbuffer(self,freqidx):
        """
        read u and v buffer from user
        """
        rawu = np.fromfile(self.ufilelist[freqidx],dtype=np.csingle)
        rawv = np.fromfile(self.vfilelist[freqidx],dtype=np.csingle)
        self.rankmatrix = self.rankmatlist[freqidx]
        return rawu,rawv

    def convertuvbuf2uvlist(self,u,v):
        """
        read u and v buffer from user
        u and v are a single array
        """
        rowsum = np.sum(self.rankmatrix,axis=1)
        self.ubuffer = np.split(u,(np.cumsum(rowsum)[:-1])*self.nb )
        self.ulist = [[] for _ in range(self.mt)]
        for idx,ubuf in enumerate(self.ubuffer):
            assert(ubuf.shape[0] == rowsum[idx] * self.nb)
            ubuflist = np.split( ubuf, (np.cumsum(self.rankmatrix[idx,:])[:-1])*self.nb )
            for x in ubuflist:
                tmpx = x.reshape((-1,self.nb)).T
                self.ulist[idx].append(tmpx)

        colsum = np.sum(self.rankmatrix,axis=0)
        self.vbuffer = np.split(v, (np.cumsum(colsum)[:-1])*self.nb)
        self.vlist = [[] for _ in range(self.mt)]
        for idx,vbuf in enumerate(self.vbuffer):
            assert(vbuf.shape[0] == colsum[idx] * self.nb)
            tmpx = vbuf.reshape((self.nb,colsum[idx])).T
            vtilelist = np.split(tmpx, np.cumsum(self.rankmatrix[:,idx])[:-1], axis=0 )
            for curx,vtile in enumerate(vtilelist):
                assert(vtile.shape == (self.rankmatrix[curx,idx],self.nb))
                self.vlist[curx].append(vtile)
        
        
    def mmm_uvlist(self,B,transpose=False):
        assert(len(B.shape) == 2)
        padding_dimx = 0
        padding_dimy = 0
        if not transpose:
            assert(self.n == B.shape[0])
            padding_dimx = self.nt * self.nb
            padding_dimy = self.mt * self.nb
        else:
            assert(self.m == B.shape[0])
            padding_dimx = self.mt * self.nb
            padding_dimy = self.nt * self.nb
        xpadding = np.zeros((padding_dimx,B.shape[1]),np.csingle)
        xpadding[:B.shape[0],:B.shape[1]] = B
        ybuffer = np.zeros((padding_dimy,B.shape[1]),dtype=np.csingle)
        if not transpose:
            for i in range(self.mt):
                for j in range(self.nt):
                    ybuffer[i*self.nb:(i+1)*self.nb] += self.ulist[i][j] @ self.vlist[i][j] @ xpadding[j*self.nb:(j+1)*self.nb]
            ybuffer = ybuffer[:self.m]
        else:
            for i in range(self.nt):
                for j in range(self.mt):
                    ybuffer[i*self.nb:(i+1)*self.nb] += self.vlist[j][i].T @ self.ulist[j][i].T @ xpadding[j*self.nb:(j+1)*self.nb]
            ybuffer = ybuffer[:self.n]
        return ybuffer

    def mmm(self,B,transpose=False):
        if not transpose:
            assert(len(B.shape) == 2)
            colB = B.shape[1]
            yulist = []
            paddingB = np.zeros((self.nt*self.nb,colB),dtype=self.dtype)
            paddingB[:B.shape[0],:B.shape[1]] = B
            for i in range(self.nt):
                mergeoutput = self.vbuffer[i] @ paddingB[i*self.nb:(i+1)*self.nb,:]
                yulist.append(np.split(mergeoutput,np.cumsum(self.rankmatrix[:,i][:-1]),axis=0))
            ybuf = []
            for i in range(self.mt):
                mergeyu = []
                for j in range(len(yulist)):
                    mergeyu.append(yulist[j][i])
                mergeyu = np.vstack(mergeyu)
                ybuf.append(self.ubuffer[i] @ mergeyu)
            yres = np.vstack(ybuf)
        return yres[:self.m,:]

def hilbertIndexing():
    ny, nx, nz = 200, 330, 155
    y, x, z = np.arange(ny)*15., np.arange(nx)*15., np.arange(nz)*15.
    # data size
    nt, nrx, nry = 1126, 177, 90
    dt = 0.004
    dr = 20
    nfmax = 260
    # fk
    nfft=nt
    nfftk=2**8
    rho_sep = 1000.0 # density at separation level
    vel_sep = 1750.0 # velocity at separation level
    critical = 0.9
    ntaper = 9
    cutoff = 1e7
    ## Acquisition
    srcx = np.arange(300,x[-1]-300, 20)
    srcy = np.arange(300,y[-1]-300, 20)
    SRCY, SRCX = np.meshgrid(srcy, srcx, indexing='ij')
    SRCX, SRCY = SRCX.ravel(), SRCY.ravel()
    # shift to original point and scale down
    SRCPoints = [(int(x[0]/20.),int(x[1]/20.)) for x in zip(SRCX-300,SRCY-300)]
    recx = np.arange(700,x[-1]-700, 20)
    recy = np.arange(600,y[-1]-600, 20)
    RECY, RECX = np.meshgrid(recy, recx, indexing='ij')
    RECX, RECY = RECX.ravel(), RECY.ravel()
    # shift to original point and scale down
    RECPoints = [(int(x[0]/20.),int(x[1]/20.)) for x in zip(RECX-700,RECY-600)]
    nsrc, nrec = SRCX.size, RECX.size
    hilbert_curve = HilbertCurve(12, 2)
    hilbertcodes = hilbert_curve.distances_from_points(SRCPoints)
    srcidx = np.argsort(hilbertcodes).astype(np.int32)
    hilbert_curve = HilbertCurve(12, 2)
    hilbertcodes = hilbert_curve.distances_from_points(RECPoints)
    recidx = np.argsort(hilbertcodes).astype(np.int32)
    return srcidx, recidx

class TilematrixGPU_Ove3D:
    def __init__(self, m, n, nb, synthetic=False, acc=None, freqlist=None, 
                 datafolder=None , ranklist=None, order="hilbert", streamsize=1):
        self.m = m
        self.n = n
        self.nb = nb
        self.mt = (self.m + nb -1) // nb
        self.nt = (self.n + nb -1) // nb
        self.synthetic = synthetic
        if not synthetic:
            self.acc = acc
            self.datafolder = datafolder
            self.order = order
            # load rank matrix
            self.rankfilelist =["{}/Mode8_Order{}_{}_Rmat_nb{}_acc{}.bin"
                    .format(datafolder,order,freq,nb,acc) for freq in freqlist]
            self.ufilelist =["{}/Mode8_Order{}_{}_Ubases_nb{}_acc{}.bin"
                    .format(datafolder,order,freq,nb,acc) for freq in freqlist]
            self.vfilelist =["{}/Mode8_Order{}_{}_Vbases_nb{}_acc{}.bin"
                    .format(datafolder,order,freq,nb,acc) for freq in freqlist]
            self.rankmatlist = [np.fromfile(rankfile,dtype=np.int32).reshape((self.nt,self.mt)).T
                for rankfile in self.rankfilelist]
            self.freqlist = freqlist
        else:
            assert(ranklist != None)
            self.rankmatlist = ranklist    
        self.cuconfiglist = [ccuTlrConfig(self.m,self.n,self.nb,list(x.flatten(order='F')),"") for x in self.rankmatlist]
        self.cuconfiglist[0].streaminit(streamsize)
        self.colB = 1
        self.srcidx,self.recidx = hilbertIndexing()
        self.reversesrcidx = np.zeros_like(self.srcidx)
        for idx,val in enumerate(self.srcidx):
            self.reversesrcidx[val] = idx

    def estimategpumemory(self):
        totalrank = np.sum([np.sum(x) for x in self.rankmatlist])
        print("UV buffer memory is ", totalrank * 2 * self.nb * 8 * 1e-9, " GB.")
        print("colB is",self.colB)
        print("RHS memory for non transpose is", self.n * self.colB * len(self.rankfilelist) * 8 * 1e-9, "GB.")
        print("RHS memory for transpose is", self.m * self.colB * len(self.rankfilelist) * 8 * 1e-9, "GB.")

    def loaduvbuffer(self):
        """
        read u and v buffer from user
        """
        if not self.synthetic:
            self.ulist = [np.fromfile(ufile,dtype=np.csingle) for ufile in self.ufilelist]
            self.vlist = [np.fromfile(vfile,dtype=np.csingle) for vfile in self.vfilelist]
            for idx,config in enumerate(self.cuconfiglist):
                ccutlrconfig_setuv(config,self.ulist[idx],self.vlist[idx])


    def setcolB(self,colB):
        self.colB = colB
        for config in self.cuconfiglist:
            ccutlrconfig_setyuyv(config,colB)
 
    def tlrmmm(self,Blist,transpose=False,conjugate=False):
        if self.order == "hilbert":
            for idx,B in enumerate(Blist):
                Blist[idx] = B[self.recidx]
        assert(len(Blist) == len(self.rankfilelist))
        Brows = self.m if transpose else self.n
        yrows = self.m if not transpose else self.n
        assert(Blist[0].shape[0] == Brows)
        colB = Blist[0].shape[1]
        B = np.concatenate([x.flatten(order='F') for x in Blist])
        B = cp.array(B) if not conjugate else cp.array(np.conjugate(B))
        y = cp.zeros(yrows*colB*len(self.rankfilelist),dtype=np.csingle)
        if not transpose:
            cucTlrmmmBatched(self.cuconfiglist,B.data.ptr,colB,y.data.ptr)
        else:
            cucTlrmmm_transBatched(self.cuconfiglist,B.data.ptr,colB,y.data.ptr)
        y = y.get() if not conjugate else np.conjugate(y.get())
        y = y.reshape((-1, yrows)).T
        ylist = np.split(y,len(Blist),axis=1)
        if self.order == "hilbert":
            for idx,yval in enumerate(ylist):
                ylist[idx] = yval[self.reversesrcidx]
        return ylist

    def tlrmvmgpuinput(self,B,transpose=False,conjugate=False):
        # if self.order == "hilbert":
        #     for idx,B in enumerate(Blist):
        #         Blist[idx] = B[self.recidx]
        # assert(len(Blist) == len(self.rankfilelist))
        # Brows = self.m if transpose else self.n
        yrows = self.m if not transpose else self.n
        
        # assert(Blist[0].shape[0] == Brows)
        # colB = Blist[0].shape[1]
        # B = np.concatenate([x.flatten(order='F') for x in Blist])
        # B = cp.array(B) if not conjugate else cp.array(np.conjugate(B))
        y = cp.zeros(yrows*1*len(self.rankfilelist),dtype=np.csingle)
        if not transpose:
            cucTlrmmmBatched(self.cuconfiglist,B.data.ptr,1,y.data.ptr)
        else:
            cucTlrmmm_transBatched(self.cuconfiglist,B.data.ptr,1,y.data.ptr)
        return y

    def __del__(self):
        self.cuconfiglist[0].streamfini()


if __name__ == "__main__":
    m = 2604
    n = 1593
    nb = 50
    mt = (m + nb - 1) // nb
    nt = (n + nb - 1) // nb
    rankmatrix = list(np.random.randint(1,10,(mt*nt)))
    tilemat = Tilematrix(rankmatrix,m,n,nb,np.csingle)
    tilemat.randomuv()
    B = (np.random.rand(n,1) + 1j * np.random.rand(n,1)).astype(np.csingle)
    ybuf = tilemat.mmm_uvlist(B,False)
    denseA = tilemat.converttodense()
    ybuf_dense = denseA @ B
    print(np.allclose(ybuf,ybuf_dense))
    B = (np.random.rand(m,1) + 1j * np.random.rand(m,1)).astype(np.csingle)
    ybuf = tilemat.mmm_uvlist(B,True)
    denseA = tilemat.converttodense()
    print(B.shape)
    print(denseA.T.shape)
    ybuf_dense = denseA.T @ B
    print(np.allclose(ybuf,ybuf_dense))