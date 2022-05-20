##################################################################
# @copyright (c) 2021- King Abdullah University of Science and
#                      Technology (KAUST). All rights reserved.
#
# Author: Yuxi Hong
# Description: A tools for generating compressed U and V bases.
# They are input of TLR-MVM.
##################################################################
import os
from os.path import join, exists
from tqdm import tqdm
import numpy as np
import pickle 
from scipy.linalg import svd
class TLRMVM_Util:
    """A TLR-MVM Utility class 
    1. compute svd for input of TLR-MVM
    3. save U and V bases
    4. save Dense matrix
    """
    def __init__(self, denseAarray, nb, datafolder, error_threshold, problemname, rankmodule) -> None:
        self.denseA = denseAarray
        self.dtype = denseAarray.dtype
        self.m = denseAarray.shape[0]
        self.n = denseAarray.shape[1]
        self.nb = nb 
        self.mtg = self.m // nb if self.m % nb == 0 else self.m // nb + 1
        self.ntg = self.n // nb if self.n % nb == 0 else self.n // nb + 1
        print("self.mtg self.ntg", self.mtg, self.ntg)
        self.paddingm = self.mtg * nb 
        self.paddingn = self.ntg * nb 
        self.datafolder = datafolder
        if not exists(self.datafolder):
            print("Folder {} not exists!".format(self.datafolder))
        self.error_threshold = error_threshold
        self.problemname = problemname
        self.rankfile = join(self.datafolder, 'compresseddata',
        '{}_Rmat_nb{}_acc{}.bin'.format(self.problemname,self.nb,self.error_threshold))
        self.rankmodule = rankmodule

    def computesvd(self):
        A = self.denseA
        padding_m = self.paddingm 
        padding_n = self.paddingn
        m = self.m 
        n = self.n 
        ntiles = self.ntg 
        mtiles = self.mtg 
        svdsavepath = join(self.datafolder, 'SVDinfo')
        if not exists(svdsavepath):
            os.mkdir(svdsavepath)
        nb = self.nb 
        svdname = join( svdsavepath, '{}_nb{}.pickle'.format(self.problemname,nb) )
        if exists(svdname):
            print("svd {} exists.".format(svdname))
            return 
        else:
            print("save svd to {}. ".format(svdname))
        bigmap = dict()
        padA = np.zeros((padding_m,padding_n),dtype=self.dtype)
        padA[:m,:n] = A
        for j in tqdm(range(ntiles)):
            for i in range(mtiles):
                curblock = padA[i*nb:(i+1)*nb, j*nb:(j+1)*nb]
                [u,s,v]  = svd(curblock)
                bigmap['{}_{}'.format(i,j)] = [u,s,v]
        with open( svdname,'wb') as f:
            pickle.dump(bigmap,  f)

    def saveX(self, xvec):
        xfile = join(self.datafolder, '{}_x.bin'.format(self.problemname))
        xvec.tofile(xfile)

    def saveUV(self):
        svdname = join( self.datafolder, 'SVDinfo', '{}_nb{}.pickle'.format(self.problemname,self.nb) )
        if not exists(svdname):
            print("please do svd to matrix first!")
        with open(svdname, 'rb') as f:
            bigmap = pickle.load(f)
        nb = self.nb 
        acc = self.error_threshold
        uvsavepath = join(self.datafolder,'compresseddata')
        if not exists(uvsavepath):
            os.mkdir(uvsavepath)
        ufile = uvsavepath + '/{}_Ubases_nb{}_acc{}.bin'.format(self.problemname,nb,acc)
        vfile = uvsavepath + '/{}_Vbases_nb{}_acc{}.bin'.format(self.problemname,nb,acc)
        rfile = uvsavepath + '/{}_Rmat_nb{}_acc{}.bin'.format(self.problemname,nb,acc)
        
        print("generate uvr file to {}.".format(uvsavepath))
        padding_m = self.paddingm 
        padding_n = self.paddingn
        m = self.m 
        n = self.n 
        ntiles = self.ntg 
        mtiles = self.mtg 
        uvsavepath = self.datafolder
        nb = self.nb 
        tmpacc = self.error_threshold
        acc = tmpacc if isinstance(tmpacc,float) else float(tmpacc)
        ApproximateA = np.zeros((padding_m, padding_n),dtype=self.dtype)
        originpadA = np.zeros((padding_m, padding_n),dtype=self.dtype)
        originpadA[:m,:n] = self.denseA
        normA = np.linalg.norm(self.denseA,'fro')
        ranklist = np.zeros(mtiles * ntiles,dtype=np.int32)
        ulist = [[] for _ in range(mtiles)]
        vlist = [[] for _ in range(mtiles)]
        precarray = np.zeros(mtiles * ntiles,dtype=np.float32)
        u_high = 1e-4
        u_low = 1e-2
        p = mtiles
        for i in tqdm(range(mtiles-1)):
            for j in range(ntiles-1):
                curblock = originpadA[i*nb:(i+1)*nb, j*nb:(j+1)*nb] 
                normblock = np.linalg.norm(curblock,'fro')
                precarray[j*mtiles + i] = (p*normblock) / normA
                [u,s,v] = bigmap['{}_{}'.format(i,j)]
                srk = 0
                erk = nb
                while srk != erk:
                    midrk = (srk + erk) // 2
                    tmpu = u[:, :midrk]
                    tmps = s[:midrk]
                    tmpv = v[:midrk, :]
                    if np.linalg.norm(curblock-(tmpu*tmps)@tmpv, ord='fro') < normA * acc:
                        erk = midrk
                    else:
                        srk = midrk+1
                if srk == 0:
                    srk = 1
                tmpu = u[:, :srk]
                tmps = s[:srk]
                tmpv = v[:srk, :]
                ApproximateA[i*nb:(i+1)*nb, j*nb:(j+1)*nb] = (tmpu*tmps)@tmpv
                us = tmpu * tmps 
                vt = tmpv
                if srk == 0:
                    ranklist[j*mtiles+i] = 1
                    ulist[i].append(np.zeros((nb,1),dtype=self.dtype))
                    vlist[i].append(np.zeros((1,nb),dtype=self.dtype))
                else:
                    ranklist[j*mtiles+i] = srk
                    ulist[i].append(us)
                    vlist[i].append(vt)
        currank = np.copy(ranklist)
        currank = currank.reshape((mtiles, ntiles)).T
        # print(currank)
        def getsrk(normA,nb, acc,u,s,v):
            srk = 0
            erk = nb
            while srk != erk:
                midrk = (srk + erk) // 2
                tmpu = u[:, :midrk]
                tmps = s[:midrk]
                tmpv = v[:midrk, :]
                if np.linalg.norm(curblock-(tmpu*tmps)@tmpv, ord='fro') < normA * acc:
                    erk = midrk
                else:
                    srk = midrk+1
            return srk

        for i in tqdm(range(mtiles)):
            for j in range(ntiles):
                if i < mtiles-1 and j < ntiles-1:
                    continue
                curblock = originpadA[i*nb:(i+1)*nb, j*nb:(j+1)*nb]
                normblock = np.linalg.norm(curblock,'fro')
                precarray[j*mtiles + i] = (p*normblock) / normA
                [u,s,v] = bigmap['{}_{}'.format(i,j)]
                if i < mtiles-1 or j < ntiles-1:
                    if i == mtiles-1:
                        presum = np.sum(currank[:,j])
                        srk = getsrk(normA, nb, acc, u,s,v)
                        while srk < nb and (srk + presum) % self.rankmodule != 0:
                            srk += 1
                        
                        if srk == nb and (srk + presum) % self.rankmodule != 0:
                            print("can't find a solution! i = mtiles")
                            exit()
                        else:
                            currank[i,j] = srk
                    elif j == ntiles-1:
                        presum = np.sum(currank[i,:])
                        srk = getsrk(normA, nb, acc, u,s,v)
                        while srk < nb and (srk + presum) % self.rankmodule != 0:
                            srk += 1
                        if srk == nb and (srk + presum) % self.rankmodule != 0:
                            print("can't find a solution! j = ntiles")
                            exit()
                        else:
                            currank[i,j] = srk
                elif i == mtiles-1 and j == ntiles-1:
                    srk = 0
                    while srk < nb and ( srk + np.sum(currank[i,:]) ) % self.rankmodule != 0 and \
                        ( srk + np.sum(currank[:,j]) ) % self.rankmodule != 0:
                        srk += 1
                    if srk == nb:
                        print("can't find a solution!")
                        exit()
                    else:
                        currank[i,j] = srk
                if srk == 0:
                    srk = self.rankmodule
                tmpu = u[:, :srk]
                tmps = s[:srk]
                tmpv = v[:srk, :]
                ApproximateA[i*nb:(i+1)*nb, j*nb:(j+1)*nb] = (tmpu*tmps)@tmpv
                us = tmpu * tmps 
                vt = tmpv
                if srk == 0:
                    ranklist[j*mtiles+i] = 1
                    ulist[i].append(np.zeros((nb,1),dtype=self.dtype))
                    vlist[i].append(np.zeros((1,nb),dtype=self.dtype))
                else:
                    ranklist[j*mtiles+i] = srk
                    ulist[i].append(us)
                    vlist[i].append(vt)
        tmpurow = []
        for x in ulist:
            tmpurow.append(np.concatenate(x,axis=1))
        finalu = np.concatenate(tmpurow,axis=1)
        finalu.T.tofile(ufile)
        tmpvcol = []
        npvlist = np.array(vlist,dtype=np.object)
        for i in range(npvlist.shape[1]):
            tmpvcol.append(np.concatenate(npvlist[:,i],axis=0))
        
        with open(vfile, 'wb') as f:
            for x in tmpvcol:
                x.T.tofile(f)
        ranklist.tofile(rfile)

        # save precarray
        precarrayfile = join(uvsavepath, 'precarray', "{}_PrecArray_nb{}.bin".format(self.problemname, nb, acc))
        precarray.tofile(precarrayfile)
        print("precarrayfile ", precarrayfile)

    def printdatainfo(self):
        print("Description of Dataset: ")
        print("problem name : {} ".format(self.problemname) )
        print("m is {} n {} nb is {} error threshold is {}.".format(self.m, self.n, self.nb, self.error_threshold))
        rankfile = join(self.datafolder, 'compresseddata', '{}_Rmat_nb{}_acc{}.bin'.format(self.problemname, self.nb, self.error_threshold))
        self.ranklist = np.fromfile(rankfile, dtype=np.int32)
        mn = self.m * self.n 
        rank = np.sum(self.ranklist)
        print("Global rank is {}, compression rate is {:.3f}%.".format( rank, 2*rank*self.nb / mn * 100))
