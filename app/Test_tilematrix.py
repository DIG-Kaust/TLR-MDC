import numpy as np
import time
import os
import argparse
import time
from mdctlr.tlrmvm.tilematrix import Tilematrix_Ove3D, TilematrixGPU, TilematrixGPU_cupy, TilematrixGPU_Ove3D
description = "unittest for tilematrix class of TLR-MDC"

parser = argparse.ArgumentParser(description=description)
parser.add_argument("--m", type=int, default=26040, help="rows")
parser.add_argument("--n", type=int, default=15930, help="cols")
parser.add_argument("--nb", type=int, default=50, help="tile size")
parser.add_argument("--transpose", type=bool, default=False, help="transpose")
args = parser.parse_args()

m = args.m
n = args.n
nb = args.nb
mt = (m + nb - 1) // nb
nt = (n + nb - 1) // nb
# rankmatrix = list(np.random.randint(1,10,(mt*nt)))
# tilemat = Tilematrix(rankmatrix,m,n,nb,np.csingle)
# tilemat.randomuv()
# B = (np.random.rand(n,1) + 1j * np.random.rand(n,1)).astype(np.csingle)
# ybuf = tilemat.mmm_uvlist(B,False)
# denseA = tilemat.converttodense()
# ybuf_dense = denseA @ B
# print(np.allclose(ybuf,ybuf_dense))

# u,v = tilemat.converttotlrmvminput()
# tilemat.setuvbuffer(u,v)
# ymmmv2 = tilemat.mmmv2(B)
# print(np.allclose(ymmmv2,ybuf_dense))

# cutilemat = TilematrixGPU_cupy(rankmatrix,m,n,nb,np.csingle)
# t0 = time.time()
# cutilemat.setuvbuffer(u,v)
# t1 = time.time()
# print("set uv: ", t1-t0)
# y = cutilemat.mmm(B)
# t2 = time.time()
# print("mmm: ", t2-t1)
# print(np.allclose(y,ybuf_dense))

# t0 = time.time()
# cutilemat = TilematrixGPU(rankmatrix,m,n,nb,np.csingle)
# cutilemat.setuvbuffer(u,v)
# cutilemat.setcolB(1)
# t1 = time.time()
# print("create TilematrixGPU set uv: ", t1-t0)
# y = cutilemat.tlrmmm(B)
# t2 = time.time()
# print("mmm: ", t2-t1)
# print(np.allclose(y,ybuf_dense))

# real dataset
freqlist = [10,11,12,13]
# random
ranklist = []
for _ in range(10):
    rankmatrix = np.random.randint(1,10,(mt*nt))
    ranklist.append(rankmatrix)

batchsize = len(freqlist)
colB = 10

# Blist = [(np.random.rand(n,colB) + 1j * np.random.rand(n,colB)).astype(np.csingle) for _ in range(batchsize)]
# cutilemat = TilematrixGPU_Ove3D(m, n, nb, 
# synthetic=False, datafolder=os.environ["STORE_PATH"],acc="0.0001",freqlist=freqlist)
# cutilemat.loaduvbuffer()
# cutilemat.setcolB(Blist[0].shape[1])
# cutilemat.estimategpumemory()
# ylist = cutilemat.tlrmmm(Blist)
# cputilemat = Tilematrix_Ove3D(m,n,nb,synthetic=True,acc="0.0001",
#     freqlist=freqlist,datafolder=os.environ["STORE_PATH"])
# for i in range(len(Blist)):
#     rawu,rawv = cputilemat.loaduvbuffer(i)
#     cputilemat.convertuvbuf2uvlist(rawu,rawv)
#     ydense = cputilemat.converttodense() @ Blist[i]    
#     print(np.allclose(ylist[i],ydense,rtol=1e-5,atol=1e-7))


# Blist = [(np.random.rand(m,colB) + 1j * np.random.rand(m,colB)).astype(np.csingle) for _ in range(batchsize)]
# cutilemat = TilematrixGPU_Ove3D(m, n, nb, 
# synthetic=False, datafolder=os.environ["STORE_PATH"],acc="0.0001",freqlist=freqlist)
# cutilemat.loaduvbuffer()
# cutilemat.setcolB(Blist[0].shape[1])
# ylist = cutilemat.tlrmmm(Blist,transpose=True)
# cputilemat = Tilematrix_Ove3D(m,n,nb,synthetic=False,acc="0.0001",
#     freqlist=freqlist,datafolder=os.environ["STORE_PATH"])
# for i in range(len(Blist)):
#     rawu,rawv = cputilemat.loaduvbuffer(i)
#     cputilemat.convertuvbuf2uvlist(rawu,rawv)
#     ydense = cputilemat.converttodense().T @ Blist[i]
#     print(np.allclose(ylist[i],ydense,rtol=1e-5,atol=1e-7))

# add hilbert
Blist = [(np.random.rand(n,colB) + 1j * np.random.rand(n,colB)).astype(np.csingle) for _ in range(batchsize)]
Blistcopy = [np.copy(x) for x in Blist]
cutilemat = TilematrixGPU_Ove3D(m, n, nb, 
synthetic=False, datafolder=os.environ["STORE_PATH"],acc="0.0001",freqlist=freqlist)
cutilemat.loaduvbuffer()
cutilemat.setcolB(Blist[0].shape[1])
cutilemat.estimategpumemory()
ylist = cutilemat.tlrmmm(Blist)
cputilemat = Tilematrix_Ove3D(m,n,nb,synthetic=True,acc="0.0001",
    freqlist=freqlist,datafolder=os.environ["STORE_PATH"])
for i in range(len(Blist)):
    rawu,rawv = cputilemat.loaduvbuffer(i)
    cputilemat.convertuvbuf2uvlist(rawu,rawv)
    ydense = cputilemat.converttodense() @ Blistcopy[i][cutilemat.recidx]    
    ydense = ydense[cutilemat.reversesrcidx]
    print(np.allclose(ylist[i],ydense,rtol=1e-5,atol=1e-7))
