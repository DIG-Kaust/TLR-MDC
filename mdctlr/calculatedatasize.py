import numpy as np
from os.path import join
def calculate(ordering, bandlength, outtype, intype, STORE_PATH):
    assert(bandlength >= 0 and bandlength <= 39)
    d = {'fp32': 8, 'fp16': 4, 'int8': 2}
    inbts = d[intype]
    outbts = d[outtype]
    inmask = np.zeros((39,39))
    outmask = np.ones((39,39))
    for i in range(39):
        for j in range(39):
            if abs(i-j) < bandlength:
                inmask[i,j] = 1
    outmask = np.subtract(outmask, inmask)
    assert(np.sum(outmask) + np.sum(inmask) == 39 * 39)
    rpath = [ join(STORE_PATH, 'compresseddata','Mode4_Order{}_Mck_freqslice_{}_Rmat_nb256_acc0.001.bin').format(ordering, i) for i in range(150)]
    totalinbytes = 0
    totaloutbytes = 0
    for x in rpath:
        curR = np.fromfile(x,dtype=np.int32).reshape(39,39).T
        totalinbytes += np.sum(np.multiply(inmask, curR)) * 256 * 2 * inbts
        totaloutbytes += np.sum(np.multiply(outmask, curR)) * 256 * 2 * outbts
    return (totalinbytes + totaloutbytes) *1e-9


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--storepath', type=str,default='None', 
        help='your dataset store path.')
    parser.add_argument('--order', type=str,default='normal', 
        help='geometry order type, str, [hilbert,normal,morton,l1]')
    parser.add_argument('--bandlength', type=int,default=10, 
        help='band length of inner band, int,[0-39]')
    parser.add_argument('--outtype', type=str,default='fp16', 
        help='precision of inner band, str,[fp32,fp16,int8]')
    parser.add_argument('--intype', type=str,default='fp32', 
        help='precison of inner band,str, [fp32,fp16,int8]')
    args = parser.parse_args()
    STORE_PATH=args.storepath
    print("Your data path: ", STORE_PATH)
    print("ordering method: ", args.order)
    print("band length: ", args.bandlength)
    if args.bandlength == 0:
        print("full out type: ", args.outtype)
    elif args.bandlength == 39:
        print("full inner type:", args.intype)
    else:
        print("outtype: ", args.outtype)
        print("intype: ", args.intype)
    size = calculate(args.order, args.bandlength, args.outtype, args.intype, STORE_PATH)
    print("size is ", size, " GB")