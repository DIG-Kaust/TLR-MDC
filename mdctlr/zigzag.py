def zigzag(start_freq, end_freq, nodesize):
    """
    Load balancing for x mpi processes.
    return a map, key is node id, value is list of frequency matrices.
    """
    splitfreqlist = []
    cnt = 0
    assert(start_freq >= 0 and start_freq < end_freq)
    nfmax = end_freq - start_freq
    reverse = False
    while cnt < nfmax:
        tmp = []
        idx = 0
        while idx < nodesize:
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

    def getfreqlist(mpirank,start_frequency = 0):
        Ownfreqlist = []
        for x in splitfreqlist:
            if len(x) > mpirank:
                Ownfreqlist.append(x[mpirank] + start_frequency)
        return Ownfreqlist

    freqmap = {}
    for x in range(nodesize):
        freqmap[x] = getfreqlist(x,start_freq)
    return freqmap
