import numpy as np
import pymorton as pm
import matplotlib.pyplot as plt
from hilbertcurve.hilbertcurve import HilbertCurve

def findclosest(p, grid):
    return np.argmin((p[0]-grid[0])**2 + (p[1]-grid[1])**2)


class GeometryArrangement:
    """Geometry rearrangement.

    Re-arrange geometry to ensure locality between neigbouring points

    Parameters
    ----------
    x : :obj:`numpy.ndarray`
        X-coordinates of grid points
    y : :obj:`numpy.ndarray`
        Y-coordinates of grid points
    normalizex : :obj:`float`, optional
        Normalize x-coordinates by this number
    normalizey : :obj:`float`, optional
        Normalize y-coordinates by this number

    """
    def __init__(self, x, y, normalizex=None, normalizey=None):
        self.x, self.y = x, y
        if normalizex is not None:
            self.x /= normalizex
        if normalizey is not None:
            self.y /= normalizey
        self.npoints = len(x)

    def _zeroarrange(self, nb):
        groups = np.tile(np.arange(self.npoints // nb),
                         nb).reshape(nb, -1).T.flatten()
        groups = np.pad(groups, (0, self.npoints-len(groups)),
                        constant_values=groups.max()+1)
        return np.arange(self.npoints), groups

    def _normarrange(self, nb, nbx, nby, kind):
        idx = []
        count = 0
        groups = -np.ones(self.npoints)

        X, Y = self.x.copy(), self.y.copy()
        while len(X) > 0:
            # find left-bottom most source
            ox, oy = 0, 0

            dist = (X - ox) ** 2 + (Y - oy) ** 2
            isel = np.argmin(dist)

            if kind == 'l1':
                # compute distance from selected source
                dist = np.sqrt(np.abs(X[isel] - X) + np.abs(Y[isel] - Y))
                # find nb closest
                distsort = np.argsort(dist)
                iclos = distsort[:nb]
                ifar = distsort[nb:]
            elif kind == 'l2':
                # compute distance from selected source
                dist = np.sqrt((X[isel] - X) ** 2 + (Y[isel] - Y) ** 2)
                # find nb closest
                distsort = np.argsort(dist)
                iclos = distsort[:nb]
                ifar = distsort[nb:]
            elif kind == 'bb':
                # define sources within bounding box
                iall = list(np.arange(len(X)))
                iclos = list(np.where((np.abs(X[isel] - X) < nbx) & (np.abs(Y[isel] - Y) < nby))[0])
                ifar = list(set(iall) - set(iclos))

            if len(iclos) == nb:
                for icl in iclos:
                    iclidx = \
                        np.where((self.x == X[icl]) & (self.y == Y[icl]))[0][0]
                    idx.append(iclidx)
                    groups[iclidx] = count
                count += 1
            X, Y = X[ifar], Y[ifar]

        # add leftover points
        ileftover = list(set(range(self.npoints)) - set(idx))
        for icl in ileftover:
            idx.append(icl)
            groups[icl] = count
        return idx, groups

    def _mortonarrange(self, nb, fast='x'):
        if fast == 'x':
            mortoncodes = np.array(
                [pm.interleave2(int(x), int(y)) for x, y in zip(self.x, self.y)])
        else:
            mortoncodes = np.array(
                [pm.interleave2(int(y), int(x)) for x, y in zip(self.x, self.y)])
        idx = np.argsort(mortoncodes).astype(np.int)
        groups = np.zeros(self.npoints)
        groups[idx] = np.arange(self.npoints) // nb
        return idx, groups

    def _hilbertarrange(self, nb, p, fast='x'):
        if fast == 'x':
            points = np.vstack([self.x, self.y]).T
        else:
            points = np.vstack([self.y, self.x]).T
        hilbert_curve = HilbertCurve(p, 2)
        hilbertcodes = hilbert_curve.distances_from_points(points)
        idx = np.argsort(hilbertcodes).astype(np.int)
        groups = np.zeros(self.npoints)
        groups[idx] = np.arange(self.npoints) // nb
        return idx, groups

    def rearrange(self, nb, nbx=None, nby=None, p=None, fast='x', kind='l2'):
        """Re-arrange geometry with different methods

        Parameters
        ----------
        nb : :obj:`int
            Tile size
        nbx : :obj:`int, optional
            Tile size in x direction
        p : :obj:`int, optional
            Exponent of ``2^p`` which represents the max length of the
            hypercube containing all the points to sort in hilbert sorting.
            Choose it bigger than the maximum point in the grid to sort.
        fast : :obj:`str, optional
            Fast axis (`x` or `y`) for morton and hilbert sorting.
        kind : :obj:`str, optional
            Resorting kind: ``bb`` bounding box, ``l1`` l1 distance sorting,
            ``l2`` l2 distance sorting, ``morton`` morton sorting,
            ``hilbert`` hilbert sorting.

        """
        if nbx is not None and nby is not None and nbx*nby != nb:
            raise ValueError('The product between nbx and nby must match nb')
        if kind == None:
            return self._zeroarrange(nb)
        elif kind in ['bb', 'l1', 'l2']:
            return self._normarrange(nb, nbx, nby, kind)
        elif kind == 'morton':
            return self._mortonarrange(nb, fast)
        elif kind == 'hilbert':
            return self._hilbertarrange(nb, p, fast)

    def selectblocks(self, isrc, ivsx, ivsy, x, y, X, Y, Xfull, Yfull, groups,
                     band=4, prec=0.1, plotflag=False):
        """Select groups in Fresnel zone of source-receiver pair

        Parameters
        ----------
        isrc : :obj:`int
            Index of source
        ivsx : :obj:`int,
            X-index of receiver
        ivsy : :obj:`int,
            Y-index of receiver
        x : :obj:`np.ndarray,
            Regular x-axis
        y : :obj:`np.ndarray,
            Regular y-axis

        """
        # Find parametric line between source and receiver
        if (X[isrc] == x[ivsx]):
            if (Y[isrc] <= y[ivsy]):
                yline = np.arange(Y[isrc], y[ivsy] + 0.1 * prec, prec)
            else:
                yline = np.arange(y[ivsy], Y[isrc] + 0.1 * prec, prec)
            xline = X[isrc] * np.ones_like(yline)
        else:
            if (X[isrc] < x[ivsx]):
                xline = np.arange(X[isrc], x[ivsx] + 0.1 * prec, prec)
            else:
                xline = np.flip(np.arange(x[ivsx], X[isrc] + 0.1 * prec, prec))
            m = (y[ivsy] - Y[isrc]) / (x[ivsx] - X[isrc])
            q = Y[isrc] - m * X[isrc]
            yline = m * xline + q
        igroup = []
        for iband in np.arange(-band // 2, band // 2 + 1):
            for xl, yl in zip(xline, yline + iband):
                if (((X[isrc] < x[ivsx]) and (xl > x[ivsx])) or (
                        (X[isrc] > x[ivsx]) and (xl < x[ivsx]))):
                    break
                iclos = findclosest((xl, yl), np.vstack((Xfull, Yfull)))
                igroup.append(int(groups[iclos]))
            for xl, yl in zip(xline + iband, yline):
                if (((Y[isrc] < y[ivsy]) and (yl > y[ivsy])) or (
                        (Y[isrc] > y[ivsy]) and (yl < y[ivsy]))):
                    break
                iclos = findclosest((xl, yl), np.vstack((Xfull, Yfull)))
                igroup.append(int(groups[iclos]))
            igroup = list(set(igroup))
            if len(igroup) > 0:
                igroups = np.hstack([np.where(groups == ig)[0] for ig in igroup])

        if plotflag:
            plt.figure(figsize=(10,10))
            plt.scatter(Xfull, Yfull, c=groups, cmap='jet', alpha=0.5)
            plt.plot(xline, yline, c='k')
            plt.scatter(Xfull[igroups], Yfull[igroups], c='w', s=50)
            plt.scatter(X[isrc], Y[isrc], c='k', s=200, marker='*', label='SRC')
            plt.scatter(x[ivsx], y[ivsy], c='k',  s=200, label='REC')
            plt.legend()
            plt.show()
        return igroup

    def computemask(self, ivsx, ivsy, x, y, Xfull, Yfull, nb, idx, groups,
                    band=4, prec=0.1, plotflag=False):
        n = Xfull.size
        mask = np.zeros((len(np.unique(groups)) - 1, len(np.unique(groups)) - 1),
                        dtype=np.int)

        for ib, isrc in enumerate(
                list(range(0, n, nb))[:len(np.unique(groups)) - 1]):
            igroup = self.selectblocks(isrc, ivsx, ivsy, x, y, Xfull[idx],
                                       Yfull[idx], Xfull, Yfull, groups,
                                       band=band, prec=prec,
                                       plotflag=False)
            mask[ib, igroup] = 1.

        if plotflag:
            plt.figure(figsize=(10, 10))
            plt.imshow(mask)
            plt.figure()
            plt.plot(np.sum(mask, axis=0), 'k')
            plt.show()
        return mask, np.sum(mask)