##################################################################
# @copyright (c) 2021- King Abdullah University of Science and
#                      Technology (KAUST). All rights reserved.
#
# Author: Yuxi Hong
# Description: generic utilities
##################################################################

import numpy as np
from scipy.spatial import Voronoi, ConvexHull


def voronoi_volumes(points):
    """Voronoi tessellation of set of scatter points
    """
    v = Voronoi(points)
    vol = np.zeros(v.npoints)
    for i, reg_num in enumerate(v.point_region):
        indices = v.regions[reg_num]
        if -1 in indices: # some regions can be opened
            vol[i] = np.inf
        else:
            vol[i] = ConvexHull(v.vertices[indices]).volume
    return v, vol


def snr(xtrue : np.ndarray, xapprox : np.ndarray):
    """Signal noise ratio of two vectors
    """
    return - 20 * np.log10(np.linalg.norm(xapprox-xtrue) / np.linalg.norm(xtrue))
