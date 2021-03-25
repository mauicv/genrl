import numpy as np


def vector_decomp(p, o, v):
    """
    returns the parallel and orthogonal distances of the point
    along the vector v from o. i.e.

    dist(p, l(t) = o + tv) = min_{t}(dist(p, o + tv))


    :param p: point to decompose into v_orthogonal and v_parallel
    :param o: origin of line.
    :param v: vector along which to decompose p
    :return: p_parallel, p_orthogonal
    """

    v = np.array(v) / np.linalg.norm(v)
    p, o = (np.array(p), np.array(o))
    p_parallel = np.dot(o - p, v)
    p_orthogonal = np.linalg.norm((o - p) - p_parallel*v)
    return -p_parallel, p_orthogonal
