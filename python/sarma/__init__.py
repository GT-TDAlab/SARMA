#!/usr/bin/env python3

from scipy.sparse import csr_matrix

from ._sarma import sps as _sps, \
                    sgo_2ds as _sgo_2ds, \
                    sgo_2dr as _sgo_2dr, \
                    nic as _nic, \
                    pal as _pal, \
                    rac as _rac, \
                    uni as _uni

class sps(_sps):
    def __init__(self, A):
        if isinstance(A, str):
            super().__init__(A)
        else:
            A = csr_matrix(A)
            super().__init__(A.indptr, A.indices, A.data.astype('uint32'), A.get_shape()[1])

def partition(A, p, alg, *args):
    A = csr_matrix(A)
    return alg(A.indptr, A.indices, A.data.astype('uint32'), A.get_shape()[1], p, *args)

def sgo(A, p, q=None):
    return partition(A, p, _sgo_2ds) if q is None else partition(A, p, _sgo_2dr, q)

def nic(A, p):
    return partition(A, p, _nic)

def pal(A, p):
    return partition(A, p, _pal)

def rac(A, p):
    return partition(A, p, _rac)

def uni(A, p):
    return partition(A, p, _uni)
