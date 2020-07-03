#!/usr/bin/env python3

from sarma import *
from scipy.io import mmread
A = mmread('../tests/system/matrices/email-Eu-core.mtx')
print(A.get_shape())
Q = sps(A)
for alg in [nic, pal, rac, uni]:
    p = alg(A, 8)
    if isinstance(p, tuple):
        L = Q.max_load(*p)
    else:
        L = Q.max_load(p, p)
    print(alg.__name__, p, L)
