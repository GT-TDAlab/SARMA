#!/usr/bin/env python3
import sys
import os
import argparse

import ctypes
import struct

from scipy.io import mmread
from scipy.sparse import csr_matrix

def convert():
    ifname = args.input
    ofname = args.output
    if os.path.isdir(ofname):
        # A directory was given instead, set the file to the same filename in the new directory
        ofname = os.path.join(ofname, '%s.bin' % (os.path.splitext(os.path.basename(ifname))[0],))

    try:
        mtx = mmread(ifname)
        csr = csr_matrix(mtx)
        vtype = 'I' if csr.shape[0] < ctypes.c_uint32(-1).value else 'Q'
        etype = 'I' if csr.getnnz() < ctypes.c_uint32(-1).value else 'Q'
        info_line = ('4' if vtype == 'I' else '8') + ' ' + ('4' if vtype == 'I' else '8') + '\n'
        with open(ofname, 'wb') if ofname != '-' else sys.stdout.buffer as f:
            f.write(info_line.encode("utf-8"))
            f.write(struct.pack(vtype + vtype, *csr.shape))
            f.write(csr.indptr.astype('uint32' if vtype == 'I' else 'uint64'))
            f.write(csr.indices.astype('uint32' if etype == 'I' else 'uint64'))
            f.write(csr.data.astype('uint32')) # Change this, maybe add data type to the info line as well.
    except Exception as e:
        print("Error converting ", ifname)
        raise e

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert matrix martket file to Sarma file csr format.")
    parser.add_argument('-i', '--input', metavar='Input mtx file', type=str, help='Path to Matrix Market file.')
    parser.add_argument('-o', '--output', default='-', metavar='PATH', type=str, help='Path to Matrix Market file.')
    parser.set_defaults(func=convert)
    args = parser.parse_args()
    args.func()
