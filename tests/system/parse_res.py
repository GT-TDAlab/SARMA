#!/usr/bin/env python3

import sys
import argparse

parser = argparse.ArgumentParser(description='Helper utility to extract cut from output')
parser.add_argument('input_file', help='Input filename containing stdout from sarma')
parser.add_argument('output_file', help='Output filename that will only contain the cuts')
args = parser.parse_args()

with open(args.input_file, 'rb') as fi:
    with open(args.output_file, 'wb') as fo:
        for line in fi:
            if line.startswith( b'Cuts:'):
                fo.write(line)
                break
        for line in fi:
            fo.write(line)
            if line.startswith( b'Max load:'):
                break
