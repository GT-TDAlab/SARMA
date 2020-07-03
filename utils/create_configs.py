#!/usr/bin/env python3
import argparse
import datetime
import os
import sys

DELIM = '\t'
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--data_dir', default='', metavar='Input to dataset directory', type=str)
    parser.add_argument('-e', '--extension', default='.mtx', metavar='Extensions of matrix files', type=str)
    parser.add_argument('-c', '--configfile', default='{}'.format(datetime.datetime.now().strftime("sarma-bench_%Y-%m-%d_%H-%M-%S.txt")), metavar='Output config file', type=str)
    parser.add_argument('-a', '--algs', default="nic,pal", metavar='Algorithms to try', type=str)
    parser.add_argument('-o', '--orders', default="nat", metavar='Orders to try', type=str)
    parser.add_argument('-p', '--p', default="16,32", metavar='P values to try', type=str)
    parser.add_argument('-s', '--seeds', default="673", metavar='Seeds to try', type=str)
    parser.add_argument('-y', '--sparsify', default="1,0.01,100", metavar='Sparsifications to try', type=str)
    # if len(sys.argv)==1:
    #     parser.print_help(sys.stderr)
    #     sys.exit(1)
    args = parser.parse_args()

    orders = args.orders.split(',')  
    algorithms = args.algs.split(',')
    ps = list(map(int, args.p.split(',')))
    seeds = list(map(int, args.seeds.split(',')))
    sparsify = list(map(float, args.sparsify.split(',')))

    # pring arguments for both 1) ensure we read and convert them correctly, 2) to notify user what we are using, if we are using defaults
    print("Dir: [" + args.data_dir + "]\t extension: [" + args.extension + "]\t configfile: [" + args.configfile + "]")
    print("Algs: [" + ', '.join(algorithms) + "]\t orders: [" + ', '.join(orders) + "]\t p: [" + ', '.join(str(p) for p in ps) + "]")
    print("Seeds: [" + ', '.join(str(s) for s in seeds) + "]\t Sparsify: [" +  ', '.join(str(s) for s in sparsify) + "]")

    # why os.dirlist doesn't work with empty string but you need to ignore to get it working
    if (args.data_dir!=''):
        graphs = os.listdir(args.data_dir)
    else:
        graphs = os.listdir()
    graphs = list(filter(lambda g: g.endswith(args.extension), graphs))
    graphs = sorted(graphs, key=lambda g: os.stat(args.data_dir + g).st_size, reverse=True)[3:]

    with open(args.configfile, 'w') as fp:
        fp.write("# {}\n".format(args.configfile));
        fp.write("# Graph{delim}graph{delim}order{delim}algorithm{delim}p{delim}prob{delim}seed{delim}load_imbalance{delim}time{delim}norm_imbalance{delim}norm_time\n".format(delim=DELIM))
        for graph in graphs:
            for order in orders:
                for algorithm in algorithms:
                    for p in ps:
                        for prob in sparsify:
                            for seed in seeds:
                                fp.write("{}{delim}{}{delim}{}{delim}{}{delim}{}{delim}{}{delim}{}{delim}{}{delim}{}{delim}{}\n"
                                    .format(os.path.join(args.data_dir, graph), order, algorithm, p, prob, seed, -1, -1, 1, 1, delim=DELIM))
