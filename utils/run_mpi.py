#!/usr/bin/env python3

import subprocess
import os
import sys
import time
import argparse
import datetime
import random
import pandas as pd

def construct_jobscript(args):
    subprocess.call(['mkdir', '-p', args["outpath"], args["outpath"] + '/jobscripts', args["outpath"] + '/outs', args["outpath"] + '/errs'])

    res = []
    
    for order in args["orders"]:
        for alg in args["algs"]:
            for p in args["p"]:
                cmdline = "mpirun -np {} {} {}/{} {} {}".format(p * p, args["exec"], args['dataset'], args["graph"], order, alg)

                if args['submit']:
                    suffix = "_{}_{}_{}".format(order, alg, p)
                    
                    jpath = args["outpath"] + '/jobscripts/' + args["jobname"] + suffix + ".pbs"
                    with open(jpath, 'w') as jfile:
                        jfile.write("#PBS -N " + args["jobname"] + "_{}_{}_{}".format(order, alg, p) + "\n")
                        jfile.write("#PBS -l " + "nodes=" + str((p * p + int(args["ppn"]) - 1) // int(args["ppn"])) + ":ppn=" + args["ppn"] + "\n")
                        jfile.write("#PBS -l " + "mem=" + args["mem"] + "GB" + "\n")
                        jfile.write("#PBS -l " + "walltime=" + args["walltime"] + ":59:59" + "\n")
                        jfile.write("#PBS -q " + args["queue"] + "\n")
                        jfile.write("#PBS -e {}/errs/".format(args["outpath"]) + args["jobname"] + suffix + ".err" "\n")
                        jfile.write("#PBS -o {}/outs/".format(args["outpath"]) + args["jobname"] + suffix + ".out" + "\n")
                        jfile.write("#PBS -m abe" + "\n")
                        # jfile.write("#PBS -M " + args_email + "\n")
                        # jfile.write('echo  ' + '"Started job "' + args_jobname + '"' + "\n\n")
                        jfile.write("cd " + args["projectpath"] + "\n")
                        jfile.write("export GRAPH_DIR=" + args["dataset"] + "\n")
                        jfile.write("module load gcc/9" + "\n")
                        jfile.write("module load python/3" + "\n")
                        jfile.write("module load mvapich2" + "\n")
                        # jfile.write("module load gurobi/9" + "\n")
                        # jfile.write("make" + "\n")
                        jfile.write(cmdline)

                    subprocess.call(['qsub', jpath])
                else:
                    print(cmdline)
                    out = subprocess.check_output(cmdline.split()).decode()
                    print(out)
                    d = {'graph': args['graph'], 'order': order, 'alg': alg, 'p': p}
                    if "spgemm" in args["exec"]:
                        for k, v in zip(['nnz_max', 'nnz_min', 'time_max', 'time_min', 'max_comm', 'max_total_comm'], out.split()):
                            d[k] = v
                    else:
                        V = out.split()
                        d["total_time"] = float(V[1])
                        d["max_time"] = float(V[3])
                        d["min_time"] = float(V[4])
                        d["mean"] = float(V[6])
                        d["stddev"] = float(V[7])
                    res.append(d)
                    print(d)
    
    return res

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--submit', action='store_true')
    parser.add_argument('-d', '--data_dir', default=os.environ['GRAPH_DIR'], metavar='Input to dataset directory', type=str)
    parser.add_argument('-q', '--queue', default='hive-interact', type=str)
    parser.add_argument('-w', '--walltime', default=1, type=int)
    parser.add_argument('-m', '--mem', default=178, type=int)
    parser.add_argument('-n', '--num_runs', default=1, type=int)
    parser.add_argument('-e', '--exec', default='build/mpi_spmv', type=str)
    parser.add_argument('-p', '--ppn', default=24, type=int)
    parser.add_argument('-o', '--output_dir', default='results/{}/'.format(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")), type=str)
    args = parser.parse_args()

    graphs = os.listdir(args.data_dir)
    graphs = list(filter(lambda g: g.endswith('.bin'), graphs))
    graphs = sorted(graphs, key=lambda g: os.stat(args.data_dir + g).st_size, reverse=True)[3:]
    random.shuffle(graphs)

    graphs = ['mouse_gene.bin', 'road_central.bin', 'mycielskian16.bin', 'com-LiveJournal.bin', 'nv2.bin', 'nlpkkt80.bin', 'NLR.bin', 'cage14.bin']
    # graphs = ['kmer_V1r.bin', 'nlpkkt200.bin', 'kmer_A2a.bin', 'stokes.bin', 'Queen_4147.bin', 'mycielskian18.bin', 'uk-2002.bin', 'kmer_P1a.bin', 'HV15R.bin', 'mawi_201512020130.bin']
    res = []
    for i in range(len(graphs)):
        res.extend(construct_jobscript({
            'submit': args.submit,
            'graph': graphs[i],
            'orders': ['nat', 'rnd'],
            'algs': ["opal", "uni", "nic", "sgo", "sgo_n", "alpha"],
            'p': [6, 12],
            'seeds': [str(random.randint(0, 1000000000)) for _ in range(args.num_runs)],
            'outpath': args.output_dir,
            'jobname': str(graphs[i]),
            'email': 'mbalin3@gatech.edu',
            'projectpath': '~/data/sarma/',
            'exec': args.exec,
            'mem': str(args.mem),
            'ppn': str(args.ppn),
            'walltime': str(args.walltime - 1),
            'dataset': args.data_dir,
            'queue': args.queue
        }))
    
        if not args.submit:
            df = pd.DataFrame(res)
            df.to_csv('results.csv')
