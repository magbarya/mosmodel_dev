#! /usr/bin/env python3

import os
import argparse
def getCommandLineArguments():
    parser = argparse.ArgumentParser(description='This python script runs a single benchmark, \
            possibly with a prefixing submit command like \"perf stat --\". \
            The script creates a new output directory in the current working directory, \
            copy the benchmark files there, and then pre_run, run, and post_run the benchmark. \
            Finally, the script deletes large files (> 1MB) residing in the output directory.')
    parser.add_argument('-s', '--num_sockets', type=int, default=2,
            help='the number of the CPU sockets')
    parser.add_argument('-t', '--num_threads', type=int, default=4,
            help='the number of threads (for multi-threaded benchmark)')
    parser.add_argument('-r', '--num_repeats', type=int, default=4,
            help='the number of repetitions (it is recommended to be >= the number of sockets)')
    parser.add_argument('-cmd', '--submit_command', type=str, default='',
            help='a command that will prefix running the benchmark, e.g., "perf stat --".')
    parser.add_argument('-c', '--clean_threshold', type=int, default=1024*1024,
            help='delete files larger than this size (in bytes) after the benchmark runs')
    parser.add_argument('-x', '--exclude_files', type=str, nargs='*', default=[],
            help='list of files to not remove')
    parser.add_argument('-f', '--force', action='store_true', default=False,
            help='run the benchmark anyway even if the output directory already exists')
    parser.add_argument('benchmark_dir', type=str, help='the benchmark directory, must contain three \
            bash scripts: pre_run.sh, run.sh, and post_run.sh')
    parser.add_argument('output_dir', type=str, help='the output directory which will be created for \
            running the benchmark on a clean slate')
    args = parser.parse_args()
    return args

from runBenchmark import BenchmarkRun
if __name__ == "__main__":
    args = getCommandLineArguments()

    repeated_runs = [BenchmarkRun(args.benchmark_dir, args.output_dir +'/repeat'+str(i+1) )
            for i in range(args.num_repeats)]
    should_pre_run = any([not run.doesOutputDirectoryExist() for run in repeated_runs])
    if should_pre_run:
        repeated_runs[0].prerun() # pre_run only once for all repeats

    i = 0
    executed_runs = []
    for run in repeated_runs: # run for each repeat
        if not run.doesOutputDirectoryExist() or args.force:
            node_cset = f'node{i}_cset'
            user = os.getenv('USER')
            cset_shield_cmd = f'sudo -E cset shield --userset={node_cset} --exec {args.submit_command}'
            print(f"executing command: ===> {cset_shield_cmd} <===")
            run.async_run(args.num_threads, cset_shield_cmd)
            executed_runs.append(run)
            i += 1
            # if all nodes are occupied
            if i == args.num_sockets:
                for run in executed_runs: # wait for all executed repeats to finish
                    run.async_wait()
                i = 0
                executed_runs = []

        else:
            print('Skipping the run because output directory', run._output_dir, 'already exists.')
            print('You can use the \'-f\' flag to suppress this message and run the benchmark anyway.')

    for run in executed_runs: # wait for all executed repeat to finish
        run.async_wait()
    for run in repeated_runs: # post_run and clean for each repeat
        if not run.doesOutputDirectoryExist():
            run.postrun()
            run.clean(args.clean_threshold, args.exclude_files)




