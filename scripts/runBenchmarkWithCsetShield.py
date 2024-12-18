#! /usr/bin/env python3

import argparse
def getCommandLineArguments():
    parser = argparse.ArgumentParser(description='This python script runs a single benchmark, \
            possibly with a prefixing submit command like \"perf stat --\". \
            The script creates a new output directory in the current working directory, \
            copy the benchmark files there, and then pre_run, run, and post_run the benchmark. \
            Finally, the script deletes large files (> 1MB) residing in the output directory.')
    parser.add_argument('-n', '--num_threads', type=int, default=4,
            help='the number of threads (for multi-threaded benchmark)')
    parser.add_argument('-r', '--num_repeats', type=int, default=4,
            help='the number of repetitions (it is recommended to be >= the number of sockets)')
    parser.add_argument('-s', '--submit_command', type=str, default='',
            help='a command that will prefix running the benchmark, e.g., "perf stat --".')
    parser.add_argument('-c', '--clean_threshold', type=int, default=1024*1024,
            help='delete files larger than this size (in bytes) after the benchmark runs')
    parser.add_argument('-x', '--exclude_files', type=str, nargs='*', default=[],
            help='list of files to not remove')
    parser.add_argument('-w', '--warmup', action='store_true', default=False,
            help='run a warmup of the benchmark')
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

    run_dir = args.output_dir +'/run_dir'
    repeated_runs = [BenchmarkRun(args.benchmark_dir, run_dir, args.output_dir +'/repeat'+str(i+1) )
            for i in range(args.num_repeats)]
    # add warmup as a separated run
    if args.warmup:
        warmup_run = BenchmarkRun(args.benchmark_dir, run_dir, args.output_dir + '/warmup')
        repeated_runs = [warmup_run] + repeated_runs

    existing_repeat_dirs = 0
    for run in repeated_runs:
        if run.doesOutputDirectoryExist():
            existing_repeat_dirs += 1
    if existing_repeat_dirs == args.num_repeats and not args.force:
        print(f'Skipping the run because output director [{args.output_dir}] already exists.')
        print('You can use the \'-f\' flag to suppress this message and run the benchmark anyway.')
        exit(0)

    # replace prerun with warmup, which runs the benchmark before other runs

    cset_shield_cmd = f'sudo -E cset shield --exec {args.submit_command}'
    for run in repeated_runs: # run for each repeat
        print('================================================')
        print(f'start producing:\n\t{run._output_dir}')
        p = run.run(args.num_threads, cset_shield_cmd)
        p.check_returncode()
        print(f'start post-running...')
        run.postrun()
        print('================================================')

    existing_repeat_dirs = 0
    for run in repeated_runs:
        if run.doesOutputDirectoryExist():
            existing_repeat_dirs += 1
    # clean only if all output directories exist
    if existing_repeat_dirs == args.num_repeats:
        run.clean(args.clean_threshold, args.exclude_files)




