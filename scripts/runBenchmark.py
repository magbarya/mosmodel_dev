#! /usr/bin/env python3

import argparse
import time
import subprocess
import shutil
import shlex

import signal
import psutil
import os
import os.path
import sys

# try to kill all subprocesses if this script is killed by a signal from the user
def killAllSubprocesses(signum, frame):
    current_process = psutil.Process()
    children = current_process.children(recursive=True)
    for child in children:
        print(f"Killing child process {child.pid}")
        try:
            os.kill(child.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass # the child process may have terminated already
    sys.exit(f"Exiting due to a {signal.Signals(signum).name} signal")

signal.signal(signal.SIGINT, killAllSubprocesses)
signal.signal(signal.SIGTERM, killAllSubprocesses)


class BenchmarkRun:
    def __init__(self, benchmark_name: str, benchmarks_root: str, output_dir: str):
        self._benchmarks_root = benchmarks_root
        self._benchmark_name = benchmark_name
        self._assertBenchmarkIsValid()

        self._output_dir = os.path.abspath(output_dir)
        if os.path.exists(self._output_dir):
            print(f'{self._benchmark_name}: output directory {self._output_dir} already exists')
            self._does_output_dir_exist = True
        else:
            print(f'{self._benchmark_name}: creating a new output directory\n\t{self._output_dir}')
            self._createNewRunDirectory(self._output_dir)
            self._does_output_dir_exist = False

        log_file_name = self._output_dir + '/benchmark.log'
        self._log_file = open(log_file_name, 'w')

    def __del__(self):
        if hasattr(self, "_log_file"):
            self._log_file.close()

    def _assertBenchmarkIsValid(self):
        self._benchmark_dir = self._benchmarks_root + '/' + self._benchmark_name
        if not os.path.exists(self._benchmark_dir):
            sys.exit(f'Error: the benchmark {self._benchmark_name} was not found in {self._benchmark_dir}.\nDid you spell it correctly?')

    def _createNewRunDirectory(self, new_output_dir: str):
        print(f'{self._benchmark_name}: copying the benchmark files')
        # symlinks are copied as symlinks with symlinks=True
        shutil.copytree(self._benchmark_dir, new_output_dir, symlinks=True)

    def doesOutputDirectoryExist(self):
        return self._does_output_dir_exist

    # prerun is required, for example, to read input files into the page-cache before run() is invoked
    def prerun(self):
        print(f'{self._benchmark_name}: prerunning')
        os.chdir(self._output_dir)
        subprocess.run('./prerun.sh', stdout=self._log_file, stderr=self._log_file, check=True)

    def run(self, num_threads: int, submit_command: str):
        print(f'{self._benchmark_name}: running\n\t{submit_command} ./run.sh')

        # override the values already in the environment
        environment_variables = os.environ.copy()
        environment_variables.update({"OMP_NUM_THREADS": str(num_threads),
            "OMP_THREAD_LIMIT": str(num_threads)})

        os.chdir(self._output_dir)
        p = subprocess.run(shlex.split(submit_command + ' ./run.sh'),
                env=environment_variables, stdout=self._log_file, stderr=self._log_file)
        return p

    # postrun is required, for example, to validate the run() outputs
    def postrun(self):
        print(f'{self._benchmark_name}: postrunning')
        os.chdir(self._output_dir)
        # sleep a bit to let the filesystem recover before running postrun.sh
        time.sleep(5)  # seconds
        subprocess.run('./postrun.sh', stdout=self._log_file, stderr=self._log_file, check=True)

    def clean(self, threshold: int = 1024*1024, exclude_files: list = []):
        print(f'{self._benchmark_name}: cleaning large files from the output directory')
        os.chdir(self._output_dir)
        for root, dirs, files in os.walk('./'):
            for name in files:
                file_path = os.path.join(root, name)
                # remove files larger than threshold (default is 1MB)
                if (not os.path.islink(file_path)) and (os.path.getsize(file_path) > threshold) and (name not in exclude_files):
                    os.remove(file_path)
        # sync to clean all pending I/O activity
        os.sync()


def getCommandLineArguments():
    parser = argparse.ArgumentParser(description='This python script runs a single CSL benchmark, \
            possibly with a prefixing submit command like \"perf stat --\". \
            The script creates a new output directory in the current working directory, \
            copy the benchmark files there, and then invoke prerun.sh, run.sh, and postrun.sh. \
            Finally, the script deletes large files (> 1MB) residing in the output directory.')
    parser.add_argument('-n', '--num_threads', type=int, default=4,
            help='use this number of threads (for multi-threaded benchmarks)')
    parser.add_argument('-s', '--submit_command', type=str, default='',
            help='prefix the benchmark run with this command (e.g., "perf stat --")')
    parser.add_argument('-c', '--clean_threshold', type=int, default=1024*1024,
            help='delete files larger than this size (in bytes) after the benchmark runs')
    parser.add_argument('-x', '--exclude_files', type=str, nargs='*', default=[],
            help='do not delete large files whose names appear in this list')
    parser.add_argument('-r', '--repeats', type=int, default=1,
            help='repeat the benchmark run these times')
    parser.add_argument('-l', '--loop_until', type=int, default=None,
            help='run the benchmark repeatedly until LOOP_UNTIL seconds have passed')
    parser.add_argument('-t', '--timeout', type=int, default=None,
            help='timeout the benchmark run to TIMEOUT seconds')
    parser.add_argument('-d', '--directory', type=str, default=None,
            help='run the benchmark in DIRECTORY (defaults to <suite_name>/<benchmark_name>)')
    parser.add_argument('benchmark_name', type=str,
            help='the full benchmark name (<suite_name>/<benchmark_name>)')
    args = parser.parse_args()

    if args.timeout and args.loop_until:
        parser.error('only one of --timeout or --loop_until can be defined')
    if args.timeout and args.timeout <= 0:
        parser.error('timeout must be a positive integer')
    if args.loop_until and args.loop_until <= 0:
        parser.error('loop_until must be a positive integer')

    return args


def findBenchmarksRoot():
    benchmarks_root = sys.path[0]
    # override benchmarks_root if supplied by an environment variable
    environment_variables = dict(os.environ)
    if 'BENCHMARKS_ROOT' in environment_variables:
        benchmarks_root = environment_variables['BENCHMARKS_ROOT']
    error_string = 'Error: the benchmarks root ' + benchmarks_root + ' was not found.\n' + \
            'The directory search path is (in the following order):\n' + \
            '(1) the BENCHMARKS_ROOT environment variable.\n' + \
            '(2) the directory containing this script, i.e., ' + sys.path[0]
    if not os.path.exists(benchmarks_root):
        sys.exit(error_string)
    return benchmarks_root


if __name__ == "__main__":
    cwd = os.getcwd()
    args = getCommandLineArguments()
    benchmarks_root = findBenchmarksRoot()
    output_dir = args.directory if args.directory is not None else args.benchmark_name
    for i in range(args.repeats):
        os.chdir(cwd)
        run = BenchmarkRun(args.benchmark_name, benchmarks_root, os.path.join(output_dir, 'repeat'+str(i)))
        if run.doesOutputDirectoryExist():
            continue  # skip existing directories
        else:
            run.prerun()
            if args.timeout:
                timeout_command = f'timeout {args.timeout}'
                p = run.run(args.num_threads, args.submit_command+' '+timeout_command)
                if p.returncode == 0: # the run ended before the timeout
                    run.postrun()
            elif args.loop_until:
                loop_forever = benchmarks_root + '/loopForever.sh'
                timeout_command = f'timeout {args.loop_until} {loop_forever}'
                run.run(args.num_threads, args.submit_command+' '+timeout_command)
                # don't check the exit status of run() because it was interrupted by timeout
                # don't call postrun() because we cannot validate a run that was interrupted by timeout
            else:
                p = run.run(args.num_threads, args.submit_command)
                p.check_returncode()
                run.postrun()
            run.clean(args.clean_threshold, args.exclude_files)

