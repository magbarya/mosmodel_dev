#! /usr/bin/env python3

import sys
import os
import time
import subprocess
import shutil
import shlex
from os.path import join, getsize, islink

class BenchmarkRun:
    def __init__(self, benchmark_dir, output_dir):
        self._benchmark_dir = benchmark_dir
        if not os.path.exists(self._benchmark_dir):
            sys.exit('Error: the benchmark path ' + self._benchmark_dir + ' was not found.')

        self._output_dir = os.getcwd() + '/' + output_dir
        if os.path.exists(self._output_dir):
            print('output directory', self._output_dir, 'already exists...')
            self._does_output_directory_exist = True
        else:
            print('creating a new output directory', self._output_dir, '...')
            print('copying the benchmark files to ' + self._output_dir + '...')
            # symlinks are copied as symlinks with symlinks=True
            shutil.copytree(self._benchmark_dir, self._output_dir, symlinks=True)
            self._does_output_directory_exist = False

        log_file_name = self._output_dir + '/benchmark.log'
        self._log_file = open(log_file_name, 'w')

    def __del__(self):
        if hasattr(self, "_log_file"):
            self._log_file.close()

    def doesOutputDirectoryExist(self):
        return self._does_output_directory_exist

    def pre_run(self, pre_run_script='pre_run.sh'):
        print('warming up before running...')
        os.chdir(self._output_dir)
        # the pre_run script will read input files to force them to reside
        # in the page-cache before run() is invoked.
        subprocess.check_call(f'./{pre_run_script}', stdout=self._log_file, stderr=self._log_file)

    def run(self, num_threads, submit_command):
        print('running the benchmark ' + self._benchmark_dir + '...')
        print('the full submit command is:\n\t' + submit_command + ' ./run.sh')
        environment_variables = {"OMP_NUM_THREADS": str(num_threads),
                "OMP_THREAD_LIMIT": str(num_threads)}
        environment_variables.update(os.environ)
        os.chdir(self._output_dir)
        self._run_process = subprocess.Popen(shlex.split(submit_command + ' ./run.sh'),
                stdout=self._log_file, stderr=self._log_file, env=environment_variables)

    def wait(self):
        print('waiting for the run to complete...')
        self._run_process.wait()
        if self._run_process.returncode != 0:
            raise subprocess.CalledProcessError(self._run_process.returncode, ' '.join(self._run_process.args))
        print('sleeping a bit to let the filesystem recover...')
        time.sleep(3) # seconds

    def post_run(self, post_run_script='post_run.sh'):
        print('validating the run outputs...')
        os.chdir(self._output_dir)
        subprocess.check_call(f'./{post_run_script}', stdout=self._log_file, stderr=self._log_file)

    def clean(self, exclude_files=[], threshold=1024*1024):
        print('cleaning large files from the output directory...')
        for root, dirs, files in os.walk('./'):
            for name in files:
                file_path = join(root, name)
                # remove files larger than threshold (default is 1MB)
                if (not islink(file_path)) and (getsize(file_path) > threshold) and (name not in exclude_files):
                    os.remove(file_path)
        print('syncing to clean all pending I/O activity...')
        os.sync()

import argparse
def getCommandLineArguments():
    parser = argparse.ArgumentParser(description='This python script runs a single benchmark, \
            possibly with a prefixing submit command like \"perf stat --\". \
            The script creates a new output directory in the current working directory, \
            copy the benchmark files there, and then pre_run, run, and post_run the benchmark. \
            Finally, the script deletes large files (> 1MB) residing in the output directory.')
    parser.add_argument('-n', '--num_threads', type=int, default=4,
            help='the number of threads (for multi-threaded benchmark)')
    parser.add_argument('-s', '--submit_command', type=str, default='',
            help='a command that will prefix running the benchmark, e.g., "perf stat --".')
    parser.add_argument('-x', '--exclude_files', type=str, nargs='*', default=[],
            help='list of files to not remove')
    parser.add_argument('-f', '--force', action='store_true', default=False,
            help='run the benchmark anyway even if the output directory already exists')
    parser.add_argument('-pre', '--pre_run_script', type=str, default='pre_run.sh',
            help='pre_run.sh script related path to benchmaark_dir')
    parser.add_argument('-post', '--post_run_script', type=str, default='post_run.sh',
            help='post_run.sh script related path to benchmaark_dir')
    parser.add_argument('benchmark_dir', type=str, help='the benchmark directory, must contain three \
            bash scripts: pre_run.sh, run.sh, and post_run.sh')
    parser.add_argument('output_dir', type=str, help='the output directory which will be created for \
            running the benchmark on a clean slate')
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = getCommandLineArguments()

    if os.path.exists(args.output_dir):
        print('Skipping the run because output directory', args.output_dir, 'already exists.')
        print('You can use the \'-f\' flag to suppress this message and run the benchmark anyway.')
    else:
        benchmark_run = BenchmarkRun(args.benchmark_dir, args.output_dir)
        benchmark_run.pre_run(args.pre_run_script)
        benchmark_run.run(args.num_threads, args.submit_command)
        benchmark_run.wait()
        benchmark_run.post_run(args.post_run_script)
        benchmark_run.clean(args.exclude_files)

