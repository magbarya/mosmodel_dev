#!/usr/bin/env python3
# import cProfile
import pandas as pd
import itertools
import numpy as np
import subprocess
import math
import os, sys

curr_file_dir = os.path.dirname(os.path.abspath(__file__))
experiments_root_dir = os.path.join(curr_file_dir, '..')
sys.path.append(experiments_root_dir)
from Utils.utils import Utils

class MosrangeExperiment:
    DEFAULT_HUGEPAGE_SIZE = 1 << 21 # 2MB 0

    def __init__(self,
                 memory_footprint_file, pebs_mem_bins_file,
                 collect_reults_cmd, results_file,
                 run_experiment_cmd, exp_root_dir,
                 num_layouts, metric_name, metric_val) -> None:
        self.last_layout_num = 0
        self.collect_reults_cmd = collect_reults_cmd
        self.results_file = results_file
        self.memory_footprint_file = memory_footprint_file
        self.pebs_mem_bins_file = pebs_mem_bins_file
        self.run_experiment_cmd = run_experiment_cmd
        self.exp_root_dir = exp_root_dir
        self.num_layouts = num_layouts
        self.metric_val = metric_val
        self.metric_name = metric_name
        self.load()

    def load(self):
        # read memory-footprints
        self.footprint_df = pd.read_csv(self.memory_footprint_file)
        self.mmap_footprint = self.footprint_df['anon-mmap-max'][0]
        self.brk_footprint = self.footprint_df['brk-max'][0]

        self.hugepage_size = MosrangeExperiment.DEFAULT_HUGEPAGE_SIZE
        self.num_hugepages = math.ceil(self.memory_footprint / self.hugepage_size) # bit vector length

        # round up the memory footprint to match the new boundaries of the new hugepage-size
        self.memory_footprint = (self.num_hugepages + 1) * self.hugepage_size
        self.brk_footprint = self.memory_footprint

        if self.pebs_mem_bins_file is None:
            print('pebs_mem_bins_file argument is missing, skipping loading PEBS results...')
            self.pebs_df = None
            self.total_misses = None
        else:
            self.pebs_df = Utils.load_pebs(self.pebs_mem_bins_file, True)
            self.total_misses = self.pebs_df['NUM_ACCESSES'].sum()

    def run_command(command, out_dir):
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)

        # Run the command
        process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        output, error = process.communicate()

        # Get the output and error messages
        output = output.decode('utf-8')
        error = error.decode('utf-8')

        # Check the return code
        return_code = process.returncode

        output_log = f'{out_dir}/benchmark.log'
        error_log = f'{out_dir}/benchmark.log'
        with open(output_log, 'w+') as out:
            out.write(output)
            out.write('============================================')
            out.write(f'the process exited with status: {return_code}')
            out.write('============================================')
        with open(error_log, 'w+') as err:
            err.write(error)
            err.write('============================================')
            err.write(f'the process exited with status: {return_code}')
            err.write('============================================')
        if return_code != 0:
            # Print the output and error
            print('============================================')
            print(f'Failed to run the following command with exit code: {return_code}')
            print(f'Command line: {command}')
            print('Output:', output)
            print('Error:', error)
            print('Return code:', return_code)
            print('============================================')

        return return_code

    def collect_results(collect_reults_cmd, results_file):
        print(f'** collecting results: {collect_reults_cmd}')

        # Extract the directory path
        results_dir = os.path.dirname(results_file)
        # Create the directory if it doesn't exist
        if not os.path.exists(results_dir):
            os.makedirs(results_dir)

        ret_code = MosrangeExperiment.run_command(collect_reults_cmd, results_dir)
        if ret_code != 0:
            raise RuntimeError(f'Error: collecting experiment results failed with error code: {ret_code}')
        if os.path.exists(results_file):
            results_df = Utils.load_dataframe(results_file)
        else:
            results_df = pd.DataFrame()

        return results_df

    def predictTlbMisses(self, mem_layout):
        assert self.pebs_df is not None
        expected_tlb_coverage = self.pebs_df.query(f'PAGE_NUMBER in {mem_layout}')['NUM_ACCESSES'].sum()
        expected_tlb_misses = self.total_misses - expected_tlb_coverage
        print(f'[DEBUG]: mem_layout of size {len(mem_layout)} has an expected-tlb-coverage={expected_tlb_coverage} and expected-tlb-misses={expected_tlb_misses}')
        return expected_tlb_misses

    def predictTlbCoverage(self, mem_layout):
        assert self.pebs_df is not None
        expected_tlb_coverage = self.pebs_df.query(f'PAGE_NUMBER in {mem_layout}')['TLB_COVERAGE'].sum()
        return expected_tlb_coverage

    def generate_layout_from_pebs(self, pebs_coverage, pebs_df):
        mem_layout = []
        total_weight = 0
        for index, row in pebs_df.iterrows():
            page = row['PAGE_NUMBER']
            weight = row['TLB_COVERAGE']
            if (total_weight + weight) < (pebs_coverage + 0.5):
                mem_layout.append(page)
                total_weight += weight
            if total_weight >= pebs_coverage:
                break
        return mem_layout
    
    def split_pages_to_working_sets(self, upper_pages, lower_pages):
        pebs_set = set(self.pebs_df['PAGE_NUMBER'].to_list())
        upper_set = set(upper_pages)
        lower_set = set(lower_pages)
        all_set = lower_set | upper_set | pebs_set
        all = list(all_set)

        union_set = lower_set | upper_set
        union = list(union_set)
        intersection = list(lower_set & upper_set)
        only_in_lower = list(lower_set - upper_set)
        only_in_upper = list(upper_set - lower_set)
        not_in_upper = list(all_set - upper_set)

        #assert (len(only_in_lower) == 0 and len(only_in_upper) > 0), f'Unexpected behavior: the lower layout ({lower["layout"]}) is included in the upper layout ({upper["layout"]})'
        #print('******************************************')

        not_in_pebs = list(all_set - pebs_set)
        out_union_based_on_pebs = list(pebs_set - union_set)
        out_union = list(all_set - union_set)

        return only_in_upper, only_in_lower, out_union, all

    def get_layout_results(self, layout_name):
        results_df = MosrangeExperiment.collect_results(self.collect_reults_cmd, self.results_file)
        layout_results = results_df[results_df['layout'] == layout_name].iloc[0]
        return layout_results

    def fill_buckets(self, buckets_weights, start_from_tail=False, fill_min_buckets_first=True):
        assert self.pebs_df is not None

        group_size = len(buckets_weights)
        group = [ [] for _ in range(group_size) ]
        df = self.pebs_df.sort_values('TLB_COVERAGE', ascending=start_from_tail)

        threshold = 2
        i = 0
        for index, row in df.iterrows():
            page = row['PAGE_NUMBER']
            weight = row['TLB_COVERAGE']
            selected_weight = None
            selected_index = None
            completed_buckets = 0
            # count completed buckets and find bucket with minimal remaining
            # space to fill, i.e., we prefer to place current page in the
            # bicket that has the lowest remaining weight/space
            for i in range(group_size):
                if buckets_weights[i] <= 0:
                    completed_buckets += 1
                elif buckets_weights[i] >= weight - threshold:
                    if selected_index is None:
                        selected_index = i
                        selected_weight = buckets_weights[i]
                    elif fill_min_buckets_first and buckets_weights[i] < selected_weight:
                        selected_index = i
                        selected_weight = buckets_weights[i]
                    elif not fill_min_buckets_first and buckets_weights[i] > selected_weight:
                        selected_index = i
                        selected_weight = buckets_weights[i]
            if completed_buckets == group_size:
                break
            # if there is a bucket that has a capacity of current page, add it
            if selected_index is not None:
                group[selected_index].append(page)
                buckets_weights[selected_index] -= weight
        return group

    def create_endpoint_layouts(self):
        all_4kb_pages = []
        all_2mb_pages = [i for i in range(self.num_hugepages)]
        mem_layouts = [all_4kb_pages, all_2mb_pages]
        return mem_layouts

    def moselect_initial_samples(self):
        # desired weights for each group layout
        buckets_weights = [56, 28, 14]
        group = self.fill_buckets(buckets_weights)
        mem_layouts = []
        # create eight layouts as all subgroups of these three group layouts
        for subset_size in range(len(group)+1):
            for subset in itertools.combinations(group, subset_size):
                subset_pages = list(itertools.chain(*subset))
                mem_layouts.append(subset_pages)
        return mem_layouts

    def generate_random_layout(self):
        mem_layout = []
        random_mem_layout = np.random.randint(2, size=self.num_hugepages)
        for i in range(len(random_mem_layout)):
            if random_mem_layout[i] == 1:
                mem_layout.append(i)
        return mem_layout

    def get_all_measurements(self):
        res_df = MosrangeExperiment.collect_results(self.collect_reults_cmd, self.results_file)
        if res_df.empty:
            return res_df
        for index, row in res_df.iterrows():
            layout_name = row['layout']
            mem_layout_pages = Utils.load_layout_hugepages(layout_name, self.exp_root_dir)
            res_df.iloc[index]['hugepages'] = mem_layout_pages
        return res_df
    
    def get_previous_run_samples(self):
        res_df = self.get_all_measurements();
        self.last_layout_num = len(res_df)
        return res_df

    def get_surrounding_layouts(self, res_df):
        # sort pebs by stlb-misses
        df = res_df.sort_values(self.metric_name, ascending=True)
        # Find the index of the closest value to X
        closest_index = (df[self.metric_name] - self.metric_val).abs().idxmin()
        # Find the two rows that surround the given stlb-misses value
        lower_layout = df.iloc[closest_index-1]
        upper_layout = df.iloc[closest_index+1]
        return lower_layout, upper_layout

    def generate_initial_samples(self):
        res_df = self.get_previous_run_samples()
        num_prev_samples = len(res_df)
        mem_layouts = self.moselect_initial_samples()
        # mem_layouts = self.create_endpoint_layouts()
        
        for i, mem_layout in enumerate(mem_layouts):
            if i < num_prev_samples:
                continue
            print(f'** Producing initial sample #{i} using a memory layout with {len(mem_layout)} (x2MB) hugepages')
            self.last_layout_num += 1
            layout_name = f'layout{self.last_layout_num}'
            layout_res = self.run_workload(mem_layout, layout_name)
        res_df = self.get_all_measurements()
        return res_df
    
    def generate_layout_from_base(self, base_pages, search_space, coverage):
        expected_coverage = coverage - self.predictTlbCoverage(base_pages)
        df = self.pebs_df.query(f'PAGE_NUMBER in {search_space}').sort_values('TLB_COVERAGE', ascending=False)
        layout = self.generate_layout_from_pebs(expected_coverage, df)
        if layout:
            return layout+base_pages
        else:
            return []

    def generate_layouts(self, next_coverage, alpha, gamma, U):
        mem_layouts = []
        even_gamma = [p for p in gamma if p%2==0]
        even_U = [p for p in U if p%2==0]
        layout = self.generate_layout_from_base(alpha, gamma, next_coverage)
        if layout:
            mem_layouts.append(layout)
        layout = self.generate_layout_from_base(alpha, U, next_coverage)
        if layout:
            mem_layouts.append(layout)
        layout = self.generate_layout_from_base(alpha, even_gamma, next_coverage)
        if layout:
            mem_layouts.append(layout)
        layout = self.generate_layout_from_base(alpha, even_U, next_coverage)
        if layout:
            mem_layouts.append(layout)
        
        return mem_layouts

    def run_workload(self, mem_layout, layout_name):
        Utils.write_layout(layout_name, mem_layout, self.exp_root_dir, self.brk_footprint, self.mmap_footprint)

        print('--------------------------------------')
        print(f'** Running {layout_name} with {len(mem_layout)} hugepages')
        out_dir = f'{self.exp_root_dir}/{layout_name}'
        run_bayesian_cmd = f'{self.run_experiment_cmd} {layout_name}'
        ret_code = MosrangeExperiment.run_command(run_bayesian_cmd, out_dir)
        if ret_code != 0:
            raise RuntimeError(f'Error: running {layout_name} failed with error code: {ret_code}')
        layout_res = self.get_layout_results(layout_name)
        tlb_misses = layout_res['stlb_misses']
        runtime = layout_res['cpu_cycles']
        print(f'\tResults: runtime={runtime/1e9:.2f} Billion cycles , stlb-misses={tlb_misses/1e9:.2f} Billions')
        print('--------------------------------------')
        return layout_res

    def run(self):
        # Define the initial data samples (X and Y pairs) for Bayesian optimization
        res_df = self.generate_initial_samples()

        num_layouts = max(0, (self.num_layouts - len(res_df)))
        if num_layouts == 0:
            print('================================================')
            print(f'No more layouts to run for the experiment:\n{self.exp_root_dir}')
            print('================================================')
            return
        
        while num_layouts > 0:
            res_df = self.get_all_measurements()
            # find the sourrounding layouts of the given stlb-misses
            lower_layout_r, upper_layout_r = self.get_surrounding_layouts(res_df)
            alpha, beta, gamma, U = self.split_pages_to_working_sets(
                upper_layout_r['hugepages'], lower_layout_r['hugepages'])
            upper_mem_layout = upper_layout_r['hugepages']
            lower_mem_layout = lower_layout_r['hugepages']
            upper_coverage = self.predictTlbCoverage(upper_mem_layout)
            lower_coverage = self.predictTlbCoverage(lower_mem_layout)
            next_coverage = (upper_coverage + lower_coverage) / 2
            
            mem_layouts = self.generate_layouts(next_coverage, alpha, gamma, U)
            for mem_layout in mem_layouts:
                self.last_layout_num += 1
                layout_name = f'layout{self.last_layout_num}'
                layout_res = self.run_workload(mem_layout, layout_name)
                num_layouts -= 1
                if num_layouts == 0:
                    break

        print('================================================')
        print(f'Finished running MosRange process for:\n{self.exp_root_dir}')
        print('================================================')

import argparse
def parseArguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('-mem', '--memory_footprint', default='memory_footprint.txt')
    parser.add_argument('-pebs', '--pebs_mem_bins', default=None)
    parser.add_argument('-exp', '--exp_root_dir', required=True)
    parser.add_argument('-res', '--results_file', required=True)
    parser.add_argument('-c', '--collect_reults_cmd', required=True)
    parser.add_argument('-x', '--run_experiment_cmd', required=True)
    parser.add_argument('-n', '--num_layouts', required=True, type=int)
    parser.add_argument('-d', '--debug', action='store_true')
    parser.add_argument('-m', '--metric', choices=['stlb_misses', 'stlb_hits', 'walk_cycles'], default='stlb_misses')
    parser.add_argument('-v', '--metric_value', type=float, required=True)
    return parser.parse_args()

if __name__ == "__main__":
    args = parseArguments()

    # profiler = cProfile.Profile()
    # profiler.enable()
    exp = MosrangeExperiment(args.memory_footprint, args.pebs_mem_bins,
                             args.collect_reults_cmd, args.results_file,
                             args.run_experiment_cmd, args.exp_root_dir,
                             args.num_layouts, args.metric, args.metric_value)
    exp.run()
    # profiler.disable()
    # profiler.dump_stats('profile_results.prof')
    # profiler.print_stats()
