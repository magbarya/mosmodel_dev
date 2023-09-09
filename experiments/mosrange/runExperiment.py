#!/usr/bin/env python3
# import cProfile
import pandas as pd
import itertools
import numpy as np
import subprocess
import math
import os, sys
import logging
import random

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
        self.layouts = []
        self.layout_names = []
        self.search_pebs_threshold = 0.5
        logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
        self.load()

    def load(self):
        # read memory-footprints
        self.footprint_df = pd.read_csv(self.memory_footprint_file)
        self.mmap_footprint = self.footprint_df['anon-mmap-max'][0]
        self.brk_footprint = self.footprint_df['brk-max'][0]

        self.hugepage_size = MosrangeExperiment.DEFAULT_HUGEPAGE_SIZE
        self.num_hugepages = math.ceil(self.brk_footprint / self.hugepage_size) # bit vector length

        # round up the memory footprint to match the new boundaries of the new hugepage-size
        self.memory_footprint = (self.num_hugepages + 1) * self.hugepage_size
        self.brk_footprint = self.memory_footprint

        self.all_pages = [i for i in range(self.num_hugepages)]
        if self.pebs_mem_bins_file is None:
            logging.error('pebs_mem_bins_file argument is missing, skipping loading PEBS results...')
            self.pebs_df = None
        else:
            self.pebs_df = Utils.load_pebs(self.pebs_mem_bins_file, True)
            self.pebs_pages = list(set(self.pebs_df['PAGE_NUMBER'].to_list()))
            self.pages_not_in_pebs = list(set(self.all_pages) - set(self.pebs_pages))
            self.total_misses = self.pebs_df['NUM_ACCESSES'].sum()
        # load results file
        self.results_df = self.get_runs_measurements()
         
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
            # log the output and error
            logging.error('============================================')
            logging.error(f'Failed to run the following command with exit code: {return_code}')
            logging.error(f'Command line: {command}')
            logging.error('Output:', output)
            logging.error('Error:', error)
            logging.error('Return code:', return_code)
            logging.error('============================================')

        return return_code

    def collect_results(collect_reults_cmd, results_file):
        # Extract the directory path
        results_dir = os.path.dirname(results_file)
        # Create the directory if it doesn't exist
        if not os.path.exists(results_dir):
            logging.debug(f'creating new directory: {results_dir}')
            os.makedirs(results_dir)
        else:
            logging.debug(f'collecting results to the existing directory: {results_dir}')

        logging.debug(f'running collect-results script: {collect_reults_cmd}')
        ret_code = MosrangeExperiment.run_command(collect_reults_cmd, results_dir)
        if ret_code != 0:
            raise RuntimeError(f'Error: collecting experiment results failed with error code: {ret_code}')
        if os.path.exists(results_file):
            results_df = Utils.load_dataframe(results_file)
        else:
            logging.warning(f'could not find results file: {results_file}')
            results_df = pd.DataFrame()
        logging.info(f'** results of {len(results_df)} layouts were collected **')

        return results_df

    def predictTlbMisses(self, mem_layout):
        assert self.pebs_df is not None
        expected_tlb_coverage = self.pebs_df.query(f'PAGE_NUMBER in {mem_layout}')['NUM_ACCESSES'].sum()
        expected_tlb_misses = self.total_misses - expected_tlb_coverage
        logging.debug(f'mem_layout of size {len(mem_layout)} has an expected-tlb-coverage={expected_tlb_coverage} and expected-tlb-misses={expected_tlb_misses}')
        return expected_tlb_misses

    def pebsTlbCoverage(self, mem_layout):
        assert self.pebs_df is not None
        df = self.pebs_df.query(f'PAGE_NUMBER in {mem_layout}')
        expected_tlb_coverage = df['TLB_COVERAGE'].sum()
        return expected_tlb_coverage

    def realMetricCoverage(self, layout_results):
        min_val = self.all_2mb_r[self.metric_name]
        max_val = self.all_4kb_r[self.metric_name]
        layout_val = layout_results[self.metric_name]
        coverage = (max_val - layout_val) / (max_val - min_val)
        coverage *= 100
        return coverage

    def generate_layout_from_pebs(self, pebs_coverage, pebs_df):
        mem_layout = []
        total_weight = 0
        for index, row in pebs_df.iterrows():
            page = row['PAGE_NUMBER']
            weight = row['TLB_COVERAGE']
            if (total_weight + weight) < (pebs_coverage + self.search_pebs_threshold):
                mem_layout.append(page)
                total_weight += weight
            if total_weight >= pebs_coverage:
                break
        # could not find subset of pages to add that leads to the required coverage
        if total_weight < (pebs_coverage - self.search_pebs_threshold):
            logging.debug(f'generate_layout_from_pebs(): total_weight < (pebs_coverage - self.search_pebs_threshold): {total_weight} < ({pebs_coverage} - {self.search_pebs_threshold})')
            return []
        logging.debug(f'generate_layout_from_pebs(): found layout of length: {len(mem_layout)}')
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

        not_in_pebs = list(all_set - pebs_set)
        out_union_based_on_pebs = list(pebs_set - union_set)
        out_union = list(all_set - union_set)

        return only_in_upper, only_in_lower, out_union, all

    def get_layout_results(self, layout_name):
        self.get_runs_measurements()
        layout_results = self.results_df[self.results_df['layout'] == layout_name].iloc[0]
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
        all_2mb_pages = [i for i in range(self.num_hugepages)]
        mem_layouts.append(all_2mb_pages)
        return mem_layouts

    def generate_random_layout(self):
        mem_layout = []
        random_mem_layout = np.random.randint(2, size=self.num_hugepages)
        for i in range(len(random_mem_layout)):
            if random_mem_layout[i] == 1:
                mem_layout.append(i)
        return mem_layout

    def get_runs_measurements(self):
        res_df = MosrangeExperiment.collect_results(self.collect_reults_cmd, self.results_file)
        if res_df.empty:
            return res_df
        res_df['hugepages'] = None
        for index, row in res_df.iterrows():            
            layout_name = row['layout']
            mem_layout_pages = Utils.load_layout_hugepages(layout_name, self.exp_root_dir)
            res_df.at[index, 'hugepages'] = mem_layout_pages
        res_df = res_df.query(f'layout in {self.layout_names}').reset_index(drop=True)
        logging.info(f'** kept results of {len(res_df)} collected layouts **')
        self.results_df = res_df
        
        # print results of previous runs
        for index, row in res_df.iterrows():
            logging.debug(f'{row["layout"]}: coverage={self.pebsTlbCoverage(list(row["hugepages"]))} ({len(row["hugepages"])} x hugepages), runtime={row["cpu_cycles"]} , tlb-misses={row["stlb_misses"]}')
        return res_df
    
    def get_surrounding_layouts(self, res_df):
        # sort pebs by stlb-misses
        df = res_df.sort_values(self.metric_name, ascending=True).reset_index(drop=True)
        lower_layout = None
        upper_layout = None
        for index, row in df.iterrows():
            row_metric_val = row[self.metric_name]
            if row_metric_val <= self.metric_val:
                lower_layout = row
            else:
                upper_layout = row
                break
        return lower_layout, upper_layout

    def update_range_stlb_coverage(self, res_df):
        self.all_4kb_r = None
        self.all_2mb_r = None
        max_num_hugepages = 0
        for index, row in res_df.iterrows():
            num_hugepages = len(row['hugepages'])
            if num_hugepages > max_num_hugepages:
                self.all_2mb_r = row
            if num_hugepages == 0:
                self.all_4kb_r = row
        
        min_val = self.all_2mb_r[self.metric_name]
        max_val = self.all_4kb_r[self.metric_name]
        self.metric_coverage = (max_val - self.metric_val) / (max_val - min_val)
        self.metric_coverage *= 100
        
        min_stlb_misses = self.all_2mb_r['stlb_misses']
        max_stlb_misses = self.all_4kb_r['stlb_misses']
        if self.metric_name == 'stlb_misses':
            self.range_stlb_coverage = min_stlb_misses + self.metric_coverage * (max_stlb_misses - min_stlb_misses)
        else:
            self.range_stlb_coverage = max_stlb_misses - self.metric_coverage * (max_stlb_misses - min_stlb_misses)
        
    def generate_initial_samples(self):
        res_df = self.get_runs_measurements()
        num_prev_samples = len(res_df)
        mem_layouts = self.moselect_initial_samples()
        # mem_layouts = self.create_endpoint_layouts()
        
        for i, mem_layout in enumerate(mem_layouts):
            if i < num_prev_samples:
                continue
            logging.info(f'** Producing initial sample #{i} using a memory layout with {len(mem_layout)} (x2MB) hugepages')
            self.run_next_layout(mem_layout)            
        res_df = self.get_runs_measurements()
        self.update_range_stlb_coverage(res_df)
        return res_df
    
    def generate_layout_from_base(self, base_pages, search_space, coverage):
        logging.debug(f'generate_layout_from_base(): len(base_pages)={len(base_pages)} , len(search_space)={len(search_space)} , coverage={coverage}')
        expected_coverage = coverage - self.pebsTlbCoverage(base_pages)
        df = self.pebs_df.query(f'PAGE_NUMBER in {search_space} and PAGE_NUMBER not in {base_pages}')
        df = df.sort_values('TLB_COVERAGE', ascending=False)
        logging.debug(f'generate_layout_from_base() after filtering pages: len(df)={len(df)}')
        layout = self.generate_layout_from_pebs(expected_coverage, df)
        if layout:
            return layout+base_pages
        else:
            return []

    def add_hugepages_to_base(self, next_coverage, base_pages, other_pages, all_pages):
        other_even_pages = [p for p in other_pages if p%2==0]
        all_even_pages = [p for p in all_pages if p%2==0]
        search_space_options = [other_pages, all_pages, other_even_pages, all_even_pages]
        for s in search_space_options:
            layout = self.generate_layout_from_base(base_pages, s, next_coverage)
            if layout and self.isPagesListUnique(layout, self.layouts):
                return layout
        return []

    def remove_hugepages_from_base(self, pebs_coverage, base_pages, pages_to_remove):
        mem_layout = []
        df = self.pebs_df.query(f'PAGE_NUMBER in {base_pages}')
        df = df.sort_values('TLB_COVERAGE', ascending=False)
        total_weight = df['TLB_COVERAGE'].sum()
        # if the coverage of the base_pages less than expected, 
        # then we can not remove pages from it
        if total_weight < (pebs_coverage - self.search_pebs_threshold):
            return []
        for index, row in df.iterrows():
            page = row['PAGE_NUMBER']
            if page not in pages_to_remove:
                continue
            weight = row['TLB_COVERAGE']
            if (total_weight - weight) > (pebs_coverage - self.search_pebs_threshold):
                mem_layout.append(page)
                total_weight -= weight
            if total_weight <= pebs_coverage:
                break
        # could not find subset to remove that leads to the required coverage
        if total_weight > (pebs_coverage + self.search_pebs_threshold):
            return []
        if mem_layout and self.isPagesListUnique(mem_layout, self.layouts):
            return mem_layout
        return []

    def write_layout(self, layout_name, mem_layout):
        logging.info(f'writing {layout_name} with {len(mem_layout)} hugepages')
        Utils.write_layout(layout_name, mem_layout, self.exp_root_dir, self.brk_footprint, self.mmap_footprint)
        self.layouts.append(mem_layout)
        self.layout_names.append(layout_name)
        
    def layout_was_run(self, layout_name, mem_layout):
        prev_layout_res = None
        if not self.results_df.empty:
            prev_layout_res = self.results_df.query(f'layout == "{layout_name}"')
        # prev_layout_res = self.results_df[self.results_df['layout'] == layout_name]
        if prev_layout_res is None or prev_layout_res.empty:
            # the layout does not exist in the results file
            return False, None
        prev_layout_res = prev_layout_res.iloc[0]
        prev_layout_hugepages = prev_layout_res['hugepages']
        if set(prev_layout_hugepages) != set(mem_layout):
            # the existing layout has different hugepages set than the new one
            return False, None
        
        # the layout exists and has the same hugepages set
        return True, prev_layout_res
        
    def run_workload(self, mem_layout, layout_name):
        found, prev_res = self.layout_was_run(layout_name, mem_layout)
        if found:
            logging.info(f'+++ {layout_name} already exists, skipping running it +++')
            self.layouts.append(mem_layout)
            self.layout_names.append(layout_name)
            return prev_res
        
        self.write_layout(layout_name, mem_layout)
        out_dir = f'{self.exp_root_dir}/{layout_name}'
        run_cmd = f'{self.run_experiment_cmd} {layout_name}'
        
        logging.info('-------------------------------------------')
        logging.info(f'*** start running {layout_name} with {len(mem_layout)} hugepages ***')
        logging.info(f'start running workload')
        logging.info(f'\texperiment: {out_dir}')
        logging.debug(f'\tscript: {run_cmd}')
        logging.info(f'\tlayout: {layout_name}')
        logging.info(f'\t#hugepages: {len(mem_layout)}')
        
        ret_code = MosrangeExperiment.run_command(run_cmd, out_dir)
        if ret_code != 0:
            raise RuntimeError(f'Error: running {layout_name} failed with error code: {ret_code}')
        
        layout_res = self.get_layout_results(layout_name)
        tlb_misses = layout_res['stlb_misses']
        tlb_hits = layout_res['stlb_hits']
        walk_cycles = layout_res['walk_cycles']
        runtime = layout_res['cpu_cycles']
        
        logging.info('-------------------------------------------')
        logging.info(f'Results:')
        logging.info(f'\tstlb-misses={tlb_misses/1e9:.2f} Billions')
        logging.info(f'\tstlb-hits={tlb_hits/1e9:.2f} Billions')
        logging.info(f'\twalk-cycles={walk_cycles/1e9:.2f} Billion cycles')
        logging.info(f'\truntime={runtime/1e9:.2f} Billion cycles')
        logging.info('===========================================')
        return layout_res

    def generate_and_run_layouts_v1(self, num_layouts):
        while num_layouts > 0:
            res_df = self.get_runs_measurements()
            # find the sourrounding layouts of the given stlb-misses
            lower_layout_r, upper_layout_r = self.get_surrounding_layouts(res_df)
            alpha, beta, gamma, U = self.split_pages_to_working_sets(
                upper_layout_r['hugepages'], lower_layout_r['hugepages'])
            upper_mem_layout = upper_layout_r['hugepages']
            lower_mem_layout = lower_layout_r['hugepages']
            upper_coverage = self.pebsTlbCoverage(upper_mem_layout)
            lower_coverage = self.pebsTlbCoverage(lower_mem_layout)
            next_coverage = (upper_coverage + lower_coverage) / 2
            
            mem_layouts = self.add_hugepages_to_base(next_coverage, alpha, gamma, U)
            for mem_layout in mem_layouts:
                self.run_next_layout(mem_layout)
                num_layouts -= 1
                if num_layouts == 0:
                    break
    
    def isPagesListUnique(self, pages_list, all_layouts):
        pages_set = set(pages_list)
        for l in all_layouts:
            if set(l) == pages_set:
                return False
        return True
    
    def find_general_layout(self, pebs_coverage, include_pages, exclude_pages, sort_ascending=False):
        df = self.pebs_df.query(f'PAGE_NUMBER not in {exclude_pages}')
        df = df.sort_values('TLB_COVERAGE', ascending=sort_ascending)
        search_space = df['PAGE_NUMBER'].to_list()
        layout = self.generate_layout_from_base(include_pages, search_space, pebs_coverage)
        return layout
        
    def select_layout_from_subsets(self, pebs_coverage, include_pages, exclude_pages, layouts, sort_ascending=False):
        layout = self.find_general_layout(pebs_coverage, include_pages, exclude_pages, sort_ascending)
        if not layout:
            return
        if self.isPagesListUnique(layout, self.layouts + layouts):
            layouts.append(layout)
            return
        
        # for i in range(len(layout)):
        #     self.select_layout_from_subsets(pebs_coverage, layout[i:], layout[:i], layouts, sort_ascending)
        #     if self.isPagesListUnique(layout, self.layouts + layouts):
        #         layouts.append(layout)
        #         return
        for subset_size in range(len(layout)):
            for subset in list(itertools.combinations(layout, subset_size+1)):
                cosubset = set(layout) - set(subset)
                self.select_layout_from_subsets(pebs_coverage, list(subset), list(cosubset), layouts, sort_ascending)
                if self.isPagesListUnique(layout, self.layouts + layouts):
                    layouts.append(layout)
                    return

    def select_layout_generally(self, pebs_coverage):
        logging.debug(f'select_layout_generally() -->: pebs_coverage={pebs_coverage}')
        mem_layouts = []
        self.select_layout_from_subsets(pebs_coverage, [], [], mem_layouts, False)
        for l in mem_layouts:
            if l and self.isPagesListUnique(l, self.layouts):
                return l
        mem_layouts = []
        self.select_layout_from_subsets(pebs_coverage, [], [], mem_layouts, True)
        for l in mem_layouts:
            if l and self.isPagesListUnique(l, self.layouts):
                return l
        return []
    
    def remove_hugepages_blindly(self, pebs_coverage, upper_mem_layout, lower_mem_layout, ratio=None):
        if ratio is None:
            upper_layout_pebs = self.pebsTlbCoverage(upper_mem_layout)
            lower_layout_pebs = self.pebsTlbCoverage(lower_mem_layout)
            ratio = abs(lower_layout_pebs - pebs_coverage) / abs(lower_layout_pebs - upper_layout_pebs)
            if ratio >= 1:
                return ratio, []
        
        hugepages_only_in_lower = list(set(lower_mem_layout) - set(upper_mem_layout))
        remove_set_size = int(len(hugepages_only_in_lower) * ratio)
        remove_set = hugepages_only_in_lower[:remove_set_size]
        layout = list(set(lower_mem_layout) - set(remove_set))
        
        if self.isPagesListUnique(layout, self.layouts):
            return ratio, layout
        return ratio, []
    
    def add_hugepages_blindly(self, pebs_coverage, upper_mem_layout, lower_mem_layout, ratio=None):
        if ratio is None:
            upper_layout_pebs = self.pebsTlbCoverage(upper_mem_layout)
            lower_layout_pebs = self.pebsTlbCoverage(lower_mem_layout)
            ratio = abs(pebs_coverage - upper_layout_pebs) / abs(lower_layout_pebs - upper_layout_pebs)
            if ratio >= 1:
                return ratio, []
        
        hugepages_only_in_lower = list(set(lower_mem_layout) - set(upper_mem_layout))
        add_set_size = int(len(hugepages_only_in_lower) * ratio)
        add_set = hugepages_only_in_lower[:add_set_size]
        layout = list(set(upper_mem_layout) | set(add_set))
                
        if self.isPagesListUnique(layout, self.layouts):
            return ratio, layout
        return ratio, []
    
    def select_layout_blindly(self, upper_mem_layout, num_hugepages_to_add=None):
        layout = []
        add_set = list(set(self.all_pages) - set(upper_mem_layout))
        if num_hugepages_to_add is None:
            num_hugepages_to_add = math.ceil(len(add_set) / 2)
        num_hugepages_to_add = max(1, num_hugepages_to_add)
        num_hugepages_to_add = min(num_hugepages_to_add, len(add_set))
        while not layout or not self.isPagesListUnique(layout, self.layouts):
            random_subset = random.sample(add_set, num_hugepages_to_add)
            layout = list(set(upper_mem_layout) | set(random_subset))
        return layout
    
    def run_next_layout(self, mem_layout):
        self.last_layout_num += 1
        layout_name = f'layout{self.last_layout_num}'
        logging.info(f'run workload under {layout_name} with {len(mem_layout)} hugepages')
        last_result = self.run_workload(mem_layout, layout_name)
        return last_result
    
    def generate_and_run_layouts(self, option=1):
        '''
        upper-layout is the layout with the higher runtime (mostly with less hugepages than lower-layout)
        lower-layout is the layout with the lower runtime (mostly with more hugepages than upper-layout)
        Generally:
        | Layout | runtime | TLB-misses | walk-cycles | TLB-hits | coverage |
        =====================================================================
        | upper  | higher  | higher     | higher      | lower    | lower    |
        ---------------------------------------------------------------------
        | lower  | lower   | lower      | lower       | higher   | higher   |
        ---------------------------------------------------------------------
        '''
        res_df = self.get_runs_measurements()
        lower_layout_r, upper_layout_r = self.get_surrounding_layouts(res_df)
        upper_mem_layout = upper_layout_r['hugepages']
        lower_mem_layout = lower_layout_r['hugepages']
        alpha, beta, gamma, U = self.split_pages_to_working_sets(upper_mem_layout, lower_mem_layout)            
        upper_pebs_coverage = self.pebsTlbCoverage(upper_mem_layout)
        lower_pebs_coverage = self.pebsTlbCoverage(lower_mem_layout)
        next_coverage = (upper_pebs_coverage + lower_pebs_coverage) / 2
        delta_pebs_coverage = abs(upper_pebs_coverage - lower_pebs_coverage) / 2
        upper_real_coverage = self.realMetricCoverage(upper_layout_r)
        lower_real_coverage = self.realMetricCoverage(lower_layout_r)
        
        num_extra_pages = len(self.pages_not_in_pebs)//2
        remove_ratio = None
        add_ratio = None
        last_result = None
        num_hugepages_to_add = None
        
        logging.debug('-------------------------------------------------')
        logging.debug('[DEBUG] next layout configuration:')
        logging.debug(f'\t previous layouts: {len(res_df)}')
        logging.debug(f'\t upper-layout:')
        logging.debug(f'\t\t {upper_layout_r["layout"]}: pebs-coverage={upper_pebs_coverage} , real-coverage={upper_real_coverage} , tlb-misses={upper_layout_r["stlb_misses"]} , size={len(upper_mem_layout)}')
        logging.debug(f'\t lower-layout:')
        logging.debug(f'\t\t {lower_layout_r["layout"]}: pebs-coverage={lower_pebs_coverage} , real-coverage={lower_real_coverage} , tlb-misses={lower_layout_r["stlb_misses"]} , size={len(lower_mem_layout)}')
        logging.debug(f'\t all-4KB layout:')
        logging.debug(f'\t\t {self.all_4kb_r["layout"]}: runtime={self.all_4kb_r["cpu_cycles"]} , tlb-misses={self.all_4kb_r["stlb_misses"]} , tlb-hits={self.all_4kb_r["stlb_hits"]} , walk-cycles={self.all_4kb_r["walk_cycles"]}')
        logging.debug(f'\t all-2MB layout:')
        logging.debug(f'\t\t {self.all_2mb_r["layout"]}: runtime={self.all_2mb_r["cpu_cycles"]} , tlb-misses={self.all_2mb_r["stlb_misses"]} , tlb-hits={self.all_2mb_r["stlb_hits"]} , walk-cycles={self.all_2mb_r["walk_cycles"]}')
        logging.debug(f'\t next layout:')
        logging.debug(f'\t\t coverage={next_coverage} , range_stlb_coverage={self.range_stlb_coverage} , metric_coverage={self.metric_coverage}')
        logging.debug(f'\t Debug info:')
        logging.debug(f'\t\t num-pebs-pages={len(self.pebs_df)}')
        logging.debug('-------------------------------------------------')
        
        while self.last_layout_num < self.num_layouts:
            if option == 1:
                # remove hugepages from lower layout
                logging.info(f'=====> select layout by removing hugepages from lower layout (option#1) <======')
                layout = self.remove_hugepages_from_base(next_coverage, lower_mem_layout, beta)
                if layout:
                    last_result = self.run_next_layout(layout)
                    last_result_real_coverage = self.realMetricCoverage(last_result)
                    if last_result_real_coverage > lower_real_coverage:
                        delta_pebs_coverage *= 1.2
                        next_coverage = lower_pebs_coverage - delta_pebs_coverage
                        continue
                    elif last_result_real_coverage < upper_real_coverage:
                        delta_pebs_coverage *= 0.7
                        next_coverage = lower_pebs_coverage - delta_pebs_coverage
                        continue
                    else:
                        self.generate_and_run_layouts(option=1)
                        continue
                else:
                    logging.info('could not select layout by removing hugepages from lower layout (option#1)')
                    option += 1
                    self.generate_and_run_layouts(option)
                    return
            elif option == 2:
                # add hugepages to upper layout
                logging.info(f'=====> select layout by adding hugepages to upper layout (option#2) <======')
                layout = self.add_hugepages_to_base(next_coverage, upper_mem_layout, gamma, U)
                if layout:
                    last_result = self.run_next_layout(layout)
                    last_result_real_coverage = self.realMetricCoverage(last_result)
                    if last_result_real_coverage > lower_real_coverage:
                        delta_pebs_coverage *= 0.8
                        next_coverage = upper_pebs_coverage + delta_pebs_coverage
                        continue
                    elif last_result_real_coverage < upper_real_coverage:
                        delta_pebs_coverage *= 1.3
                        next_coverage = upper_pebs_coverage + delta_pebs_coverage
                        continue
                    else:
                        self.generate_and_run_layouts(option=1)
                        return
                else:
                    logging.info('could not select layout by adding hugepages to upper layout (option#2)')
                    option += 1
                    self.generate_and_run_layouts(option)
                    return
            elif option == 3:
                # generate layout generally
                logging.info(f'=====> select layout generally (option#3) <======')
                layout = self.select_layout_generally(next_coverage)
                if layout:
                    last_result = self.run_next_layout(layout)
                    last_result_real_coverage = self.realMetricCoverage(last_result)
                    if last_result_real_coverage > lower_real_coverage:
                        delta_pebs_coverage *= 0.8
                        next_coverage = upper_pebs_coverage + delta_pebs_coverage
                        continue
                    elif last_result_real_coverage < upper_real_coverage:
                        delta_pebs_coverage *= 1.3
                        next_coverage = upper_pebs_coverage + delta_pebs_coverage
                        continue
                    else:
                        self.generate_and_run_layouts(option=1)
                        return
                else:
                    logging.info('could not select layout generally (option#3)')
                    option += 1
                    self.generate_and_run_layouts(option)
                    return
            elif option == 4:
                # generate layout blindly
                logging.info(f'=====> select layout by removing hugepages blindly (option#4) <======')
                remove_ratio, layout = self.remove_hugepages_blindly(next_coverage, upper_mem_layout, lower_mem_layout, remove_ratio)
                if layout:
                    last_result = self.run_next_layout(layout)
                    last_result_real_coverage = self.realMetricCoverage(last_result)
                    if last_result_real_coverage > lower_real_coverage:
                        remove_ratio *= 1.2
                        continue
                    elif last_result_real_coverage < upper_real_coverage:
                        remove_ratio *= 0.7
                        continue
                    else:
                        self.generate_and_run_layouts(option)
                        return
                else:
                    logging.info('could not select by removing hugepages layout blindly (option#4)')
                    option += 1
                    self.generate_and_run_layouts(option)
                    continue
            elif option == 5:
                # generate layout blindly
                logging.info(f'=====> select layout by adding hugepages blindly (option#5) <======')
                add_ratio, layout = self.add_hugepages_blindly(next_coverage, upper_mem_layout, lower_mem_layout, add_ratio)
                if layout:
                    last_result = self.run_next_layout(layout)
                    last_result_real_coverage = self.realMetricCoverage(last_result)
                    if last_result_real_coverage > lower_real_coverage:
                        add_ratio *= 0.8
                        continue
                    elif last_result_real_coverage < upper_real_coverage:
                        add_ratio *= 1.3
                        continue
                    else:
                        self.generate_and_run_layouts(option)
                        return
                else:
                    logging.info('could not select layout by adding hugepages blindly (option#5)')
                    option += 1
                    self.generate_and_run_layouts(option)
                    continue
            elif option == 6:
                # select layout blindly
                logging.info(f'=====> select layout blindly (option#6) <======')
                layout = self.select_layout_blindly(upper_mem_layout, num_hugepages_to_add)
                if layout:
                    last_result = self.run_next_layout(layout)
                    last_result_real_coverage = self.realMetricCoverage(last_result)
                    if last_result_real_coverage > lower_real_coverage:
                        num_hugepages_to_add *= 0.8
                        continue
                    elif last_result_real_coverage < upper_real_coverage:
                        num_hugepages_to_add *= 1.3
                        continue
                    else:
                        self.generate_and_run_layouts(option)
                        return
                else:
                    logging.info('could not select layout blindly (option#6)')
                    # self.generate_and_run_layouts(option)
                    continue
            else:
                # assert
                assert False,'unknown option!'

    def run(self):
        # Define the initial data samples
        res_df = self.generate_initial_samples()

        num_layouts = max(0, (self.num_layouts - len(res_df)))
        if num_layouts == 0:
            logging.info('================================================')
            logging.info(f'No more layouts to run for the experiment:\n{self.exp_root_dir}')
            logging.info('================================================')
            return
        
        self.generate_and_run_layouts()

        logging.info('================================================')
        logging.info(f'Finished running MosRange process for:\n{self.exp_root_dir}')
        logging.info('================================================')

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
