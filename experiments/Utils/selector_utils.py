#!/usr/bin/env python3
# import cProfile
import pandas as pd
import subprocess
import math
import os
import logging
from Utils.collect_results import CollectResults
from Utils.layout_utils import LayoutUtils

class Selector:
    DEFAULT_HUGEPAGE_SIZE = 1 << 21 # 2MB 0

    def __init__(self,
                 memory_footprint_file, pebs_mem_bins_file,
                 exp_root_dir, results_dir,
                 run_experiment_cmd,
                 num_layouts, num_repeats,
                 metric_name='stlb_misses',
                 rebuild_pebs=True,
                 skip_outliers=False,
                 run_endpoint_layouts=True) -> None:
        self.memory_footprint_file = memory_footprint_file
        self.pebs_mem_bins_file = pebs_mem_bins_file
        self.exp_root_dir = exp_root_dir
        self.num_layouts = num_layouts
        self.num_repeats = num_repeats
        self.collectResults = CollectResults(exp_root_dir, results_dir, num_repeats)
        self.run_experiment_cmd = run_experiment_cmd
        self.metric_name = metric_name
        self.rebuild_pebs = rebuild_pebs
        self.skip_outliers = skip_outliers
        self.run_endpoint_layouts = run_endpoint_layouts
        self.last_layout_num = 0
        self.num_generated_layouts = 0
        self.layouts = []
        self.layout_names = []
        logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
        self.__load()

    def __load(self):
        # read memory-footprints
        self.footprint_df = pd.read_csv(self.memory_footprint_file)
        self.mmap_footprint = self.footprint_df['anon-mmap-max'][0]
        self.brk_footprint = self.footprint_df['brk-max'][0]

        self.hugepage_size = Selector.DEFAULT_HUGEPAGE_SIZE
        self.num_hugepages = math.ceil(self.brk_footprint / self.hugepage_size) # bit vector length

        # round up the memory footprint to match the new boundaries of the new hugepage-size
        self.memory_footprint = (self.num_hugepages + 1) * self.hugepage_size
        self.brk_footprint = self.memory_footprint

        self.all_pages = [i for i in range(self.num_hugepages)]
        if self.pebs_mem_bins_file is None:
            logging.warning('pebs_mem_bins_file argument is missing, skipping loading PEBS results...')
            self.pebs_df = None
        else:
            self.pebs_df = Selector.load_pebs(self.pebs_mem_bins_file, True)
            self.pebs_pages = list(set(self.pebs_df['PAGE_NUMBER'].to_list()))
            self.pages_not_in_pebs = list(set(self.all_pages) - set(self.pebs_pages))
            self.total_misses = self.pebs_df['NUM_ACCESSES'].sum()
        # load results file
        self.results_df, _ = self.collect_results(filter_results=False)

        self.all_4kb_layout = []
        self.all_2mb_layout = [i for i in range(self.num_hugepages)]
        self.all_pebs_pages_layout = self.pebs_pages
        
        if self.run_endpoint_layouts:
            self.run_endpoint_layouts()

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

    def collect_results(self, filter_results=True):
        logging.debug(f'collecting results to the directory: {self.results_dir}')
        results_df, found_outliers = self.collectResults.collectResults(False, True)
        logging.info(f'** results of {len(results_df)} layouts were collected **')
        if found_outliers:
            logging.info(f'-- outliers were found and removed --')
            
        if results_df:
            results_df['hugepages'] = None
            for index, row in results_df.iterrows():
                layout_name = row['layout']
                mem_layout_pages = LayoutUtils.load_layout_hugepages(layout_name, self.exp_root_dir)
                results_df.at[index, 'hugepages'] = mem_layout_pages
            if filter_results:
                results_df = results_df.query(f'layout in {self.layout_names}').reset_index(drop=True)
                logging.info(f'** kept results of {len(results_df)} collected layouts **')

            # print results of previous runs
            for index, row in results_df.iterrows():
                logging.debug(f'{row["layout"]}: coverage={self.pebsTlbCoverage(list(row["hugepages"]))} ({len(row["hugepages"])} x hugepages), runtime={row["cpu_cycles"]} , tlb-misses={row["stlb_misses"]}')

        return results_df, found_outliers

    def load_pebs(pebs_out_file, normalize=True):
        # read mem-bins
        pebs_df = pd.read_csv(pebs_out_file, delimiter=',')

        if not normalize:
            pebs_df = pebs_df[['PAGE_NUMBER', 'NUM_ACCESSES']]
            pebs_df = pebs_df.reset_index()
            pebs_df = pebs_df.sort_values('NUM_ACCESSES', ascending=False)
            return pebs_df

        # filter and eep only brk pool accesses
        pebs_df = pebs_df[pebs_df['PAGE_TYPE'].str.contains('brk')]
        if pebs_df.empty:
            raise SyntaxError('Input file does not contain page accesses information about the brk pool!')
        pebs_df = pebs_df[['PAGE_NUMBER', 'NUM_ACCESSES']]

        # transform NUM_ACCESSES from absolute number to percentage
        total_access = pebs_df['NUM_ACCESSES'].sum()
        pebs_df['TLB_COVERAGE'] = pebs_df['NUM_ACCESSES'].mul(100).divide(total_access)
        pebs_df = pebs_df.sort_values('TLB_COVERAGE', ascending=False)
        
        pebs_df = pebs_df.reset_index()
        
        return pebs_df
    
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

    def realMetricCoverage(self, layout_results, metric_name=None):
        if metric_name is None:
            metric_name = self.metric_name
        layout_metric_val = layout_results[metric_name]
        all_2mb_metric_val = self.all_2mb_r[metric_name]
        all_4kb_metric_val = self.all_4kb_r[metric_name]
        min_val = min(all_2mb_metric_val, all_4kb_metric_val)
        max_val = max(all_2mb_metric_val, all_4kb_metric_val)

        # either metric_val or metric_coverage will be provided
        delta = max_val - min_val
        coverage = ((max_val - layout_metric_val) / delta) * 100

        return coverage
    
    def run_endpoint_layouts(self):
        mem_layouts = [self.all_4kb_layout, self.all_2mb_layout, self.all_pebs_pages_layout]
        for i, mem_layout in enumerate(mem_layouts):
            logging.info(f'** Producing sample #{i} using endpoint memory layout with {len(mem_layout)} (x2MB) hugepages')
            self.run_next_layout(mem_layout)
        self.update_endpoint_results(self.results_df)
        return self.results_df
    
    def update_endpoint_results(self, res_df):
        all_4kb_set = set(self.all_4kb_layout)
        all_2mb_set = set(self.all_2mb_layout)
        all_pebs_set= set(self.all_pebs_pages_layout)
        for index, row in res_df.iterrows():
            hugepages_set = set(row['hugepages'])
            if hugepages_set == all_4kb_set:
                self.all_4kb_r = row
            elif hugepages_set == all_2mb_set:
                self.all_2mb_r = row
            elif hugepages_set == all_pebs_set:
                self.all_pebs_r = row

        all_2mb_metric_val = self.all_2mb_r[self.metric_name]
        all_4kb_metric_val = self.all_4kb_r[self.metric_name]
        self.metric_min_val = min(all_2mb_metric_val, all_4kb_metric_val)
        self.metric_max_val = max(all_2mb_metric_val, all_4kb_metric_val)

        # either metric_val or metric_coverage will be provided
        self.metric_range_delta = self.metric_max_val - self.metric_min_val
        
        # add missing pages to pebs
        if self.rebuild_pebs:
            self.add_missing_pages_to_pebs()

    def select_layout_from_pebs_gradually(self, pebs_coverage, pebs_df):
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
            logging.debug(f'select_layout_from_pebs_gradually(): total_weight < (pebs_coverage - self.search_pebs_threshold): {total_weight} < ({pebs_coverage} - {self.search_pebs_threshold})')
            return []
        logging.debug(f'select_layout_from_pebs_gradually(): found layout of length: {len(mem_layout)}')
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
        layout_results = self.results_df[self.results_df['layout'] == layout_name].iloc[0]
        return layout_results

    def get_surrounding_layouts(self, res_df, metric_name, metric_val):
        # sort pebs by stlb-misses
        df = res_df.sort_values(metric_name, ascending=True).reset_index(drop=True)
        lower_layout = None
        upper_layout = None
        for index, row in df.iterrows():
            row_metric_val = row[metric_name]
            if row_metric_val <= metric_val:
                lower_layout = row
            else:
                upper_layout = row
                break
        return lower_layout['hugepages'], upper_layout['hugepages']

    def add_missing_pages_to_pebs(self):
        pebs_pages = self.pebs_df['PAGE_NUMBER'].tolist()
        missing_pages = list(set(self.all_2mb_layout) - set(pebs_pages))
        #self.total_misses
        all_pebs_real_coverage = self.realMetricCoverage(self.all_pebs_r, 'stlb_misses')
        # normalize pages recorded by pebs based on their real coverage
        ratio = all_pebs_real_coverage / 100
        self.pebs_df['TLB_COVERAGE'] *= ratio
        self.pebs_df['NUM_ACCESSES'] = (self.pebs_df['NUM_ACCESSES'] * ratio).astype(int)
        # add missing pages with a unified coverage ratio
        missing_pages_total_coverage = 100 - all_pebs_real_coverage
        total_missing_pages = len(missing_pages)
        if total_missing_pages == 0:
            return
        missing_pages_coverage_ratio = missing_pages_total_coverage / total_missing_pages
        # update total_misses acording to the new ratio
        old_total_misses = self.total_misses
        self.total_misses *= ratio
        missing_pages_total_misses = self.total_misses - old_total_misses
        missing_pages_misses_ratio = missing_pages_total_misses / total_missing_pages
        # update pebs_df dataframe
        missing_pages_df = pd.DataFrame(
            {'PAGE_NUMBER': missing_pages,
             'NUM_ACCESSES': missing_pages_misses_ratio,
             'TLB_COVERAGE': missing_pages_coverage_ratio})
        self.pebs_df = pd.concat([self.pebs_df, missing_pages_df], ignore_index=True)

    def write_layout(self, layout_name, mem_layout):
        logging.info(f'writing {layout_name} with {len(mem_layout)} hugepages')
        LayoutUtils.write_layout(layout_name, mem_layout, self.exp_root_dir, self.brk_footprint, self.mmap_footprint)
        self.layouts.append(mem_layout)
        self.layout_names.append(layout_name)

    def isPagesListUnique(self, pages_list, all_layouts):
        pages_set = set(pages_list)
        for l in all_layouts:
            if set(l) == pages_set:
                return False
        return True

    def find_layout_results(self, layout):
        for index, row in self.results_df.iterrows():
            prev_layout_hugepages = row['hugepages']
            if set(prev_layout_hugepages) == set(layout):
                return True, row
        return False, None
    
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
        # if the layout's measurements were found
        if found:
            logging.info(f'+++ {layout_name} already exists, skip running it +++')
            self.layouts.append(mem_layout)
            self.layout_names.append(layout_name)
            return prev_res
        found, prev_res = self.find_layout_results(mem_layout)
        # if layout file was found but it was not run
        if found:
            self.num_generated_layouts -= 1
            self.last_layout_num -= 1
            return prev_res

        self.write_layout(layout_name, mem_layout)
        out_dir = f'{self.exp_root_dir}/{layout_name}'
        run_cmd = f'{self.run_experiment_cmd} {layout_name}'

        logging.info('===========================================')
        logging.info(f'*** start running {layout_name} for {out_dir} ***')
        logging.info(f'\texperiment: {out_dir}')
        logging.info(f'\tlayout: {layout_name}')
        logging.info(f'\t#hugepages: {len(mem_layout)}')
        logging.debug(f'\tscript: {run_cmd}')

        found_outliers = True
        while found_outliers:
            ret_code = Selector.run_command(run_cmd, out_dir)
            if ret_code != 0:
                raise RuntimeError(f'Error: running {layout_name} failed with error code: {ret_code}')
            self.results_df, found_outliers = self.collect_results()
            if self.skip_outliers:
                break

        layout_res = self.get_layout_results(layout_name)
        tlb_misses = layout_res['stlb_misses']
        tlb_hits = layout_res['stlb_hits']
        walk_cycles = layout_res['walk_cycles']
        runtime = layout_res['cpu_cycles']
        
        if 'hugepages' not in layout_res:
            layout_res['hugepages'] = mem_layout

        logging.info('-------------------------------------------')
        logging.info(f'Results:')
        logging.info(f'\tstlb-misses={tlb_misses/1e9:.2f} Billions')
        logging.info(f'\tstlb-hits={tlb_hits/1e9:.2f} Billions')
        logging.info(f'\twalk-cycles={walk_cycles/1e9:.2f} Billion cycles')
        logging.info(f'\truntime={runtime/1e9:.2f} Billion cycles')
        logging.info('===========================================')
        return layout_res

    def run_next_layout(self, mem_layout):
        self.num_generated_layouts += 1
        self.last_layout_num += 1
        layout_name = f'layout{self.last_layout_num}'
        logging.info(f'run workload under {layout_name} with {len(mem_layout)} hugepages')
        last_result = self.run_workload(mem_layout, layout_name)
        return last_result
    
    def run_layouts(self, layouts):
        results_df = pd.DataFrame()
        for l in layouts:
            r = self.run_next_layout(l)
            results_df = results_df.append(r)
        return results_df
 