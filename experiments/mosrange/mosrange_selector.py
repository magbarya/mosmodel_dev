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
from Utils.selector_utils import Selector

class MosrangeSelector(Selector):
    def __init__(self, memory_footprint_file, 
                 pebs_mem_bins_file,
                 exp_root_dir, 
                 results_dir,
                 run_experiment_cmd,
                 num_layouts, 
                 num_repeats,
                 metric_name,
                 metric_val, 
                 metric_coverage) -> None:
        self.num_generated_layouts = 0
        self.metric_val = metric_val
        self.metric_coverage = metric_coverage
        self.search_pebs_threshold = 0.5
        self.last_lo_layout = None
        self.last_hi_layout = None
        self.last_layout_result = None
        self.last_runtime_range = 0
        self.head_pages_coverage_threshold = 2
        super().__init__(memory_footprint_file, 
                         pebs_mem_bins_file,
                         exp_root_dir, 
                         results_dir,
                         run_experiment_cmd,
                         num_layouts, 
                         num_repeats,
                         metric_name,
                         rebuild_pebs=True,
                         skip_outliers=False,
                         generate_endpoints=True)
        # Set the seed for reproducibility (optional)
        random.seed(42)
        self.logger = logging.getLogger(__name__)
        self.update_metric_values()

    def update_metric_values(self):
        if self.metric_val is None:
            self.metric_val = self.metric_max_val - self.metric_range_delta * (self.metric_coverage / 100)
        else:
            self.metric_coverage = ((self.metric_max_val - self.metric_val) / self.metric_range_delta) * 100
 
    def get_surrounding_layouts(self, res_df, surrounding_percentile=0.01, layout_pair_idx=0):
        df = res_df.sort_values(self.metric_name, ascending=True).reset_index(drop=True)
        delta = surrounding_percentile * self.metric_val
        
        lo_layouts_df = df.query(f'{self.metric_name} < {self.metric_val}')
        assert len(lo_layouts_df) > 0
        lo_layouts_max = lo_layouts_df[self.metric_name].max()
        lo_layouts_min = lo_layouts_max - delta
        
        hi_layouts_df = df.query(f'{self.metric_name} >= {self.metric_val}')
        assert len(hi_layouts_df) > 0
        hi_layouts_min = hi_layouts_df[self.metric_name].min()
        hi_layouts_max = hi_layouts_min + delta
        
        lo_layouts_df = lo_layouts_df.query(f'{self.metric_name} >= {lo_layouts_min}')
        hi_layouts_df = hi_layouts_df.query(f'{self.metric_name} <= {hi_layouts_max}')
        assert len(lo_layouts_df) > 0
        assert len(hi_layouts_df) > 0
        all_pairs_df = lo_layouts_df.merge(hi_layouts_df, how='cross', suffixes=['_lo', '_hi'])
        all_pairs_df['runtime_diff'] = abs(all_pairs_df['cpu_cycles_lo'] - all_pairs_df['cpu_cycles_hi'])
        all_pairs_df = all_pairs_df.sort_values('runtime_diff', ascending=False).reset_index(drop=True)
        if layout_pair_idx >= len(all_pairs_df):
            layout_pair_idx = 0
        selected_pair = all_pairs_df.iloc[layout_pair_idx]
        
        lo_layout = selected_pair['hugepages_lo']
        hi_layout = selected_pair['hugepages_hi']
        
        return lo_layout, hi_layout

    def __try_select_layout_random_order(self, pebs_df, pebs_coverage, 
                                         include_pages=[], epsilon=0.5, exclude_pages=None):
        include_pages_weight = self.pebsTlbCoverage(include_pages)
        rem_weight = pebs_coverage - include_pages_weight
        if rem_weight < 0:
            return []
        
        self.logger.debug(f'--> __try_select_layout_random_order(): pebs_coverage={pebs_coverage} , #include_pages={len(include_pages)} , include_pages_coverage={include_pages_weight} , epsilon={epsilon}')
        
        if exclude_pages:
            pebs_df = pebs_df.query(f'PAGE_NUMBER not in {exclude_pages}')
        if include_pages:
            pebs_df = pebs_df.query(f'PAGE_NUMBER not in {include_pages}')
        
        pebs_df = pebs_df.sample(len(pebs_df))
        
        mem_layout = include_pages.copy()
        for index, row in pebs_df.iterrows():
            page = row['PAGE_NUMBER']
            weight = row['TLB_COVERAGE']
            if weight <= (rem_weight + epsilon):
                mem_layout.append(page)
                rem_weight -= weight
            if rem_weight <= epsilon:
                break
        # could not find subset of pages to add that leads to the required coverage
        if rem_weight > epsilon:
            rem_pages = self.try_select_layout(pebs_df, pebs_coverage, include_pages=mem_layout)
            if rem_pages:
                mem_layout += rem_pages
            else:
                self.logger.debug(f'<-- __try_select_layout_random_order: could not select layout for pebs_coverage={pebs_coverage} with epsilon={epsilon}')
                return []
        self.logger.debug(f'<-- __try_select_layout_random_order: found layout with {len(mem_layout)} hugepages')
        return mem_layout
     
    def __try_select_layout_random_size(self, pebs_df, 
                                      pebs_coverage,
                                      include_pages=[], 
                                      epsilon=0.5, 
                                      exclude_pages=None, 
                                      sort_ascending=False,
                                      num_samples=None):        
        if num_samples is None:
            num_samples = random.randint(1, len(pebs_df))
        random_pebs_df = pebs_df.sample(num_samples)
        # if there is no enough pages in pebs_DF to get the desired coverage
        if random_pebs_df['TLB_COVERAGE'].sum() < pebs_coverage:
            return []

        layout = self.try_select_layout(pebs_df=random_pebs_df, 
                                        pebs_coverage=pebs_coverage, 
                                        include_pages=include_pages, 
                                        epsilon=epsilon,
                                        exclude_pages=exclude_pages, 
                                        sort_ascending=sort_ascending)
        return layout

    def try_select_layout_random(self, pebs_df, pebs_coverage,
                                 include_pages=[], exclude_pages=None, 
                                 sort_ascending=False, tmp_layouts=[],
                                 randomization='order'):
        layout = []
        for i in range(self.num_layouts):
            if randomization == 'order':
                layout = self.__try_select_layout_random_order(pebs_df=pebs_df,
                                                               pebs_coverage=pebs_coverage, 
                                                               include_pages=include_pages, 
                                                               epsilon=2,
                                                               exclude_pages=exclude_pages)
            elif randomization == 'size':
                layout = self.__try_select_layout_random_size(pebs_df=pebs_df, 
                                                            pebs_coverage=pebs_coverage, 
                                                            include_pages=include_pages, 
                                                            epsilon=2, 
                                                            exclude_pages=exclude_pages,
                                                            sort_ascending=sort_ascending,
                                                            num_samples=len(pebs_df)-i)
            if layout and self.isPagesListUnique(layout, self.layouts + tmp_layouts):
                return layout
        return []

    def try_select_layout_dynamic_epsilon(self, pebs_df, pebs_coverage, 
                                          include_pages=[], max_epsilon=2, 
                                          exclude_pages=None, sort_ascending=False):
        include_pages_weight = self.pebsTlbCoverage(include_pages)
        if include_pages_weight > pebs_coverage:
            return []
        epsilon = 0.5
        while epsilon < max_epsilon:
            layout = self.try_select_layout(pebs_df, pebs_coverage, include_pages, epsilon, exclude_pages, sort_ascending)
            if layout and self.isPagesListUnique(layout, self.layouts):
                return layout
            epsilon += 0.5
        return []
    
    def try_select_layout(self, pebs_df, pebs_coverage, include_pages=[], epsilon=0.5, exclude_pages=None, sort_ascending=False):
        include_pages_weight = self.pebsTlbCoverage(include_pages)
        self.logger.debug(f'** try_select_layout(): pebs_coverage={pebs_coverage} , #include_pages={len(include_pages)} , include_pages_coverage={include_pages_weight} , epsilon={epsilon}')
        
        rem_weight = pebs_coverage - include_pages_weight
        if rem_weight < 0:
            self.logger.debug(f'<-- try_select_layout(): the include_pages set has a weight higher than expected - [{include_pages_weight} > {pebs_coverage}]')
            return []
        
        self.logger.debug(f'--> try_select_layout(): pebs_coverage={pebs_coverage} , #include_pages={len(include_pages)} , include_pages_coverage={include_pages_weight} , epsilon={epsilon}')
        
        if exclude_pages:
            pebs_df = pebs_df.query(f'PAGE_NUMBER not in {exclude_pages}')
        if include_pages:
            pebs_df = pebs_df.query(f'PAGE_NUMBER not in {include_pages}')
        pebs_df = pebs_df.sort_values('TLB_COVERAGE', ascending=sort_ascending)
        
        mem_layout = include_pages.copy()
        for index, row in pebs_df.iterrows():
            page = row['PAGE_NUMBER']
            weight = row['TLB_COVERAGE']
            if weight <= (rem_weight + epsilon):
                mem_layout.append(page)
                rem_weight -= weight
            if rem_weight <= epsilon:
                break
        # could not find subset of pages to add that leads to the required coverage
        if rem_weight > epsilon:
            self.logger.debug(f'<-- try_select_layout(): could not select layout for pebs_coverage={pebs_coverage} with epsilon={epsilon}')
            return []
        self.logger.debug(f'<-- try_select_layout(): found layout with {len(mem_layout)} hugepages')
        return mem_layout

    def combine_layouts(self, layout1, layout2):
        layout1_set = set(layout1)
        layout2_set = set(layout2)
        only_in_layout1 = list(layout1_set - layout2_set)
        only_in_layout2 = list(layout2_set - layout1_set)
        in_both = list(layout1_set & layout2_set)
        
        mem_layout = in_both

        # first chance: try to select half their combined weight
        search_space = only_in_layout1 + only_in_layout2
        weight = self.pebsTlbCoverage(search_space)
        expected_coverage = weight / 2
        pebs_df = self.pebs_df.query(f'PAGE_NUMBER in {search_space}')
        subset = self.try_select_layout_dynamic_epsilon(pebs_df, expected_coverage)
        if subset:
            mem_layout += subset
            return list(set(mem_layout))

        # second chance: try to combine layouts randomally        
        rand_layout = self.combine_layouts_semi_random(layout1, layout2)
        if rand_layout and self.isPagesListUnique(rand_layout, self.layouts):
            return rand_layout
        
        # third chance: try to combine layouts blindly
        only_in_layout1_df = self.pebs_df.query(f'PAGE_NUMBER in {only_in_layout1}').sort_values('TLB_COVERAGE')
        only_in_layout1 = only_in_layout1_df['PAGE_NUMBER'].to_list()
        only_in_layout2_df = self.pebs_df.query(f'PAGE_NUMBER in {only_in_layout2}').sort_values('TLB_COVERAGE')
        only_in_layout2 = only_in_layout2_df['PAGE_NUMBER'].to_list()
        mem_layout += only_in_layout1[0::2]
        mem_layout += only_in_layout2[0::2]

        return list(set(mem_layout))
    
    def combine_layouts_semi_random(self, layout1, layout2):
        layout1_set = set(layout1)
        layout2_set = set(layout2)
        only_in_layout1 = list(layout1_set - layout2_set)
        only_in_layout2 = list(layout2_set - layout1_set)
        in_both = list(layout1_set & layout2_set)
        
        mem_layout = in_both
        
        # define the search space to add left pages from
        search_space = only_in_layout1 + only_in_layout2
        pebs_df = self.pebs_df.query(f'PAGE_NUMBER in {search_space}')
        # Determine the maximum number of rows to select
        max_rows_to_select = len(pebs_df)  # Maximum number of rows available
        
        # try to select half their combined weight
        weight = self.pebsTlbCoverage(search_space)
        expected_coverage = weight / 2
        for i in range(self.num_layouts):
            # Select a random number of rows (between 1 and max_rows_to_select)
            n = random.randint(1, max_rows_to_select)
            # Select n random rows from the DataFrame
            random_pebs_df = pebs_df.sample(n)
            subset = self.try_select_layout(random_pebs_df, expected_coverage, epsilon=1)
            if subset:
                mem_layout += subset
                return list(set(mem_layout))
        
        return []

    def calculate_runtime_range(self):
        epsilon = 0.005
        metric_low_val = self.metric_val * (1 - epsilon)
        metric_hi_val = self.metric_val * (1 + epsilon)
        range_df = self.results_df.query(f'{metric_hi_val} >= {self.metric_name} >= {metric_low_val}')
        
        if len(range_df) < 2:
            return 0

        max_runtime = range_df['cpu_cycles'].max()
        min_runtime = range_df['cpu_cycles'].min()
        range_percentage = (max_runtime - min_runtime) / min_runtime
        range_percentage = round(range_percentage*100, 2)
        
        return range_percentage

    def is_result_within_target_range(self, layout_res):
        if layout_res is None:
            return False
        diff = abs(layout_res[self.metric_name] - self.metric_val)
        diff_ratio = diff / self.metric_val
        return diff_ratio < 0.01
    
    
    def select_initial_layouts_random(self, tmp_layouts=[], max_num_layouts=10):
        self.logger.info(f'--> select_initial_layouts() entry')
        
        mem_layouts = []
        for i in range(max_num_layouts):
            layout = self.try_select_layout_random(self.pebs_df, self.metric_coverage, 
                                                   tmp_layouts=tmp_layouts+mem_layouts, 
                                                   randomization='size')
            if layout and self.isPagesListUnique(layout, (self.layouts+tmp_layouts+mem_layouts)):
                mem_layouts.append(layout)
                if len(mem_layouts) >= max_num_layouts:
                    break
        
        self.logger.info(f'<-- select_initial_layouts() exit: selected #{len(mem_layouts)} layouts')
        return mem_layouts
    
    def select_initial_layouts_odd_even_pagenumbers(self, tmp_layouts=[], max_num_layouts=10):
        self.logger.info(f'--> select_initial_layouts() entry')
        
        mem_layouts = []
        even_pebds_df = self.pebs_df.query(f'PAGE_NUMBER % {2} == 0')
        odd_pebds_df = self.pebs_df.query(f'PAGE_NUMBER % {2} == 1')
        for i in range(max_num_layouts):
            if i%2:
                layout = self.try_select_layout_random(even_pebds_df, self.metric_coverage, tmp_layouts=tmp_layouts+mem_layouts)
            else:
                layout = self.try_select_layout_random(odd_pebds_df, self.metric_coverage, tmp_layouts=tmp_layouts+mem_layouts)
            if layout and self.isPagesListUnique(layout, (self.layouts+tmp_layouts+mem_layouts)):
                mem_layouts.append(layout)
        
        self.logger.info(f'<-- select_initial_layouts() exit: selected #{len(mem_layouts)} layouts')
        return mem_layouts
    
    def select_initial_layouts_pagenumbers_multiplies(self, tmp_layouts=[], max_num_layouts=10):
        self.logger.info(f'--> select_initial_layouts() entry')
        
        mem_layouts = []
        for i in range(3, 17):
            pebs_df = self.pebs_df.query(f'PAGE_NUMBER % {i} == 0')
            if pebs_df.empty:
                continue
            layout = self.try_select_layout_random(pebs_df, self.metric_coverage, tmp_layouts=tmp_layouts+mem_layouts)
            if layout and self.isPagesListUnique(layout, (self.layouts+tmp_layouts+mem_layouts)):
                mem_layouts.append(layout)
                if len(mem_layouts) >= max_num_layouts:
                    break
        
        self.logger.info(f'<-- select_initial_layouts() exit: selected #{len(mem_layouts)} layouts')
        return mem_layouts
    
    def select_initial_layouts_minimal_tail_pages(self, tmp_layouts=[], max_num_layouts=10):
        self.logger.info(f'--> select_initial_layouts() entry')
        
        pebs_df = self.orig_pebs_df.query(f'TLB_COVERAGE >= {self.head_pages_coverage_threshold}')
        head_rows = pebs_df.sort_values('TLB_COVERAGE', ascending=False).head(10)
        head_pages = head_rows['PAGE_NUMBER'].to_list()
        # create eight layouts as all subgroups of these three group layouts
        all_subsets = []
        for subset_size in range(len(head_pages)+1):
            for subset in itertools.combinations(head_pages, subset_size):
                layout = list(subset)
                layout_pebs = self.pebsTlbCoverage(layout)
                layout.sort()
                all_subsets.append({'hugepages': layout, 'pebs_coverage': layout_pebs})
        all_subsets_df = pd.DataFrame.from_records(all_subsets)
        all_subsets_df['diff'] = self.metric_coverage - all_subsets_df['pebs_coverage']
        all_subsets_df = all_subsets_df.query('diff >= 0').sort_values('diff', ascending=True)
        all_subsets_df = all_subsets_df.head(max_num_layouts)
        
        mem_layouts = []
        base_layouts = all_subsets_df['hugepages'].to_list()
        
        for include_pages in base_layouts:
            exclude_pages = list(set(head_pages) - set(include_pages))
            layout = self.try_select_layout_random(self.pebs_df, 
                                                   self.metric_coverage, 
                                                   include_pages=include_pages, 
                                                   exclude_pages=exclude_pages, 
                                                   tmp_layouts=tmp_layouts)
            if layout and self.isPagesListUnique(layout, (self.layouts+tmp_layouts+mem_layouts)):
                mem_layouts.append(layout)
        
        self.logger.info(f'<-- select_initial_layouts() exit: selected #{len(mem_layouts)} layouts')
        return mem_layouts

    def select_initial_layouts_headpages_subgroups(self, tmp_layouts=[], max_num_layouts=10, subset_size_ascending=False):
        self.logger.info(f'--> select_initial_layouts() entry')
        
        pebs_df = self.orig_pebs_df.query(f'TLB_COVERAGE >= {self.head_pages_coverage_threshold}')
        head_rows = pebs_df.sort_values('TLB_COVERAGE', ascending=False).head(10)
        head_pages = head_rows['PAGE_NUMBER'].to_list()
        mem_layouts = []
        # create eight layouts as all subgroups of these three group layouts
        if subset_size_ascending:
            subsets_size_range = range(len(head_pages)+1)
        else:
            subsets_size_range = range(len(head_pages), 0, -1)
        for subset_size in subsets_size_range:
            for subset in itertools.combinations(head_pages, subset_size):
                include_pages = list(subset)
                exclude_pages = list(set(head_pages) - set(include_pages))
                self.logger.debug(f'** try to find initial layout that contains pages: {include_pages} **')
                layout = self.try_select_layout_dynamic_epsilon(self.pebs_df, self.metric_coverage, 
                                                                include_pages, max_epsilon=5, exclude_pages=exclude_pages)
                if layout and self.isPagesListUnique(layout, (self.layouts+tmp_layouts+mem_layouts)):
                    mem_layouts.append(layout)
                    if len(mem_layouts) >= max_num_layouts:
                        break
                    # # for each subset size will select only one layout (which is enough)
                    # break
            if len(mem_layouts) >= max_num_layouts:
                break
        
        self.logger.info(f'<-- select_initial_layouts() exit: selected #{len(mem_layouts)} layouts')
        return mem_layouts
    
    def select_initial_layouts_headpages_candidates(self, tmp_layouts=[], max_num_layouts=10):
        self.logger.info(f'--> select_initial_layouts() entry')
        
        pebs_df = self.orig_pebs_df.query(f'TLB_COVERAGE >= {self.head_pages_coverage_threshold}')
        head_rows = pebs_df.sort_values('TLB_COVERAGE', ascending=False).head(10)
        head_pages = head_rows['PAGE_NUMBER'].to_list()
        mem_layouts = []
        # create eight layouts as all subgroups of these three group layouts
        for subset_size in range(len(head_pages)+1):
            for subset in itertools.combinations(head_pages, subset_size):
                include_pages = list(subset)
                exclude_pages = list(set(head_pages) - set(include_pages))
                self.logger.debug(f'** try to find initial layout that contains pages: {include_pages} **')
                layout = self.try_select_layout_dynamic_epsilon(self.pebs_df, self.metric_coverage, 
                                                                include_pages, max_epsilon=5, exclude_pages=exclude_pages)
                if layout and self.isPagesListUnique(layout, (self.layouts+tmp_layouts+mem_layouts)):
                    mem_layouts.append(layout)
                    # for each subset size will select only one layout (which is enough)
                    break
        
        self.logger.info(f'<-- select_initial_layouts() exit: selected #{len(mem_layouts)} layouts')
        return mem_layouts
    
    def select_initial_layouts(self):
        mem_layouts = []
        debug_info = []
        mem_layouts_len = 0
        prev_len = 0
        
        prev_len = len(mem_layouts)
        mem_layouts += self.select_initial_layouts_odd_even_pagenumbers(mem_layouts)
        mem_layouts_len = len(mem_layouts) - mem_layouts_len
        mem_layouts_len = len(mem_layouts) - prev_len
        debug_info.append({'method': 'odd_even_pagenumbers', 'num_layouts': mem_layouts_len})
        
        # prev_len = len(mem_layouts)
        # mem_layouts += self.select_initial_layouts_pagenumbers_multiplies(mem_layouts)
        # mem_layouts_len = len(mem_layouts) - mem_layouts_len
        # mem_layouts_len = len(mem_layouts) - prev_len
        # debug_info.append({'method': 'pagenumbers_multiplies', 'num_layouts': mem_layouts_len})
        
        prev_len = len(mem_layouts)
        mem_layouts += self.select_initial_layouts_random(mem_layouts)
        mem_layouts_len = len(mem_layouts) - mem_layouts_len
        mem_layouts_len = len(mem_layouts) - prev_len
        debug_info.append({'method': 'random', 'num_layouts': mem_layouts_len})
        
        prev_len = len(mem_layouts)
        mem_layouts += self.select_initial_layouts_headpages_candidates(mem_layouts)
        mem_layouts_len = len(mem_layouts) - mem_layouts_len
        mem_layouts_len = len(mem_layouts) - prev_len
        debug_info.append({'method': 'headpages_candidates', 'num_layouts': mem_layouts_len})
        
        # prev_len = len(mem_layouts)
        # mem_layouts += self.select_initial_layouts_headpages_subgroups(mem_layouts, subset_size_ascending=True)
        # mem_layouts_len = len(mem_layouts) - prev_len
        # debug_info.append({'method': 'headpages_subgroups_ascending', 'num_layouts': mem_layouts_len})
        
        # prev_len = len(mem_layouts)
        # mem_layouts += self.select_initial_layouts_headpages_subgroups(mem_layouts, subset_size_ascending=False)
        # mem_layouts_len = len(mem_layouts) - mem_layouts_len
        # mem_layouts_len = len(mem_layouts) - prev_len
        # debug_info.append({'method': 'headpages_subgroups_descending', 'num_layouts': mem_layouts_len})
        
        prev_len = len(mem_layouts)
        mem_layouts += self.select_initial_layouts_minimal_tail_pages(mem_layouts)
        mem_layouts_len = len(mem_layouts) - mem_layouts_len
        mem_layouts_len = len(mem_layouts) - prev_len
        debug_info.append({'method': 'minimal_tail_pages', 'num_layouts': mem_layouts_len})
        
        print('=======================================================')
        from_layout = 4
        for i in debug_info:
            method = i['method']
            num_layouts = i['num_layouts']
            to_layout = from_layout + num_layouts-1
            print(f'{method} method has #{num_layouts} layouts: layout{from_layout}--layout{to_layout}')
            from_layout += num_layouts
        print('=======================================================')
            
        
        return mem_layouts
    
    def combine_surrounding_layouts(self, results_df):
        surrounding_percentile = 0.01
        while surrounding_percentile < 1:
            for idx in range(self.num_layouts):
                lower_layout, upper_layout = self.get_surrounding_layouts(results_df, surrounding_percentile, idx)
                # if the same surrounding layouts selected, then try to find another pair 
                if self.last_lo_layout is not None \
                    and self.last_hi_layout is not None \
                    and set(self.last_lo_layout) == set(lower_layout) \
                    and set(self.last_hi_layout) == set(upper_layout):
                    continue
                else:
                    mem_layout = self.combine_layouts(lower_layout, upper_layout)
                    if mem_layout and self.isPagesListUnique(mem_layout, self.layouts):
                        self.last_lo_layout = lower_layout
                        self.last_hi_layout = upper_layout
                        return mem_layout
            surrounding_percentile += 0.01
        return []
            
    def select_next_layout(self):
        # if last result is within expected range, then use the same base layouts
        if self.is_result_within_target_range(self.last_layout_result):
            # use last surrounding layouts
            mem_layout = self.combine_layouts(self.last_lo_layout, self.last_hi_layout)
            if mem_layout and self.isPagesListUnique(mem_layout, self.layouts):
                return mem_layout
        
        # otherwise, last result is not within expected range, then select new base layouts
        mem_layout = self.combine_surrounding_layouts(self.results_df)
        if mem_layout and self.isPagesListUnique(mem_layout, self.layouts):
            return mem_layout
        
        assert False
        
    def pause():
        print('=============================')
        print('press any key to continue ...')
        print('=============================')
        input()
        
    def log_metadata(self):
        self.logger.info('=================================================================')
        self.logger.info(f'** Metadata: **')
        self.logger.info(f'\t metric_name: {self.metric_name}')
        self.logger.info(f'\t metric_coverage: {round(self.metric_coverage, 2)}%')
        self.logger.info(f'\t metric_val: {Utils.format_large_number(self.metric_val)}')
        self.logger.info(f'\t metric_min_val: {Utils.format_large_number(self.metric_min_val)}')
        self.logger.info(f'\t metric_max_val: {Utils.format_large_number(self.metric_max_val)}')
        self.logger.info(f'\t metric_range_delta: {Utils.format_large_number(self.metric_range_delta)}')
        self.logger.info(f'\t #pages_in_pebs: {len(self.pebs_pages)}')
        self.logger.info(f'\t #pages_not_in_pebs: {len(self.pages_not_in_pebs)}')
        self.logger.info(f'\t #layouts: {self.num_layouts}')
        self.logger.info(f'\t #repeats: {self.num_repeats}')
        self.logger.info('=================================================================')
    
    def custom_log_layout_result(self, layout_res, old_result=False):
        if old_result:
            return
        # if endpoints were not run already, then skip
        if not hasattr(self, 'all_2mb_r'):
            return
        self.logger.info(f'\texpected-coverage={Utils.format_large_number(self.metric_coverage)}')
        self.logger.info(f'\treal-coverage={Utils.format_large_number(self.realMetricCoverage(layout_res))}')
        self.logger.info(f'\texpected-{self.metric_name}={Utils.format_large_number(self.metric_val)}')
        self.logger.info(f'\treal-{self.metric_name}={Utils.format_large_number(layout_res[self.metric_name])}')
        self.logger.info(f'\tis_result_within_target_range={self.is_result_within_target_range(layout_res)}')
        prev_runtime_range = self.last_runtime_range
        curr_runtime_range = self.calculate_runtime_range()
        self.last_runtime_range = curr_runtime_range
        runtime_range_improvement = curr_runtime_range - prev_runtime_range
        self.logger.info(f'\tprev_runtime_range={prev_runtime_range}%')
        self.logger.info(f'\tcurr_runtime_range={curr_runtime_range}%')
        self.logger.info(f'\truntime_range_improvement={runtime_range_improvement}%')
        
    def generate_initial_layouts(self):
        # self.num_generated_layouts = 0
        # Define the initial data samples
        init_layouts = self.select_initial_layouts()
        
        self.logger.info('=======================================================')
        self.logger.info(f'==> start running #{len(init_layouts)} initial layouts')
        
        res_df = self.run_layouts(init_layouts)
        
        self.logger.info(f'<== completed running #{len(init_layouts)} initial layouts')
        self.logger.info('=======================================================')
        
        return res_df
    
    def generate_next_layout(self):
        # MosrangeSelector.pause()
        self.logger.info('=======================================================')
        self.logger.info(f'==> start selecting next layout: #{self.last_layout_num+1}')
        
        layout = self.select_next_layout()
        
        self.logger.info(f'<== finished selecting next layout: #{self.last_layout_num+1}')
        self.logger.info('=======================================================')
        self.logger.info(f'==> start running next layout: #{self.last_layout_num+1}')
        
        self.last_layout_result = self.run_next_layout(layout)
        self.last_runtime_range = self.calculate_runtime_range()
        
        self.logger.info(f'<== completed running next layout: #{self.last_layout_num}')
        self.logger.info('=======================================================')
    
    def run(self):
        self.log_metadata()
        
        self.generate_initial_layouts()
        
        self.num_generated_layouts = 0
        while self.num_generated_layouts < self.num_layouts:
            self.generate_next_layout()

        self.logger.info('=================================================================')
        self.logger.info(f'Finished running MosRange process for:\n{self.exp_root_dir}')
        self.logger.info('=================================================================')
      
        
    # def generate_layout_from_base(self, base_pages, search_space, coverage, sort=True):
    #     self.logger.debug(f'generate_layout_from_base(): len(base_pages)={len(base_pages)} , len(search_space)={len(search_space)} , coverage={coverage}')
    #     expected_coverage = coverage - self.pebsTlbCoverage(base_pages)
    #     pebs_df = self.pebs_df.query(f'PAGE_NUMBER in {search_space} and PAGE_NUMBER not in {base_pages}')
    #     if sort:
    #         pebs_df = pebs_df.sort_values('TLB_COVERAGE', ascending=False)
    #     self.logger.debug(f'generate_layout_from_base() after filtering pages: len(pebs_df)={len(pebs_df)}')
    #     layout = self.generate_layout_from_pebs(expected_coverage, pebs_df)
    #     if layout:
    #         return layout+base_pages
    #     else:
    #         return []

    # def add_hugepages_to_base(self, next_coverage, base_pages, other_pages, all_pages):
    #     other_even_pages = [p for p in other_pages if p%2==0]
    #     all_even_pages = [p for p in all_pages if p%2==0]
    #     search_space_options = [other_pages, all_pages, other_even_pages, all_even_pages]
    #     for s in search_space_options:
    #         layout = self.generate_layout_from_base(base_pages, s, next_coverage)
    #         if layout and self.isPagesListUnique(layout, self.layouts):
    #             return layout
    #     return []

    # def remove_hugepages_from_base(self, pebs_coverage, base_pages, pages_to_remove):
    #     mem_layout = []
    #     df = self.pebs_df.query(f'PAGE_NUMBER in {base_pages}')
    #     df = df.sort_values('TLB_COVERAGE', ascending=False)
    #     total_weight = df['TLB_COVERAGE'].sum()
    #     # if the coverage of the base_pages less than expected,
    #     # then we can not remove pages from it
    #     if total_weight < (pebs_coverage - self.search_pebs_threshold):
    #         return []
    #     for index, row in df.iterrows():
    #         page = row['PAGE_NUMBER']
    #         if page not in pages_to_remove:
    #             continue
    #         weight = row['TLB_COVERAGE']
    #         if (total_weight - weight) > (pebs_coverage - self.search_pebs_threshold):
    #             mem_layout.append(page)
    #             total_weight -= weight
    #         if total_weight <= pebs_coverage:
    #             break
    #     # could not find subset to remove that leads to the required coverage
    #     if total_weight > (pebs_coverage + self.search_pebs_threshold):
    #         return []
    #     if mem_layout and self.isPagesListUnique(mem_layout, self.layouts):
    #         return mem_layout
    #     return []

    # def constrained_layout_selection(self, pebs_coverage, include_pages, exclude_pages, sort_ascending=False):
    #     pebs_df = self.pebs_df.query(f'PAGE_NUMBER not in {exclude_pages}')
    #     pebs_df = pebs_df.sort_values('TLB_COVERAGE', ascending=sort_ascending)
    #     search_space = pebs_df['PAGE_NUMBER'].to_list()
    #     layout = self.generate_layout_from_base(include_pages, search_space, pebs_coverage)
    #     return layout

    # def scaleExpectedCoverage(self, layout_res, base_layout):
    #     pebs_coverage = self.pebsTlbCoverage(layout_res['hugepages'])
    #     real_coverage = self.realMetricCoverage(layout_res)
    #     base_pebs = self.pebsTlbCoverage(base_layout)
    #     base_res = self.find_layout_results(base_layout)
    #     base_real = self.realMetricCoverage(base_res)
    #     pebs_delta = pebs_coverage - base_pebs
    #     real_gap = real_coverage - base_real

    #     if real_gap <= 0:
    #         desired_coverage = min(100, base_pebs + pebs_delta * 2)
    #         base_layout = layout_res['hugepages']
    #         return desired_coverage, base_layout
    #     return None, None
    
    # def layoutComposedOfHeadPages(self, layout):
    #     layout_coverage = self.pebsTlbCoverage(layout)
    #     layout_pages = self.pebs_df.query(f'PAGE_NUMBER in {layout}')
    #     layout_pages = layout_pages.sort_values('TLB_COVERAGE', ascending=False).head(10)
    #     headpages_coverage = layout_pages['TLB_COVERAGE'].sum()
    #     return headpages_coverage >= (layout_coverage/2)

    # def realToPebsCoverage(self, layout_res, layout_expected_real):
    #     layout = layout_res['hugepages']
    #     layout_pebs = self.pebsTlbCoverage(layout)
    #     layout_real = self.realMetricCoverage(layout_res)
        
    #     if self.layoutComposedOfHeadPages(layout):
    #         scaled_desired_coverage = layout_expected_real - layout_real + layout_pebs
    #         return scaled_desired_coverage

    #     # prevent division by zero and getting numerous ratio in
    #     # the calculation of expected_to_real
    #     layout_real = max(1, layout_real)
    #     expected_to_real = layout_expected_real / layout_real
    #     scaled_desired_coverage = layout_pebs * expected_to_real
    #     return scaled_desired_coverage
    
    # def select_next_layout_v2(self):
    #     lower_layout, upper_layout = self.get_surrounding_layouts(self.results_df)
        
    #     base_layout = upper_layout
    #     next_coverage = self.metric_coverage
    #     layout = None
    #     while not layout:
    #         A, B, C, U = self.split_pages_to_working_sets(upper_layout, lower_layout)
    #         layout = self.generate_layout_from_base(base_layout, C, next_coverage)
    #         if layout and self.isPagesListUnique(layout, self.layouts):
    #             layout_res = self.run_next_layout(layout)
    #         else:
    #             break
    #         next_coverage, base_layout = self.scaleExpectedCoverage(layout_res, base_layout)
    #         if base_layout is None:
    #             next_coverage = self.realToPebsCoverage(self, layout_res, self.metric_coverage)
    #             base_layout = upper_layout
    