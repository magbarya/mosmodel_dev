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
                 metric_coverage,
                 range_epsilon=0.01) -> None:
        self.num_generated_layouts = 0
        self.metric_val = metric_val
        self.metric_coverage = metric_coverage
        self.range_epsilon = range_epsilon
        self.search_pebs_threshold = 0.5
        self.last_lo_layout = None
        self.last_hi_layout = None
        self.last_layout_result = None
        self.last_runtime_range = 0
        self.head_pages_coverage_threshold = 2
        self.num_initial_layouts = 0
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

    def get_surrounding_pair(self, res_df, layout_pair_idx=0, by='cpu_cycles', ascending=False, return_full_result=False):
        all_pairs_df = self.get_surrounding_layouts(res_df, by, ascending)

        if layout_pair_idx >= len(all_pairs_df):
            layout_pair_idx = 0
        selected_pair = all_pairs_df.iloc[layout_pair_idx]

        self.logger.debug(f'-------------------------------------------------------')
        self.logger.debug(f'get_surrounding_layouts: selected layouts [{selected_pair["layout_lo"]} , {selected_pair["layout_hi"]}]')
        self.logger.debug(f'\t selected layouts:')
        self.logger.debug(f'\n{selected_pair}')
        self.logger.debug(f'-------------------------------------------------------')

        lo_layout = selected_pair[f'hugepages_lo']
        hi_layout = selected_pair[f'hugepages_hi']
        if return_full_result:
            cols = selected_pair.keys()
            # remove duplicates
            cols = list(set(cols))
            lo_cols = [c for c in cols if c.endswith('_lo')]
            hi_cols = [c for c in cols if c.endswith('_hi')]
            # split the result to two serieses
            lo_layout = selected_pair[lo_cols]
            hi_layout = selected_pair[hi_cols]
            # remove suffixes
            lo_cols = [c.replace('_lo', '') for c in lo_cols]
            hi_cols = [c.replace('_hi', '') for c in hi_cols]
            # rename columns (by removing the _lo and _hi suffixes)
            lo_layout = lo_layout.set_axis(lo_cols)
            hi_layout = hi_layout.set_axis(hi_cols)

        return lo_layout, hi_layout

    def get_surrounding_layouts(self, res_df, by='cpu_cycles', ascending=False):
        df = res_df.sort_values(self.metric_name, ascending=True).reset_index(drop=True)
        lo_layouts_df = df.query(f'{self.metric_name} < {self.metric_val}')
        assert len(lo_layouts_df) > 0

        hi_layouts_df = df.query(f'{self.metric_name} >= {self.metric_val}')
        assert len(hi_layouts_df) > 0

        all_pairs_df = lo_layouts_df.merge(hi_layouts_df, how='cross', suffixes=['_lo', '_hi'])
        all_pairs_df[f'{by}_diff'] = abs(all_pairs_df[f'{by}_lo'] - all_pairs_df[f'{by}_hi'])
        all_pairs_df = all_pairs_df.sort_values(f'{by}_diff', ascending=ascending).reset_index(drop=True)

        return all_pairs_df

    def __try_select_layout_random_order(self, pebs_df, pebs_coverage,
                                         include_pages=[], epsilon=0.5, exclude_pages=None):
        return self.try_select_layout(pebs_df=pebs_df, pebs_coverage=pebs_coverage, include_pages=include_pages,
                                      epsilon=2, exclude_pages=exclude_pages, sample_pebs=True)

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
        self.logger.debug(f'try_select_layout_random: pebs_coverage={pebs_coverage}')
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
                self.logger.debug(f'try_select_layout_random: found layout with #{len(layout)} hugepages')
                return layout
            elif layout:
                self.logger.debug(f'try_select_layout_random: found existing layout with #{len(layout)} hugepages')
        return []

    def try_select_layout_dynamic_epsilon(self, pebs_df, pebs_coverage,
                                          include_pages=[], max_epsilon=2,
                                          exclude_pages=None, sort_ascending=False):
        include_pages_weight = self.pebsTlbCoverage(include_pages, pebs_df)
        if include_pages_weight > pebs_coverage:
            return []
        epsilon = 0.5
        while epsilon < max_epsilon:
            layout = self.try_select_layout(pebs_df, pebs_coverage, include_pages, epsilon, exclude_pages, sort_ascending)
            if layout and self.isPagesListUnique(layout, self.layouts):
                return layout
            epsilon += 0.5
        return []

    def try_select_layout(self, pebs_df, pebs_coverage, include_pages=[], epsilon=0.5,
                          exclude_pages=None, sort_ascending=False, sample_pebs=False):
        include_pages_weight = self.pebsTlbCoverage(include_pages, pebs_df)
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

        if sample_pebs:
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
            self.logger.debug(f'<-- try_select_layout(): could not select layout for pebs_coverage={pebs_coverage} with epsilon={epsilon}')
            return []
        self.logger.debug(f'<-- try_select_layout(): found layout with {len(mem_layout)} hugepages')
        return mem_layout

    def combine_layouts(self, layout1, layout2, pebs_df=None):
        if pebs_df is None:
            pebs_df = self.pebs_df
        layout1_set = set(layout1)
        layout2_set = set(layout2)
        only_in_layout1 = list(layout1_set - layout2_set)
        only_in_layout2 = list(layout2_set - layout1_set)
        in_both = list(layout1_set & layout2_set)

        mem_layout = in_both

        # first chance: try to select half their combined weight
        search_space = only_in_layout1 + only_in_layout2
        weight = self.pebsTlbCoverage(search_space, pebs_df)
        expected_coverage = weight / 2
        ss_pebs_df = pebs_df.query(f'PAGE_NUMBER in {search_space}')
        subset = self.try_select_layout_dynamic_epsilon(ss_pebs_df, expected_coverage)
        if subset:
            mem_layout += subset
            return list(set(mem_layout))

        # second chance: try to combine layouts randomally
        rand_layout = self.combine_layouts_semi_random(layout1, layout2, pebs_df)
        if rand_layout and self.isPagesListUnique(rand_layout, self.layouts):
            return rand_layout

        # third chance: try to combine layouts blindly
        mem_layout += only_in_layout1[0::2]
        mem_layout += only_in_layout2[0::2]

        return list(set(mem_layout))

    def combine_layouts_semi_random(self, layout1, layout2, pebs_df=None):
        if pebs_df is None:
            pebs_df = self.pebs_df
        layout1_set = set(layout1)
        layout2_set = set(layout2)
        only_in_layout1 = list(layout1_set - layout2_set)
        only_in_layout2 = list(layout2_set - layout1_set)
        in_both = list(layout1_set & layout2_set)

        mem_layout = in_both

        # define the search space to add left pages from
        search_space = only_in_layout1 + only_in_layout2
        pebs_df = pebs_df.query(f'PAGE_NUMBER in {search_space}')
        # Determine the maximum number of rows to select
        max_rows_to_select = len(pebs_df)  # Maximum number of rows available

        # try to select half their combined weight
        weight = self.pebsTlbCoverage(search_space, pebs_df)
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
        metric_low_val = self.metric_val * (1 - self.range_epsilon)
        metric_hi_val = self.metric_val * (1 + self.range_epsilon)
        range_df = self.results_df.query(f'{metric_hi_val} >= {self.metric_name} >= {metric_low_val}')

        if len(range_df) < 2:
            return 0

        max_runtime = range_df['cpu_cycles'].max()
        min_runtime = range_df['cpu_cycles'].min()
        range_percentage = (max_runtime - min_runtime) / min_runtime
        range_percentage = round(range_percentage*100, 2)

        return range_percentage

    def is_result_within_target_range_v2(self, layout_res):
        if layout_res is None:
            return False
        diff = abs(layout_res[self.metric_name] - self.metric_val)
        diff_ratio = diff / self.metric_val
        return diff_ratio < 0.01

    def is_result_within_target_range(self, layout_res):
        if layout_res is None:
            return False
        min_val = self.metric_val * (1 - self.range_epsilon)
        max_val = self.metric_val * (1 + self.range_epsilon)
        val = layout_res[self.metric_name]
        return min_val <= val <= max_val

    def get_layounts_within_target_range(self):
        min_val = self.metric_val * (1 - self.range_epsilon)
        max_val = self.metric_val * (1 + self.range_epsilon)
        res = self.results_df.query(f'{min_val} <= {self.metric_name} <= {max_val}')
        return res

    def get_highest_runtime_layout_in_range(self):
        range_layouts = self.get_layounts_within_target_range()
        hi_runtime = range_layouts['cpu_cycles'].max()
        hi_layout = range_layouts[range_layouts['cpu_cycles'] == hi_runtime].iloc[0]
        return hi_layout['hugepages']

    def get_lowest_runtime_layout_in_range(self):
        range_layouts = self.get_layounts_within_target_range()
        lo_runtime = range_layouts['cpu_cycles'].min()
        lo_layout = range_layouts[range_layouts['cpu_cycles'] == lo_runtime].iloc[0]
        return lo_layout['hugepages']

    def select_initial_layouts_weighted_random(self, tmp_layouts=[], max_num_layouts=10):
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

    def select_initial_layouts_minimal_tail_pages(self, tmp_layouts=[], max_num_layouts=5, pebs_df=None):
        if pebs_df is None:
            pebs_df = self.orig_pebs_df
        self.logger.info(f'--> select_initial_layouts() entry')

        # head_pebs_df = pebs_df.query(f'TLB_COVERAGE >= {self.head_pages_coverage_threshold}')
        # head_rows = head_pebs_df.sort_values('TLB_COVERAGE', ascending=False).head(5)
        head_rows = pebs_df.sort_values('TLB_COVERAGE', ascending=False).head(5)
        head_pages = head_rows['PAGE_NUMBER'].to_list()
        # create eight layouts as all subgroups of these three group layouts
        all_subsets = []
        for subset_size in range(len(head_pages)+1):
            for subset in itertools.combinations(head_pages, subset_size):
                layout = list(subset)
                layout_pebs = self.pebsTlbCoverage(layout, pebs_df)
                layout.sort()
                all_subsets.append({'hugepages': layout, 'pebs_coverage': layout_pebs})
        all_subsets_df = pd.DataFrame.from_records(all_subsets)
        # all_subsets_df['diff'] = self.metric_coverage - all_subsets_df['pebs_coverage']
        # all_subsets_df = all_subsets_df.query('diff >= 0').sort_values('diff', ascending=True)
        all_subsets_df['diff'] = abs(self.metric_coverage - all_subsets_df['pebs_coverage'])
        all_subsets_df = all_subsets_df.sort_values('diff', ascending=True)
        all_subsets_df = all_subsets_df.head(max_num_layouts)

        mem_layouts = []
        base_layouts = all_subsets_df['hugepages'].to_list()

        for include_pages in base_layouts:
            exclude_pages = list(set(head_pages) - set(include_pages))
            layout = self.try_select_layout_random(pebs_df,
                                                   self.metric_coverage,
                                                   include_pages=include_pages,
                                                   exclude_pages=exclude_pages,
                                                   tmp_layouts=tmp_layouts)
            if layout and self.isPagesListUnique(layout, (self.layouts+tmp_layouts+mem_layouts)):
                mem_layouts.append(layout)

        self.logger.info(f'<-- select_initial_layouts() exit: selected #{len(mem_layouts)} layouts')
        return mem_layouts

    def select_initial_layouts(self):
        mem_layouts = []
        debug_info = []
        mem_layouts_len = 0
        prev_len = 0

        # prev_len = len(mem_layouts)
        # mem_layouts += self.select_initial_layouts_weighted_random(mem_layouts, 5)
        # mem_layouts_len = len(mem_layouts) - mem_layouts_len
        # mem_layouts_len = len(mem_layouts) - prev_len
        # debug_info.append({'method': 'weighted_random', 'num_layouts': mem_layouts_len})

        prev_len = len(mem_layouts)
        mem_layouts += self.select_initial_layouts_minimal_tail_pages(mem_layouts, 10, self.orig_pebs_df)
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

        self.num_initial_layouts = len(mem_layouts)
        return mem_layouts

    def combine_surrounding_layouts(self, results_df):
        surrounding_percentile = self.range_epsilon
        while surrounding_percentile < 1:
            for idx in range(self.num_layouts):
                lower_layout, upper_layout = self.get_surrounding_pair(results_df, surrounding_percentile, idx)
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

    def select_desired_layout(self, base_layout, base_layout_real_coverage, pebs_df=None):
        if pebs_df is None:
            pebs_df = self.pebs_df
        base_coverage =self.pebsTlbCoverage(base_layout, pebs_df)
        # use abs(diff) + base to cover rare cases where the base_coverage could be higher than the metric_coverage
        diff_real = self.metric_coverage - base_layout_real_coverage
        assert diff_real > 0 # base_layout was selected such that it's one of the nearest layouts with lower coverage
        layout_pebs_step = 4
        layout_pebs = min(100, base_coverage + layout_pebs_step)
        prev_layout_real = base_layout_real_coverage
        prev_layout_pebs = layout_pebs
        last_pebs_step = 0
        while True:
            self.logger.info(f'select_desired_layout: trying to select layout with pebs-coverage={layout_pebs}')
            layout = self.try_select_layout_dynamic_epsilon(pebs_df, layout_pebs,
                                                            include_pages=base_layout, max_epsilon=2,
                                                            exclude_pages=None, sort_ascending=False)
            self.logger.info('=======================================================')
            self.logger.info(f'==> select_desired_layout: selecting layout: #{self.last_layout_num+1} with #{len(layout)} hugepages')

            if not layout or not self.isPagesListUnique(layout, self.layouts):
                self.logger.info(f'<== select_desired_layout: could not find new layout')
                self.logger.info('=======================================================')
                return False

            # run the layout
            self.last_layout_result = self.run_next_layout(layout)

            # check if the goal is acheived, i.e., layout result falls in the range
            if self.is_result_within_target_range(self.last_layout_result):
                self.logger.info('=======================================================')
                self.logger.info(f'+++ layout{self.last_layout_num} result falls within desired range +++')
                self.logger.info('=======================================================')
                return True

            # adapt the layout_pebs according to last layout result
            layout_real = self.realMetricCoverage(self.last_layout_result)
            if layout_real > self.metric_coverage:
                layout_pebs_step = (last_pebs_step + layout_pebs_step) / 2
                layout_pebs = base_coverage + layout_pebs_step
                layout_pebs = max(0, layout_pebs)
                self.logger.info(f'*** real coverage={layout_real} > metric_coverage={self.metric_coverage} ==> pebs: {prev_layout_pebs}-->{layout_pebs}')
            else:
                last_pebs_step = layout_pebs_step
                layout_pebs_step *= 1.5
                layout_pebs = base_coverage + layout_pebs_step
                layout_pebs = min(layout_pebs, 100)
                self.logger.info(f'*** real coverage={layout_real} < metric_coverage={self.metric_coverage} ==> pebs: {prev_layout_pebs}-->{layout_pebs}')

            self.logger.info(f'prev real coverage: {prev_layout_real} ==> {layout_real} last layout real coverage')
            self.logger.info(f'prev pebs coverage: {prev_layout_pebs} ==> {layout_pebs} next pebs coverage')
            prev_layout_real = layout_real
            prev_layout_pebs = layout_pebs

    def select_desired_layout_v2(self):
        self.logger.info('=======================================================')
        self.logger.info(f'==> select_desired_layout: selecting layout: #{self.last_layout_num+1}')

        layout = self.select_next_layout()

        self.logger.info(f'<== select_desired_layout: finished selecting layout: #{self.last_layout_num+1}')
        self.logger.info('=======================================================')

        self.last_layout_result = self.run_next_layout(layout)
        self.last_runtime_range = self.calculate_runtime_range()
        if self.is_result_within_target_range(self.last_layout_result):
            self.logger.info('=======================================================')
            self.logger.info(f'+++ layout{self.last_layout_num} result falls within desired range +++')
            self.logger.info('=======================================================')
            return True
        return False

    def combine_layout_and_test_in_range(self, layout1, layou2, pebs_df=None):
        if pebs_df is None:
            pebs_df = self.pebs_df
        mem_layout = self.combine_layouts(layout1, layou2, pebs_df)
        if not mem_layout or not self.isPagesListUnique(mem_layout, self.layouts):
            return False
        # run the layout
        self.last_layout_result = self.run_next_layout(mem_layout)

        # check if the goal is acheived, i.e., layout result falls in the range
        if self.is_result_within_target_range(self.last_layout_result):
            self.logger.info('=======================================================')
            self.logger.info(f'+++ layout{self.last_layout_num} result falls within desired range +++')
            self.logger.info('=======================================================')
            return True
        return False

    def find_desired_layout(self):
        pebs_df = self.pebs_df
        # pebs_df = self.orig_pebs_df
        while True:
            last_hi_layout_name = None
            all_pairs_df = self.get_surrounding_layouts(res_df=self.results_df, by=self.metric_name, ascending=True)
            # try to select layout that yields a data point at the desired metric_val
            for idx in range(len(all_pairs_df)):
                pair = all_pairs_df.iloc[idx]
                hi_layout_name = pair['layout_hi']
                if hi_layout_name == last_hi_layout_name:
                    continue
                last_hi_layout_name = hi_layout_name
                # 1st selection method by expanding layouts with lower values
                hi_layout = pair['hugepages_hi']
                hi_real_coverage = self.realCoverage(pair[f'{self.metric_name}_hi'], self.metric_name)
                if self.select_desired_layout(hi_layout, hi_real_coverage, pebs_df):
                    break
            all_pairs_df = self.get_surrounding_layouts(res_df=self.results_df, by=self.metric_name, ascending=True)
            # try to select layout that yields a data point at the desired metric_val
            for idx in range(len(all_pairs_df)):
                pair = all_pairs_df.iloc[idx]
                lo_layout = pair['hugepages_lo']
                hi_layout = pair['hugepages_hi']
                # 2nd selection method by merging surrounding layouts of the desired metric_val
                self.combine_layout_and_test_in_range(lo_layout, hi_layout, pebs_df)
                if self.is_result_within_target_range(self.last_layout_result) and self.num_generated_layouts >= (self.num_layouts // 2):
                    return True
            if self.is_result_within_target_range(self.last_layout_result):
                return True

    def get_tail_pages(self, threshold=0.01, total_threshold=2):
        tail_pages_df = self.pebs_df.query(f'TLB_COVERAGE < {threshold}')
        tail_pages_df = tail_pages_df.sort_values('TLB_COVERAGE', ascending=True)
        tail_pages_df['tail_cumsum'] = tail_pages_df['TLB_COVERAGE'].cumsum()
        tail_pages_df = tail_pages_df.query(f'tail_cumsum <= {total_threshold}')
        tail_pages = tail_pages_df['PAGE_NUMBER'].to_list()
        return tail_pages

    def get_tail_pages_groups(self, threshold=0.01, total_threshold=2):
        tail_pages = self.get_tail_pages(threshold, total_threshold)
        num_tail_pages = len(tail_pages)
        assert num_tail_pages > self.num_layouts
        num_groups = self.num_layouts
        group_size = num_tail_pages // num_groups
        groups = [tail_pages[i : i+group_size] for i in range(0, num_tail_pages, group_size)]

        return groups

    def add_tails_pages_func(self, base_pages, tested_tail_pages, subset):
        layout = list(set(base_pages + tested_tail_pages + subset))
        return layout

    def remove_tails_pages_func(self, base_pages, tested_tail_pages, subset):
        layout = list(set(base_pages) - set(tested_tail_pages) - set(subset))
        return layout

    def binary_search_tail_pages_selector(self, base_pages, tail_pages, create_layout_func):
        def evaluate_subset(subset, tested_tail_pages):
            layout = create_layout_func(base_pages, tested_tail_pages, subset)
            if layout and self.isPagesListUnique(layout, self.layouts):
                self.last_layout_result = self.run_next_layout(layout)
                return self.is_result_within_target_range(self.last_layout_result)
            return True

        result = []
        def search(left, right):
            if left >= right:
                return

            mid = (left + right) // 2
            left_subset = tail_pages[left:mid]
            right_subset = tail_pages[mid:right]

            left_in_range = evaluate_subset(left_subset, result)
            if left_in_range:
                # If the left subset is under the threshold, add it to the result
                result.extend(left_subset)
            else:
                search(left, mid)

            right_in_range = evaluate_subset(right_subset, result)
            if right_in_range:
                # If the left subset is under the threshold, add it to the result
                result.extend(right_subset)
            else:
                search(mid, right)

            if left_in_range and right_in_range:
                return

        # Start the search with the entire list
        search(0, len(tail_pages))
        layout = create_layout_func(base_pages, result, [])
        return result, layout

    def generate_layouts(self):
        tail_pages = self.get_tail_pages()
        range_layouts_df = self.get_layounts_within_target_range()
        for index, row in range_layouts_df.iterrows():
            layout = row['hugepages']
            self.last_layout_result = self.run_next_layout(layout)
            res, base_layout = self.binary_search_tail_pages_selector(layout, tail_pages, self.remove_tails_pages_func)
            res, base_layout = self.binary_search_tail_pages_selector(layout, tail_pages, self.add_tails_pages_func)

    def generate_layouts_v2(self):
        self.num_generated_layouts = 0

        tail_pages = []
        skipped_layouts = []
        groups = self.get_tail_pages_groups()
        for g in groups:
            hi_layout = self.get_highest_runtime_layout_in_range()
            layout = list(set(hi_layout + tail_pages + g))
            if layout and self.isPagesListUnique(layout, self.layouts):
                self.last_layout_result = self.run_next_layout(layout)
                if self.is_result_within_target_range(self.last_layout_result):
                    tail_pages += g
                    continue
            skipped_layouts.append(g)

        while self.num_generated_layouts < self.num_layouts:
            if len(skipped_layouts) == 0:
                break
            skipped_pages = skipped_layouts.pop()
            if len(skipped_pages) <= 1:
                continue
            g1 = skipped_pages[0::2]
            g2 = skipped_pages[1::2]
            for g in [g1,g2]:
                hi_layout = self.get_highest_runtime_layout_in_range()
                layout = list(set(hi_layout + tail_pages + g))
                if layout and self.isPagesListUnique(layout, self.layouts):
                    self.last_layout_result = self.run_next_layout(layout)
                    if self.is_result_within_target_range(self.last_layout_result):
                        tail_pages += g
                        continue
                skipped_layouts.append(g)

    def run(self):
        self.log_metadata()

        self.generate_initial_layouts()
        self.find_desired_layout()
        self.generate_layouts()
        self.generate_layouts_v2()

        self.logger.info('=================================================================')
        self.logger.info(f'Finished running MosRange process for:\n{self.exp_root_dir}')
        self.logger.info('=================================================================')
        # MosrangeSelector.pause()
