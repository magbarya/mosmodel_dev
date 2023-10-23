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
                         run_endpoint_layouts=True)
        self.num_generated_layouts = 0
        self.metric_val = metric_val
        self.metric_coverage = metric_coverage
        self.search_pebs_threshold = 0.5
        logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
        self.update_metric_values()

    def update_metric_values(self):
        if self.metric_val is None:
            self.metric_val = self.metric_max_val - self.metric_range_delta * (self.metric_coverage / 100)
        else:
            self.metric_coverage = ((self.metric_max_val - self.metric_val) / self.metric_range_delta) * 100

    def select_initial_layouts(self):
        head_rows = self.pebs_df.sort_values('TLB_COVERAGE', ascending=False).head(10)
        head_pages = head_rows['PAGE_NUMBER'].to_list()
        mem_layouts = []
        # create eight layouts as all subgroups of these three group layouts
        for subset_size in range(len(head_pages)+1):
            for subset in itertools.combinations(head_pages, subset_size):
                include_pages = list(subset)
                layout = self.try_select_layout_dynamic_epsilon(self.pebs_df, self.metric_coverage, include_pages)
                if layout:
                    mem_layouts.append(layout)
        return mem_layouts

    def get_surrounding_layouts(self, res_df, surrounding_percentile=0.01):
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
        all_pairs_df = lo_layouts_df.merge(hi_layouts_df, how='cross', suffixes=['_lo', '_hi'])
        all_pairs_df['runtime_diff'] = abs(all_pairs_df['cpu_cycles_lo'] - all_pairs_df['cpu_cycles_hi'])
        max_pair = all_pairs_df[all_pairs_df['runtime_diff'] == all_pairs_df['runtime_diff'].max()].iloc[0]
        
        lo_layout = max_pair['hugepages_lo']
        hi_layout = max_pair['hugepages_hi']
        
        return lo_layout, hi_layout
        
    def generate_layout_from_base(self, base_pages, search_space, coverage, sort=True):
        logging.debug(f'generate_layout_from_base(): len(base_pages)={len(base_pages)} , len(search_space)={len(search_space)} , coverage={coverage}')
        expected_coverage = coverage - self.pebsTlbCoverage(base_pages)
        pebs_df = self.pebs_df.query(f'PAGE_NUMBER in {search_space} and PAGE_NUMBER not in {base_pages}')
        if sort:
            pebs_df = pebs_df.sort_values('TLB_COVERAGE', ascending=False)
        logging.debug(f'generate_layout_from_base() after filtering pages: len(pebs_df)={len(pebs_df)}')
        layout = self.generate_layout_from_pebs(expected_coverage, pebs_df)
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

    def constrained_layout_selection(self, pebs_coverage, include_pages, exclude_pages, sort_ascending=False):
        pebs_df = self.pebs_df.query(f'PAGE_NUMBER not in {exclude_pages}')
        pebs_df = pebs_df.sort_values('TLB_COVERAGE', ascending=sort_ascending)
        search_space = pebs_df['PAGE_NUMBER'].to_list()
        layout = self.generate_layout_from_base(include_pages, search_space, pebs_coverage)
        return layout

    def try_select_layout_dynamic_epsilon(self, pebs_df, pebs_coverage, include_pages=[], max_epsilon=2, exclude_pages=None, sort_ascending=False):
        epsilon = 0.5
        while epsilon < max_epsilon:
            layout = self.try_select_layout(pebs_df, pebs_coverage, include_pages, epsilon, exclude_pages, sort_ascending)
            if layout and self.isPagesListUnique(layout, self.layouts):
                return layout
            epsilon += 0.5
        return []
     
    def try_select_layout(self, pebs_df, pebs_coverage, include_pages=[], epsilon=0.5, exclude_pages=None, sort_ascending=False):
        include_pages_weight = self.pebsTlbCoverage(include_pages)
        rem_weight = pebs_coverage - include_pages_weight
        assert rem_weight >= 0
        
        if exclude_pages:
            pebs_df = pebs_df.query(f'PAGE_NUMBER not in {exclude_pages}')
        if include_pages:
            pebs_df = pebs_df.query(f'PAGE_NUMBER not in {include_pages}')
        pebs_df = pebs_df.sort_values('TLB_COVERAGE', ascending=sort_ascending)
        
        mem_layout = include_pages
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
            logging.debug(f'try_select_layout(): could not select layout for pebs_coverage={pebs_coverage} with epsilon={epsilon}')
            return []
        logging.debug(f'try_select_layout(): found layout with {len(mem_layout)} hugepages')
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
            return mem_layout
        
        # second chance: 
        only_in_layout1_df = self.pebs_df.query(f'PAGE_NUMBER in {only_in_layout1}').sort_values('TLB_COVERAGE')
        only_in_layout1 = only_in_layout1_df['PAGE_NUMBER'].to_list()
        only_in_layout2_df = self.pebs_df.query(f'PAGE_NUMBER in {only_in_layout2}').sort_values('TLB_COVERAGE')
        only_in_layout2 = only_in_layout2_df['PAGE_NUMBER'].to_list()
        mem_layout += only_in_layout1[0::2]
        mem_layout += only_in_layout2[0::2]

        return mem_layout

    def scaleExpectedCoverage(self, layout_res, base_layout):
        pebs_coverage = self.pebsTlbCoverage(layout_res['hugepages'])
        real_coverage = self.realMetricCoverage(layout_res)
        base_pebs = self.pebsTlbCoverage(base_layout)
        base_res = self.find_layout_results(base_layout)
        base_real = self.realMetricCoverage(base_res)
        pebs_delta = pebs_coverage - base_pebs
        real_gap = real_coverage - base_real

        if real_gap <= 0:
            desired_coverage = min(100, base_pebs + pebs_delta * 2)
            base_layout = layout_res['hugepages']
            return desired_coverage, base_layout
        return None, None
    
    def layoutComposedOfHeadPages(self, layout):
        layout_coverage = self.pebsTlbCoverage(layout)
        layout_pages = self.pebs_df.query(f'PAGE_NUMBER in {layout}')
        layout_pages = layout_pages.sort_values('TLB_COVERAGE', ascending=False).head(10)
        headpages_coverage = layout_pages['TLB_COVERAGE'].sum()
        return headpages_coverage >= (layout_coverage/2)
        
    def realToPebsCoverage(self, layout_res, layout_expected_real):
        layout = layout_res['hugepages']
        layout_pebs = self.pebsTlbCoverage(layout)
        layout_real = self.realMetricCoverage(layout_res)
        
        if self.layoutComposedOfHeadPages(layout):
            scaled_desired_coverage = layout_expected_real - layout_real + layout_pebs
            return scaled_desired_coverage

        # prevent division by zero and getting numerous ratio in
        # the calculation of expected_to_real
        layout_real = max(1, layout_real)
        expected_to_real = layout_expected_real / layout_real
        scaled_desired_coverage = layout_pebs * expected_to_real
        return scaled_desired_coverage
    
    def select_next_layout_v2(self):
        lower_layout, upper_layout = self.get_surrounding_layouts(self.res_df)
        
        base_layout = upper_layout
        next_coverage = self.metric_coverage
        layout = None
        while not layout:
            A, B, C, U = self.split_pages_to_working_sets(upper_layout, lower_layout)
            layout = self.generate_layout_from_base(base_layout, C, next_coverage)
            if layout and self.isPagesListUnique(layout, self.layouts):
                layout_res = self.run_next_layout(layout)
            else:
                break
            next_coverage, base_layout = self.scaleExpectedCoverage(layout_res, base_layout)
            if base_layout is None:
                next_coverage = self.realToPebsCoverage(self, layout_res, self.metric_coverage)
                base_layout = upper_layout
    
    def select_next_layout(self):
        surrounding_percentile = 0.01
        while surrounding_percentile <= 1:
            lower_layout, upper_layout = self.get_surrounding_layouts(self.res_df)
            mem_layout = self.combine_layouts(lower_layout, upper_layout, surrounding_percentile)
            if mem_layout and self.isPagesListUnique(mem_layout, self.layouts):
                return mem_layout
            surrounding_percentile += 0.01
        assert False

    def pause():
        print('=============================')
        print('press any key to continue ...')
        print('=============================')
        input()
        
    def run(self):
        self.num_generated_layouts = 0
        
        # Define the initial data samples
        init_layouts = self.select_initial_layouts()
        res_df = self.run_layouts(init_layouts)
        
        while self.num_generated_layouts < self.num_layouts:
            MosrangeSelector.pause()
            layout = self.select_next_layout()
            self.run_next_layout(layout)

        logging.info('================================================')
        logging.info(f'Finished running MosRange process for:\n{self.exp_root_dir}')
        logging.info('================================================')
      
