#!/usr/bin/env python3
import pandas as pd
from layout_generator import *

import argparse
def parseArguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--memory_footprint', default='memory_footprint.txt')
    parser.add_argument('-p', '--pebs_mem_bins', default='mem_bins_2mb.csv')
    parser.add_argument('-b', '--max_budget', type=int, default=50)
    parser.add_argument('-l', '--layout', required=True)
    parser.add_argument('-d', '--exp_dir', required=True)
    parser.add_argument('-n', '--results_file', required=True)
    return parser.parse_args()

if __name__ == "__main__":
    args = parseArguments()

    # read memory-footprints
    footprint_df = pd.read_csv(args.memory_footprint)
    mmap_footprint = footprint_df['anon-mmap-max'][0]
    brk_footprint = footprint_df['brk-max'][0]

    LayoutGeneratorUtils.setPoolsFootprints(brk_footprint, mmap_footprint)

    results_df = LayoutGeneratorUtils.loadDataframe(args.results_file)

    pebs_df = LayoutGeneratorUtils.normalizePebsAccesses(args.pebs_mem_bins)

    max_gap = 200 / args.max_budget
    layout_generator = LayoutGenerator(pebs_df, results_df, args.layout, args.exp_dir, max_gap, args.max_budget)
    layout_generator.generateLayout()
