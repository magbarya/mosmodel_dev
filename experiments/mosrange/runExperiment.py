#!/usr/bin/env python3
# import cProfile
import logging
from mosrange_selector import MosrangeSelector

import argparse
def parseArguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('-mem', '--memory_footprint', default='memory_footprint.txt')
    parser.add_argument('-pebs', '--pebs_mem_bins', default=None)
    parser.add_argument('-exp', '--exp_root_dir', required=True)
    parser.add_argument('-res', '--results_dir', required=True)
    parser.add_argument('-x', '--run_experiment_cmd', required=True)
    parser.add_argument('-n', '--num_layouts', required=True, type=int)
    parser.add_argument('-r', '--num_repeats', required=True, type=int)
    parser.add_argument('-m', '--metric', choices=['stlb_misses', 'stlb_hits', 'walk_cycles', 'walk_active', 'walk_pending'], default='stlb_misses')
    parser.add_argument('-v', '--metric_value', type=float, default=None)
    parser.add_argument('-c', '--metric_coverage', type=int, default=None)
    parser.add_argument('-d', '--debug', action='store_true')
    return parser.parse_args()

if __name__ == "__main__":
    args = parseArguments()

    if args.metric_value is None and args.metric_coverage is None:
        raise ValueError('Should provide either metric_value or metric_coverage arguments: None was provided!')
    if args.metric_value is not None and args.metric_coverage is not None:
        raise ValueError('Should provide either metric_value or metric_coverage arguments: Both were provided!')

    logging_level = logging.INFO
    if args.debug:
        logging_level = logging.DEBUG
    logging.basicConfig(level=logging_level, format='[%(name)s:%(levelname)s] %(message)s')

    exp = MosrangeSelector(args.memory_footprint,
                             args.pebs_mem_bins,
                             args.exp_root_dir,
                             args.results_dir,
                             args.run_experiment_cmd,
                             args.num_layouts,
                             args.num_repeats,
                             args.metric,
                             args.metric_value,
                             args.metric_coverage,
                             debug=args.debug)
    exp.run()
