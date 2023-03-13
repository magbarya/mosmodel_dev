#! /usr/bin/env python3

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('-m', '--mean_file', default='mean.csv', nargs='+',
        help='the input CSV file containing mean values')
parser.add_argument('-s', '--std_file', nargs='*',
        help='an optional CSV file containing stdev values')
parser.add_argument('-o', '--output', help='the output file name')
parser.add_argument('-n', '--normalize', choices=[None, 'by-y', 'separate'],
        default=None, help='how to normalize the data columns')
parser.add_argument('-x', '--x-metric', choices=['walk_cycles', 'tlb_misses'],
        default='tlb_misses', help='which event to use in the x-axis')
parser.add_argument('-y', '--y-metric', default='cpu-cycles',
       help='which event to use in the y-axis')
args = parser.parse_args()

import pandas as pd
from performance_statistics import PerformanceStatistics

def readSingle(mean_file, std_file, y_metric, x_metric):
    if x_metric == 'tlb_misses':
        metric_func = PerformanceStatistics.getStlbMisses
    elif x_metric == 'walk_cycles':
        metric_func = PerformanceStatistics.getWalkDuration
    else:
        raise Exception('Unknown x-metric: ' + x_metric)

    mean_ps = PerformanceStatistics(mean_file, 'layout')
    mean_df = mean_ps.getDataFrame()
    mean_df[x_metric] = metric_func(mean_ps)
    mean_df = mean_df[[x_metric, y_metric]]

    std_df = pd.DataFrame()
    if std_file:
        std_ps = PerformanceStatistics(std_file, 'layout')
        std_df = std_ps.getDataFrame()
        std_df[y_metric+'_std'] = std_df[y_metric]
        std_df[x_metric+'_std'] = metric_func(std_ps)
        std_df = std_df[[x_metric+'_std', y_metric+'_std']]

    output_df = pd.concat([mean_df, std_df], axis='columns')
    output_df.sort_values(x_metric, inplace=True)
    return output_df

mean_files = args.mean_file
std_files = args.std_file
if args.std_file is None:
    std_files = len(mean_files) * [None]
else:
    assert(len(mean_files) == len(std_files))

output_dfs = []
for mean_file, std_file in zip(mean_files, std_files):
    output_df = readSingle(mean_file, std_file, args.y_metric, args.x_metric)
    output_dfs.append(output_df)

output_df = pd.concat(output_dfs)

if args.normalize:
    max_metric = output_df[args.y_metric].max()
    max_x_metric = output_df[args.x_metric].max()
    if args.normalize == 'by-y':
        output_df = output_df / max_metric
        # we normalized the entire DataFrame, including the std columns
    elif args.normalize == 'separate':
        output_df[args.y_metric] = output_df[args.y_metric] / max_metric
        output_df[args.x_metric] = output_df[args.x_metric] / max_x_metric
        if args.std_file:
            output_df[args.y_metric+'_std'] = output_df[args.y_metric+'_std'] / max_metric
            output_df[args.x_metric+'_std'] = output_df[args.x_metric+'_std'] / max_x_metric

output_df.to_csv(args.output, float_format='%.3f')

