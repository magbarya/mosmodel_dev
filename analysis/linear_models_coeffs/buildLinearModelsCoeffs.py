#! /usr/bin/env python3

import numpy as np
import pandas as pd

import sys
import os

def loadDataframe(mean_file):
    df = pd.read_csv(mean_file, index_col='layout')
    df.fillna(0, inplace=True)
    df['l2_tlb_misses_completed'] = df['dtlb_load_misses.walk_completed'] + df['dtlb_store_misses.walk_completed']
    df['CPI'] = df['cycles'] / df['instructions']
    df['MPKI'] = 1000 * df['l2_tlb_misses_completed'] / df['instructions']
    return df

def calculateYanivCoeffs(df_4kb, df_2mb):
    delta_y = df_4kb['CPI'] - df_2mb['CPI']
    delta_x = df_4kb['MPKI'] - df_2mb['MPKI']
    A = delta_y / delta_x
    B = df_4kb['CPI'] - A * df_4kb['MPKI']
    return A,B

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('-i', '--mean_file', default='mean.csv')
parser.add_argument('-o', '--output',  default='linear_models_coeffs.csv')
args = parser.parse_args()

res_df = pd.DataFrame(columns=['yaniv_A', 'yaniv_B'])
df = loadDataframe(args.mean_file)
df_4kb = df.loc['layout4kb']
df_2mb = df.loc['layout2mb']
yaniv_A, yaniv_B = calculateYanivCoeffs(df_4kb, df_2mb)
res_df.loc[0] = {'yaniv_A':yaniv_A, 'yaniv_B':yaniv_B}

res_df.to_csv(args.output)

