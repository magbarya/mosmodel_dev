#!/usr/bin/env python3
import cProfile
import pandas as pd
from skopt import gp_minimize
from skopt.space import Integer, Space
from skopt.utils import use_named_args
import numpy as np
from bitarray import bitarray
import subprocess
import math
import os, sys
sys.path.append(os.path.dirname(os.path.abspath(__file__))+'/..')
from Utils.utils import Utils

import argparse
def parseArguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--memory_footprint', default='memory_footprint.txt')
    parser.add_argument('-p', '--pebs_mem_bins', default='mem_bins_2mb.csv')
    parser.add_argument('-e', '--exp_root_dir', required=True)
    parser.add_argument('-r', '--results_file', required=True)
    parser.add_argument('-c', '--collect_reults_cmd', required=True)
    parser.add_argument('-x', '--run_bayesian_cmd', required=True)
    parser.add_argument('-n', '--num_layouts', required=True)
    parser.add_argument('-d', '--debug', action='store_true')
    return parser.parse_args()

# based on https://arxiv.org/pdf/1807.02811.pdf
MAX_DIMENSIONS = 20
DEFAULT_HUGEPAGE_SIZE = 1 << 21 # 2MB
last_layout_num = 0

if __name__ == "__main__":
    args = parseArguments()

    # read memory-footprints
    footprint_df = pd.read_csv(args.memory_footprint)
    mmap_footprint = footprint_df['anon-mmap-max'][0]
    brk_footprint = footprint_df['brk-max'][0]
    memory_footprint = brk_footprint
        
    hugepage_size = DEFAULT_HUGEPAGE_SIZE
    num_hugepages = math.ceil(memory_footprint / hugepage_size) # bit vector length
    num_default_hugepages = math.ceil(memory_footprint / DEFAULT_HUGEPAGE_SIZE)
    
    dimension_size_in_bits = 64 #sys.getsizeof(int)
    dimension_capacity = 2**dimension_size_in_bits
    # the num_dimensions is calculated for (num_hugepages + 1) because
    # an additional bit may be required when converting a binary number to gray code
    num_dimensions = math.ceil((num_hugepages + 1) / dimension_size_in_bits)
    if num_dimensions > MAX_DIMENSIONS:
        max_num_hugepages = MAX_DIMENSIONS * dimension_size_in_bits
        hugepage_size = Utils.round_up(math.ceil(memory_footprint / max_num_hugepages), DEFAULT_HUGEPAGE_SIZE)
        num_hugepages = math.ceil(memory_footprint / hugepage_size)
    # update num_dimensions and layout_bit_vector_length in case we exceeded the MAX_DIMESNIONS
    layout_bit_vector_length = num_hugepages
    gray_layout_bit_vector_length = layout_bit_vector_length + 1
    num_dimensions = math.ceil(gray_layout_bit_vector_length / dimension_size_in_bits)
    hugepages_in_compressed_hugepage = hugepage_size // DEFAULT_HUGEPAGE_SIZE
    # Define the search space
    dimension_min_val = 0
    dimension_max_val = dimension_capacity - 1
    last_dimension_size_in_bits = num_hugepages - ((num_dimensions-1) * dimension_size_in_bits)
    last_dimension_max_val = 2**last_dimension_size_in_bits
    dimensions = [Integer(dimension_min_val, dimension_max_val, name=f'mem_region_{i}') for i in range(num_dimensions - 1)] 
    dimensions += [Integer(dimension_min_val, last_dimension_max_val, name=f'mem_region_{num_dimensions-1}')]
    
    if False:
        print(f'num_dimensions: {num_dimensions}')
        print(f'memory_footprint: {memory_footprint}')
        print(f'num_hugepages: {num_hugepages}')
        print(f'hugepage_size: {hugepage_size}')
        print(f'dimension_size_in_bits: {dimension_size_in_bits}')
        print(f'last_dimension_size_in_bits: {last_dimension_size_in_bits}')
        print(f'hugepages_in_compressed_hugepage: {hugepages_in_compressed_hugepage}')
        print(f'layout_bit_vector_length: {layout_bit_vector_length}')
        sys.exit(1)

    pebs_df = Utils.load_pebs(args.pebs_mem_bins, False)
    total_misses = pebs_df['NUM_ACCESSES'].sum()

def convert_to_gray(binary):
    if isinstance(binary, str):
        binary = bitarray(binary)
    elif str(binary).isnumeric():
        binary = bitarray(bin(binary)[2:])
    gray = bitarray(0)
    gray.append(binary[0])
    for i in range(1, len(binary)):
        gray.append(binary[i] ^ binary[i-1])
    return gray

def set_bits(bitarray_obj, bits_val):
    bits_to_set = bin(bits_val)[2:]
    bitarray_bits = bitarray_obj.to01()
    max_len = max(len(bits_to_set), len(bitarray_bits))
    bits_to_set = bits_to_set.zfill(max_len)
    bitarray_bits = bitarray_bits.zfill(max_len)
    new_bitarray = bitarray(bits_to_set) | bitarray(bitarray_bits)
    return new_bitarray

def convert_mem_layout_to_gray(mem_layout_hugepages):
    mem_layout_bin = bitarray(gray_layout_bit_vector_length)
    mem_layout_bin.setall(0)
    # createa one long bit-vector that represents the memory layout
    for p in mem_layout_hugepages:
        # mem_layout_bin = set_bits(mem_layout_bin, p)
        aggregated_p = p // hugepages_in_compressed_hugepage
        mem_layout_bin[aggregated_p] = 1
    # reverse the string to make it readable as binary string
    mem_layout_bin.reverse()
    # convert to gray-code
    gray_mem_layout = convert_to_gray(mem_layout_bin)
    gray_mem_layout.reverse()
    return gray_mem_layout

def convert_from_gray(gray):
    if isinstance(gray, str):
        gray = bitarray(gray)
    elif str(gray).isnumeric():
        gray = bitarray(bin(gray)[2:])
    binary = bitarray(0)
    binary.append(gray[0])
    for i in range(1, len(gray)):
        binary.append(binary[i-1] ^ gray[i])
    return binary

def convert_dimensions_to_mem_layout_bin(mem_layout_dimensions):
    gray_mem_layout = bitarray(0)
    for i in range(len(mem_layout_dimensions)):
        gray_word = bin(mem_layout_dimensions[i])[2:]
        padding_size = dimension_size_in_bits
        if i == (len(mem_layout_dimensions) - 1):
            padding_size = last_dimension_size_in_bits
        padded_word = gray_word.zfill(padding_size)
        gray_mem_layout.extend(padded_word)
    gray_mem_layout.reverse()
    mem_layout = convert_from_gray(gray_mem_layout)
    mem_layout.reverse()
    return mem_layout

def decompress_memory_layout(mem_layout_dimensions):
    hugepages_bit_vector = convert_dimensions_to_mem_layout_bin(mem_layout_dimensions)
    mem_layout_hugepages = []
    for i in range(len(hugepages_bit_vector)):
        if hugepages_bit_vector[i] == 1:
            for k in range(hugepages_in_compressed_hugepage):
                hugepage_idx = i * hugepages_in_compressed_hugepage + k
                mem_layout_hugepages.append(hugepage_idx)
    return mem_layout_hugepages
    
def compress_memory_layout(mem_layout_hugepages):
    gray_mem_layout = convert_mem_layout_to_gray(mem_layout_hugepages)    
    
    compressed_mem_layout = [0] * num_dimensions
    for i in range(num_dimensions):
        dimension_start_idx = i*dimension_size_in_bits
        dimension_end_idx = dimension_start_idx + dimension_size_in_bits
        if dimension_start_idx >= len(gray_mem_layout):
            print('WARNING: memory layout size in gray code is smaller than in normal binary code')
            sys.exit(1)
            break
        if i == (num_dimensions - 1):
            dimension_end_idx = dimension_start_idx + last_dimension_size_in_bits
        gray_i = gray_mem_layout[dimension_start_idx:dimension_end_idx]
        gray_i.reverse()
        gray_i_number = int(gray_i.to01(), 2)
        compressed_mem_layout[i] = gray_i_number
    
    return compressed_mem_layout
    
def run_command(command, out_dir):
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
        print('Output:', output)
        print('Error:', error)
        print('Return code:', return_code)
    
    return return_code

def collect_results():
    print('-------------------------------------------')
    print('collecting results....')
    print(args.collect_reults_cmd)
    
    # Extract the directory path
    results_dir = os.path.dirname(args.results_file)
    # Create the directory if it doesn't exist
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
        
    ret_code = run_command(args.collect_reults_cmd, results_dir)
    if ret_code != 0:
        raise RuntimeError(f'Error: collecting experiment results failed with error code: {ret_code}')
    if os.path.exists(args.results_file):
        results_df = Utils.load_dataframe(args.results_file)
    else:
        results_df = pd.DataFrame()
    
    print('-------------------------------------------')
    
    return results_df

def get_layout_tlb_misses(layout_name):
    results_df = collect_results()
    tlb_misses = results_df[results_df['layout'] == layout_name]['stlb_misses'].iloc[0]
    return tlb_misses

def run_workload(compressed_mem_layout, layout_name):
    mem_layout = decompress_memory_layout(compressed_mem_layout)
    Utils.write_layout(layout_name, mem_layout, args.exp_root_dir, brk_footprint, mmap_footprint)
    
    print('--------------------------------------')
    print(f'Running workload under memory layout with {len(mem_layout)} hugepages')
    print('--------------------------------------')
    out_dir = f'{args.exp_root_dir}/{layout_name}'
    layout_file = f'{args.exp_root_dir}/layouts/{layout_name}.csv'
    run_bayesian_cmd = args.run_bayesian_cmd.replace('OUT_DIR', out_dir)
    run_bayesian_cmd = run_bayesian_cmd.replace('CONFIG_FILE', layout_file)
    ret_code = run_command(run_bayesian_cmd, out_dir)
    if ret_code != 0:
        raise RuntimeError(f'Error: running {layout_name} failed with error code: {ret_code}')
    tlb_misses = get_layout_tlb_misses(layout_name)
    return tlb_misses
    
# Define the objective function using named arguments and the use_named_args decorator
@use_named_args(dimensions)
def objective_function(**params):
    mem_layout = [params[f'mem_region_{i}'] for i in range(num_dimensions)]
    last_layout_num += 1
    layout_name = f'layout{last_layout_num}'
    return run_workload(mem_layout, layout_name)

def predictTlbMisses(mem_layout):
    expected_tlb_coverage = pebs_df.query(f'PAGE_NUMBER in {mem_layout}')['NUM_ACCESSES'].sum()
    expected_tlb_misses = total_misses - expected_tlb_coverage
    print(f'[DEBUG]: mem_layout of size {len(mem_layout)} has an expected-tlb-coverage={expected_tlb_coverage} and expected-tlb-misses={expected_tlb_misses}')
    return expected_tlb_misses

from scipy.special import roots_chebyt
def chebyshev_initial_samples(dimensions, num_samples, min_val=0, max_val=dimension_max_val):
    '''
    Generate initial samples for Bayesian optimization using 
    Chebyshev distribution with discrete integer dimensions.
    Use roots_chebyt to obtain the Chebyshev nodes, 
    scales the values to match the desired range, 
    and rounds them to the nearest integer to align 
    with the Integer dimension.
    '''
    samples = np.zeros((num_samples, dimensions))
    for i in range(dimensions):
        if i == (dimensions - 1):
            max_val = last_dimension_max_val
        roots = roots_chebyt(num_samples)
        scaled_samples = 0.5 * (np.array(roots[0]) + 1) * (max_val - min_val) + min_val
        samples[:, i] = np.round(scaled_samples).astype(int)
    return samples

def generate_random_layout(layouts_space_len):
    mem_layout = []
    random_mem_layout = np.random.randint(2, size=layouts_space_len)
    for i in range(len(random_mem_layout)):
        if random_mem_layout[i] == 1:
            mem_layout.append(i)
    return mem_layout

def random_initial_samples(layouts_space_len, num_initial_layouts):
    mem_layouts = []
    for i in range(num_initial_layouts):
        random_mem_layout = generate_random_layout(layouts_space_len)
        mem_layouts.append(random_mem_layout)
    return mem_layouts

def base_mem_layouts():
    base_pages_layout = []
    hugepages_layout = [i for i in range(num_default_hugepages)]
    mem_layouts = [base_pages_layout, hugepages_layout]
    return mem_layouts

def get_previous_run_samples():
    X0 = []
    Y0 = []
    res_df = collect_results()
    if res_df.empty:
        return X0, Y0
    for index, row in res_df.iterrows():
        layout_name = row['layout']
        mem_layout_pages = Utils.load_layout_hugepages(layout_name, args.exp_root_dir)
        runtime = row['cpu_cycles']
        X0.append(mem_layout_pages)
        Y0.append(runtime)
        last_layout_num += 1
    return X0, Y0

def generate_initial_samples(layouts_space_len, num_initial_points):
    X0, Y0 = get_previous_run_samples()
    if X0:
        return X0, Y0
    
    mem_layouts = base_mem_layouts()
    # mem_layouts = random_initial_samples(layouts_space_len, num_initial_points)
    # mem_layouts = chebyshev_initial_samples(layouts_space_len, num_initial_points)
    for i, mem_layout in enumerate(mem_layouts):
        print(f'==== Producing initial sample #{i} from a layout with {len(mem_layout)*hugepages_in_compressed_hugepage} (x2MB) hugepages ====')
        compressed_mem_layout = compress_memory_layout(mem_layout)
        X0.append(compressed_mem_layout)
        last_layout_num += 1
        layout_name = f'layout{last_layout_num}'
        tlb_misses = run_workload(compressed_mem_layout, layout_name)
        Y0.append(tlb_misses) # evaluate the objective function for each sample
    return X0, Y0
        
def createLayout(layouts_space_len, num_layouts, initial_points=None):
    if initial_points is None:
        initial_points = layouts_space_len * 3
        # initial_points = 100
    # Define the initial data samples (X and Y pairs) for Bayesian optimization
    X0, Y0 = generate_initial_samples(layouts_space_len, initial_points)    
    # Perform Bayesian optimization with the initial data samples
    result = gp_minimize(objective_function,  # the objective function to minimize
                        dimensions=dimensions,  # the search space
                        acq_func='EI',  # the acquisition function
                        n_calls=num_layouts,  # the number of evaluations of f including at x0
                        x0=X0,  # the initial data samples
                        y0=Y0)  # the initial data sample evaluations

    # print("result:", result)
    print("Best TLB misses:", result.fun)
    compressed_best_layout = [int(x) for x in result.x]
    print("Best memory layout (compressed):", compressed_best_layout)
    decompressed_best_layout = decompress_memory_layout(compressed_best_layout)
    print(f"Best memory layout ({len(decompressed_best_layout)} items):")
    if len(decompressed_best_layout) <= 20:
        print(decompressed_best_layout)
    else:
        print(decompressed_best_layout[:10], '...', decompressed_best_layout[-10:])

if __name__ == "__main__":
    # profiler = cProfile.Profile()
    # profiler.enable()
    createLayout(layout_bit_vector_length, args.num_layouts)
    # profiler.disable()
    # profiler.dump_stats('profile_results.prof')
    # profiler.print_stats()
