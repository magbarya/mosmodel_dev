BENCHMARK_PATH := /csl/a.mohammad/CSL/benchmarks/xsbench/unionized-16GB
PRE_RUN_SCRIPT_NAME ?= ./pre_run.sh
POST_RUN_SCRIPT_NAME ?= ./post_run.sh

export MOSRANGE_METRIC_COVERAGE ?= 50
export MOSRANGE_METRIC_NAME ?= stlb_misses
export DEFAULT_NUM_LAYOUTS := 60
export DEFAULT_NUM_OF_REPEATS := 4

export SKIP_MOSALLOC_TEST := 1
export PERF_RECORD_FREQUENCY := 1024


