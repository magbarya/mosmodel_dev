#! /bin/bash

# Get the directory of the script
script_dir="$(dirname "$(readlink -f "$0")")"

general_events="ref-cycles,cpu-cycles,instructions,"

data_cache_events=`perf list | grep -o "mem_load.*_retired\.l[1-3]_\S*" | sort -u | tr '\n' ',i'`
l2tlb_walk_events=`perf list | grep -o "dtlb_[a-z]*_misses\.walk_[a-z]*" | sort -u | tr '\n' ',i'`
l2tlb_retired_misses=`perf list | grep -o "mem.*_retired\.stlb_miss_[a-z]*" | sort -u | tr '\n' ',i'`
l2tlb_hits=`perf list | grep -o "dtlb_.*stlb_hit" | sort -u | tr '\n' ',i'`

perf_events="${general_events}${data_cache_events}${l2tlb_walk_events}${l2tlb_retired_misses}${l2tlb_hits}"
perf_events=${perf_events%?} # remove the trailing , charachter

echo ${perf_events} > ${script_dir}/perf_events_list.txt


