#! /bin/bash

if (( $# < 1 )); then
    echo "Usage: $0 \"command_to_execute\""
    exit -1
fi

command="$@"

prefix_perf_command="perf stat --field-separator=, --output=perf.out"
general_events="cycles,instructions,"
tlb_events=mem_inst_retired.all_loads,mem_inst_retired.all_stores,mem_inst_retired.stlb_miss_loads,mem_inst_retired.stlb_miss_stores,dtlb_load_misses.walk_completed,dtlb_store_misses.walk_completed

perf_command="$prefix_perf_command --event $general_events$tlb_events -- "

time_format="seconds-elapsed,%e\nuser-time-seconds,%U\n"
time_format+="kernel-time-seconds,%S\nmax-resident-memory-kb,%M"
time_command="time --format=$time_format --output=time.out"

submit_command="$perf_command $time_command"
echo "Running the following command:"
echo "$submit_command $command"
$submit_command $command

