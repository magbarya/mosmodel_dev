#! /bin/bash

if (( $# < 1 )); then
    echo "Usage: $0 \"command_to_execute\""
    exit -1
fi

command="$@"

general_events="cpu-cycles,instructions,"

cache_events="L1-dcache-loads,L1-dcache-stores,L1-dcache-load-misses,L1-dcache-store-misses"
cache_events+=",LLC-loads,LLC-stores,LLC-load-misses,LLC-store-misses,"

prefix_perf_command="perf stat --field-separator=, --output=perf.out"
# extract architecture specific dtlb and energy events from 'ocperf list' output
dtlb_events=`perf list | \grep -o "dtlb_.*_misses.*[a-zA-Z].\b" | sort -u | tr '\n' ','`
dtlb_events=${dtlb_events%?} # remove the trailing , charachter

perf_command="$prefix_perf_command --event $general_events$cache_events$dtlb_events -- "

# We also measure energy if the system allows it.
energy_events=`perf list | \grep -o "\w*\/energy.*\/" | sort -u | tr '\n' ',i'`
energy_events=${energy_events%?} # remove the trailing , charachter

if [[ -z "$energy_events" ]]; then
    echo "this CPU does not support energy events"
else
    echo "this CPU supports energy events"
    echo "skipping profiling energy events..."
    # perf_command+="$prefix_perf_command --event $energy_events --cpu=0 -- "
fi

time_format="seconds-elapsed,%e\nuser-time-seconds,%U\n"
time_format+="kernel-time-seconds,%S\nmax-resident-memory-kb,%M"
time_command="time --format=$time_format --output=time.out"

submit_command="$perf_command $time_command"
echo "Running the following command:"
echo "$submit_command $command"
$submit_command $command

