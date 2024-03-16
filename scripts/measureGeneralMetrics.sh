#! /bin/bash

# Get the directory of the script
script_dir="$(dirname "$(readlink -f "$0")")"

if (( $# < 1 )); then
    echo "Usage: $0 \"command_to_execute\""
    exit -1
fi

command="$@"

# Path to the perf_events_list file
perf_events_file="$script_dir/perf_events_list.txt"

# Check if the perf_events_list file exists
if [ ! -f "$perf_events_file" ]; then
    echo "Error: perf_events file '$perf_events_file' not found." >&2
    exit 1
fi

# Read the perf_events from the file
perf_events=$(< "$perf_events_file")
perf_command="perf stat --field-separator=, --output=perf.out --event $perf_events -- "

time_format="seconds-elapsed,%e\nuser-time-seconds,%U\n"
time_format+="kernel-time-seconds,%S\nmax-resident-memory-kb,%M"
time_command="time --format=$time_format --output=time.out"

submit_command="$perf_command $time_command"
echo "Running the following command:"
echo "$submit_command $command"
$submit_command $command

