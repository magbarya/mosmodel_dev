#!/bin/bash

# disable intel p-states: we don't apply this as we observed weird behavior where the CPU runs with the min frequency
# echo "Disable Intel P-States"
# echo off | sudo tee /sys/devices/system/cpu/intel_pstate/status

# Set CPU frequency to the maximum for all CPUs
echo "Setting CPU frequency to maximum..."
sudo cpupower frequency-set --governor performance
# sudo cpupower frequency-set --max `cpupower frequency-info -l | grep -oP '\d+\.\d+' | sort -n | tail -1`
sudo cpupower frequency-set --max `cpupower frequency-info -l | grep -oP '\d+\.?\d*' | sort -n | tail -1`
for cpu in /sys/devices/system/cpu/cpu*; do
    echo performance | sudo tee "$cpu/cpufreq/scaling_governor"
done

# Disable Transparent Huge Pages (THP)
echo "Disable Transparent Huge Pages (THP)"
echo never | sudo tee /sys/kernel/mm/transparent_hugepage/enabled
echo never | sudo tee /sys/kernel/mm/transparent_hugepage/defrag

# Disable automatic NUMA balancing
echo "Disabling automatic NUMA balancing..."
echo 0 | sudo tee /proc/sys/kernel/numa_balancing

# Disable hyper-threading
echo "Disabling hyper-threading..."
echo off | sudo tee /sys/devices/system/cpu/smt/control

# Disable C-states
echo "Disabling C-states..."
for cpu in /sys/devices/system/cpu/cpu*/cpuidle/state[1-3]/disable; do
    echo 1 | sudo tee $cpu;
done

# disable intel turbo
echo "Disable Intel CPU Turbo..."
echo 1 | sudo tee /sys/devices/system/cpu/intel_pstate/no_turbo

# drop page cache
echo "Drop caches..."
echo 3 | sudo tee /proc/sys/vm/drop_caches

# disable watchdog
echo "Turn off watchdog..."
echo 0 | sudo tee /proc/sys/kernel/watchdog

# extend vmstat interval
echo "Extend vmstat_update interval to 1M seconds"
echo 1000000 | sudo tee /proc/sys/vm/stat_interval

# disable ASLR
echo "Disable ASLR"
echo 0 | sudo tee /proc/sys/kernel/randomize_va_space

# Disable Kernel Samepage Merging (KSM)
echo "Disable Kernel Samepage Merging (KSM)"
echo 0 | sudo tee /sys/kernel/mm/ksm/run

echo "All operations completed to set CPUs operating in max performance."

