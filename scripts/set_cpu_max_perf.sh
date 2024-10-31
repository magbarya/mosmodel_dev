#!/bin/bash

# Set CPU frequency to the maximum for all CPUs
echo "Setting CPU frequency to maximum..."
sudo cpupower frequency-set --governor performance
# sudo cpupower frequency-set --max `cpupower frequency-info -l | grep -oP '\d+\.\d+' | sort -n | tail -1`
sudo cpupower frequency-set --max `cpupower frequency-info -l | grep -oP '\d+\.?\d*' | sort -n | tail -1`

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

echo "All operations completed to set CPUs operating in max performance."


