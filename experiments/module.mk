MODULE_NAME := experiments
SUBMODULES := \
	memory_footprint \
	single_page_size \
	pebs_tlb_miss_trace \
	fixed_selector \
	genetic_selector \
	moselect \
	bayesian_optimization \
	mosrange \
	growing_window_2m \
	random_window_2m \
	sliding_window \
	manual_layouts \
	vanilla

EXPERIMENTS_MODULE_NAME := $(MODULE_NAME)
EXPERIMENTS_SUBMODULES := $(addprefix $(EXPERIMENTS_MODULE_NAME)/,$(SUBMODULES))
SUBMODULES := $(EXPERIMENTS_SUBMODULES)

##### mosalloc paths
RUN_MOSALLOC_TOOL := $(ROOT_DIR)/mosalloc/runMosalloc.py
RESERVE_HUGE_PAGES := $(ROOT_DIR)/mosalloc/reserveHugePages.sh
MOSALLOC_MAKEFILE := $(ROOT_DIR)/mosalloc/CMakeLists.txt
export MOSALLOC_TOOL := $(ROOT_DIR)/mosalloc/src/libmosalloc.so

##### scripts

COLLECT_RESULTS := $(SCRIPTS_ROOT_DIR)/collectResults.py
CHECK_PARANOID := $(SCRIPTS_ROOT_DIR)/checkParanoid.sh
CSET_SHIELD_CPUS_SCRIPT := $(SCRIPTS_ROOT_DIR)/cset_shield_node_cpus.sh
SET_CPU_MAX_PERF := $(SCRIPTS_ROOT_DIR)/set_cpu_max_perf.sh
SET_THP := $(SCRIPTS_ROOT_DIR)/setTransparentHugePages.sh
SET_CPU_MEMORY_AFFINITY := $(SCRIPTS_ROOT_DIR)/setCpuMemoryAffinity.sh
MEASURE_GENERAL_METRICS := $(SCRIPTS_ROOT_DIR)/measureGeneralMetrics.sh
RUN_BENCHMARK := $(SCRIPTS_ROOT_DIR)/runBenchmark.py
RUN_BENCHMARK_WITH_SLURM := $(SCRIPTS_ROOT_DIR)/runBenchmarkWithSlurm.py
RUN_BENCHMARK_WITH_CSET_SHIELD := $(SCRIPTS_ROOT_DIR)/runBenchmarkWithCsetShield.py
RUN_WITH_CONDA := $(SCRIPTS_ROOT_DIR)/run_with_conda.sh
COLLECT_MEMORY_FOOTPRINT := $(SCRIPTS_ROOT_DIR)/collectMemoryFootprint.py

###### global constants

export EXPERIMENTS_ROOT := $(ROOT_DIR)/$(MODULE_NAME)
export EXPERIMENTS_TEMPLATE := $(EXPERIMENTS_ROOT)/template.mk
export EXPERIMENTS_VARS_TEMPLATE := $(EXPERIMENTS_ROOT)/template_vars.mk
NUMBER_OF_SOCKETS := $(shell ls -d /sys/devices/system/node/node*/ | wc -w)
export BOUND_MEMORY_NODE := $$(( $(NUMBER_OF_SOCKETS) - 1 ))
export NUMBER_OF_SOCKETS := $(shell ls -d /sys/devices/system/node/node*/ | wc -w)
export NUMBER_OF_CORES_PER_SOCKET := $(shell ls -d /sys/devices/system/node/node0/cpu*/ | wc -w)
export OMP_NUM_THREADS := $(NUMBER_OF_CORES_PER_SOCKET) 
export OMP_THREAD_LIMIT := $(OMP_NUM_THREADS) 
export EXPERIMENTS_ROOT_DIR := $(ROOT_DIR)/$(MODULE_NAME)

CSET_SHIELD_CPUS := $(CSET_SHIELD_CPUS_SCRIPT) $(BOUND_MEMORY_NODE)

#### recipes and rules for prerequisites

.PHONY: experiments-prerequisites perf numactl cpuset mosalloc test-run-mosalloc-tool

mosalloc: $(MOSALLOC_TOOL)
$(MOSALLOC_TOOL): $(MOSALLOC_MAKEFILE)
	$(APT_INSTALL) cmake libgtest-dev
	cd $(dir $<)
	cmake .
	if [[ $$SKIP_MOSALLOC_TEST == 0 ]]; then \
		make -j && ctest -VV; \
	else \
		make -j; \
	fi

$(MOSALLOC_MAKEFILE):
	git submodule update --init --progress

experiments-prerequisites: perf numactl cpuset mosalloc

PERF_PACKAGES := linux-tools
KERNEL_VERSION := $(shell uname -r)
PERF_PACKAGES := $(addsuffix -$(KERNEL_VERSION),$(PERF_PACKAGES))
APT_INSTALL := sudo apt install -y
perf:
	$(CHECK_PARANOID)
	$(APT_INSTALL) "$(PERF_PACKAGES)"

numactl:
	$(APT_INSTALL) $@

cpuset:
	$(APT_INSTALL) $@
	$(CSET_SHIELD_CPUS)
	$(SET_CPU_MAX_PERF)

TEST_RUN_MOSALLOC_TOOL := $(SCRIPTS_ROOT_DIR)/testRunMosallocTool.sh
test-run-mosalloc-tool: $(RUN_MOSALLOC_TOOL) $(MOSALLOC_TOOL)
	$(TEST_RUN_MOSALLOC_TOOL) $<

#### recipes and rules for creating run_and_collect_results.sh script

CUSTOM_RUN_EXPERIMENT_TEMPLATE := $(EXPERIMENTS_ROOT_DIR)/run_benchmark.sh.template
CUSTOM_RUN_EXPERIMENT_SCRIPT := $(EXPERIMENTS_ROOT_DIR)/run_benchmark.sh
CUSTOM_COLLECT_RESULTS_TEMPLATE := $(EXPERIMENTS_ROOT_DIR)/collect_results.sh.template
CUSTOM_COLLECT_RESULTS_SCRIPT := $(EXPERIMENTS_ROOT_DIR)/collect_results.sh

NUM_OF_REPEATS ?= 4
NUMBER_OF_THREADS ?= $(NUMBER_OF_CORES_PER_SOCKET)
RESULTS_ROOT_DIR := $(ROOT_DIR)/results

$(CUSTOM_COLLECT_RESULTS_SCRIPT): | $(CUSTOM_COLLECT_RESULTS_TEMPLATE)
	cp $| $@
	sed -i "s,__COLLECT_RESULTS_SCRIPT__,$(COLLECT_RESULTS),g" $@
	sed -i "s,__EXPERIMENTS_ROOT_DIR__,$(EXPERIMENTS_ROOT_DIR),g" $@
	sed -i "s,__RESULTS_ROOT_DIR__,$(RESULTS_ROOT_DIR),g" $@
	sed -i "s,__EXPERIMENT_NAME__,$(EXPERIMENT_NAME),g" $@
	sed -i "s,__NUM_OF_REPEATS__,$(NUM_OF_REPEATS),g" $@
	chmod 755 $@

$(CUSTOM_RUN_EXPERIMENT_SCRIPT): $(CUSTOM_RUN_EXPERIMENT_TEMPLATE)
	cp $< $@
	sed -i "s,__COLLECT_RESULTS_SCRIPT__,$(COLLECT_RESULTS),g" $@
	sed -i "s,__EXPERIMENTS_ROOT_DIR__,$(EXPERIMENTS_ROOT_DIR),g" $@
	sed -i "s,__EXPERIMENT_NAME__,$(EXPERIMENT_NAME),g" $@
	sed -i "s,__NUM_OF_REPEATS__,$(NUM_OF_REPEATS),g" $@
	sed -i "s,__NUM_OF_THREADS__,$(NUMBER_OF_THREADS),g" $@
	sed -i "s,__RUN_BENCHMARK_SCRIPT__,$(RUN_BENCHMARK_WITH_CSET_SHIELD),g" $@
	sed -i "s,__MEASURE_GENERAL_METRICS_SCRIPT__,$(MEASURE_GENERAL_METRICS),g" $@
	sed -i "s,__RUN_MOSALLOC_TOOL__,$(RUN_MOSALLOC_TOOL),g" $@
	sed -i "s,__MOSALLOC_TOOL__,$(MOSALLOC_TOOL),g" $@
	sed -i "s,__EXTRA_ARGS_FOR_MOSALLOC__,$(EXTRA_ARGS_FOR_MOSALLOC),g" $@
	sed -i "s,__BENCHMARK_PATH__,$(BENCHMARK_PATH),g" $@
	chmod 755 $@

#### recipes and rules for calculating the benchmark memory footprint

MEMORY_FOOTPRINT_FILE := $(MODULE_NAME)/memory_footprint.csv

$(MEMORY_FOOTPRINT_FILE): | experiments/memory_footprint/layout4kb
	$(COLLECT_MEMORY_FOOTPRINT) $| --output=$@

$(MODULE_NAME)/clean:
	rm -f $(MEMORY_FOOTPRINT_FILE)

### include common makefile

### define RESULT_DIRS to hold all created result directories
RESULT_DIRS :=

include $(ROOT_DIR)/common.mk

