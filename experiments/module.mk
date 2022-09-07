MODULE_NAME := experiments
SUBMODULES := \
	memory_footprint \
	single_page_size \
	pebs_tlb_miss_trace \
	static_auto_mosalloc \
	dynamic_auto_mosalloc \
	moselect \
	dynamic_grouping \
	runtime_range \
	growing_window_2m \
	random_window_2m \
	sliding_window \
	extra_sliding_window \
	extra_random_window_2m \
	manual \
	subgroups_windows \
	subgroups_uniformly_windows \
	subgroups_head_pages \
	genetic_scan \
	smart_genetic_scan
SUBMODULES := $(addprefix $(MODULE_NAME)/,$(SUBMODULES))

##### mosalloc paths
RUN_MOSALLOC_TOOL := $(ROOT_DIR)/mosalloc/runMosalloc.py
RESERVE_HUGE_PAGES := $(ROOT_DIR)/mosalloc/reserveHugePages.sh
MOSALLOC_MAKEFILE := $(ROOT_DIR)/mosalloc/CMakeLists.txt
export MOSALLOC_TOOL := $(ROOT_DIR)/mosalloc/src/libmosalloc.so

##### scripts

COLLECT_RESULTS := $(SCRIPTS_ROOT_DIR)/collectResults.py
CHECK_PARANOID := $(SCRIPTS_ROOT_DIR)/checkParanoid.sh
SET_THP := $(SCRIPTS_ROOT_DIR)/setTransparentHugePages.sh
SET_CPU_MEMORY_AFFINITY := $(SCRIPTS_ROOT_DIR)/setCpuMemoryAffinity.sh
MEASURE_GENERAL_METRICS := $(SCRIPTS_ROOT_DIR)/measureGeneralMetrics.sh
RUN_BENCHMARK := $(SCRIPTS_ROOT_DIR)/runBenchmark.py
RUN_BENCHMARK_WITH_SLURM := $(SCRIPTS_ROOT_DIR)/runBenchmarkWithSlurm.py
COLLECT_MEMORY_FOOTPRINT := $(SCRIPTS_ROOT_DIR)/collectMemoryFootprint.py

###### global constants

export EXPERIMENTS_ROOT := $(ROOT_DIR)/$(MODULE_NAME)
export EXPERIMENTS_TEMPLATE := $(EXPERIMENTS_ROOT)/template.mk
NUMBER_OF_SOCKETS := $(shell ls -d /sys/devices/system/node/node*/ | wc -w)
export BOUND_MEMORY_NODE := $$(( $(NUMBER_OF_SOCKETS) - 1 ))
export NUMBER_OF_SOCKETS := $(shell ls -d /sys/devices/system/node/node*/ | wc -w)
export NUMBER_OF_CORES_PER_SOCKET := $(shell ls -d /sys/devices/system/node/node0/cpu*/ | wc -w)
export OMP_NUM_THREADS := $(NUMBER_OF_CORES_PER_SOCKET) 
export OMP_THREAD_LIMIT := $(OMP_NUM_THREADS) 

define configuration_array
$(addprefix configuration,$(shell seq 1 $1))
endef

#### recipes and rules for prerequisites

.PHONY: experiments-prerequisites perf numactl mosalloc test-run-mosalloc-tool

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

experiments-prerequisites: perf numactl mosalloc

PERF_PACKAGES := linux-tools
KERNEL_VERSION := $(shell uname -r)
PERF_PACKAGES := $(addsuffix -$(KERNEL_VERSION),$(PERF_PACKAGES))
APT_INSTALL := sudo apt install -y
perf:
	$(CHECK_PARANOID)
	$(APT_INSTALL) "$(PERF_PACKAGES)"

numactl:
	$(APT_INSTALL) $@

TEST_RUN_MOSALLOC_TOOL := $(SCRIPTS_ROOT_DIR)/testRunMosallocTool.sh
test-run-mosalloc-tool: $(RUN_MOSALLOC_TOOL) $(MOSALLOC_TOOL)
	$(TEST_RUN_MOSALLOC_TOOL) $<

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

