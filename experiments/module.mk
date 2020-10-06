MODULE_NAME := experiments
SUBMODULES := \
	memory_footprint \
	single_page_size \
	growing_window_2m \
	random_window_2m \
	sliding_window
SUBMODULES := $(addprefix $(MODULE_NAME)/,$(SUBMODULES))

##### mosalloc paths
RUN_MOSALLOC_TOOL := $(ROOT_DIR)/mosalloc/runMosalloc.py
RESERVE_HUGE_PAGES := $(ROOT_DIR)/mosalloc/reserveHugePages.sh
MOSALLOC_MAKEFILE := $(ROOT_DIR)/mosalloc/CMakeLists.txt
export MOSALLOC_TOOL := $(ROOT_DIR)/mosalloc/src/libmosalloc.so

##### scripts

COLLECT_RESULTS_SCRIPT := $(SCRIPTS_ROOT_DIR)/collectResults.py
CHECK_PARANOID := $(SCRIPTS_ROOT_DIR)/checkParanoid.sh
SET_THP := $(SCRIPTS_ROOT_DIR)/setTransparentHugePages.sh
SET_CPU_MEMORY_AFFINITY := $(SCRIPTS_ROOT_DIR)/setCpuMemoryAffinity.sh
MEASURE_GENERAL_METRICS := $(SCRIPTS_ROOT_DIR)/measureGeneralMetrics.sh
RUN_BENCHMARK_SCRIPT := $(SCRIPTS_ROOT_DIR)/runBenchmark.py

###### global constants

export EXPERIMENTS_ROOT := $(ROOT_DIR)/$(MODULE_NAME)
export EXPERIMENTS_TEMPLATE := $(EXPERIMENTS_ROOT)/template.mk
export BOUND_MEMORY_NODE := 1

define configuration_array
$(addprefix configuration,$(shell seq 1 $1))
endef

#### recipes and rules for prerequisites

$(MOSALLOC_TOOL): $(MOSALLOC_MAKEFILE) cmake
	cd $(dir $<)
	cmake .
	make

$(MOSALLOC_MAKEFILE):
	git submodule update --init --progress

.PHONY: experiments-prerequisites perf numactl cmake

experiments-prerequisites: perf numactl

PERF_PACKAGES := linux-tools
KERNEL_VERSION := $(shell uname -r)
PERF_PACKAGES := $(addsuffix -$(KERNEL_VERSION),$(PERF_PACKAGES))
APT_INSTALL := sudo apt install -y
perf:
	$(CHECK_PARANOID)
	$(APT_INSTALL) "$(PERF_PACKAGES)"

numactl:
	$(APT_INSTALL) $@

cmake:
	$(APT_INSTALL) $@

TEST_RUN_MOSALLOC_TOOL := $(SCRIPTS_ROOT_DIR)/testRunMosallocTool.sh
.PHONY: test-run-mosalloc-tool
test-run-mosalloc-tool: $(RUN_MOSALLOC_TOOL) $(MOSALLOC_TOOL)
	$(TEST_RUN_MOSALLOC_TOOL) $<

include $(ROOT_DIR)/common.mk

