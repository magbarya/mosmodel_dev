MOSRANGE_EXPERIMENT_NAME := mosrange
MODULE_NAME := experiments/$(MOSRANGE_EXPERIMENT_NAME)
MOSRANGE_EXPERIMENT_ROOT_DIR := $(ROOT_DIR)/$(MODULE_NAME)
MOSRANGE_RESULTS_DIR := $(ROOT_DIR)/results/$(MOSRANGE_EXPERIMENT_NAME)

MOSRANGE_NUM_LAYOUTS ?= 50
MOSRANGE_NUM_OF_REPEATS ?= 4

NUM_LAYOUTS := $(MOSRANGE_NUM_LAYOUTS)
NUM_OF_REPEATS := $(MOSRANGE_NUM_OF_REPEATS)

PEBS_FILE := analysis/pebs_tlb_miss_trace/mem_bins_2mb.csv
MOSRANGE_RUN_BENCHMARK := $(MODULE_NAME)/run_benchmark.sh
MOSRANGE_COLLECT_RESULTS := $(MODULE_NAME)/collect_results.sh
RUN_MOSRANGE_EXP_SCRIPT := $(MODULE_NAME)/runExperiment.py

$(MOSRANGE_RUN_BENCHMARK): $(CUSTOM_RUN_EXPERIMENT_SCRIPT).$(MOSRANGE_EXPERIMENT_NAME)
	cp $< $@
$(CUSTOM_RUN_EXPERIMENT_SCRIPT).$(MOSRANGE_EXPERIMENT_NAME): EXPERIMENT_NAME := $(MOSRANGE_EXPERIMENT_NAME)
$(CUSTOM_RUN_EXPERIMENT_SCRIPT).$(MOSRANGE_EXPERIMENT_NAME): NUM_OF_REPEATS := $(MOSRANGE_NUM_OF_REPEATS)
$(CUSTOM_RUN_EXPERIMENT_SCRIPT).$(MOSRANGE_EXPERIMENT_NAME): | $(CUSTOM_RUN_EXPERIMENT_SCRIPT)
	cp $| $@

$(MOSRANGE_COLLECT_RESULTS): $(CUSTOM_COLLECT_RESULTS_SCRIPT).$(MOSRANGE_EXPERIMENT_NAME)
	cp $< $@
$(CUSTOM_COLLECT_RESULTS_SCRIPT).$(MOSRANGE_EXPERIMENT_NAME): EXPERIMENT_NAME := $(MOSRANGE_EXPERIMENT_NAME)
$(CUSTOM_COLLECT_RESULTS_SCRIPT).$(MOSRANGE_EXPERIMENT_NAME): NUM_OF_REPEATS := $(MOSRANGE_NUM_OF_REPEATS)
$(CUSTOM_COLLECT_RESULTS_SCRIPT).$(MOSRANGE_EXPERIMENT_NAME): | $(CUSTOM_COLLECT_RESULTS_SCRIPT)
	cp $| $@

$(MODULE_NAME): $(MEMORY_FOOTPRINT_FILE) $(PEBS_FILE) $(MOSRANGE_COLLECT_RESULTS) $(MOSRANGE_RUN_BENCHMARK)
# $(MODULE_NAME): $(MEMORY_FOOTPRINT_FILE) $(MOSRANGE_COLLECT_RESULTS) $(MOSRANGE_RUN_BENCHMARK)
	$(RUN_MOSRANGE_EXP_SCRIPT) \
		--memory_footprint=$(MEMORY_FOOTPRINT_FILE) \
		--exp_root_dir=$(MOSRANGE_EXPERIMENT_ROOT_DIR) \
		--results_file=$(MOSRANGE_RESULTS_DIR)/median.csv \
		--collect_reults_cmd=$(MOSRANGE_COLLECT_RESULTS) \
		--run_experiment_cmd=$(MOSRANGE_RUN_BENCHMARK) \
		--num_layouts=$(MOSRANGE_NUM_LAYOUTS) \
		--pebs_mem_bins=$(MEM_BINS_2MB_CSV_FILE)

.PHONY: $(MODULE_NAME)
