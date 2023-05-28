BAYESIAN_EXPERIMENT_NAME := bayesian_optimization
MODULE_NAME := experiments/$(BAYESIAN_EXPERIMENT_NAME)
BAYESIAN_EXPERIMENT_ROOT_DIR := $(ROOT_DIR)/$(MODULE_NAME)
BAYESIAN_RESULTS_DIR := $(ROOT_DIR)/results/$(BAYESIAN_EXPERIMENT_NAME)

BAYESIAN_NUM_LAYOUTS ?= 50
NUM_LAYOUTS := $(BAYESIAN_NUM_LAYOUTS)

BAYESIAN_NUM_OF_REPEATS ?= 4
NUM_OF_REPEATS := $(BAYESIAN_NUM_OF_REPEATS)

BAYESIAN_RUN_BENCHMARK := $(MODULE_NAME)/run_benchmark.sh
BAYESIAN_COLLECT_RESULTS := $(MODULE_NAME)/collect_results.sh
RUN_BAYESIAN_EXP_SCRIPT := $(MODULE_NAME)/runBayesianExperiment.py

$(BAYESIAN_RUN_BENCHMARK): $(CUSTOM_RUN_EXPERIMENT_SCRIPT).$(BAYESIAN_EXPERIMENT_NAME)
	cp $< $@
$(CUSTOM_RUN_EXPERIMENT_SCRIPT).$(BAYESIAN_EXPERIMENT_NAME): EXPERIMENT_NAME := $(BAYESIAN_EXPERIMENT_NAME)
$(CUSTOM_RUN_EXPERIMENT_SCRIPT).$(BAYESIAN_EXPERIMENT_NAME): | $(CUSTOM_RUN_EXPERIMENT_SCRIPT)
	cp $| $@

$(BAYESIAN_COLLECT_RESULTS): $(CUSTOM_COLLECT_RESULTS_SCRIPT).$(BAYESIAN_EXPERIMENT_NAME)
	cp $< $@
$(CUSTOM_COLLECT_RESULTS_SCRIPT).$(BAYESIAN_EXPERIMENT_NAME): EXPERIMENT_NAME := $(BAYESIAN_EXPERIMENT_NAME)
$(CUSTOM_COLLECT_RESULTS_SCRIPT).$(BAYESIAN_EXPERIMENT_NAME): | $(CUSTOM_COLLECT_RESULTS_SCRIPT)
	cp $| $@

# $(MODULE_NAME): $(MEMORY_FOOTPRINT_FILE) analysis/pebs_tlb_miss_trace/mem_bins_2mb.csv $(BAYESIAN_COLLECT_RESULTS) $(BAYESIAN_RUN_BENCHMARK)
$(MODULE_NAME): $(MEMORY_FOOTPRINT_FILE) $(BAYESIAN_COLLECT_RESULTS) $(BAYESIAN_RUN_BENCHMARK)
	$(RUN_BAYESIAN_EXP_SCRIPT) \
		--memory_footprint=$(MEMORY_FOOTPRINT_FILE) \
		--pebs_mem_bins=$(MEM_BINS_2MB_CSV_FILE) \
		--exp_root_dir=$(BAYESIAN_EXPERIMENT_ROOT_DIR) \
		--results_file=$(BAYESIAN_RESULTS_DIR)/median.csv \
		--collect_reults_cmd=$(BAYESIAN_COLLECT_RESULTS) \
		--run_experiment_cmd=$(BAYESIAN_RUN_BENCHMARK) \
		--num_layouts=$(BAYESIAN_NUM_LAYOUTS)

.PHONY: $(MODULE_NAME)

# CREATE_BAYESIAN_LAYOUTS_SCRIPT := $(MODULE_NAME)/createLayouts.py
# $(LAYOUT_FILES): $(BAYESIAN_EXPERIMENT)/layouts/%.csv: $(MEMORY_FOOTPRINT_FILE) analysis/pebs_tlb_miss_trace/mem_bins_2mb.csv
# 	mkdir -p results/$(BAYESIAN_EXPERIMENT_NAME)
# 	$(COLLECT_RESULTS) --experiments_root=$(BAYESIAN_EXPERIMENT) --repeats=$(NUM_OF_REPEATS) \
# 		--output_dir=$(BAYESIAN_RESULTS) --skip_outliers
# 	$(CREATE_BAYESIAN_LAYOUTS_SCRIPT) \
# 		--memory_footprint=$(MEMORY_FOOTPRINT_FILE) \
# 		--pebs_mem_bins=$(MEM_BINS_2MB_CSV_FILE) \
# 		--layout=$* \
# 		--max_budget=$(BAYESIAN_NUM_LAYOUTS) \
# 		--exp_dir=$(dir $@)/.. \
# 		--results_file=$(BAYESIAN_RESULTS)/median.csv
