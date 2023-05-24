BAYESIAN_EXPERIMENT_NAME := bayesian_optimization
MODULE_NAME := experiments/$(BAYESIAN_EXPERIMENT_NAME)
BAYESIAN_EXPERIMENT_ROOT_DIR := $(ROOT_DIR)/$(MODULE_NAME)
BAYESIAN_RESULTS_DIR := $(ROOT_DIR)/results/$(BAYESIAN_EXPERIMENT_NAME)

# BAYESIAN_NUM_OF_REPEATS ?= 4
# BAYESIAN_NUM_LAYOUTS ?= 50
# NUM_LAYOUTS := $(BAYESIAN_NUM_LAYOUTS)
# NUM_OF_REPEATS := $(BAYESIAN_NUM_OF_REPEATS)

COLLECT_BAYESIAN_RESULTS_COMMAND := $(COLLECT_RESULTS) \
		--experiments_root=$(BAYESIAN_EXPERIMENT_ROOT_DIR) --repeats=$(NUM_OF_REPEATS) \
		--output_dir=$(BAYESIAN_RESULTS_DIR) --skip_outliers
RUN_BAYESIAN_EXP_COMMANMD := $(RUN_BENCHMARK_WITH_SLURM) --num_threads=$(NUMBER_OF_THREADS) --num_repeats=$(NUM_OF_REPEATS) \
		--submit_command "$(MEASURE_GENERAL_METRICS)  \
		$(RUN_MOSALLOC_TOOL) --library $(MOSALLOC_TOOL) $(EXTRA_ARGS_FOR_MOSALLOC) -cpf CONFIG_FILE" -- \
		$(BENCHMARK_PATH) OUT_DIR

dummy:
	echo $(RUN_BAYESIAN_EXP_COMMANMD)

RUN_BAYESIAN_EXP_SCRIPT := $(MODULE_NAME)/runBayesianExperiment.py
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
