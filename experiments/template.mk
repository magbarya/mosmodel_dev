DEFAULT_NUM_LAYOUTS ?= 9
DEFAULT_NUM_OF_REPEATS ?= 1
NUMBER_OF_THREADS ?= $(NUMBER_OF_CORES_PER_SOCKET)

ifndef NUM_LAYOUTS
NUM_LAYOUTS := $(DEFAULT_NUM_LAYOUTS)
endif # ifndef NUM_LAYOUTS

ifndef LAYOUTS
LAYOUTS := $(shell seq 1 $(NUM_LAYOUTS))
LAYOUTS := $(addprefix layout,$(LAYOUTS)) 
endif #ifndef LAYOUTS

ifndef NUM_OF_REPEATS
NUM_OF_REPEATS := $(DEFAULT_NUM_OF_REPEATS)
endif # ifndef NUM_OF_REPEATS

ifndef NUMBER_OF_THREADS
NUMBER_OF_THREADS ?= NUMBER_OF_CORES_PER_SOCKET
endif

$(MODULE_NAME)% : NUM_LAYOUTS := $(NUM_LAYOUTS)
$(MODULE_NAME)% : NUM_OF_REPEATS := $(NUM_OF_REPEATS)
$(MODULE_NAME)%: EXTRA_ARGS_FOR_MOSALLOC := $(EXTRA_ARGS_FOR_MOSALLOC)

EXPERIMENT_DIR := $(MODULE_NAME)
RESULT_DIR := $(subst experiments,results,$(EXPERIMENT_DIR))
RESULT_DIRS += $(RESULT_DIR)

LAYOUTS_DIR := $(EXPERIMENT_DIR)/layouts
LAYOUT_FILES := $(addprefix $(LAYOUTS_DIR)/,$(LAYOUTS))
LAYOUT_FILES := $(addsuffix .csv,$(LAYOUT_FILES))

$(LAYOUTS_DIR): $(LAYOUT_FILES)

EXPERIMENTS := $(addprefix $(EXPERIMENT_DIR)/,$(LAYOUTS)) 
RESULTS := $(addsuffix /mean.csv,$(RESULT_DIR)) 

REPEATS := $(shell seq 1 $(NUM_OF_REPEATS))
REPEATS := $(addprefix repeat,$(REPEATS)) 

EXPERIMENT_REPEATS := $(foreach experiment,$(EXPERIMENTS),$(foreach repeat,$(REPEATS),$(experiment)/$(repeat)))
MEASUREMENTS := $(addsuffix /perf.out,$(EXPERIMENT_REPEATS))

$(EXPERIMENT_DIR): $(MEASUREMENTS)
$(EXPERIMENTS): $(EXPERIMENT_DIR)/layout%: $(foreach repeat,$(REPEATS),$(addsuffix /$(repeat)/perf.out,$(EXPERIMENT_DIR)/layout%))
$(EXPERIMENT_REPEATS): %: %/perf.out

define MEASUREMENTS_template =
$(EXPERIMENT_DIR)/$(1)/$(2)/perf.out: %/perf.out: $(EXPERIMENT_DIR)/layouts/$(1).csv | experiments-prerequisites 
	echo ========== [INFO] start producing: $$@ ==========
	$$(RUN_BENCHMARK) --num_threads=$$(NUMBER_OF_THREADS) \
		--submit_command \
		"$$(MEASURE_GENERAL_METRICS) $$(SET_CPU_MEMORY_AFFINITY) $$(BOUND_MEMORY_NODE) \
		$$(RUN_MOSALLOC_TOOL) --library $$(MOSALLOC_TOOL) -cpf $$(ROOT_DIR)/$$< $$(EXTRA_ARGS_FOR_MOSALLOC)" -- \
		$$(BENCHMARK_PATH) $$*
endef

define SLURM_EXPS_template =
$(EXPERIMENT_DIR)/$(1)/$(2)/perf.out: %/$(2)/perf.out: $(EXPERIMENT_DIR)/layouts/$(1).csv | experiments-prerequisites 
	echo ========== [INFO] start producing: $$@ ==========
	$$(RUN_BENCHMARK_WITH_SLURM) --num_threads=$$(NUMBER_OF_THREADS) num_repeats=$$(NUM_OF_REPEATS) \
		--submit_command "$$(MEASURE_GENERAL_METRICS)  \
		$$(RUN_MOSALLOC_TOOL) --library $$(MOSALLOC_TOOL) -cpf $$(ROOT_DIR)/$$< $$(EXTRA_ARGS_FOR_MOSALLOC)" -- \
		$$(BENCHMARK_PATH) $$*
endef

ifndef SLURM
$(foreach layout,$(LAYOUTS),$(foreach repeat,$(REPEATS),$(eval $(call MEASUREMENTS_template,$(layout),$(repeat)))))
else
$(foreach layout,$(LAYOUTS),$(foreach repeat,$(REPEATS),$(eval $(call SLURM_EXPS_template,$(layout),$(repeat)))))
endif

results: $(RESULT_DIR)
$(RESULTS): LAYOUT_LIST := $(call array_to_comma_separated,$(LAYOUTS))
$(RESULT_DIR): $(RESULTS)
$(RESULTS): results/%/mean.csv: experiments/%
	mkdir -p $(dir $@)
	$(COLLECT_RESULTS) --experiments_root=$< --layouts=$(LAYOUT_LIST) --output_dir=$(dir $@)

DELETED_TARGETS := $(EXPERIMENTS) $(EXPERIMENT_REPEATS) $(LAYOUTS_DIR)
CLEAN_TARGETS := $(addsuffix /clean,$(DELETED_TARGETS))
$(CLEAN_TARGETS): %/clean: %/delete
$(MODULE_NAME)/clean: $(CLEAN_TARGETS)


