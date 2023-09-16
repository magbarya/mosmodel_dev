include $(EXPERIMENTS_VARS_TEMPLATE)

define MEASUREMENTS_template =
$(EXPERIMENT_DIR)/$(1)/$(2)/perf.out: %/perf.out: $(EXPERIMENT_DIR)/layouts/$(1).csv | experiments-prerequisites 
	echo ========== [INFO] allocate/reserve hugepages ==========
	$$(SET_CPU_MEMORY_AFFINITY) $$(BOUND_MEMORY_NODE) $$(RUN_MOSALLOC_TOOL) --library $$(MOSALLOC_TOOL) -cpf $$(ROOT_DIR)/$$< /bin/date
	echo ========== [INFO] start producing: $$@ ==========
	$$(RUN_BENCHMARK) --num_threads=$$(NUMBER_OF_THREADS) \
		--submit_command \
		"$$(MEASURE_GENERAL_METRICS) $$(SET_CPU_MEMORY_AFFINITY) $$(BOUND_MEMORY_NODE) \
		$$(RUN_MOSALLOC_TOOL) --library $$(MOSALLOC_TOOL) -cpf $$(ROOT_DIR)/$$< $$(EXTRA_ARGS_FOR_MOSALLOC)" -- \
		$$(BENCHMARK_PATH) $$*
endef

define SLURM_EXPS_template =
$(EXPERIMENT_DIR)/$(1)/$(2)/perf.out: %/$(2)/perf.out: $(EXPERIMENT_DIR)/layouts/$(1).csv | experiments-prerequisites 
	echo ========== [INFO] allocate/reserve hugepages ==========
	for ((i=0; i < $$(NUMBER_OF_SOCKETS); i++)) do
		srun -- $$(RUN_MOSALLOC_TOOL) --library $$(MOSALLOC_TOOL) -cpf $$(ROOT_DIR)/$$< sleep 2 &
	done
	wait
	echo ========== [INFO] hugepages allocation is done for all nodes ==========
	echo ========== [INFO] start producing: $$@ ==========
	$$(RUN_BENCHMARK_WITH_SLURM) --num_threads=$$(NUMBER_OF_THREADS) --num_repeats=$$(NUM_OF_REPEATS) \
		--submit_command "$$(MEASURE_GENERAL_METRICS)  \
		$$(RUN_MOSALLOC_TOOL) --library $$(MOSALLOC_TOOL) -cpf $$(ROOT_DIR)/$$< $$(EXTRA_ARGS_FOR_MOSALLOC)" -- \
		$$(BENCHMARK_PATH) $$*
endef

ifndef SLURM
$(foreach layout,$(LAYOUTS),$(foreach repeat,$(REPEATS),$(eval $(call MEASUREMENTS_template,$(layout),$(repeat)))))
else
$(foreach layout,$(LAYOUTS),$(foreach repeat,$(REPEATS),$(eval $(call SLURM_EXPS_template,$(layout),$(repeat)))))
endif
