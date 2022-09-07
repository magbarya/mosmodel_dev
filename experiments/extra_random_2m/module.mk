MODULE_NAME := experiments/extra_random_window_2m

include $(EXPERIMENTS_TEMPLATE)

$(LAYOUT_FILES): $(MEMORY_FOOTPRINT_FILE)
	$(CREATE_RANDOM_WINDOW_LAYOUTS) \
		--memory_footprint=$(MEMORY_FOOTPRINT_FILE) --num_layouts=$(NUM_LAYOUTS) \
		--output=$(dir $@)/..

