MODULE_NAME := experiments/dynamic_grouping

DYNAMIC_GROUPING_NUM_OF_REPEATS := $(NUMBER_OF_SOCKETS)
NUM_LAYOUTS := 9
NUM_OF_REPEATS := $(DYNAMIC_GROUPING_NUM_OF_REPEATS)
undefine LAYOUTS #allow the template to create new layouts based on the new NUM_LAYOUTS

include $(EXPERIMENTS_TEMPLATE)

CREATE_DYNAMIC_GROUPING_LAYOUTS_SCRIPT := $(MODULE_NAME)/createLayouts.py
$(LAYOUT_FILES): $(MEMORY_FOOTPRINT_FILE) analysis/pebs_tlb_miss_trace/mem_bins_2mb.csv
	$(CREATE_DYNAMIC_GROUPING_LAYOUTS_SCRIPT) \
		--memory_footprint=$(MEMORY_FOOTPRINT_FILE) \
		--pebs_mem_bins=$(MEM_BINS_2MB_CSV_FILE) \
		--layout=layout1 \
		--layouts_dir=$(dir $@)/..

override undefine NUM_LAYOUTS
override undefine NUM_OF_REPEATS
override undefine LAYOUTS