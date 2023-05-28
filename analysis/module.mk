MODULE_NAME := analysis
SUBMODULES := \
	single_page_size \
	growing_window_2m \
	random_window_2m \
	sliding_window \
	extra_random_2m \
	linear_models_coeffs \
	pebs_tlb_miss_trace \
	subgroups_windows \
	subgroups_uniformly_windows \
	subgroups_head_pages \
	static_auto_mosalloc \
	dynamic_auto_mosalloc \
	moselect \
	bayesian_optimization \
	dynamic_grouping \
	runtime_range \
	genetic_scan \
	smart_genetic_scan \
	mosmodel \
	extra_data \
	all_data
SUBMODULES := $(addprefix $(MODULE_NAME)/,$(SUBMODULES))

ARRANGE_DATA_TO_PLOT := $(MODULE_NAME)/arrangeDataToPlot.py
SCATTER_PLOT := $(MODULE_NAME)/plotScatter.gp
WHISKER_PLOT := $(MODULE_NAME)/plotWhisker.gp
POLY_PLOT := $(MODULE_NAME)/assessPolynomialModels.py
BUILD_OVERHEAD := $(MODULE_NAME)/buildOverheadSummary.py

COMMON_ANALYSIS_MAKEFILE := $(MODULE_NAME)/common.mk

include $(ROOT_DIR)/common.mk

