MODULE_NAME := analysis/dynamic_auto_mosalloc
SUBMODULES := 

$(MODULE_NAME)% : NUM_OF_REPEATS := $(DYNAMIC_AUTO_MOSALLOC_NUM_OF_REPEATS)

include $(COMMON_ANALYSIS_MAKEFILE)
