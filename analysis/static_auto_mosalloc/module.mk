MODULE_NAME := analysis/auto_mosalloc
SUBMODULES := 

$(MODULE_NAME)% : NUM_OF_REPEATS := $(AUTO_MOSALLOC_NUM_OF_REPEATS)

include $(COMMON_ANALYSIS_MAKEFILE)
