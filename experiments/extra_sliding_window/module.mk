EXTRA_SLIDING_WINDOW_MODULE_NAME := experiments/extra_sliding_window
EXTRA_SLIDING_WINDOW_WEIGHTS := 50 70 90

define extra_sliding-makefiles
EXTRA_SLIDING_WINDOW_WEIGHT := $(1)
include $(EXTRA_SLIDING_WINDOW_MODULE_NAME)/template.mk
endef

$(foreach w,$(EXTRA_SLIDING_WINDOW_WEIGHTS),$(eval $(call extra_sliding-makefiles,$(w))))
