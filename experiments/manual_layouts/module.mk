MANUAL_LAYOUTS_EXPERIMENT_NAME := manual_layouts
MODULE_NAME := experiments/$(MANUAL_LAYOUTS_EXPERIMENT_NAME)

MANUAL_LAYOUTS_EXPERIMENT := $(MODULE_NAME)
MANUAL_LAYOUTS_RESULTS := $(subst experiments,results,$(MANUAL_LAYOUTS_EXPERIMENT))

MANUAL_LAYOUTS_DIR := $(MANUAL_LAYOUTS_EXPERIMENT_NAME)/layouts
# Gather all CSV files in the layouts directory and remove the .csv suffix
MANUAL_LAYOUTS := $(patsubst %.csv, %, $(wildcard $(MANUAL_LAYOUTS_DIR)/*.csv))
# Count the number of layout files
MANUAL_NUM_LAYOUTS := $(words $(LAYOUTS))

NUM_LAYOUTS := $(MANUAL_NUM_LAYOUTS)
LAYOUTS := $(MANUAL_LAYOUTS)

include $(EXPERIMENTS_TEMPLATE)

override undefine NUM_LAYOUTS
override undefine LAYOUTS
