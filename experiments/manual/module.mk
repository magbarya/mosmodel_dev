MODULE_NAME := experiments/manual

LAYOUTS := $(shell cd $(MODULE_NAME)/layouts && ls *.csv | cut -d . -f 1)
NUM_LAYOUTS := $(shell cd $(MODULE_NAME)/layouts && ls *.csv | wc -l)

include $(EXPERIMENTS_TEMPLATE)

undefine NUM_LAYOUTS
undefine LAYOUTS

