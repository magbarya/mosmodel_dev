MODULE_NAME := experiments/vanilla
VANILLA_LAYOUTS ?= all_4kb
LAYOUTS := $(VANILLA_LAYOUTS)

VANILLA_EXPERIMENT := $(MODULE_NAME)

VANILLA_RUN = 1
include $(EXPERIMENTS_TEMPLATE)
undefine VANILLA_RUN

$(MODULE_NAME)/clean:
	rm -rf experiments/vanilla/all_4kb

