
# --------------------------------------------------------------------
# 1) Check if user or environment set ISOLATED_CPUS. If so, compare it
#    to the system’s actual /sys/devices/system/cpu/isolated.
# --------------------------------------------------------------------

# For convenience, read the system’s actual isolated CPU string:
SYSTEM_ISOLATED_CPUS := $(shell cat /sys/devices/system/cpu/isolated 2>/dev/null)

ifeq ($(strip $(ISOLATED_CPUS)),)
  $(warning "ISOLATED_CPUS is not set — skipping isolation checks.")
else
  # Compare user-supplied ISOLATED_CPUS to what's in /sys/devices/system/cpu/isolated
  ifneq ($(strip $(ISOLATED_CPUS)),$(strip $(SYSTEM_ISOLATED_CPUS)))
    $(error "Mismatch: The system indicates isolated CPUs = '$(SYSTEM_ISOLATED_CPUS)' but you set ISOLATED_CPUS = '$(ISOLATED_CPUS)'")
  endif

  # --------------------------------------------------------------------
  # 2) Make sure that ISOLATED_MEMORY_NODE is set and that all isolated
  #    CPUs indeed belong to that node.
  # --------------------------------------------------------------------
  ifeq ($(strip $(ISOLATED_MEMORY_NODE)),)
    $(error "ISOLATED_MEMORY_NODE is not set, but ISOLATED_CPUS is set!")
  endif

  # Expand CPU ranges like "2-3,5" => "2 3 5"
  define expand_cpu_ranges
python3 -c "import sys
s = sys.argv[1]
expanded = []
for part in s.split(','):
    if '-' in part:
        start, end = part.split('-')
        for c in range(int(start), int(end)+1):
            expanded.append(str(c))
    else:
        expanded.append(part)
print(' '.join(expanded))" "$(1)"
  endef

  # Expand the user's isolated CPU list
  EXPANDED_ISOLATED := $(shell $(expand_cpu_ranges) "$(ISOLATED_CPUS)")

  # Now check each CPU’s node via symlink in /sys/devices/system/cpu/cpuX/node*
  # If any CPU is not on node ISOLATED_MEMORY_NODE, error out.
  CHECK_NODE_MISMATCH := $(shell \
    iso_list="$(EXPANDED_ISOLATED)"; \
    node_ok=1; \
    for core in $$iso_list; do \
      node_path="$$(readlink -f /sys/devices/system/cpu/cpu$$core/node* 2>/dev/null || true)"; \
      node_id="$${node_path##*node}"; \
      # If there's no node_path or node_id differs from our target node -> mismatch
      if [ -z "$$node_id" ] || [ "$$node_id" != "$(ISOLATED_MEMORY_NODE)" ]; then \
        node_ok=0; \
        echo "CPU $$core is on node $$node_id, not node $(ISOLATED_MEMORY_NODE)!" 1>&2; \
      fi; \
    done; \
    if [ "$$node_ok" = 0 ]; then echo "FAIL"; fi \
  )

  ifeq ($(CHECK_NODE_MISMATCH),FAIL)
    $(error "One or more isolated CPUs are not on node $(ISOLATED_MEMORY_NODE).")
  else
    $(info "All isolated CPUs match node $(ISOLATED_MEMORY_NODE).")
  endif
endif

