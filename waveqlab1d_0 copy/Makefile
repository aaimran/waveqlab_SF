# =============================================================================
# waveqlab1d — Build & Install Makefile
#
# Usage:
#   make install       Build libffi + Python 3.13.3, create venv, install pkgs
#   make verify        Source env and run the baseline simulation test
#   make clean-build   Remove source build trees (keeps installed prefixes)
#   make uninstall     Remove all installs and venv (asks for confirmation)
#   make help          Show this help
#
# All paths are configured in the Variables section below.
# =============================================================================

# -----------------------------------------------------------------------------
# Variables
# -----------------------------------------------------------------------------

# Root directory for all installations (must be writable)
INSTALL_ROOT    := /work/aimran

# libffi
LIBFFI_VERSION  := 3.4.6
LIBFFI_TARBALL  := $(INSTALL_ROOT)/libffi-$(LIBFFI_VERSION).tar.gz
LIBFFI_SRC      := $(INSTALL_ROOT)/libffi-$(LIBFFI_VERSION)
LIBFFI_PREFIX   := $(INSTALL_ROOT)/libffi

# Python
PY_VERSION      := 3.13.3
PY_TARBALL      := $(INSTALL_ROOT)/Python-$(PY_VERSION).tgz
PY_SRC          := $(INSTALL_ROOT)/Python-$(PY_VERSION)
PY_PREFIX       := $(INSTALL_ROOT)/python3.13
PY_BIN          := $(PY_PREFIX)/bin/python3.13
PY_DYNLOAD      := $(PY_PREFIX)/lib/python3.13/lib-dynload

# Virtual environment
VENV_DIR        := $(INSTALL_ROOT)/wql1d
VENV_PY         := $(VENV_DIR)/bin/python
VENV_PIP        := $(VENV_DIR)/bin/pip
VENV_ENV        := $(VENV_DIR)/env.sh

# pip cache (avoids re-downloading on reinstall)
PIP_CACHE       := /scratch/aimran/pip-cache

# Package versions (pinned for reproducibility)
PKG_NUMPY       := numpy==2.4.4
PKG_NUMBA       := numba==0.65.0
PKG_LLVMLITE    := llvmlite==0.47.0
PKG_MATPLOTLIB  := matplotlib==3.10.8

# Simulation code
SIM_DIR         := /scratch/aimran/FNO/waveqlab_SF/waveqlab1d
SIM_SCRIPT      := rupture_1d.py
SIM_INPUT       := input/rupture_1d_SW.in

NPROC           := $(shell nproc)

# Colours for output
GREEN  := \033[0;32m
YELLOW := \033[0;33m
RESET  := \033[0m

# -----------------------------------------------------------------------------
# Phony targets
# -----------------------------------------------------------------------------

.PHONY: all install libffi python venv packages verify \
        clean-build uninstall help

all: install

# -----------------------------------------------------------------------------
# Top-level install target
# -----------------------------------------------------------------------------

install: libffi python venv packages env-script
	@echo ""
	@printf "$(GREEN)=== Installation complete ===$(RESET)\n"
	@echo ""
	@echo "  Activate:  source $(VENV_ENV)"
	@echo "  Test:      make verify"
	@echo ""

# -----------------------------------------------------------------------------
# Step 1 — libffi
# -----------------------------------------------------------------------------

$(LIBFFI_PREFIX)/lib64/libffi.so: $(LIBFFI_TARBALL)
	@printf "$(YELLOW)>>> Building libffi $(LIBFFI_VERSION)$(RESET)\n"
	@test -d $(LIBFFI_SRC) || tar -xzf $(LIBFFI_TARBALL) -C $(INSTALL_ROOT)
	cd $(LIBFFI_SRC) && ./configure \
	    --prefix=$(LIBFFI_PREFIX) \
	    --with-pic \
	    2>&1 | tail -3
	cd $(LIBFFI_SRC) && $(MAKE) -j$(NPROC)
	cd $(LIBFFI_SRC) && $(MAKE) install
	@printf "$(GREEN)>>> libffi installed at $(LIBFFI_PREFIX)$(RESET)\n"

libffi: $(LIBFFI_PREFIX)/lib64/libffi.so
	@test -f $< && echo "  libffi.so present: $<" || (echo "ERROR: libffi .so missing" && exit 1)

# -----------------------------------------------------------------------------
# Step 2 — Python 3.13.3
# -----------------------------------------------------------------------------

$(PY_BIN): $(PY_TARBALL) $(LIBFFI_PREFIX)/lib64/libffi.so
	@printf "$(YELLOW)>>> Building Python $(PY_VERSION)$(RESET)\n"
	@test -d $(PY_SRC) || tar -xzf $(PY_TARBALL) -C $(INSTALL_ROOT)
	cd $(PY_SRC) && ./configure \
	    --prefix=$(PY_PREFIX) \
	    --with-ensurepip=install \
	    --enable-shared \
	    LDFLAGS="-L$(LIBFFI_PREFIX)/lib64 -Wl,-rpath,$(LIBFFI_PREFIX)/lib64 -Wl,-rpath,$(PY_PREFIX)/lib" \
	    CPPFLAGS="-I$(LIBFFI_PREFIX)/include" \
	    PKG_CONFIG_PATH="$(LIBFFI_PREFIX)/lib/pkgconfig:$(LIBFFI_PREFIX)/lib64/pkgconfig" \
	    2>&1 | tail -3
	cd $(PY_SRC) && $(MAKE) -j$(NPROC)
	@# --- workaround: _sha2 fails to link on this system (HACL* issue).
	@# --- Copy all successfully built extensions manually into lib-dynload.
	@mkdir -p $(PY_DYNLOAD)
	@for f in $(PY_SRC)/build/lib.linux-x86_64-3.13/*.so; do \
	    base=$$(basename $$f); \
	    cp -f "$$f" "$(PY_DYNLOAD)/$$base" && echo "  installed $$base"; \
	done
	@printf "$(GREEN)>>> Python $(PY_VERSION) installed at $(PY_PREFIX)$(RESET)\n"

python: $(PY_BIN)
	@LD_LIBRARY_PATH=$(PY_PREFIX)/lib:$(LIBFFI_PREFIX)/lib64 \
	    $(PY_BIN) -c "import _ctypes; print('  _ctypes OK')"

# -----------------------------------------------------------------------------
# Step 3 — Virtual environment
# -----------------------------------------------------------------------------

$(VENV_PY): $(PY_BIN)
	@printf "$(YELLOW)>>> Creating venv at $(VENV_DIR)$(RESET)\n"
	@$(PY_BIN) -m venv $(VENV_DIR)
	@printf "$(GREEN)>>> venv created$(RESET)\n"

venv: $(VENV_PY)

# -----------------------------------------------------------------------------
# Step 4 — Install Python packages
# -----------------------------------------------------------------------------

$(VENV_DIR)/.packages-installed: $(VENV_PY)
	@printf "$(YELLOW)>>> Installing Python packages$(RESET)\n"
	LD_LIBRARY_PATH=$(PY_PREFIX)/lib:$(LIBFFI_PREFIX)/lib64 \
	    $(VENV_PIP) install \
	    --cache-dir $(PIP_CACHE) \
	    $(PKG_NUMPY) $(PKG_NUMBA) $(PKG_LLVMLITE) $(PKG_MATPLOTLIB)
	@touch $@
	@printf "$(GREEN)>>> Packages installed$(RESET)\n"

packages: $(VENV_DIR)/.packages-installed
	@LD_LIBRARY_PATH=$(PY_PREFIX)/lib:$(LIBFFI_PREFIX)/lib64 \
	    $(VENV_PY) -c "import numpy, numba, matplotlib; \
	    print('  numpy', numpy.__version__, \
	          ' numba', numba.__version__, \
	          ' matplotlib', matplotlib.__version__)"

# -----------------------------------------------------------------------------
# Step 5 — Write env.sh activation script
# -----------------------------------------------------------------------------

env-script: $(VENV_DIR)/env.sh

$(VENV_DIR)/env.sh: $(VENV_DIR)/.packages-installed
	@printf "$(YELLOW)>>> Writing env.sh$(RESET)\n"
	@printf '%s\n' \
	    '#!/usr/bin/env bash' \
	    '# wql1d environment setup — source this file before running simulations:' \
	    '#   source $(VENV_DIR)/env.sh' \
	    '' \
	    'export WQL1D_ROOT=$(VENV_DIR)' \
	    'export PYTHON_ROOT=$(PY_PREFIX)' \
	    'export LIBFFI_ROOT=$(LIBFFI_PREFIX)' \
	    '' \
	    'export PATH="$${WQL1D_ROOT}/bin:$${PYTHON_ROOT}/bin:$${PATH}"' \
	    'export LD_LIBRARY_PATH="$${PYTHON_ROOT}/lib:$${LIBFFI_ROOT}/lib64$${LD_LIBRARY_PATH:+:$$LD_LIBRARY_PATH}"' \
	    '' \
	    'export NUMBA_CACHE_DIR="$${WQL1D_ROOT}/.numba_cache"' \
	    'export NUMBA_NUM_THREADS="$${NUMBA_NUM_THREADS:-$$(nproc)}"' \
	    'export OMP_NUM_THREADS="$${OMP_NUM_THREADS:-$$(nproc)}"' \
	    '' \
	    'alias python="$${WQL1D_ROOT}/bin/python"' \
	    'alias pip="$${WQL1D_ROOT}/bin/pip"' \
	    '' \
	    'echo "wql1d env active -- Python $$($${WQL1D_ROOT}/bin/python --version 2>&1 | cut -d'"'"' '"'"' -f2), numba $$($${WQL1D_ROOT}/bin/python -c '"'"'import numba; print(numba.__version__)'"'"' 2>/dev/null), threads=$${NUMBA_NUM_THREADS}"' \
	    > $@
	@chmod +x $@
	@printf "$(GREEN)>>> env.sh written at $@$(RESET)\n"

# -----------------------------------------------------------------------------
# verify — run the baseline simulation
# -----------------------------------------------------------------------------

verify:
	@printf "$(YELLOW)>>> Running baseline simulation$(RESET)\n"
	cd $(SIM_DIR) && \
	    LD_LIBRARY_PATH=$(PY_PREFIX)/lib:$(LIBFFI_PREFIX)/lib64 \
	    NUMBA_NUM_THREADS=1 \
	    $(VENV_PY) $(SIM_SCRIPT) $(SIM_INPUT)
	@printf "$(GREEN)>>> Baseline simulation passed$(RESET)\n"

# -----------------------------------------------------------------------------
# clean-build — remove only build trees, keep installs
# -----------------------------------------------------------------------------

clean-build:
	@printf "$(YELLOW)>>> Removing build trees$(RESET)\n"
	rm -rf $(LIBFFI_SRC) $(PY_SRC)
	@printf "$(GREEN)>>> Build trees removed$(RESET)\n"

# -----------------------------------------------------------------------------
# uninstall — remove everything (asks for confirmation)
# -----------------------------------------------------------------------------

uninstall:
	@echo "WARNING: This will permanently delete:"
	@echo "  $(LIBFFI_PREFIX)"
	@echo "  $(PY_PREFIX)"
	@echo "  $(VENV_DIR)"
	@read -p "Type YES to confirm: " confirm && [ "$$confirm" = "YES" ]
	rm -rf $(LIBFFI_PREFIX) $(PY_PREFIX) $(VENV_DIR)
	@printf "$(GREEN)>>> Uninstall complete$(RESET)\n"

# -----------------------------------------------------------------------------
# help
# -----------------------------------------------------------------------------

help:
	@echo ""
	@echo "waveqlab1d installation targets:"
	@echo ""
	@echo "  make install       Build libffi + Python $(PY_VERSION), create wql1d venv, install packages"
	@echo "  make libffi        Build and install libffi $(LIBFFI_VERSION) only"
	@echo "  make python        Build and install Python $(PY_VERSION) only (requires libffi)"
	@echo "  make venv          Create the wql1d virtual environment only"
	@echo "  make packages      Install Python packages into the venv only"
	@echo "  make env-script    (Re)write the env.sh activation script"
	@echo "  make verify        Run the baseline SW rupture simulation"
	@echo "  make clean-build   Remove libffi and Python source build trees"
	@echo "  make uninstall     Remove all installed prefixes and the venv"
	@echo ""
	@echo "Configuration (edit at top of Makefile):"
	@echo "  INSTALL_ROOT  = $(INSTALL_ROOT)"
	@echo "  PY_VERSION    = $(PY_VERSION)"
	@echo "  LIBFFI_VER    = $(LIBFFI_VERSION)"
	@echo "  VENV_DIR      = $(VENV_DIR)"
	@echo "  SIM_DIR       = $(SIM_DIR)"
	@echo ""
