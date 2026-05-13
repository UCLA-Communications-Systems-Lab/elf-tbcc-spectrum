CXX = nvcc
CXXFLAGS := -shared -Xcompiler -fPIC -O3
GENCODE := -gencode arch=compute_75,code=sm_75 \
           -gencode arch=compute_80,code=sm_80 \
           -gencode arch=compute_89,code=sm_89 \
           -gencode arch=compute_89,code=compute_89

# parent directory containing the cgbn folder (may need to modify if not using colab)
CGBN_INCLUDE ?= /content/CGBN/include
GMP_LIB := -lgmp

SRC_DIR := compile
LIB_DIR := lib
LIB := $(LIB_DIR)/libfoldshift.so

SRC_STD := $(SRC_DIR)/distance_spectrum.cu
SRC_CGBN := distance_spectrum_cgbn.cu

CONFIG ?=
K_VAL = $(shell python3 -c "import yaml; print(yaml.safe_load(open('$(CONFIG)'))['bch_config']['K'])")

.PHONY: all clean run compile check_config compile-kernel compile-cgbn

all: run

check_config:
ifndef CONFIG
	$(error CONFIG is required. Example usage: make compile CONFIG=config/k31n62v6.yaml)
endif

compile: check_config | $(LIB_DIR)
	@if [ $(K_VAL) -gt 64 ]; then \
		$(MAKE) compile-cgbn; \
	else \
		$(MAKE) compile-kernel; \
	fi

compile-kernel:
	$(CXX) $(CXXFLAGS) $(GENCODE) $(SRC_STD) -o $(LIB)

compile-cgbn:
	$(CXX) $(CXXFLAGS) $(GENCODE) -I$(CGBN_INCLUDE) $(GMP_LIB) $(SRC_CGBN) -o $(LIB)

run: compile
	@if [ $(K_VAL) -gt 64 ]; then \
		echo "K = $(K_VAL) > 64, running CGBN kernel with $(CONFIG)"; \
		python3 test_cgbn.py $(CONFIG); \
	else \
		echo "K = $(K_VAL) <= 64, running regular kernel with $(CONFIG)"; \
		python3 test_foldshift.py $(CONFIG); \
	fi

$(LIB_DIR):
	mkdir -p $(LIB_DIR)

clean:
	rm -f $(LIB)