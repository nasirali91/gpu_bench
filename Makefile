# Makefile for GPU Benchmark

# Compiler and flags
NVCC = nvcc
NVCC_FLAGS = -gencode arch=compute_80,code=sm_80
CUDA_PATH ?= /usr/local/cuda
NVML_PATH ?= $(CUDA_PATH)

# Directories
SRC_DIR = src
BIN_DIR = bin
OBJ_DIR = obj

# Include and library paths
INCLUDES = -I$(CUDA_PATH)/include -I$(NVML_PATH)/include
LDFLAGS = -lcudart -lnvidia-ml -lcurand

# Target executable
TARGET = $(BIN_DIR)/gpu_bench

# Source files
SRC = $(wildcard $(SRC_DIR)/*.cu)

# If no source files found, provide a helpful error message
ifeq ($(SRC),)
    $(error No CUDA source files found in $(SRC_DIR) directory)
endif

# Object files
OBJ = $(patsubst $(SRC_DIR)/%.cu,$(OBJ_DIR)/%.o,$(SRC))

# Create directories if they don't exist
$(shell mkdir -p $(BIN_DIR) $(OBJ_DIR))

# Default target
all: $(TARGET)

# Rule to build the executable
$(TARGET): $(OBJ)
	$(NVCC) $(NVCC_FLAGS) $(INCLUDES) -o $@ $^ $(LDFLAGS)

# Rule to compile source files
$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cu
	$(NVCC) $(NVCC_FLAGS) $(INCLUDES) -c $< -o $@

# Clean target
clean:
	rm -rf $(BIN_DIR) $(OBJ_DIR) *.csv

# Run target with default parameters
run: $(TARGET)
	$(TARGET) 5 10 1000 1 80

# Run target with custom parameters
# Usage: make custom_run ARGS="<sampling_period> <num_runs> <delay_between_runs> <scale> <percent>"
custom_run: $(TARGET)
	$(TARGET) $(ARGS)

# Help target
help:
	@echo "GPU Benchmark Makefile"
	@echo "----------------------"
	@echo "Available targets:"
	@echo "  all        : Build the executable (default)"
	@echo "  clean      : Remove the executable and object files"
	@echo "  run        : Run the benchmark with default parameters"
	@echo "  custom_run : Run the benchmark with custom parameters"
	@echo "               Usage: make custom_run ARGS=\"<sampling_period> <num_runs> <delay_between_runs> <scale> <percent>\""
	@echo "  help       : Display this help message"
	@echo ""
	@echo "Source files: $(SRC)"
	@echo "Object files: $(OBJ)"
	@echo "Executable: $(TARGET)"
	@echo ""
	@echo "Example usage:"
	@echo "  make                          # Build the executable"
	@echo "  make run                      # Run with default parameters"
	@echo "  make custom_run ARGS=\"5 20 500 2 50\"  # Run with custom parameters"

# Phony targets
.PHONY: all clean run custom_run help
