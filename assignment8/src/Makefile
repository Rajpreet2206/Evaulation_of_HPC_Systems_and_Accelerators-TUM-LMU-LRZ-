NODE = $(shell uname -n | cut -c1-3)
# ThunderX  or icelake systems



ifeq ($(NODE),ice) || ($(NODE),thx)  
	#@module load cuda/11.1.1
	CXX = nvcc
	CXX_FLAGS = -std=c++17 -arch=sm_70 -Xcompiler -fopenmp
else # Rome systems
	CXX = hipcc
	CXX_FLAGS = -std=c++17 -Wall -Wextra --offload-arch=gfx908
endif

# LIKWID_BASE ?= $(shell dirname $(dir $(shell which likwid-perfctr)))
# CXX_FLAGS += -DLIKWID_PERFMON -I$(LIKWID_BASE)/include
# LDFLAGS := -L$(LIKWID_BASE)/lib -llikwid

# ThunderX  or icelake systems
ifeq ($(NODE),ice) || ($(NODE),thx)  
all: assignment8

assignment8: assignment8.cu
	${CXX} ${CXX_FLAGS} $< -o $@ ${LDFLAGS}
else # Rome systems
all: assignment8_hip

assignment8_hip: assignment8_hip.cu
	${CXX} ${CXX_FLAGS} $< -o $@
endif 

# For this assignment we don't have CI tests.

clean:
ifeq ($(NODE),ice) || ($(NODE),thx) #ThunderX or icelake systems
	rm -f assignment8
else # Rome systems
	rm -f assignment8_hip
endif 
