CXX=c++
CXX_FLAGS= -std=c++17 -Wall -Wextra -O3 -march=native -g

LIKWID_BASE ?= $(shell dirname $(dir $(shell which likwid-perfctr)))
CXX_FLAGS += -DLIKWID_PERFMON -I$(LIKWID_BASE)/include
LDFLAGS := -L$(LIKWID_BASE)/lib-llikwid

all: list c2c list_access_latency
list: list.c
	${CXX} ${CXX_FLAGS} $< -o list 

c2c: c2c.c
	${CXX} -O2 -fopenmp $< -o c2c ${LDFLAGS}

clean:
	rm -f list c2c perf_*
