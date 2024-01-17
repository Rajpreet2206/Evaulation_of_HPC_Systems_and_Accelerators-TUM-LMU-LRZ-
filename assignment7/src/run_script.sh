#!/bin/bash

module load gcc
module load likwid

export LD_LIBRARY_PATH=$(dirname $(dirname $(which likwid-perfctr)))/lib:$LD_LIBRARY_PATH

if [[ $HOSTNAME == "ice1" ]]; then
	module load compiler/latest
	ARCH="-march=native -mprefer-vector-width=512"
	#COMPLIST=(icpc g++)
	COMPILER=g++
elif [[ $HOSTNAME == "thx2.srve" ]]; then
	module load llvm
	COMPILER=g++
	ARCH="-march=native"
elif [[ $HOSTNAME == "milan2" ]]; then
	COMPILER=g++
	ARCH="-march=native"
else
	COMPILER=g++
	ARCH="-march=native"
fi

make clean
make CXX=$COMPILER list

# 2^30 = 1073741824
likwid-pin -c 23 ./list 1073741824 1073741824 > results/1c_${HOSTNAME}.txt
