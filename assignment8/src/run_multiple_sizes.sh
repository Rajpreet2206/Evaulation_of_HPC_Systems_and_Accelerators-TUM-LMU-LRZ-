#!/bin/bash

sizes=( 1 2 4 8 16 32 64 128 256 512 1024 2048 4096 8192  )

for size in "${sizes[@]}"
do
    ./assignment8 $size >> results/3e_unpinned.txt
done

