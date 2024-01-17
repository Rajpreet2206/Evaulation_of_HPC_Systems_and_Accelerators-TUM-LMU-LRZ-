# Assignment 3 Report
Group: 104

## Rooflinemodel Ice1
For the roofline model, we need the theoretical max [L1D-cache bandwidth](https://en.wikichip.org/wiki/intel/microarchitectures/sunny_cove), as the vector can be entirely loaded from L1-cache:

- 2x 64 B/cycle load + 1x64 B/cycle store
- 2.4 GHz per core (probably overclocked to 3.1 GHz)
- Bandwidth =$(2*64 + 64) Bytes/cycle * 2,4 *10^9 cylce/s ≈ 460 GB/s$
- As mentioned in the [zulip discussion](https://zulip.in.tum.de/#narrow/stream/1911-BeastLab-2023WS/topic/L1-Bandwidth.20ice1), it is probably closer to 256 GB/s


We also need the calculate how many FLOP can be processed at once.
For this we use only 1 thread. As the SIMD-length is at 512 bit and we can use 2 FLOP per cycle (due to FMA) this results in $\frac{1024}{64}$  FLOP/cycle = 16 FLOP/cycle that could theoretically be processed.

Now we can get the theoretical FLOP/s per thread: 
$2,4*10^9\ cycle/s * 16\ FLOP/cycle = 38,4\ GFLOP/s$ (double that with both threads) 

So now we can calculate the Flops/Byte where L1D Bandwidth and theoretical max FLOP/s meet.

For this we can just divide the max GFLOP/s by the max bandwidth of the L1D-cache.

$$
I_{intersection}=\frac{38,4\ GFLOP/s}{460\ GB/s} \approx 0,08 FLOP/Byte
$$

This seems quite low, so a more suitable estimate for the operational intensity is probably:

$$
I_{intersection}=\frac{38,4\ GFLOP/s}{256\ GB/s} \approx 0,15 FLOP/Byte
$$

And with both threads:
$$
I_{intersection}=\frac{76,8\ GFLOP/s \times 2}{256\ GB/s} \approx 0,3 FLOP/Byte
$$

### 1 a) 
After running perf list we were most interested in two categories: 
* Hardware events 
* Hardware cache events

We opted to use the following events:
* **instructions**: We need an indicator of how many FLOP have been exectuted
* **L1-dcache-[loads, stores, load_misses]**: We have chosen a size of the vector ($2^{10}$) that will fit into the L1-cache, so all the L1 cache accesses are needed to calculate the computational intensity.

### 1 b)
We used this command to get all the different counts:
```
 perf stat -e instructions,L1-dcache-stores,L1-dcache-loads,L1-dcache-load-misses ./triad-ice1 10
```

Ice1 we got the following output:
```
N_max = 1024 (unaligned memory)
N,GFLOPS,Repetitions,Threads
128,24.17,67108864,144
256,54.20,67108864,144
512,49.81,33554432,144
1024,27.46,8388608,144

 Performance counter stats for './triad-ice1 10':

 2,004,489,702,715      instructions
   154,737,625,346      L1-dcache-stores
   838,036,436,364      L1-dcache-loads
     3,000,116,533      L1-dcache-load-misses     #    0.36% of all L1-dcache hits

       8.235739913 seconds time elapsed

     725.812912000 seconds user
       0.091993000 seconds sys

```

Resulting in a operational intensity $I$ of: 
$$
 I_{perfstat} = \frac{2,004,489,702,715}{154,737,625,346 + 838,036,436,364 + 3,000,116,533} \approx 2\  FLOP/Byte
$$

This is a overestimation as floating point instructions aren't the only instructions that are executed.

You can see from the roofline Model that it already seems to be computebound.

![Roofline model for ice1 - single thread](../src/img/ex01b_roofline_ice1.png)

### 1 c)
Adding all the vector percentages up using the annotate feature of perf report:

#### Event instructions:
98.05 % in triad: 2036635398489
vmovsd: 0.29 % + 3.00 % + 20.41 % + 0.01 % + 1.60 % + 5.82 % + 0.79 % + 1.77 % + 0.44 % + 0.49 % + 0.24 % + 0.55 % + 0.18 % + 0.02 % + 0.45 % + 0.22 % + 0.05 % + 0.38 % + 0.02 % + 0.02% = 36.75 %
vfmadd132sd: 7.13 % + 0.54 % + 0.05 % + 0.08 % + 0.08 % + 0.10% + 0.12 % + 0.02 % = 8.12 % 

Total flop operations $\approx$ 44.78 %
=> 2036635398489 * 0,4478 = 912005331443

#### Event L1-dcache-stores 
99.85 % is in triad: 157068116791
vmovsd + vfmadd132sd: 0.24 + 3.50 + 7.05 + 20.79 + 0.01 + 1.90 + 0.44 + 2.39 + 1.22 + 0.08 + 2.74 + 0.56 + 0.12 + 0.44 + 0.35 + 0.12 + 0.83 + 0.06 + 0.71 + 0.18 + 0.09 + 0.06 + 0.68 + 0.01 + 0.06 + 0.01 + 0.02 = 44,66 %

=> 157068116791 * 0.4466 = 70146620959

#### Event L1-dcache-loads
99.32% in triad: 844826156116
vmovsd + vfmadd132sd: 0.23 + 2.70 + 7.06 + 20 + 0.01 + 1.55 + +0.39 + 5.46 + 0.80 + 0.06 + 1.53 + 0.63 + 0.07 + 0.98 + 0.28 + 0.06 + 0.40 + 0.21 + 0.02 + 0.08 + 0.39 + 0.18 + 0.04 + 0.12+ 0.35 + 0.01 + 0.02 + 0.01 = 43.64 %

=> 844826156116 * 0.4364 = 368682134529

#### Event L1-dcache-load-misses
98.99 % in triad: 3422970486
vmovsd + vfmadd132sd: 0.49 + 11.33 + 24.24 + 7.41 + 5.21 + 4.32 + 3.62 + 6 + 1.44 + 1.31 + 3.76 + 0.63 + 0.06+0.03 +0.92 + 0.30 + 0.03 + 0.04 + 0.06 + 0.78 + 0.03 + 0.34 + 0.22 + 1.01 + 0.01 = 73.59 %

=> 3422970486 * 0.7359 = 2518963981

So accounting for the actual FLOP that were executed, the total Operational Intensity $I$ is:
$$
I_{perfrecord} = \frac{912005331443}{70146620959 + 368682134529 + 2518963981} = 2.06640997837 FLOP/Byte
$$

Increadibly, it is very similar to the $I_{perfstat}$. 

### 2 )
#### a )
The final result can be seen in ../src/triad.c

#### b ) 
To find out the groups:
```
likwid-perfctr -a
```
Which revealed the following output:
```
    Group name  Description
--------------------------------------------------------------------------------
           UPI  UPI data traffic
          DATA  Load to store ratio
        ENERGY  Power and Energy consumption
        BRANCH  Branch prediction miss rate/ratio
           TMA  Top down cycle allocation
           MEM  Memory bandwidth in MBytes/s
        MEM_DP  Overview of arithmetic and main memory performance
            L2  L2 cache bandwidth in MBytes/s
  CYCLE_STALLS  Cycle Activities (Stalls)
        MEM_SP  Overview of arithmetic and main memory performance
            L3  L3 cache bandwidth in MBytes/s
        DIVIDE  Divide unit information
   MEM_FREERUN  Memory bandwidth in MBytes/s
CYCLE_ACTIVITY  Cycle Activities
     FLOPS_AVX  Packed AVX MFLOP/s
       L2CACHE  L2 cache miss rate/ratio
      FLOPS_DP  Double Precision MFLOP/s
         CLOCK  Power and Energy consumption
      FLOPS_SP  Single Precision MFLOP/s
```

We can now measure the Bandwidth of the other Caches (L2-L3) but weirdly enough not L1. We can also look at the FLOPS in double and single precision.

So relevant groups would be: 
* L2
* L3
* FLOPS_DP
* MEM

### c ) 
lscpu reveals the size of the Caches:

```sh
lscpu
```
------------------
```
...
L1d cache:                       3.4 MiB
L1i cache:                       2.3 MiB
L2 cache:                        90 MiB
L3 cache:                        108 MiB
...
```

To fill the L1d cache, at most 3.4 MiB can be used.
So it can fill $\frac{3.4 * 2^{20}}{8} Floats= 445644\ Float$ 

So the last power we can fit is $2^{18}$.

For L2 it is a litte more difficult. First the L1d cache needs to be filled, then the L2 needs to be filled. This results in:

$\frac{93.4*2^{20}}{8} = 12242124 \ Float$
So we the last power of 2 that yields this is $2^{23}$.

For main memory we need to fill all 3 caches and the last power of 2 that can be held by the caches is $2^{24}$


So N will be:
* for L1 - N=10
* for L2 - N=20
* for L3 - N=24
* for main mem - N=26

We also disabled the loop in triad that multiplies N by 2 and just set N to N_max.

| Size | # Threads | #MFLOPS | #Memory BW | #L3 BW | L2 BW |
|------|-----------|--------|------------|--------|-------|
|  < L1  |      1     |   8916.48  |    57 MB/s      |  0.0936 MB/s   |1.82 MB/s   |
|  > L1   |     1      |  1876.44  |    61.43 MB/s   |  60068.94 MB/s |37253.07 MB/s |
|  > L2   |     1      |   912.84  |    13505 MB/s   |  29171.37 MB/s |18199.42 MB/s |
|  > L3   |     1      |   903.33  |    13760 MB/s   |  13760.44 MB/s | 17698.93 MB/s |
|  < sum L1 | 72 | 0.0001 | 132.45 MB/s | 1108 MB/s  | 1879.37 MB/s|
|  > sum L1 | 72 | 0.0601 | 173.18 MB/s | 1003.99 MB/s| 1894.29 MB/s|
|  > sum L2 | 72 | 0.9611 | 1305.84 MB/s | 1187.73 MB/s | 1907 MB/s|
|  > sum L3 | 72 | 3.5330 | 4826.15 MB/s | 1201.14 MB/s | 1887.46 MB/s |

#### d) Interpretation
Based on the FLOPs, the code used vector instructions for the multi core version of the code and used regular floating point operations on the single core version.

It seems that the Memory bandwidth seems to be the problem.

The system has a write-back policy. 

### 3) 
Running nsys nvprov yielded 2 different files:
* report1.nsys-rep
* report1.sqlite
Looking at sqlite file through a db viewer, nothing of value seems to be in there. This is also confirmed when later on running ```nsys analyze``` on the report files, as every test is being skipped.

Trying out ncu didn't even give us a report. 
This might be a problem with the GPU as the second GPU didn't even show up this time (nvidia-smi).

Problems continued on milan as the files that were returned didn't yield anything we could analyze.

### 4)
We used iprof ./executeable to get this table:
```
                     Name |     Time | Time(%) | Calls |  Average |      Min |      Max | Error | 
 cuDevicePrimaryCtxRetain | 151.71ms |  98.46% |     1 | 151.71ms | 151.71ms | 151.71ms |     0 | 
        cuMemAllocHost_v2 | 935.75us |   0.61% |     1 | 935.75us | 935.75us | 935.75us |     0 | 
     cuDeviceGetAttribute | 747.07us |   0.48% |    12 |  62.26us |    188ns | 739.13us |     0 | 
       cuModuleLoadDataEx | 247.10us |   0.16% |     1 | 247.10us | 247.10us | 247.10us |     0 | 
      cuModuleGetFunction | 164.28us |   0.11% |     3 |  54.76us |  11.14us |  96.81us |     0 | 
            cuMemAlloc_v2 | 120.67us |   0.08% |     5 |  24.13us |   2.39us | 101.20us |     0 | 
        cuMemAllocManaged |  39.30us |   0.03% |     1 |  39.30us |  39.30us |  39.30us |     0 | 
         cuMemsetD32Async |  36.62us |   0.02% |     2 |  18.31us |  10.73us |  25.89us |     0 | 
     cuMemcpyHtoDAsync_v2 |  20.01us |   0.01% |     2 |  10.01us |   6.79us |  13.22us |     0 | 
          cuDeviceGetName |  12.38us |   0.01% |     1 |  12.38us |  12.38us |  12.38us |     0 | 
            cuMemcpyAsync |   9.53us |   0.01% |     1 |   9.53us |   9.53us |   9.53us |     0 | 
   cuPointerGetAttributes |   8.36us |   0.01% |     3 |   2.79us |    670ns |   6.95us |     0 | 
          cuCtxGetCurrent |   6.55us |   0.00% |     1 |   6.55us |   6.55us |   6.55us |     0 | 
                   cuInit |   6.34us |   0.00% |     2 |   3.17us |   2.27us |   4.08us |     0 | 
          cuCtxSetCurrent |   5.46us |   0.00% |     4 |   1.36us |    443ns |   2.86us |     0 | 
       cuFuncGetAttribute |   4.82us |   0.00% |     1 |   4.82us |   4.82us |   4.82us |     0 | 
         cuGetErrorString |   2.35us |   0.00% |     1 |   2.35us |   2.35us |   2.35us |     0 | 
       cuDriverGetVersion |   1.77us |   0.00% |     3 | 589.00ns |    261ns |    781ns |     0 | 
         cuDeviceGetCount |   1.63us |   0.00% |     2 | 813.00ns |    781ns |    845ns |     0 | 
cuDeviceComputeCapability |   1.58us |   0.00% |     1 |   1.58us |   1.58us |   1.58us |     0 | 
              cuDeviceGet |   1.48us |   0.00% |     2 | 738.50ns |    498ns |    979ns |     0 | 
           cuLaunchKernel |          |   0.00% |     1 |          |          |          |     1 | 
                    Total | 154.09ms | 100.00% |    51 |                                      1 | 
```
