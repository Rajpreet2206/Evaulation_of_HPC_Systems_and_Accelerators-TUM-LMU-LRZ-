## Assignment 5 - MxM on GPU
GROUP: 104

#### 1) Offloading with Optimal Data Transfer Policy
The data initialization **on** GPUs is considered because it gave the best performance as compared to 
initilization on CPUs. On milan2, the usage of AMD GPUs were confirmed by using the command
```
watch -n1 rocm-smi
```
while running the code from command line.
There are 2 nvidia A100 GPUs on ice1 system, therefore we use
```
watch -n1 nvidia-smi
```
to observde the gpu utilization while running the code.
The M X M computation if offloaded to GPUs correctly. To initialize on GPUs, we used the following code snippet
```cpp
      a[TWO_D_ACCESS(i,j,N)] = 0;// Initialize on GPUs
      b[TWO_D_ACCESS(i,j,N)] = (i*j)/(N*50);// Initialize on GPUs
      c[TWO_D_ACCESS(i,j,N)] = (i*j)/(N*50); // Initialize on GPUs
```
#### 2) Optimizations
### Variant1: 
To store the elements of multiplicands B and C in row-major layout, the initialization shown in the previous part is already able to do that.
```cpp
b[TWO_D_ACCESS(i,j,N)] = (i*j)/(N*50); // Initialize on GPUs
c[TWO_D_ACCESS(i,j,N)] = (i*j)/(N*50); // Initialize on GPUs
```
### Variant2:
For a row-major layout of matrix B and column major layout of matrix C, the initialization with `(i*N)+j` and `(j*N)+i` can be used respectively.
```cpp
// Initialize matrix B with row-major layout
b[TWO_D_ACCESS(i, j, N)] = (i * N) + j;
// Initialize matrix C with column-major layout
c[TWO_D_ACCESS(i, j, N)] = (j * N) + i;
```
#### 2a) Applying Cache Blocking
The cache blocking or loop tiling is used here to enhance cache locality and improve the performance of the matrix multiplication kernel using the `mm_kernel` function.
The code uses four nested loops to iterate over the matrices for multiplication. 

For column major vs. row major we don't really have to rewrite the initialization as 
indices (i,j) and (j,i) contain the same value. But just out of completeness, we adapted it in the following way:

```c
      // c[i*N + j] = (i*j)/(N*50);  // save C in row major order
      c[j*N + i] = (i*j)/(N*50);  // save C in column major order
```

We also have to adjust the access in the multiplication:

```c
#define TWO_D_ACCESS_COLMAJOR(row, col, width) ((width) * (col) + (row))
```

This alone increases the FLOPS on milan:
![Row major vs col major](../src/results/ex02/without_any_improvements/col_vs_row_major_init.png)


To add cache blocking, we copy over the code from assignment 4:

```c
for( int i=0; i<N; i+=TILE_SIZE ) {
  for( int k=0; k<N; k+=TILE_SIZE ) {
    for( int j=0; j<N; j+=TILE_SIZE ) {
      for(int ii=0; ii<TILE_SIZE; ++ii) {
        for(int kk=0; kk<TILE_SIZE; ++kk) {
          for(int jj=0; jj<TILE_SIZE; ++jj) {
            // row-major access
            // a[TWO_D_ACCESS(i+ii, j+jj, N)] += b[TWO_D_ACCESS(i+ii, k+kk, N)] * c[TWO_D_ACCESS(k+kk, j+jj, N)]; 
            
            // col-major access
            a[TWO_D_ACCESS(i+ii, j+jj, N)] += b[TWO_D_ACCESS(i+ii, k+kk, N)] * c[TWO_D_ACCESS_COLMAJOR(k+kk, j+jj, N)]; 
          }
        }
      }
    }
  }
}

```

Actually using this without collapse yieled very bad results and adding collapse will yield wrong results. So we adjusted the code the following way to get a correct result:
 ```c
  #pragma omp distribute parallel for schedule(static) collapse(4) num_threads(numThreads)
  for( int i=0; i<N; i+=TILE_SIZE ) {
    for( int j=0; j<N; j+=TILE_SIZE ) {
      for(int ii=0; ii<TILE_SIZE; ++ii) {
        for(int jj=0; jj<TILE_SIZE; ++jj) {
          for( int k=0; k<N; k+=TILE_SIZE ) {
            for(int kk=0; kk<TILE_SIZE; ++kk) {
              if (rowMajor){
                a[TWO_D_ACCESS(i+ii, j+jj, N)] += b[TWO_D_ACCESS(i+ii, k+kk, N)] * c[TWO_D_ACCESS(k+kk, j+jj, N)]; 
              } else {
                a[TWO_D_ACCESS(i+ii, j+jj, N)] += b[TWO_D_ACCESS(i+ii, k+kk, N)] * c[TWO_D_ACCESS_COLMAJOR(k+kk, j+jj, N)]; 
              }
            }
          }
        }
      }
    }
  }
 ```
 This worked because it stopped different threads overwriting old results. This is why the original code also uses ```collapse(2)``` and not ```collapse(3)```.

 As most of the matrix sizes are a multiple of 10 and 50, these are the only TILE_SIZES that we considered.

The results can be seen here:
![ex02a](../src/img/ex02a_gpu_colMajor_tiled_50.png)

rowMajor with tileSize 50 it was the quickest.


#### 2b) Appropriate Loop Scheduling Policies and Vectorization Directives
On ice1 system, only static schedule kind is supported for GPU. So, we used `schedule(static,1)` for loop scheduling of our code to achieve coallesced memory accesses.

```cpp
void mm_kernel(double* a, double* b, double* c, int N, int numTeams, int numThreads, int blockSize) {
    #pragma omp target
    #pragma omp teams num_teams(numTeams) thread_limit(numThreads)
    #pragma omp distribute parallel for schedule(static,1) collapse(2) num_threads(numThreads)  
    for (int i0 = 0; i0 < N; i0 += blockSize) {
        for (int j0 = 0; j0 < N; j0 += blockSize) {
            for (int k0 = 0; k0 < N; k0 += blockSize) {
                #pragma omp simd
                for (int i = i0; i < std::min(i0 + blockSize, N); ++i) {
                    #pragma omp simd
                    for (int j = j0; j < std::min(j0 + blockSize, N); ++j) {
                        for (int k = k0; k < std::min(k0 + blockSize, N); ++k) {
                            a[TWO_D_ACCESS(i, j, N)] += b[TWO_D_ACCESS(i, k, N)] * c[TWO_D_ACCESS(j, k, N)];
                        }
                    }
                }
            }
        }
    }
}
```

The same code did not work on Milan2 System because of compiler's inability to vertorize using this pragma `#pragma omp simd`

Also as seen in the codesnipped in 2a) the collapse could be expanded to 4 if we just move the k loop further down.

But all in all, the directives that are used are already very sensible.
Especially scheduling static is the that we could adjust. Even though static,1  is supposed to do a coallesced access, chaning the the splitting parameter to a number bigger than 1, gave good results in exercise 2.

![ex02b](../src/img/ex02b_gpu_rowMajor_tiled_50_static_1.png)

As you can see the changing the splitting argument to something different, actually worsened the outcome. So static, 1 will be used.

#### 2c) Measurement of FLOP rates and memory bandwidth utilization of our code
![Sequential](../src/2c.png)
![Sequential2](../src/2c_memory_bw.png)

The plots below are for Milan2 System. The zig-zag behaviour is not intuitive for us. This phenomenon is observed for both the systems ice1 and milan2. The possible explanation is about the un-optimized `numTeams` and `numThreads`. Because in the later cases, we observe linear improvement with matrix size N.
![Sequantial3](../src/2c_milan2.png)
![Sequantial4](../src/2c_milan2_memory_bw.png)


#### 2d) Results comparison of M X M between CPU and GPU
Without any optimization of loop iterations, (i.e. the best version **ikj**) we are able to surpass the CPU's best performance by a large factor.
 The highest MFLOPS was reached at around 8000 on CPU but with the GPU offloading, we are able to reach **50000** MFLOPS.  


#### 3) Execution Configuration on Target Device
### 3a) 
![Sequential](../src/3a.png)
![Sequential](../src/3a_memory_bw.png)
The parallelism is achieved by using OpenMP clauses on the GPUs by setting the number of teams = 100 and the number of threads = 128. This was also monitored with `smi` commands.

### 3b) League and Team size combinations from the following set = {(t,T)} = {8,16,32,40,48,64,72,80,88,96,104,112,120,128,256,512,1024} X {32,64,80,128,256,512,1024}

The most interesting matrix size was N=1750 that we selected for each thread and team size combination, as stated in the task.
Making the observation from the scatter plot for various teams and threads combinations, the optimum team number vs thread number configuration at which we get the maximum performance is **(512, 32)**. The MFLOPS for this optimal configuration is around 97000 MFLOPS.
The results in the plot are without vectorization, i.e. without `#pragma omp simd`.
![Sequential](../src/3b.png)

![Sequential](../src/3b_simd.png)
The `#pragma omp simd` still gives the optimum configuration as **(512, 32)** along with the similar maximum performance in MFLOPS i.e. 97000 MFLOPS. 

### 3c)
`Cache Utilization`: The (512, 32) configuration is effective in maximizing cache usage, leading to better performance.

`Vectorization`: The observed configuration might be conducive to efficient vectorization of the computation.

`Memory Bandwidth`: The (512, 32) configuration aligns well with the memory architecture of the Intel ICE machine (ice1), reducing memory access latency and improving overall throughput.

`Thread Synchronization`: This specific configuration minimizes contention and synchronization overhead among threads, allowing for better parallelization.

`Task Granularity`: The configuration could be well-suited to the granularity of tasks assigned to each team and thread. If the tasks are too fine-grained, there could be increased overhead, whereas if they are too coarse-grained, there might be load imbalance.

![Sequantial](../src/3b_simd2.png)

###)
The plot above is with a `blockSize=2` which showed best performance along with `#pragma omp simd` but without various teams and threads combinations. `It has crossed **165 GFLOPS** for the configuration **(512, 80)**.` Hence, this is the best performance for Matrix Multiplication on ice1 considering **Variant1** according to the assignment.  

