## Excercise 1
### a)
Each iteration does a fused-multiply-add instruction, which increases the floating point operations per cycle by a factor of 2.
This results in the number of FLOPs being 2\*N and if you consider the repetitions R it would be 2\*N\*R.

### b)
#### i)
We have 3 loads, 1 store operation => 3 * 8 bytes (loads) + 8 bytes (store) => 32 bytes transfered per iterations.
We also have 2 FLOPs/iteration => 2 * x = 32 bytes/iteration => x = 16. 

With regards to the cache, this means that we normally need to allocate space for 4 floating points per iteration.

#### ii) 
Write-allocate means, that for each write space is allocated in the cache. If this were to be disabled, it would reduce the number of cache operations by one floating point.
This would reduce it to: 2 * x = 24 => factor: 12

### c)
#### i) 
The REP paramater repeats each triad loop REP times. As measurements aren't always consistent, averaging over the number of repetitions, makes the result more reliable. 

#### ii)
As there is no nowait clause for this loop, it also synchronizes the threads after each iteration of the repetition loop.

#### iii) 
As floating point operations are usually not commutative, it could be that the parallelization creates bug that is very hard to find. This can be used to double-check this.


### d)
#### i)
We should always use the _SC_LEVEL_1_DCACHE_LINESIZE as it can vary from system to system, so using a systemcall to figure it out helps. Also the linesize is important as the address should always be a multiple of the linesize. => adress % linesize =0 


#### ii)
The restrict keyword restricts the modification of memory space by restricting its access to only the single pointer that we used. This helps us to get robust behavior of memory/objects and their modification. Especially with multiple threads this can be useful.

### e)
#### i)
##### Effect of static
When we use static scheduling policy, the object is divided in a static manner. This means, if we have access to n threads, we divide the object (e.g. array) into n pieces, and the scheduling of each piece is done in an incremental fashion. So thread i is always connected to the pieces that lie at index % n = i.  

##### Effect of nowait clause
The default behavior of a for pramga is to synchronize threads after each iteration. This can add more time, if synchronization is not needed. So the nowait clause disables this synchronization. 


=> It is for measurement purposes, and to run the triad on different threads for all the REP values.

#### ii)
Because of the NUMA-nodes and the first touch policy. By making sure that each thread actually touches the memory that it will use later on first (by using static), we make sure that the first touch policy doesn't slow us down.


## Excercise 2

As our group is 104 and 104 % 4 = 0, we should use the milan2 system.
For the compiler we used the module: gcc/13.1.0-gcc-12.2.1-uxarjdm

### a) 
To make sure that the system is only utilizing one thread, we set the corresponding environment variable to 1. Then we compile the file, making sure that the CFLAGS are not accidentally set:

```sh
export OMP_NUM_THREADS=1
make CC=gcc CFLAGS="-fopenmp -march=native"
./traid-milan2 26 > results.csv
```

![Sequential](../src/logs/sequential_performance_a.6141.png)


### b) 
#### i)
Here further compilier optimizations are tested. For this we only adapt the CFLAGS to add the optimizations:

```sh
export OMP_NUM_THREADS=1
make CC=gcc CFLAGS="-fopenmp -march=native -O<X>"
./traid-milan2 26 > results.csv
```
where <X> is either:
- O2:
    ![Sequential](../src/logs/sequential_performance_b1_o2_results.png)
- O3:
    ![Sequential](../src/logs/sequential_performance_b1_o3_results.png)
- Ofast:
    ![Sequential](../src/logs/sequential_performance_b1_ofast_results.png)

As you can see the O2 flag performed the worst, and O3 and Ofast performed similar. We chose to use the O3 flag for the rest of the submission.
#### ii)

Now other compilers and further optimizations flags should be tested. But as most of the optimizations optimize for size or for debugging - we opted out to just test another compiler.

For this we used clang, where the system only offers one version. Trying to compile using clang was quite difficult as it didn't find the header file for omp.h. 

Resolving this issue took some steps. First we tried including the path of the gcc version of the omp.h header. For this we found it using the find command and adding it manually to the include-path.


```sh
find / -name omp.h 2>/dev/null
make CC=clang CFLAGS="-I/usr/lib64/gcc/x86_64-suse-linux/7/include/omp.h -fopenmp -march=native" 
```
This however, lead to further error-messages. So the gcc-version seems to not be compatible. 
Then there was also a llvm version and as clang is based on llvm, this made more sense to include:

```sh
make CC=clang CFLAGS="-I/opt/rocm-5.7.1/llvm/lib/clang/17.0.0/include/ -march=native -fopenmp -O<X>"
```

This finally worked and produced the following graphs:

![Sequential](../src/logs/sequential_performance_b1_o2_results_clang_comp.png)

![Sequential](../src/logs/sequential_performance_b1_o3_results_clang_comp.png)

![Sequential](../src/logs/sequential_performance_b1_ofast_results_clang_comp.png)

GCC seems to be better at almost every optimiziation we can see. Clang only outperforms it with -O2.
#### d)
![Sequential](../src/logs/2d.png)
## Excercise 3

### a) 
![Sequential](../src/logs/parallel_a/threads.png)
Here you can see that more threads usually means better performance, even though 128 and 256 seem similar. This is expected as we have 128 pyhsical cores with 2 threads each. So the thread parallelization only adds a little more performance. 
### b)

#### i)
![Sequential](../src/logs/parallel_b_i.png)
The scaling seems fine ans we hit a ceiling at around 128. 

#### ii)
![Sequential](../src/logs/parallel_b_ii/parallel_b_ii.png)
Similar things happened here but the rising didn't go as quickly as before. This can be because we now now prefer threads over cores.

#### iii)
![Sequential](../src/logs/parallel_b_iii/parallel_b_iii.png)
This yields very bad results and we only get a little more than 10 flops. Here you can see the problems of the first touch policy.

#### iv)
![Sequential](../src/logs/parallel_b_iv/parallel_b_iii.png)
This also creates a problem with the first touch policy and it is even more aggressive. 
#### v)
![Sequential](../src/logs/parallel_b_v/parallel_b_v.png)
Here you can see that is also not as good as we have to wait for each thread to synchronize.

