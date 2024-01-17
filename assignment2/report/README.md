# Assignment 4 Report
Group: 104

## 1) Compiler usage for offloading

As milan2 uses AMD gpus we can use the command 
```bash
watch -n1 rocm-smi 
```
to observe the gpu usage while running the code in another terminal.

There are 2 nvidia A100 GPUs on the ice1 system and therefore we use
```bash
watch -n1 nvidia-smi
```
to see the gpu utilization.

Both of them showed full gpu utilization of a single gpu. Thanks to openmp offloading, the code ran on both GPU architectures without changing the code. 

## 2) Workload Distribution

The number of threads is limited by the parameters ```numTheads``` and to use all the threads one could remove this limitation. We didn't change this but we figured out the maximum number of threads by using ```omp_get_max_threads()```. We left the number of threads at their default value (32). To make sure that the all fixed gpu threads are used, we can introduce a schedule. For this we opted for ```schedule(static,1)``` as this will enable coalesced memory access. 

Later on we figurued out, that we can use the ```distribute``` keyword to distribute the workload to the teams, which gave a huge increase in performance.

## 3) Evaluation Data Transfer

For exercise 3, we had to try out the different variants of data initialization. 
Variant 1 initializes the data on the CPU - this is done by default.
Variant 2 however initializes the vectors on the device. For this we used the following code:
```cpp
#pragma omp target enter data map(alloc: a[0:datasetSize], b[0:datasetSize], c[0:datasetSize], d[0:datasetSize])
#pragma omp target 
for (unsigned long i = 0; i < datasetSize; i++) {
    a[i] = b[i] = c[i] = d[i] = i;
}
```
After the computation is done on the GPU, we used the following pragma to copy the vector back to RAM and delete the unneeded vectors from VRAM.

```cpp
#pragma omp target exit data map(from: a[0:datasetSize])
#pragma omp target exit data map(delete: b[0:datasetSize], c[0:datasetSize], d[0:datasetSize])
```

The results were not quite astounding, as different variants don't seem to make a difference.

![Both variants of the initialization](../src/finished/ex03/ex03.png)

One can see, that initializing on the GPU is a little better than on the CPU. This intuitively makes sense, as we spare time copying data from and to the device. We even expected a bigger difference here. Maybe with the pragma distribute directive this would make a bigger difference, but we didn't have time to check this assumption.

As we seemed to have better performance using variant 2, we kept initializing the data on the GPU for the rest of the exercises (except exercise 7 (multigpu)) 
### 4) Loop Scheduling Policy
For this exercise we were supposed to test the effects of the scheduling policy that enables coalesced memory access. For this we either added the schedule(static,1) directive or we didn't:
```c
#pragma omp parallel for num_threads(numThreads) schedule(static, 1)
```

The results can be seen here:

![Both variants of the initialization](../src/finished/ex04/ex04.png)

Adding the directive didn't really change the performance of the triad function. This indiciates that the coalesced memory access is actually the default. 

## 5) Execution Configuration on Target Device
### a) 
Here we figured out that the ```distribute``` directive actually gives a huge 
![distributed directive](../src/finished/ex05_b_teams/ex05_distributed_vs_not_distributed.png).
The memory bandwidth is now in TB/s! (indicated by the 1e6 ontop). We kept using this directive after and including exercise 5c). As the distributed keyword is distributing the workload to each team, we decided against rerunning it for 5b), as there the number of teams if fixed to one. 

### b) 
Fixing the number of teams to 1 and changing the number of threads lead to the following graphs:

For ice1:
![ice fix teams but vary threads](../src/finished/ex05_b/ex03_ice.png)
Here after 256 teams are reached, the performance stays stable. So the number of teams t* will be fixed to 256 after this. Using ```omp_get_max_threads();``` this wasn't surprising as this yielded: 384. So after 256 the performance should either increase slightly or stay the same.

For milan2:
![milan fix teams but vary threads](../src/finished/ex05_b/ex03_milan.png)
For milan2 this number is even lower: 193. You can see that after 256, the performance doesn't further increase.

So from now on the threads are fixed to 256 for both machines.

### c)
Here we have to change the number of teams.
This can be seen here:

- Ice1:
![ice varying teams](../src/finished/ex05_distributed_teams/ex05_ice1_teams.png)

It seems like after 80 teams, the performance didn't really further increase. As there is no real definition of what a team actually is, this might just reflect the number of thread blocks that are available.

- Milan2:
![ice varying teams](../src/finished/ex05_distributed_teams/ex05_milan2_teams.png)

Here it doesn't actually increase later, after 160 teams. Which might also reflect the number of accessible threadblocks. 

### d)
For t* we use 256 on both machines. And for T* we use 200, even though they didn't actually have a lot more performance, they seemed to be on top of all of the evaluations.

## 6) 
We have tested multiple changes for cpu vs. gpu. But for all of them, the GPU outperformes it by a large margain. Even so much that the differences in the CPU performance aren't even visible.

E.g.:
![CPU spread](../src/finished/ex06/ex04_spread_bind.png)
This shows the spread binding of the gpu.
Or compared to the close binding:
![CPU close](../src/finished/ex06/ex04_close_bind_milan2.png)
No difference can be seen.

## 7) 
For the multi gpu experiment we initialized the data on the CPU and then did a static distribution of the workload to each device.
So each device got a chunk of datasetSize / numDevices.

For this we used the ```#pragma omp target device(k)```directive.

![GPU](../src/finished/ex07/ex07_multi.png)

Funnily enough our performance for milan actually decreased, but this is most likely because of a conflict where another person also used the GPU.

For ice1. One can see that it pretty much doubles the previous measurement in its peak. This is because there are 2 A100 in there.

