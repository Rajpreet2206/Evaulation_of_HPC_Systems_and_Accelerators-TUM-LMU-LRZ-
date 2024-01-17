# Assignment 6 Report
Group: 104

## 1 a)
The resulting assembly code might look something like this:
```s
loop_dot_product:
  mov    eax,DWORD PTR [rbp-0xc]  ; Load i into eax
  cdqe                           ; Sign-extend eax to rax
  mov    edx,DWORD PTR [rbp+rax*4-0x1a0]  ; Load a[i] into edx
  mov    eax,DWORD PTR [rbp-0xc]  ; Load i into eax
  cdqe                           ; Sign-extend eax to rax
  mov    eax,DWORD PTR [rbp+rax*4-0x330]  ; Load b[i] into eax
  imul   eax,edx                 ; Multiply a[i] and b[i]
  add    DWORD PTR [rbp-0x8],eax ; Add the result to sum
  add    DWORD PTR [rbp-0xc],0x1 ; Increment i
  cmp    DWORD PTR [rbp-0xc],0x63 ; Compare i with 99
  jle    loop_dot_product        ; Jump back to loop_dot_product if i <= 99
```


* i) 
The multiply can be handled out of order, but the addition needs to be in order. So the issue of mul instructions can be done in a superscalar fashion whereas the addition needs to wait for the multiply to finish. 

Therefore we can actually measure the latency of the add instruction by just measuring the number of cylces it tkaes for the loop to finish. 

* ii) 
To derive the add instruction latency from the execution time and the number of clock cycles, you can use the following relationship:
Execution Time = Number of Instructions × Cycle Time

where:

Execution Time is the total time taken to execute the loop.
Number of Instructions is the total number of instructions executed in the loop.
Cycle Time is the time taken to execute one clock cycle.
For the given loop:

for (i = 0; i < N; i++) {
    sum += a[i] * b[i];
}


Assuming that the only significant operation in the loop is the addition (sum += a[i] * b[i]), let's consider this as one instruction. The loop consists of N iterations, so the total number of instructions (Number of Instructions would be N. 
Now, rearranging the formula to solve for the cycle time:

Cycle Time = Execution Time / Number of Instructions
​
If you have the execution time and the number of clock cycles, you can calculate the cycle time. Once you have the cycle time, you can use it to find the add instruction latency.

Add Instruction Latency = Cycle Time


*iii)
To circumvent this, we can actually calculate more multiplications in one iteration of the for loop:

```cpp
for (int i = 0; i < N; i+=4){
    sum += a[i] * b[i] + 
            a[i+1] * b[i+1] +
            a[i+2] * b[i+2] + 
            a[i+3] * b[i+3];
}
if (N % 4 != 0){
    for (int i = N - (N % 4); i < N; i++) {
        sum += a[i] * b[i];
    } 
}
```

This was about 2 times faster.


## 1 b) Theoretical question
* i)
To fully utilize a pipeline, you want to have enough independent instructions in the pipeline to avoid stalls due to dependencies. The number of independent dependency chains needed to fully utilize a pipeline is influenced by the depth of the pipeline and the instruction latency.

Assuming an instruction latency of n cycles, and considering a simplified scenario where all instructions in a chain are dependent on the result of the previous instruction, you would need at least n independent dependency chains to fully utilize the pipeline. The possible reasons are:
1> Pipeline Latency

2> Dependency Chains

3> Overlap of Execution

* ii)
If there are m available execution ports on the core for a given instruction, it can potentially increase the number of independent dependency chains needed to fully utilize the pipeline. The idea is to have enough independent instructions in the pipeline to keep all available execution ports busy in each cycle. The following changes can be observed:
1> Increase in Parallelism

2> Number of Independent Dependency Chains would be equal to or greater than the number of available execution ports(m)

3> Overlap of Execution

4> Optimization for Parellelism needed

* iii)
The instruction throughput lower than one cycle per instruction is possible due to following reasons:
1> Dependencies and Pipeling

2> Resource Contention

3> Instruction Mix

4> Branch Misprediction

5> Memory Access Latency

6> Out-of-Order Execution


# 2) Measuring Instruction Latency and Throughput

## a) understanding the code

* i) CHAINS controlls the number of dependency chains. As val[j] is always overwritten, the next operation on val[j] depends on the last operation on val[j]. So there is a dependecy there.

* ii) If we increase the number of Chains we will create more independent dependency chains, which could be run in parallel (depending on the superscalarity). Basically it fills up the pipeline and enables more parallelism by using superscalarity.

If CHAINS is bigger than thes superscalarity can handle, we will have to run these chains sequentially.

* iii) 
Instruction throughput = number of instructions in dependecy chain / execution time in clock cylces

Here we can use REP as the number of instructions in the dependecy chain and 


## b) running benchmark
i)

The results for each arch can be seen here:

![milan](../src/CPI_milan2.png)
![ice1](../src/CPI_ice1.png)
![cs2](../src/CPI_cs2.png)
![thx2](../src/CPI_thx2.srve.png)



==> we sadly didn't have enough time this week to fully do the exercises :(