
## Assignment 7

### 1. Compile and run

* For the part of `list`, you can simply run the test as follow. Where, the output will be directed into `perf_list.txt`.
```bash
./list 268435456 1048576 | tee perf_list.txt
```

* For the part of `c2c`, you can check the arguments in the code. Without specifying arguments, the test will take the default; otherwise, you can specify, e.g.,
```bash
./c2c 20 30 0
```

The output in this code can be written into the file as you want. To do this, you can check the util function with the comment. Feel free to modify it for a more beautiful output format.

### 2. Plot the results

CI is not used in this assignment. Use the Python script as a guide for the result figures. Feel free to modify these scripts to plot it (in /util).
* For generating graphs - the first part `list`,

```bash
python3 generate_chart_list.py --performance-data reference_output_list.txt --output-file perf_ref_list.pdf
```

* For generating graphs - the second part `c2c`,
```bash
python3 generate_chart_c2c.py reference_output_c2c.csv
```
