from matplotlib import pyplot as plt
import matplotlib as mpl
import seaborn as sns
import pandas as pd
import numpy as np
import sys

# check num arguments
if len(sys.argv) < 2:
    print('Usage: python plot_heatmap.py <file>')
    exit(1)

# read input file
filename = sys.argv[1]
df = pd.read_csv(filename, header=None)

# add columns and indices
N = df.shape[0]
columns = []
indices = []
for i in range(N):
    label = 'C' + str(i)
    columns.append(label)
    indices.append(label)
df.columns = columns
df.index = indices

# check the matrix
print(df)

# for masking null values
cmap = mpl.colormaps.get_cmap('Greens')
cmap.set_bad("black")

# plot the heatmap
plt.figure()
hm = sns.heatmap(df, cmap=cmap, linewidths=0.30, annot=False,  fmt=".1f", mask=(df==0.0))
# remove mask=(df==0.0) if not necessary

# output = './heatmap_avg_c2clatency_laptop.svg'
output = './heatmap_avg_c2clatency_laptop.pdf'
# output = './heatmap_avg_c2clatency_a64fx.pdf'
# output = './heatmap_avg_c2clatency_rome2.pdf'
# output = './heatmap_avg_c2clatency_ice2.pdf'
# output = './heatmap_avg_c2clatency_thx2.pdf'

# save to file
print('--------------------------------------------')
print('Write to file: {}'.format(output))
plt.title('laptop_4_cores', size=14)
# plt.title('a64fx_48_cores', size=14)
# plt.title('rome2_128_cores', size=14)
# plt.title('ice2_72_cores', size=14)
# plt.title('thx2_64_cores', size=14)

plt.savefig(output, bbox_inches='tight')
print('--------------------------------------------')
