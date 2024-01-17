import sys
import re
from argparse import ArgumentParser

try:
    import seaborn as sns
    import matplotlib.pyplot as plt
    import pandas as pd
except ImportError:
    print('Failed to load dependencies. Please ensure that seaborn, matplotlib, and pandas can be loaded.', file=sys.stderr)
    exit(-1)

if __name__ == '__main__':

    parser = ArgumentParser(description='Generate performance charts.')
    parser.add_argument('--performance-data', type=str, default="perf_data.txt")
    parser.add_argument('--output-file', type=str, default="perf_data.svg")

    args = parser.parse_args()

    perf_data_file = open(args.performance_data, 'r')
    perf_data = { "Data size": [], "MFLOPS": []}

    for line in perf_data_file:
      matchData = re.match("MFLOPS\(N=\s*(\d+)\): (\d+\.\d+)", line)
      perf_data["Data size"].append(int(matchData.group(1)))
      perf_data["MFLOPS"].append(float(matchData.group(2)))

    pandas_data = pd.DataFrame(perf_data)

    fig = plt.figure(figsize=(16, 6))

    # Measured performance
    axes = plt.subplot(1, 2, 1)

    sns.lineplot(data=pandas_data, x="Data size", y="MFLOPS", marker='o')
    axes.set_title('Measured Performance')

    # Max ylim
    max_lim = max([ax.get_ylim()[1] for ax in fig.get_axes()])
    for ax in fig.get_axes():
        ax.set_ylim([0, max_lim])

    plt.xticks(range(0, 2001, 200))

    fig.tight_layout()

    plt.savefig(args.output_file)
    print(f"Wrote output to {args.output_file}")

