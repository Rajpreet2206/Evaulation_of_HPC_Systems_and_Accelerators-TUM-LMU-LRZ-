import sys
from argparse import ArgumentParser

try:
    import seaborn as sns
    import matplotlib.pyplot as plt
    import pandas as pd
except ImportError:
    print('Failed to load dependencies. Please ensure that seaborn, matplotlib, and pandas can be loaded.', file=sys.stderr)
    exit(-1)

if __name__ == '__main__':

    parser = ArgumentParser(description='Generate performance charts for ci stage.')
    parser.add_argument('--performance-data', type=str, required=True)
    parser.add_argument('--output-file', type=str, required=True)

    result = parser.parse_args()

    # Read data
    data = pd.read_fwf(result.performance_data, delimiter=' |')
    fig = plt.figure(figsize=(10, 7))

    # Measured performance
    axes = plt.subplot()
    print(data)
    sns.lineplot(data=data, x="Datasize", y="Latency[ns]", hue="SumType", marker='o')
    axes.set_xscale('log')
    axes.set_title('Measured Latency [ns]')

    # Reference performance

    fig.tight_layout()

    plt.savefig(result.output_file)
    print(f"Wrote output to {result.output_file}")
