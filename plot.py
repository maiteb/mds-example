import argparse

import pandas as pd
import matplotlib.pyplot as plt
plt.style.use('dark_background')


def _parse_file_argument():
    parser = argparse.ArgumentParser("MDS Export to PNG Tool")
    parser.add_argument('--result_csv',
                        help='csv filename')
    parser.add_argument('--label_column',
                        help='csv\'s label column')
    args = parser.parse_args()
    return args

def _plot_distribution(df):
    fig = plt.figure()
    ax = plt.subplot(aspect='equal')
    scatter = ax.scatter(df['x'], df['y'], color='#FFCD00', s=30, alpha=.4)
    plt.xlim(-1, 1)
    plt.ylim(-1, 1)

    return fig, ax, scatter

if __name__ == '__main__':
    args = _parse_file_argument()
    data = pd.read_csv(args.result_csv)

    label_column = args.label_column
    data.drop(label_column, axis=1, inplace=True)
    _plot_distribution(data)
    plt.savefig('mds.png', dpi=120)
