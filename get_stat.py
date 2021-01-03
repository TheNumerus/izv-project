from download import DataDownloader
import matplotlib.pyplot as plt
import numpy as np
import argparse
from typing import Tuple, List
import os
import errno


def plot_stat(data_source: Tuple[List[str], List[np.ndarray]], fig_location=None, show_figure=False):
    """Renders plot with basic stats about accidents

    :param data_source source of data

    :key fig_location: location to save file to
    :key show_figure: show figure
    """
    fig, ax = plt.subplots(5, sharey=True)

    fig.set_size_inches((8, 10))
    fig.suptitle("Počet nehod podle kraje za rok")

    graph_data = {str(x): {} for x in range(2016, 2021)}

    for i in range(data_source[1][0].size):
        reg = data_source[1][0][i]
        date = data_source[1][4][i]
        year = np.datetime_as_string(date, unit='Y')

        if reg not in graph_data[year]:
            graph_data[year][reg] = 1
        else:
            graph_data[year][reg] += 1

    for i, x in enumerate(range(2016, 2021)):
        if i == 2:
            ax[i].set(ylabel="počet nehod", xlabel="kraj", title=f"{x}")
        else:
            ax[i].set(xlabel="kraj", title=f"{x}")
        keys = graph_data[str(x)].keys()
        crashes = graph_data[str(x)].values()
        ax[i].grid(axis='y')
        if x != 2020:
            ax[i].get_xaxis().set_visible(False)

        crashes_sorted = sorted(crashes, reverse=True)
        number = [crashes_sorted.index(x) + 1 for i, x in enumerate(crashes)]

        bar = ax[i].bar(keys, crashes)

        for bar_i, rect in enumerate(bar):
            height = rect.get_height()
            if height > 20000:
                ax[i].annotate('{}'.format(number[bar_i]),
                               xy=(rect.get_x() + rect.get_width() / 2, height),
                               xytext=(0, -3),
                               textcoords="offset points",
                               color='white',
                               ha='center', va='top')
            else:
                ax[i].annotate('{}'.format(number[bar_i]),
                               xy=(rect.get_x() + rect.get_width() / 2, height),
                               xytext=(0, 3),
                               textcoords="offset points",
                               color='black',
                               ha='center', va='bottom')

    if fig_location is not None:
        try:
            folder_path, _ = os.path.split(fig_location)
            if folder_path != '':
                os.makedirs(folder_path)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise
        plt.savefig(fig_location, dpi=200)

    if show_figure:
        plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--fig_location')
    parser.add_argument('--show_figure', default=False, action="store_true")
    args = parser.parse_args()

    data_source = DataDownloader().get_list(
        ['PHA', 'STC', 'JHC', 'PLK', 'KVK', 'ULK', 'LBK', 'HKK', 'PAK', 'OLK', 'MSK', 'JHM', 'ZLK', 'VYS'])
    plot_stat(data_source, show_figure=args.show_figure, fig_location=args.fig_location)
