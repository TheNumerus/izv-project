import errno
import os
import datetime

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pickle
import gzip


def _save_and_show(fig_location: str, show_figure: bool):
    """
     Helper function for saving and showing graphs
    :param fig_location: file to save to
    :param show_figure: show figure on screen
    :return:
    """
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


def get_dataframe(filename: str = "accidents.pkl.gz", verbose: bool = False) -> pd.DataFrame:
    """
    Parses data from given file into pandas DataFrame
    :param filename: path to file with input data
    :param verbose: if true, prints data size diffs
    :return: parsed data
    """
    file = gzip.open(filename)
    data = pickle.load(file)
    df = pd.DataFrame(data)

    # print old sizes
    if verbose:
        size = 0
        for col in df.columns:
            size += df[col].memory_usage(deep=True, index=False)
        print(f"old_size={size / 1024 / 1024:.1f} MB")

    # date convert
    df['p2a'] = df['p2a'].astype('datetime64')
    df.rename({'p2a': 'date'})

    # convert rest
    # got these by printing unique value count
    cols_to_convert = ['p', 'q', 'k', 'j', 'p36', 'weekday(p2a)', 'p6', 't', 'p2b', 'o', 'l', 'r', 'p37']
    for col in cols_to_convert:
        df[col] = df[col].astype('category')

    # if all unique value can fit into -128..=127, why not compress them
    cols_to_i8 = ['p5a', 'p58', 'p55a', 'p57', 'p7', 'p8', 'p9', 'p10', 'p11', 'p13a', 'p13b', 'p13c', 'p15', 'p16', 'p17', 'p18', 'p19',
                  'p20', 'p21', 'p22', 'p23', 'p24', 'p27', 'p28', 'p34', 'p35', 'p39', 'p44', 'p45a', 'p47', 'p48a', 'p49', 'p50a', 'p50b',
                  'p51', 'p52']
    for col in cols_to_i8:
        df[col] = df[col].astype('int8')

    # print new sizes
    if verbose:
        size = 0
        for col in df.columns:
            size += df[col].memory_usage(deep=True, index=False)
        print(f"new_size={size / 1024 / 1024:.1f} MB")

    return df


def plot_conseq(df: pd.DataFrame, fig_location: str = None, show_figure: bool = False):
    """
    Renders graph of accidents consequences
    :param df: data source
    :param fig_location: file to save to
    :param show_figure: show figure on screen
    """
    fig, ax = plt.subplots(4, 1)
    fig.set_size_inches((8, 10))
    fig.suptitle("Následky nehod podle krajů")

    sums = df.groupby(['region']).sum().reset_index()

    sns.set_palette("colorblind")

    sns.barplot(data=sums, x="region", y="p13a", order=df['region'].value_counts().index, ax=ax[0])
    ax[0].set(ylabel="počet mrtvých", xlabel=None)

    sns.barplot(data=sums, x="region", y="p13b", order=df['region'].value_counts().index, ax=ax[1])
    ax[1].set(ylabel="počet těžce zraněných", xlabel=None)

    sns.barplot(data=sums, x="region", y="p13c", order=df['region'].value_counts().index, ax=ax[2])
    ax[2].set(ylabel="počet lehce zraněných", xlabel=None)

    sns.countplot(data=df, x="region", order=df['region'].value_counts().index, ax=ax[3])
    ax[3].set(ylabel="počet nehod")

    # set background and grids
    for i in range(0, 4):
        ax[i].set(facecolor=(0.9, 0.9, 0.9))
        ax[i].grid(axis="y", alpha=0.3)

    _save_and_show(fig_location, show_figure)


def plot_damage(df: pd.DataFrame, fig_location: str = None, show_figure: bool = False):
    """
    Renders graph of accident count by damage caused and accident cause
    :param df: data source
    :param fig_location: file to save to
    :param show_figure: show figure on screen
    """
    fig, ax = plt.subplots(2, 2)
    fig.set_size_inches((10, 10))
    fig.suptitle("Počty nehod podle kraje, příčiny nehody a velikosti škody")

    regions = ['MSK', 'JHM', 'OLK', 'PHA']
    sns.set_palette("colorblind")

    for index, region in enumerate(regions):
        data = pd.DataFrame(df[(df.region == region)])

        cost_labels = ('< 50', '50 - 200', '200 - 500', '500 - 1000', '> 1000')
        data['p53'] = pd.cut(data['p53'], bins=[0, 500, 2000, 5000, 10000, 1000000], include_lowest=True,
                             labels=cost_labels)

        labels = ('nezaviněná řidičem', 'nepřiměřená rychlost jízdy', 'nesprávné předjíždění', 'nedání přednosti v jízdě',
                  'nesprávný způsob jízdy', 'technická závada vozidla')
        data['p12'] = pd.cut(data['p12'], bins=[0, 200, 300, 400, 500, 600, 700], include_lowest=True,
                             labels=labels)

        data = data.groupby(['p12', 'p53']).size().reset_index()
        print(data)

        ax[(index // 2, index % 2)].set(yscale="log", title=region)
        sns.barplot(data=data, y=0, x='p53', hue='p12', ci=None, ax=ax[(index // 2, index % 2)], bottom=0.1)

        # set background and grids
        ax[(index // 2, index % 2)].set(facecolor=(0.9, 0.9, 0.9), ylabel=None, xlabel=None)
        ax[(index // 2, index % 2)].grid(axis="y", alpha=0.3)
        ax[(index // 2, index % 2)].legend().remove()

    ax[(0, 1)].legend(prop={'size': 6})
    ax[(0, 0)].set(ylabel='Počet nehod')
    ax[(1, 1)].set(xlabel='Škoda v tis. Kč')

    _save_and_show(fig_location, show_figure)


def plot_surface(df: pd.DataFrame, fig_location: str = None, show_figure: bool = False):
    """
    Renders graph of accidents on surfaces in time
    :param df: data source
    :param fig_location: file to save to
    :param show_figure: show figure on screen
    """
    fig, ax = plt.subplots(2, 2)
    fig.set_size_inches((12, 8))
    fig.suptitle("Počty nehod podle kraje a povrchu")

    regions = ['MSK', 'JHM', 'OLK', 'PHA']
    sns.set_palette("colorblind")

    rename = {
        0: 'jiný',
        1: 'suchý neznečištěný',
        2: 'suchý znečištěný',
        3: 'mokrý',
        4: 'bláto',
        5: 'náledí, ujetý sníh - posypané',
        6: 'náledí, ujetý sníh - neposypané',
        7: 'rozlitý olej, nafta apod.',
        8: 'souvislý sníh',
        9: 'náhlá změna stavu',
    }

    for index, region in enumerate(regions):
        data = pd.DataFrame(df[(df.region == region)])
        i = (index // 2, index % 2)

        counts = pd.crosstab(data.p2a, data.p16)
        counts.rename(columns=rename, inplace=True)
        counts = counts.groupby(pd.Grouper(freq='M')).sum()

        sns.lineplot(data=counts, ax=ax[i], dashes=False)

        # set background and grids
        ax[i].set(facecolor=(0.95, 0.95, 0.95), ylabel=None, xlabel=None, title=region)
        ax[i].grid(axis="y", alpha=0.3)
        ax[i].legend().remove()

    ax[(0, 0)].set(ylabel='Počet nehod')
    ax[(1, 1)].set(xlabel='Datum nehody')
    h, l = ax[(0, 0)].get_legend_handles_labels()

    plt.subplots_adjust(bottom=0.15)
    fig.legend(h, l, loc="lower center", ncol=5)

    _save_and_show(fig_location, show_figure)


if __name__ == "__main__":
    df = get_dataframe("data/accidents.pkl.gz", True)
    # plot_conseq(df, show_figure=True)
    # plot_damage(df, show_figure=True)
    plot_surface(df, show_figure=True)
