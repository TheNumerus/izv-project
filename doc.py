import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from typing import Tuple, List


def time_map(val):
    """
    Maps input time string to hour integer if valid
    :param val: time string
    :return: time integer
    """
    normalized = int(val) // 100
    if normalized == 25:
        return None
    return normalized


def grab_stat(df: pd.DataFrame, stat: str):
    """
    Creates dictionary with hour keys and assoc. values
    :param df: source dataframe
    :param stat: dataframe series to use
    :return: dictionary with hour keys and assoc. values
    """
    times = {}

    for hour, row in df.iterrows():
        hour = int(hour)
        times[hour] = row[stat]
    return times


def list_from_dict(d) -> Tuple[List, List]:
    """
    Returns two tuples with x and y values to plot
    :param d: dictionary to use
    :return: tuples with x and y values to plot
    """
    x = []
    y = []
    for key in sorted(d):
        x.append(key)
        y.append(d[key])
    # loop
    x.append(24)
    y.append(y[0])
    return x, y


if __name__ == "__main__":
    # load data
    df = pd.read_pickle("accidents.pkl.gz")

    # create new series in dataframe
    df['hour'] = df['p2b'].map(time_map)

    # drop all invalid times
    df = df.dropna(subset=['hour'])

    # create grouped views for later use
    # p19 is visibility stat
    # 1 and 4 are good visibility in day and night
    grouped_bad_vis = df[(df.p19 != 1) & (df.p19 != 4)].groupby(['hour'])
    grouped = df.groupby(['hour'])

    # print
    print(f'total accidents = {df.shape[0]}')
    print(f'accidents at night = {df[((df.hour > 20) | (df.hour < 6))].shape[0]}')
    print(f'accidents in bad visibility = {df[(df.p19 != 1) & (df.p19 != 4)].shape[0]}')
    print(f'accidents in bad visibility at night = {df[(df.p19 != 1) & (df.p19 != 4) & ((df.hour > 20) | (df.hour < 6))].shape[0]}')

    x, y = list_from_dict(grab_stat(grouped.count(), 'region'))
    x_1, y_1 = list_from_dict(grab_stat(grouped_bad_vis.count(), 'region'))

    fig, ax = plt.subplots(1, 1, figsize=(10, 5))

    # plot per hour counts
    ax.plot(x, y, label="Počet nehod v danou hodinu")
    ax.plot(x_1, y_1, label="Počet nehod v danou hodinu se špatnou viditelností", color='gold')

    # figure options
    ax.set_xlim(0, 24)
    ax.set_ylim(0, 30000)
    ax.set_ylabel('Počet nehod')
    ax.set_xlabel('Hodina')
    ax.set_xticks(np.arange(0, len(x) + 1, 2))
    ax.grid(axis='y', alpha=0.2)

    # plot ratio
    ax_2 = ax.twinx()
    ax_2.plot(x, [y_1[i] / y[i] for i in range(len(x))], color='black', linestyle='dashed', alpha=0.2)

    # second axis options
    ax_2.set_ylim(0, 1)
    ax_2.set_ylabel('Poměr nehod se špatnou viditelností')
    ax_2.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))

    # finalize
    ax.legend(loc='upper left')
    fig.tight_layout()

    plt.savefig('fig.png', dpi=200)

    # now print table
    print("hodina,celkem,špatná viditelnost")
    for i in range(4):
        print(f'{i * 6} - {(i + 1) * 6},{sum(y[i * 6:i * 6 + 6])},{sum(y_1[i * 6:i * 6 + 6])}')
