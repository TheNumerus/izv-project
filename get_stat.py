from download import DataDownloader


def plot_stat(data_source, fig_location=None, show_figure=False):
    data_source = DataDownloader().get_list(['MSK', 'JHM', 'OLK'])
    pass
