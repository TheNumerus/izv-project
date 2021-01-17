import os
import errno
import pandas as pd
import geopandas
import matplotlib.pyplot as plt
import contextily
from sklearn.cluster import KMeans
import numpy as np

_reg = 'MSK'
_map_src = contextily.providers.Stamen.TonerLite


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


def make_geo(df: pd.DataFrame) -> geopandas.GeoDataFrame:
    """
     Converts DataFrame into valid GeoDataFrame

     :param df: input DataFrame
     :return: converted GeoDataFrame
    """

    df = df[(df.d.notnull()) & (df.e.notnull())]
    df = df[['p5a', 'd', 'e', 'region']]
    gdf = geopandas.GeoDataFrame(df, geometry=geopandas.points_from_xy(df.d, df.e), crs="EPSG:5514")
    gdf = gdf.to_crs(epsg=3857)
    return gdf


def plot_geo(gdf: geopandas.GeoDataFrame, fig_location: str = None, show_figure: bool = False):
    """
    Draws two graphs with accident locations

    :param gdf: data source
    :param fig_location: file to save to
    :param show_figure: show figure on screen
    """

    gdf = gdf[gdf.region == _reg]
    fig, ax = plt.subplots(1, 2)
    fig.set_size_inches((16, 8))

    gdf[gdf.p5a == 1].plot(ax=ax[0], markersize=2, color='red')
    gdf[gdf.p5a == 2].plot(ax=ax[1], markersize=2, color='green')

    # visuals
    ax[0].set_axis_off()
    ax[1].set_axis_off()
    ax[0].set(title='Nehody v MSK kraji v obci')
    ax[1].set(title='Nehody v MSK kraji mimo obec')
    fig.set_tight_layout(True)
    contextily.add_basemap(ax[0], source=_map_src, crs=gdf.crs.to_string())
    contextily.add_basemap(ax[1], source=_map_src, crs=gdf.crs.to_string())

    _save_and_show(fig_location, show_figure)


def plot_cluster(gdf: geopandas.GeoDataFrame, fig_location: str = None, show_figure: bool = False):
    """
    Draws graph with accident clusters

    :param gdf: data source
    :param fig_location: file to save to
    :param show_figure: show figure on screen
    """
    gdf = gdf[gdf.region == _reg]
    fig, ax = plt.subplots(1, 1)
    fig.set_size_inches((12, 10))

    clusters = KMeans(n_clusters=10).fit(np.array([gdf.geometry.x, gdf.geometry.y]).transpose())
    _unique, counts = np.unique(clusters.labels_, return_counts=True)

    df_centers = pd.DataFrame(clusters.cluster_centers_)
    df_centers['counts'] = counts
    df_centers['counts_size'] = counts / 10
    gdf_centers = geopandas.GeoDataFrame(df_centers, geometry=geopandas.points_from_xy(df_centers[0], df_centers[1]), crs="EPSG:5514")

    gdf.plot(ax=ax, markersize=2, color='grey', alpha=0.1)
    gdf_centers.plot(ax=ax, column='counts', markersize='counts_size', alpha=0.5, legend=True)

    ax.set_axis_off()
    fig.set_tight_layout(True)
    contextily.add_basemap(ax, source=_map_src, crs=gdf.crs.to_string())

    _save_and_show(fig_location, show_figure)


if __name__ == "__main__":
    gdf = make_geo(pd.read_pickle("data/accidents.pkl.gz"))
    plot_geo(gdf, "geo1.png", True)
    plot_cluster(gdf, "geo2.png", True)
