import numpy as np
from copy import deepcopy
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.colorbar import ColorbarBase
from matplotlib.cbook import boxplot_stats


def clean_outliers_fliers(data):
    """
    function for clipping the input data based on the outliers marked as fliers in boxplot data
    """
    data_clean = deepcopy(data)
    stat = boxplot_stats(data)[0]
    data_clean[data < stat["whislo"]] = stat["whislo"]
    data_clean[data > stat["whishi"]] = stat["whishi"]
    data_no_outlier = data[
        np.logical_and(data >= stat["whislo"], data <= stat["whishi"])
    ]
    return data_clean, data_no_outlier


def convert2rgb(mat, cmap_name="coolwarm", zero_centered=True):
    """
    converts the given matrix to RGB values
    """
    cmap = plt.get_cmap(cmap_name)
    if zero_centered:
        max_scalar = np.max(np.abs(mat))
        rgb_values = (max_scalar + mat) / (2 * max_scalar)
    else:
        min_scalar = np.min(mat)
        max_scalar = np.max(mat)
        rgb_values = (mat - min_scalar) / (max_scalar - min_scalar)
    return cmap(rgb_values).squeeze()[:, :-1]


def plot_colorbar(ax, data, cmap="coolwarm", ori="vertical", zero_centered=True):
    """
    lazy plotting of a colormap corresponding to the given data
    """
    if zero_centered:
        max_scalar = np.max(np.abs(data))
        norm = Normalize(vmin=-max_scalar, vmax=max_scalar)
    else:
        norm = Normalize(vmin=min(data), vmax=max(data))
    _ = ColorbarBase(ax, cmap=cmap, norm=norm, orientation=ori)


def plot_line_mean_se(
    x, mean_values, std_errors, n_se, color="blue", alpha=0.3, label=""
):
    """
    plots a line with errors as shaded area
    """
    plt.plot(x, mean_values, color=color, label=label)
    plt.fill_between(
        x,
        mean_values - n_se * std_errors,
        mean_values + n_se * std_errors,
        color=color,
        alpha=alpha,
    )


def plot_boxplot_paired(
    data,
    xticks,
    ylabel,
    datapoints=None,
    paired=None,
    pair_linewidth=0.1,
    datapoint_size=3,
    alpha=0.5,
    datapoints_color="lightskyblue",
    jitter_std=0.05,
    notch=True,
    palette="colorblind",
    showfliers=False,
):
    """
    custom function for plotting boxplots
    """

    ax = sns.boxplot(data, notch=notch, palette=palette, showfliers=False)

    ax.set_xticklabels(xticks)
    plt.ylabel(ylabel)

    datapoints = [] if datapoints is None else datapoints

    for i_data in datapoints:

        if not showfliers:
            outliers = [
                y for stat in boxplot_stats(data[i_data]) for y in stat["fliers"]
            ]
            data_i = [d for d in data[i_data] if d not in outliers]
        else:
            data_i = data[i_data]

        n_points = len(data_i)
        plt.plot(
            np.ones(n_points) * i_data + np.random.randn(n_points) * jitter_std,
            data_i,
            ".",
            color=datapoints_color,
            markersize=datapoint_size,
        )

        mean_i = np.mean(data[i_data])
        plt.plot(i_data, mean_i, ".", color="red", markersize=datapoint_size * 10)

    if paired is not None:
        if not showfliers:
            outliers_0 = [
                y for stat in boxplot_stats(data[paired[0]]) for y in stat["fliers"]
            ]
            outliers_1 = [
                y for stat in boxplot_stats(data[paired[1]]) for y in stat["fliers"]
            ]
        for d1, d2 in zip(data[paired[0]], data[paired[1]]):
            if showfliers or (
                not showfliers and d1 not in outliers_0 and d2 not in outliers_1
            ):
                x = np.array(list(paired))
                y = np.array([d1, d2])
                plt.plot(x, y, "-", linewidth=pair_linewidth, alpha=alpha)

    plt.grid(True, zorder=0)
    ax.set_axisbelow(True)


def compute_auc(data, skip_ids):
    ave_prob = []
    for i_heatmap, heatmap_type in enumerate(data.keys()):
        ave_prob.append(
            [
                np.mean(prob)
                for i, prob in enumerate(data[heatmap_type])
                if i not in skip_ids
            ]
        )

    return ave_prob
