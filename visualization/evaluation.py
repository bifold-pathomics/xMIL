import matplotlib.pyplot as plt
import numpy as np
from visualization.utils import plot_line_mean_se, plot_boxplot_paired


def plot_ave_prob_boxplots(
    probs_plot, box_labels, palette, skip_ids, ylabel, showfliers=False, verbose=False
):
    ave_prob = []

    for i_heatmap, heatmap_type in enumerate(probs_plot.keys()):
        ave_prob.append(
            [
                np.mean(prob)
                for i, prob in enumerate(probs_plot[heatmap_type])
                if i not in skip_ids
            ]
        )

    plot_boxplot_paired(
        ave_prob,
        box_labels,
        ylabel,
        datapoints=list(range(len(box_labels))),
        paired=None,
        jitter_std=0.05,
        datapoints_color="black",
        pair_linewidth=0.001,
        datapoint_size=1,
        alpha=0.2,
        notch=False,
        palette=palette,
        showfliers=showfliers,
    )


def plot_perturbation_curve(
    probs_plot,
    flip_steps,
    skip_ids,
    palette,
    xlabel,
    ylabel,
    heatmap_labels,
    n_se=1,
    std=False,
):
    for i_heatmap, heatmap_type in enumerate(probs_plot.keys()):
        mean_vals = []
        se_vals = []
        for i_step in flip_steps:

            vals_i = [
                p[i_step]
                for i_p, p in enumerate(probs_plot[heatmap_type])
                if i_p not in skip_ids
            ]
            mean_vals.append(np.mean(vals_i))
            if std:
                se_vals.append(np.std(vals_i))
            else:
                se_vals.append(np.std(vals_i) / np.sqrt(len(vals_i)))

        plot_line_mean_se(
            flip_steps,
            np.array(mean_vals),
            np.array(se_vals),
            n_se=n_se,
            color=palette[i_heatmap],
            alpha=0.1,
            label=heatmap_labels[i_heatmap],
        )

        plt.ylim(0, 1)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)

        plt.legend()
        plt.grid(True)
