import os
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import ttest_rel
from sklearn import metrics
from visualization.utils import (
    plot_line_mean_se,
    compute_SemiBluecher_faithfulness,
    plot_boxplot_with_significance,
    plot_boxplot_paired,
)


def plot_perturbation_curves(
    y1,
    y2,
    rate,
    skipped,
    slide_ids,
    xlabel="% of total patches",
    save_dir=None,
    save_name=None,
    save_format="pdf",
):
    n_slides = len(y1)
    n_plot = 40
    i_plot = 0
    ind_end = 0
    while ind_end < n_slides:
        ind_end = min((i_plot + 1) * n_plot, n_slides)
        ind_plot = range(i_plot * n_plot, ind_end)

        w = 4
        fig = plt.figure(figsize=(3, w * len(ind_plot)))
        plt.subplots_adjust(hspace=0.5)

        for i_subplot, i_slide in enumerate(ind_plot):
            if i_slide not in skipped:
                y = y1[i_slide]
                n_flip_steps = len(y)

                if xlabel == "% of total patches":
                    x = [
                        round(100 * (1 - (1 - rate) ** i), 1)
                        for i in range(n_flip_steps)
                    ]
                else:
                    x = [i for i in range(n_flip_steps)]

                ax = plt.subplot(len(ind_plot), 1, i_subplot + 1)

                color = "tab:red"
                ax.set_xlabel(xlabel)
                ax.set_ylabel("change in class prob", color=color)
                ax.plot(x, y1[i_slide], color=color)
                ax.tick_params(axis="y", labelcolor=color)

                ax2 = (
                    ax.twinx()
                )  # instantiate a second axes that shares the same x-axis

                color = "tab:blue"
                ax2.set_ylabel(
                    "class probability", color=color
                )  # we already handled the x-label with ax1
                ax2.plot(x, y2[i_slide], color=color)
                ax2.tick_params(axis="y", labelcolor=color)
                plt.title(f"batch no {i_slide}: {slide_ids[i_slide]}")

        if save_dir is not None:
            save_path = os.path.join(save_dir, f"{save_name}_{i_plot}.{save_format}")
            plt.savefig(save_path, bbox_inches="tight")
            plt.close()
        else:
            plt.show()
        i_plot += 1


def plot_spaghetti(y, skip_ids, alpha, xlabel, ylabel, ax):
    for i, p1 in enumerate(y):
        if i not in skip_ids:
            n_flip_steps = len(p1)
            x = np.arange(n_flip_steps)

            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            ax.plot(x, p1, alpha=alpha)
            plt.grid(True)


def plot_ave_prob_boxplots(
    ax,
    probs_plot,
    box_labels,
    palette,
    skip_ids,
    ylabel,
    with_sig=False,
    showfliers=False,
    significance_bars=True,
    verbose=False,
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
    if with_sig:
        plot_boxplot_with_significance(
            ax,
            ave_prob,
            box_labels,
            ylabel,
            palette,
            showfliers=showfliers,
            significance_bars=significance_bars,
            verbose=verbose,
        )
    else:
        plot_boxplot_paired(
            ave_prob,
            box_labels,
            ylabel,
            violinplot=False,
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


def plot_SemiBluecher_boxplots(
    ax,
    probs_add,
    probs_drop,
    box_labels,
    palette,
    skip_ids,
    ylabel,
    showfliers=False,
    significance_bars=True,
    verbose=False,
):
    ave_auc = compute_SemiBluecher_faithfulness(probs_add, probs_drop, skip_ids)

    plot_boxplot_with_significance(
        ax,
        list(ave_auc.values()),
        box_labels,
        ylabel,
        palette,
        showfliers=showfliers,
        significance_bars=significance_bars,
        verbose=verbose,
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

        # plt.ylim(0, 1.1)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)

        plt.legend()
        plt.grid(True)


def plot_perturbation_curve_SemiBluecher(
    probs_add, probs_drop, i_heatmap, flip_steps, skip_ids, color
):
    heatmap_type = list(probs_add.keys())[i_heatmap]

    mean_vals_add = []
    mean_vals_drop = []
    for i_step in flip_steps:

        vals_i = [
            p[i_step]
            for i_p, p in enumerate(probs_add[heatmap_type])
            if i_p not in skip_ids
        ]
        mean_vals_add.append(np.mean(vals_i))

        vals_i = [
            p[i_step]
            for i_p, p in enumerate(probs_drop[heatmap_type])
            if i_p not in skip_ids
        ]
        mean_vals_drop.append(np.mean(vals_i))

    plt.plot(flip_steps, mean_vals_drop, color=color, label="heatmap_type")
    plt.plot(flip_steps, mean_vals_add[::-1], color=color, label="heatmap_type")
