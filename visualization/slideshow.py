import os
import math
import ast
import textwrap

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from visualization.utils import convert2rgb, plot_colorbar


def build_overlay(patches, size, patch_ids, slide_dim, overlay_rgb, background="black"):
    """
    (c) modified from https://github.com/hense96/patho-preprocessing
    Args:
        patches: the dataframe containing the metadata of the patches of this slide
        size: the desired size of the overlay
        patch_ids: the patch IDs for which we want to build the overlay and have the overlay RGB values
        slide_dim: slide.dimensions if slide is the openslide object
        overlay_rgb: [n-patch x 3] The RGB values of the patches in patch_ids
        background: background color for the overlay ['black', 'white']

    Returns: The PIL image of the overlay

    """
    if background == "black":
        overlay_image = np.zeros((size[0], size[1], 3))
    elif background == "white":
        overlay_image = np.ones((size[0], size[1], 3))
    else:
        raise ValueError(f"Unsupported background color for overlay: {background}")
    for i, id_ in enumerate(patch_ids):
        this_patch = patches[patches["patch_id"] == id_]
        x_coord, y_coord = ast.literal_eval(this_patch["position_abs"].item())
        patch_size = this_patch["patch_size_abs"]
        ds_x_coord = int(x_coord * (size[0] / slide_dim[0]))
        ds_y_coord = int(y_coord * (size[1] / slide_dim[1]))
        ds_patch_size_x = int(math.ceil(patch_size * (size[0] / slide_dim[0])))
        ds_patch_size_y = int(math.ceil(patch_size * (size[1] / slide_dim[1])))
        overlay_image[
            ds_x_coord : (ds_x_coord + ds_patch_size_x),
            ds_y_coord : (ds_y_coord + ds_patch_size_y),
            :,
        ] = overlay_rgb[i, :]
    return Image.fromarray(np.uint8(np.transpose(overlay_image, (1, 0, 2)) * 255))


def heatmap_PIL(
    patches,
    size,
    patch_ids,
    slide_dim,
    score_values,
    cmap_name="coolwarm",
    background="black",
    zero_centered=True,
):
    """
    builds the PIL image of the attention values.
    Args:
        patches: the dataframe containing the metadata of the patches of this slide
        size: the desired size of the heatmap
        patch_ids: the patch IDs for which we want to build the overlay and have the overlay RGB values
        slide_dim: slide.dimensions if slide is the openslide object
        score_values: The attention values to be converted to a PIL image
        cmap_name: colormap
        background: background color for the overlay ['black', 'white']
        zero_centered: if True, the heatmap colors will be centered at score 0.0

    Returns: The PIL image of the attention image and the RGB values corresponding to the attention values

    """
    scores_rgb = convert2rgb(
        score_values, cmap_name=cmap_name, zero_centered=zero_centered
    )
    img = build_overlay(patches, size, patch_ids, slide_dim, scores_rgb, background)
    return img, scores_rgb


def overlay(bg, fg, alpha=64):
    """
    Creates an overlay of the given foreground on top of the given background using PIL functionality.
    """
    bg = bg.copy()
    fg = fg.copy()
    fg.putalpha(alpha)
    bg.paste(fg, (0, 0), fg)
    return bg


def plot_PIL(ax, im, cmap="coolwarm"):
    """
    lazy plotting of PIL images without axis ticks
    """
    img = ax.imshow(im, cmap=cmap)
    ax.axis("off")
    return img


def image_with_colorbar(
    img,
    scores,
    slide_name=None,
    label=None,
    pred_score=None,
    cmap="coolwarm",
    zero_centered=True,
):
    fig = plt.figure()
    gs = fig.add_gridspec(2, 1, width_ratios=[5], height_ratios=[20, 1])

    ax_image = fig.add_subplot(gs[0])
    plot_PIL(ax_image, img)

    ax_colorbar = fig.add_subplot(gs[1])
    _ = plot_colorbar(
        ax_colorbar, scores, cmap=cmap, ori="horizontal", zero_centered=zero_centered
    )

    title_text = ""
    if slide_name is not None:
        title_text += slide_name
    if label is not None:
        title_text += f", label: {label}"
    if pred_score is not None:
        title_text += f", prediction: {pred_score:.4f}"
    if len(title_text) > 0:
        ax_image.set_title(title_text, fontsize=8)

    return fig


def heatmap_with_slide(
    slide_thumbnail, heatmap_PIL, slide_name=None, label=None, pred_score=None
):
    """
    Plots the original slide and the heatmap next to each other.

    :param slide_thumbnail: Image of the slide.
    :param heatmap_PIL: Image of the slide heatmap.
    :param slide_name: (str) ID of the slide.
    :param label: (int/float/str) A label of the slide.
    :param pred_score: (float) A prediction score assigned to the slide.
    :return: matplotlib figure
    """
    fig, axes = plt.subplots(1, 2)

    plot_PIL(axes[0], slide_thumbnail)
    if label is not None:
        axes[0].set_title(f"label: {label}", fontsize=8)

    plot_PIL(axes[1], heatmap_PIL)
    if pred_score is not None:
        axes[1].set_title(f"prediction: {pred_score:.4f}", fontsize=8)

    if slide_name is not None:
        fig.suptitle(slide_name, fontsize=12)

    return fig


def slide_heatmap_thumbnail(
    slide,
    patches,
    patch_ids,
    patch_scores,
    slide_name=None,
    label=None,
    target_names=None,
    pred_score=None,
    annotation=None,
    side_by_side=True,
    size=(2048, 2048),
    cmap_name="coolwarm",
    background="black",
    zero_centered=True,
    title_wrap_width=40,
):
    """ "
    Plots a thumbnail of the original slide with the heatmap.

    :param slide: openslide Slide object
    :param patches: dataframe containing the metadata of the patches of this slide
    :param patch_ids: (list-like) the patch IDs for which we want to build a heatmap, corresponding to the patch scores
    :param patch_scores: (list-like) the patch scores to be visualized in the heatmap
    :param slide_name: (str) ID of the slide (optional)
    :param label: (int/float/str or list of int/float/str) label(s) of the slide (optional)
    :param target_names: (list of str) names of the targets (optional)
    :param pred_score: (float of list of floats) prediction score(s) assigned to the slide (optional)
    :param annotation: openslide object of the annotation to be added to the slide as an overlay (optional)
    :param side_by_side: (bool) if True, slide and heatmap are plotted side-by-side; if False, they are overlaid into
        a single plot
    :param size: (int, int) maximum size of the thumbnails
    :param cmap_name: (str) matplotlib colormap for the heatmap
    :param background: (str) background color for the heatmap ['black', 'white']
    :param zero_centered: (bool) if True, the heatmap colors will be centered at score 0.0
    :return: matplotlib figure
    """
    # Create thumbnails of available data
    slide_thumbnail = slide.get_thumbnail(size)
    heatmap, _ = heatmap_PIL(
        patches,
        slide_thumbnail.size,
        patch_ids,
        slide.dimensions,
        patch_scores,
        cmap_name=cmap_name,
        background=background,
        zero_centered=zero_centered,
    )
    if annotation is not None:
        slide_thumbnail = overlay(
            slide_thumbnail, annotation.get_thumbnail(slide_thumbnail.size), 40
        )

    # Plot the thumbnails
    fig = plt.figure()
    gs = fig.add_gridspec(2, 2, height_ratios=[32, 1])
    if slide_name is not None:
        fig.suptitle(slide_name, fontsize=12)

    # if labels or prediction scores are lists of values (e.g., for multi-target models), convert them to strings.
    # if target_names are given, add the target names to the labels and prediction scores
    if label is not None and isinstance(label, list):
        if (
            target_names is not None
            and isinstance(target_names, list)
            and len(target_names) == len(label)
        ):
            label = ", ".join(
                [l_name + ": " + str(l) for l, l_name in zip(label, target_names)]
            )
        else:
            label = ", ".join([str(l) for l in label])
    if pred_score is not None and isinstance(pred_score, list):
        if (
            target_names is not None
            and isinstance(target_names, list)
            and len(target_names) == len(pred_score)
        ):
            pred_score = ", ".join(
                [l_name + f": {p:.4f}" for p, l_name in zip(pred_score, target_names)]
            )
        else:
            pred_score = ", ".join([f"{p:.4f}" for p in pred_score])

    if side_by_side:
        ax_left = fig.add_subplot(gs[0, 0])
        ax_right = fig.add_subplot(gs[0, 1])
        plot_PIL(ax_left, slide_thumbnail)
        plot_PIL(ax_right, heatmap)
        if label is not None:
            # wrap the label text to avoid overlapping with the slide thumbnail
            label = textwrap.fill(label, width=title_wrap_width, break_long_words=True)
            ax_left.set_title(f"Label(s): {label}", fontsize=8)
        if pred_score is not None:
            # wrap the prediction text to avoid overlapping with the slide thumbnail
            pred_score = textwrap.fill(
                pred_score, width=title_wrap_width, break_long_words=True
            )
            ax_right.set_title(f"Prediction(s): {pred_score}", fontsize=8)
    else:
        slide_thumbnail = overlay(slide_thumbnail, heatmap, 130)
        ax_top = fig.add_subplot(gs[0, :])
        plot_PIL(ax_top, slide_thumbnail)
        title_text = ""
        if label is not None:
            title_text += f"  Label(s): {label}  "
        if pred_score is not None:
            title_text += f"  Prediction(s): {pred_score}  "
        if len(title_text) > 0:
            # wrap the title text to avoid overlapping
            title_text = textwrap.fill(
                title_text, width=title_wrap_width, break_long_words=True
            )
            ax_top.set_title(title_text, fontsize=8)

    # Create a color bar for the heatmap beneath the thumbnails
    ax_bottom = fig.add_subplot(gs[1, :])
    plot_colorbar(
        ax_bottom,
        patch_scores,
        cmap=cmap_name,
        ori="horizontal",
        zero_centered=zero_centered,
    )

    fig.tight_layout()

    return fig


def display_top_patches(
    patch_ids,
    patch_scores,
    patches_dir,
    num_patches=25,
    rows=5,
    cols=5,
    figsize=(10, 10),
):
    """
    Display the top scored patches in a grid.

    :param patch_ids: (np.array) Identifiers of the patches (num_patches,).
    :param patch_scores: (np.array) Scores assigned to the patches (num_patches,).
    :param patches_dir: (str) Directory where the patches are stored.
    :param num_patches: (int) Number of patches to display.
    :param rows: (int) Number of rows in the display.
    :param cols: (int) Number of cols in the display.
    :param figsize: (int, int) Size of the matplotlib figure.
    :return: matplotlib figure, list of top patch images
    """
    top_patch_idx = patch_scores.argsort()[::-1][:num_patches]
    top_patch_ids = patch_ids[top_patch_idx]
    top_patch_scores = patch_scores[top_patch_idx]
    top_patch_imgs = [
        Image.open(os.path.join(patches_dir, f"{patch_id}.jpg"))
        for patch_id in top_patch_ids
    ]
    fig = display_patches_in_grid(
        top_patch_imgs, rows, cols, figsize, titles=top_patch_scores
    )
    return fig, top_patch_imgs


def display_patches_in_grid(patches, rows, cols, figsize=(10, 10), titles=None):
    """
    Display a grid of patch images using Matplotlib.

    Parameters:
    - patches: List of image paths or NumPy arrays.
    - rows: Number of rows in the grid.
    - cols: Number of columns in the grid.
    - figsize: Tuple specifying the size of the figure (default is (10, 10)).
    - titles: List of titles for each image (optional).
    """
    fig, axes = plt.subplots(rows, cols, figsize=figsize)

    for i, ax in enumerate(axes.flat):
        if i < len(patches):
            if isinstance(patches[i], str):
                img = plt.imread(patches[i])
            else:
                img = patches[i]

            ax.imshow(img, cmap="gray")
            ax.axis("off")
            if titles is not None:
                ax.set_title(f"{titles[i]:.10f}")

    # Remove empty subplots, if any
    for i in range(len(patches), rows * cols):
        fig.delaxes(axes.flatten()[i])

    plt.tight_layout()

    return fig
