import os
import math
import ast
import textwrap
import json

import openslide
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from PIL import Image
from matplotlib.backends.backend_pdf import PdfPages
from copy import deepcopy
from visualization.utils import convert2rgb, plot_colorbar, clean_outliers_fliers


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
    (c) https://github.com/hense96/patho-preprocessing
    """
    bg = bg.copy()
    fg = fg.copy()
    fg.putalpha(alpha)
    bg.paste(fg, (0, 0), fg)
    return bg


def clip_heatmap(patch_scores, p, clip=True):
    heatmap_scores = deepcopy(patch_scores)

    extreme_low = np.percentile(heatmap_scores, p)
    extreme_high = np.percentile(heatmap_scores, 100 - p)
    heatmap_scores[heatmap_scores < extreme_low] = extreme_low if clip else 0
    heatmap_scores[heatmap_scores > extreme_high] = extreme_high if clip else 0
    return heatmap_scores, extreme_low, extreme_high


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


def create_visualization_pdf(
    data_loader,
    classifier,
    xmodel,
    slides_dirs,
    patches_dirs,
    results_dir,
    scores_df=None,
    annotations_dirs=None,
    thumbnail_shape=(612.0, 792.0),
    side_by_side=True,
    print_label=True,
    show_annotations=False,
    heatmap_type="attention",
    top_patches=True,
    cmap_name="coolwarm",
    preface=None,
    save_overlays=True,
    target_names=None,
    cut_top_k_percent=None,
    pdf_suffix="",
    verbose=False,
):
    """
    Creates a PDF with visualizations of the model predictions for all given slides.

    :param data_loader: (DataLoader) DataLoader object with all elements to predict on. The loader should have batch
        size 1.
    :param model: (nn.Module) Model object to fetch patch scores from.
    :param classifier: Classifier object to predict with.
    :param xmodel: Explanation model object for creating explanations
    :param slides_dirs: (list<str>) The directory where the slides are located.
    :param patches_dirs: (list<str>) The directory where the patches are located.
    :param results_dir: (str) Directory to save the visualizations in.
    :param scores_df: (pandas dataframe) dataframe with patchscores
    :param annotations_dirs: (list<str>) The directory where the annotations are located.
    :param thumbnail_shape: (float, float) Desired shape of the thumbnails.
    :param side_by_side: (bool) if True, slide and heatmap are plotted side-by-side; if False, they are overlaid into
        a single plot
    :param print_label: (bool)
    :param show_annotations: (bool) if True, annotations will be overlaid over the slide thumbnails
    :param heatmap_type: (str) How to create heatmaps, e.g., from 'attention', 'lrp', 'scores'
    :param top_patches: (boolean): if True, plot patches with the highest scores
    :param cmap_name: (str) Name of the matplotlib color map for the heatmaps.
    :param preface: (str): if given, the text will be printed at the first page of the pdf
    :param save_overlays: (bool) whether to save the slide overlays as separate files
    :param target_names: (list of str) names of the targets (optional)
    :param cut_top_k_percent: (float between 0 and 100 inclusive) if not None, the least cut_top_k_percent % and max
        cut_top_k_percent % of the data will be clipped.
    :param pdf_suffix: (str) suffix for the saved pdf
    :param verbose: (bool) Whether to print logs.
    """
    # Set up file structure
    os.makedirs(results_dir, exist_ok=True)
    pdf_filename = os.path.join(results_dir, f"test_visualizations{pdf_suffix}.pdf")
    if save_overlays:
        os.makedirs(os.path.join(results_dir, "overlays"), exist_ok=True)

    with PdfPages(pdf_filename) as pdf:

        # Create preface
        if preface:
            fig, ax = plt.subplots()  # Create a letter-sized figure
            ax.text(0.5, 0.5, preface, fontsize=6, ha="center", va="center")
            ax.axis("off")
            pdf.savefig(fig)  # Save the first page to the PDF
            plt.close(fig)

        for batch in tqdm(data_loader):

            preds, targets, loss, pred_metadata = classifier.validation_step(batch)
            n_patches = batch["bag_size"].item()
            if scores_df is not None:
                df_this_slide = scores_df[
                    (
                        scores_df["source_id"]
                        == batch["sample_ids"]["source_id"][0].item()
                    )
                    & (scores_df["slide_id"] == batch["sample_ids"]["slide_id"][0])
                ]
                patch_scores = np.array(
                    json.loads(df_this_slide[f"patch_scores_{heatmap_type}"].item())
                )
                patch_scores = patch_scores[-n_patches:]
            else:
                patch_scores = xmodel.get_heatmap(batch, heatmap_type, verbose)

            zero_centered = xmodel.get_heatmap_zero_centered(heatmap_type)

            patch_scores, _ = clean_outliers_fliers(patch_scores)

            patch_ids = batch["patch_ids"].numpy().ravel()
            pred_score = preds[0, -1]
            target = targets[0]
            patch_scores = patch_scores.squeeze()
            source_id, slide_id = (
                pred_metadata["source_id"][0],
                pred_metadata["slide_id"][0],
            )

            # Load slide and annotation
            slide_file = [
                slide_file
                for slide_file in os.listdir(slides_dirs[source_id])
                if slide_file.startswith(slide_id)
                and os.path.isfile(os.path.join(slides_dirs[source_id], slide_file))
            ][0]
            slide = openslide.open_slide(
                os.path.join(slides_dirs[source_id], slide_file)
            )
            patches_this_slide = pd.read_csv(
                os.path.join(patches_dirs[source_id], slide_id, "metadata/df.csv"),
                index_col=0,
            )

            if annotations_dirs is not None and show_annotations:
                slide_anno = openslide.open_slide(
                    os.path.join(annotations_dirs[source_id], f"{slide_id}.png")
                )
            else:
                slide_anno = None

            label_print = target.tolist() if print_label else None

            # Create overlay heatmap
            fig = slide_heatmap_thumbnail(
                slide=slide,
                patches=patches_this_slide,
                patch_ids=patch_ids,
                patch_scores=patch_scores,
                slide_name=slide_id,
                label=label_print,
                target_names=target_names,
                pred_score=pred_score.tolist(),
                annotation=slide_anno,
                side_by_side=side_by_side,
                size=thumbnail_shape,
                cmap_name=cmap_name,
                background="black",
                zero_centered=zero_centered,
            )

            pdf.savefig(fig)
            plt.close(fig)

            # Plot patches with the highest scores
            if top_patches:
                fig, _ = display_top_patches(
                    patch_ids,
                    patch_scores,
                    os.path.join(patches_dirs[source_id], slide_id),
                )
                pdf.savefig(fig)
                plt.close(fig)

            # Save overlays as png files (in a higher resolution)
            if save_overlays:
                overlay_dims = (slide.dimensions[0] // 32, slide.dimensions[1] // 32)
                heatmap_img, _ = heatmap_PIL(
                    patches=patches_this_slide,
                    size=overlay_dims,
                    patch_ids=patch_ids,
                    slide_dim=slide.dimensions,
                    score_values=patch_scores,
                    cmap_name="coolwarm",
                    background="black",
                    zero_centered=zero_centered,
                )
                heatmap_img.save(
                    os.path.join(results_dir, "overlays", f"{slide_id}.png"), "PNG"
                )
