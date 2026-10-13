import ast
import os
from pathlib import Path

import h5py
import matplotlib
import numpy as np
import pandas as pd
import pytest
import torch
from PIL import Image

matplotlib.use("Agg")

from xmil.datasets.data_handler import SlideDataHandler
from xmil.datasets.dataset_factory import DatasetFactory
from xmil.datasets.mil import MILSlideDataset


@pytest.fixture
def inputs(tmp_path):
    pd.DataFrame({"case_id": ["case"], "label": [1]}).to_csv(
        tmp_path / "case_metadata.csv", index=False
    )
    pd.DataFrame({"case_id": ["case"], "slide_id": ["slide"]}).to_csv(
        tmp_path / "slide_metadata.csv", index=False
    )
    pd.DataFrame({"case_id": ["case"], "subset": [0], "label": [1]}).to_csv(
        tmp_path / "split.csv", index=False
    )
    # Deliberately not spatially sorted: feature IDs must retain file row order.
    coords = np.array([[1024, 512], [0, 0], [512, 512]])
    features = np.arange(12, dtype=np.float32).reshape(3, 4)
    with h5py.File(tmp_path / "slide.h5", "w") as f:
        f["features"] = features
        d = f.create_dataset("coords", data=coords)
        d.attrs.update(patch_size=256, patch_size_level0=512, overlap=0)
    return dict(
        split_path=str(tmp_path / "split.csv"),
        metadata_dirs=[str(tmp_path)],
        subsets=["0"],
        patches_dirs=None,
        features_dirs=[str(tmp_path)],
        label_cols=["label"],
        data_format="trident",
    )


@pytest.mark.parametrize("preload", [False, True])
@pytest.mark.parametrize("bag_size", [2, 5])
def test_sampling_alignment_and_padding(inputs, preload, bag_size):
    ds = MILSlideDataset(**inputs, preload_features=preload, bag_size=bag_size)
    item = ds[0]
    valid = item["patch_ids"] >= 0
    ids = item["patch_ids"][valid]
    expected = torch.arange(12, dtype=torch.float32).reshape(3, 4)
    torch.testing.assert_close(item["features"][valid], expected[ids])
    torch.testing.assert_close(item["positions_x"][valid], torch.tensor([2, 0, 1])[ids])
    torch.testing.assert_close(item["positions_y"][valid], torch.tensor([1, 0, 1])[ids])
    assert (item["positions_x"][~valid] == -1).all()
    assert (item["features"][~valid] == 0).all()


def test_legacy_matches_trident(inputs, tmp_path):
    trident = MILSlideDataset(**inputs)[0]
    metadata = SlideDataHandler.load_trident_metadata(tmp_path / "slide.h5")
    folder = tmp_path / "slide/metadata"
    folder.mkdir(parents=True)
    metadata.iloc[::-1].to_csv(folder / "df.csv")
    torch.save(trident["features"], tmp_path / "slide.pt")
    legacy = MILSlideDataset(
        **{**inputs, "data_format": "legacy", "patches_dirs": [str(tmp_path)]}
    )[0]
    for key in ["features", "patch_ids", "positions_x", "positions_y", "targets"]:
        torch.testing.assert_close(legacy[key], trident[key])


def test_rejects_annotations(inputs):
    with pytest.raises(ValueError, match="annotations"):
        MILSlideDataset(**inputs, patch_filters={"has_annot": [1]})


@pytest.mark.parametrize("n_features", [0, 1])
def test_rejects_empty_or_misaligned(inputs, tmp_path, n_features):
    with h5py.File(tmp_path / "slide.h5", "a") as f:
        del f["features"]
        f["features"] = np.zeros((n_features, 4))
    with pytest.raises(ValueError, match="Empty or mismatched"):
        MILSlideDataset(**inputs)


def test_single_patch_and_overlap(inputs, tmp_path):
    with h5py.File(tmp_path / "slide.h5", "a") as f:
        attrs = dict(f["coords"].attrs)
        del f["coords"], f["features"]
        f["features"] = np.ones((1, 4), dtype=np.float32)
        f["coords"] = [[384, 768]]
        f["coords"].attrs.update({**attrs, "overlap": 64})
    item = MILSlideDataset(**inputs)[0]
    assert item["features"].shape == (1, 4)
    assert item["positions_x"].item() == 1
    assert item["positions_y"].item() == 2


def test_factory_without_patches_directory(inputs):
    args = {**inputs, "test_subsets": ["0"], "targets": ["label"]}
    del args["patches_dirs"]
    ds = DatasetFactory._build_image_dataset(args, "test")
    assert ds.data_format == "trident"
    assert ds[0]["features"].shape == (3, 4)


def test_overlay_and_wsi_patch_reading(inputs, tmp_path):
    from xmil.visualization.slideshow import build_overlay, display_top_patches
    import matplotlib.pyplot as plt

    metadata = SlideDataHandler.load_trident_metadata(tmp_path / "slide.h5")
    overlay = build_overlay(metadata, (3, 2), [0], (1536, 1024), np.array([[1, 0, 0]]))
    np.testing.assert_array_equal(np.array(overlay)[1, 2], [255, 0, 0])
    assert np.count_nonzero(np.array(overlay)) == 1

    class Slide:
        def read_region(self, location, level, size):
            assert location == (1024, 512) and level == 0 and size == (512, 512)
            return Image.new("RGBA", size, "red")

    fig, images = display_top_patches(
        np.array([0, 1]),
        np.array([0.9, 0.1]),
        None,
        num_patches=1,
        slide=Slide(),
        patch_metadata=metadata,
    )
    assert images[0].size == (256, 256)
    plt.close(fig)


def test_real_sample():
    path = os.environ.get("TRIDENT_SAMPLE")
    if not path:
        pytest.skip("Set TRIDENT_SAMPLE to check a real feature HDF5")
    metadata = SlideDataHandler.load_trident_metadata(path)
    features = SlideDataHandler.load_features(
        str(Path(path).with_suffix("")), data_format="trident"
    )
    with h5py.File(path) as f:
        expected = f["coords"][:]
        footprint = f["coords"].attrs["patch_size_level0"]
    grid = np.array([ast.literal_eval(p) for p in metadata.position_rel])
    np.testing.assert_array_equal(grid * footprint, expected)
    assert features.shape == (len(expected), 2560)
    assert torch.isfinite(features).all()
