"""Window statistics, checkpoint selection, and validation cadence on CPU."""

import ast
import importlib.util
import inspect
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

# Import the callback without loading unrelated model/training dependencies.
ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "window_callback", ROOT / "src/xmil/training/callback.py"
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
Callback = module.Callback


class Logger:
    def __init__(self):
        self.summary = {}
        self.history = []

    def set_summary(self, values):
        self.summary.update(values)

    def log_scalar(self, name, value, step):
        self.history.append((name, value, step))


def make_callback(tmp_path, criterion="perf_window_mean", size=3):
    return Callback(
        False, 1, str(tmp_path), stop_criterion=criterion, perf_window_size=size
    )


def save(callback, metrics, losses, best, logger, last=False):
    signature = inspect.signature(callback.save_checkpoint_survival)
    args = {
        name: []
        for name, param in signature.parameters.items()
        if param.default is inspect.Parameter.empty
    }
    args.update(
        epoch_no=len(metrics) - 1,
        c_index_epoch_val=metrics,
        loss_epoch_val=losses,
        model_state_dict={"weight": torch.tensor(len(metrics))},
        optimizer_state_dict={},
        best_model=best,
        logger=logger,
        last_model=last,
        return_args=True,
    )
    return callback.save_checkpoint_survival(**args)[1]


def test_full_trailing_window_and_population_std(tmp_path):
    callback = make_callback(tmp_path)
    assert callback.get_performance_window([0.5, 0.6], [1.0, 2.0]) == {}
    values = callback.get_performance_window([0.1, 0.5, 0.6, 0.7], [9.0, 3.0, 2.0, 1.0])
    assert values == pytest.approx(
        {
            "perf_window_mean": 0.6,
            "perf_window_std": np.std([0.5, 0.6, 0.7]),
            "loss_window_mean": 2.0,
        }
    )
    assert callback.get_performance_window([0.6] * 3, [1.0] * 3)["perf_window_std"] == 0


def test_window_selection_saves_endpoint_and_preserves_raw_metrics(tmp_path):
    callback = make_callback(tmp_path)
    logger = Logger()
    best = callback.get_model_dict({}, {}, -1, np.inf, 0)
    metrics = [0.9, 0.4, 0.4, 0.6, 0.7, 0.5]
    losses = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    for end in range(1, len(metrics) + 1):
        best = save(
            callback,
            metrics[:end],
            losses[:end],
            best,
            logger,
            last=end == len(metrics),
        )
        if end < 3:
            assert callback.stop_cr_counter == 0
            assert not (tmp_path / "best_model.pt").exists()
    # The first window includes an isolated .9 peak. The best full mean is
    # nevertheless the final [.6, .7, .5] window; save its .5 endpoint.
    checkpoint = torch.load(tmp_path / "best_model.pt", weights_only=False)
    assert checkpoint["epoch"] == 5
    assert checkpoint["perf_metric_val"] == 0.5
    assert checkpoint["loss_val"] == 6.0
    assert checkpoint["perf_window_mean"] == pytest.approx(0.6)
    assert checkpoint["loss_window_mean"] == 5.0
    assert checkpoint["model_state_dict"]["weight"].item() == 6
    assert logger.summary["C-Index/val.selected"] == 0.5
    assert logger.summary["loss/val.selected"] == 6.0
    assert logger.summary["C-Index/val_window_mean.selected"] == pytest.approx(0.6)
    assert logger.summary["loss/val_window_mean.selected"] == 5.0


@pytest.mark.parametrize("criterion", ["loss", "perf_metric"])
def test_existing_selection_rules_unchanged(tmp_path, criterion):
    callback = make_callback(tmp_path, criterion)
    logger = Logger()
    best = callback.get_model_dict({}, {}, -1, np.inf, 0)
    for end in range(1, 4):
        best = save(
            callback, [0.8, 0.7, 0.6][:end], [3.0, 2.0, 1.0][:end], best, logger
        )
    assert best["epoch"] == (2 if criterion == "loss" else 0)


def test_window_one_matches_raw_performance(tmp_path):
    callback = make_callback(tmp_path, size=1)
    best = callback.get_model_dict({}, {}, -1, np.inf, 0)
    for end in range(1, 4):
        best = save(callback, [0.5, 0.8, 0.6][:end], [3.0, 2.0, 1.0][:end], best, None)
    assert best["epoch"] == 1
    assert best["perf_window_std"] == 0


def test_worse_window_does_not_overwrite_selected_summary(tmp_path):
    callback = make_callback(tmp_path, size=2)
    logger = Logger()
    best = callback.get_model_dict({}, {}, -1, np.inf, 0)
    best = save(callback, [0.7, 0.8], [1.0, 2.0], best, logger)
    summary = logger.summary.copy()
    best = save(callback, [0.7, 0.8, 0.5], [1.0, 2.0, 3.0], best, logger, last=True)
    assert logger.summary == summary
    assert best["epoch"] == 1
    assert torch.load(tmp_path / "last_model.pt", weights_only=False)["epoch"] == 2


def test_invalid_windows(tmp_path):
    with pytest.raises(ValueError, match="at least 1"):
        make_callback(tmp_path, size=0)
    callback = make_callback(tmp_path)
    with pytest.raises(ValueError, match="nonfinite"):
        callback.get_performance_window([0.5, np.nan, 0.6], [1.0, 2.0, 3.0])


def test_validation_interval_and_final_epoch(tmp_path):
    # Execute the real survival loop without importing unrelated model modules.
    tree = ast.parse((ROOT / "src/xmil/training/loops.py").read_text())
    fn = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "train_survival_model"
    )
    scores = iter([0.5, 0.6, 0.5, 0.5, 0.8, 0.5, 0.7])
    namespace = dict(
        torch=torch,
        np=np,
        tqdm=lambda values: values,
        concordance_index_censored=lambda *a, **k: (next(scores),),
    )
    exec(
        compile(ast.Module(body=[fn], type_ignores=[]), "<survival loop>", "exec"),
        namespace,
    )
    model = torch.nn.Linear(1, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    def prediction(batch):
        preds = (
            torch.ones(2, 4) * 0.1,
            torch.ones(2, 4) * 0.9,
            torch.tensor([0.4, 0.2]),
        )
        targets = (
            torch.tensor([1.0, 2.0]),
            torch.tensor([0.0, 1.0]),
            torch.tensor([2.0, 5.0]),
        )
        return preds, targets, torch.tensor(1.0)

    classifier = SimpleNamespace(
        training_step=prediction, validation_step=lambda batch: (*prediction(batch), {})
    )
    callback = make_callback(tmp_path, size=2)
    callback.checkpoint_epoch = 2
    logger = Logger()
    _, _, best = namespace["train_survival_model"](
        model,
        classifier,
        optimizer,
        4,
        0.01,
        [{}],
        [{}],
        callback,
        n_epoch_val=2,
        logger=logger,
    )
    # Evaluations are at 0, 2, and final epoch 3, not four consecutive epochs.
    means = [
        (value, step)
        for name, value, step in logger.history
        if name == "C-Index/val_window_mean"
    ]
    assert means == [(pytest.approx(0.7), 2), (pytest.approx(0.75), 3)]
    assert best["epoch"] == 3
    assert (tmp_path / "last_model.pt").exists()
