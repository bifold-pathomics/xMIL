"""Experiment logging backends used by the training scripts."""


def add_logging_arguments(parser):
    """Add shared experiment-logging arguments to an argument parser."""
    group = parser.add_argument_group("Experiment logging")
    group.add_argument(
        "--logging-backend",
        choices=["wandb", "tensorboard"],
        default="wandb",
        help="Experiment logging backend (default: wandb).",
    )
    group.add_argument("--wandb-project", default="xmil")
    group.add_argument("--wandb-entity", default=None)
    group.add_argument("--wandb-run-name", default=None)
    group.add_argument("--wandb-group", default=None)
    group.add_argument("--wandb-tags", nargs="*", default=None)
    group.add_argument(
        "--wandb-mode",
        choices=["online", "offline", "disabled"],
        default="offline",
        help="W&B defaults to offline; select online to upload run data.",
    )
    group.add_argument(
        "--wandb-watch",
        choices=["none", "gradients", "parameters", "all"],
        default="none",
        help="Optionally log model gradients or parameters.",
    )
    group.add_argument("--wandb-watch-log-frequency", type=int, default=100)


class TensorBoardLogger:
    """Log scalar metrics to TensorBoard."""

    def __init__(self, log_dir, **_):
        from torch.utils.tensorboard import SummaryWriter

        self.writer = SummaryWriter(log_dir)

    def log_scalar(self, name, value, step):
        self.writer.add_scalar(name, value, step)

    def watch(self, model):
        pass

    def close(self):
        self.writer.close()


class WandbLogger:
    """Log scalar metrics and run configuration to Weights & Biases."""

    def __init__(
        self,
        log_dir,
        config,
        project,
        entity=None,
        run_name=None,
        group=None,
        tags=None,
        mode="online",
        watch="none",
        watch_log_frequency=100,
    ):
        try:
            import wandb
        except ImportError as exc:
            raise ImportError(
                "W&B logging is the default but the 'wandb' package is not installed. "
                "Install the project requirements or use "
                "'--logging-backend tensorboard'."
            ) from exc

        self.run = wandb.init(
            project=project,
            entity=entity,
            name=run_name,
            group=group,
            tags=tags,
            config=config,
            dir=log_dir,
            mode=mode,
        )
        self.watch_mode = watch
        self.watch_log_frequency = watch_log_frequency
        self.defined_metrics = set()

    def log_scalar(self, name, value, step):
        step_name = f"step/{name}"
        if name not in self.defined_metrics:
            self.run.define_metric(name, step_metric=step_name)
            self.defined_metrics.add(name)
        self.run.log({name: value, step_name: step})

    def watch(self, model):
        if self.watch_mode != "none":
            self.run.watch(
                model,
                log=self.watch_mode,
                log_freq=self.watch_log_frequency,
            )

    def close(self):
        self.run.finish()


def build_experiment_logger(backend, log_dir, config):
    """Build the requested experiment logger from command-line configuration."""
    if backend == "tensorboard":
        return TensorBoardLogger(log_dir=log_dir)
    if backend == "wandb":
        return WandbLogger(
            log_dir=log_dir,
            config=config,
            project=config["wandb_project"],
            entity=config.get("wandb_entity"),
            run_name=config.get("wandb_run_name"),
            group=config.get("wandb_group"),
            tags=config.get("wandb_tags"),
            mode=config["wandb_mode"],
            watch=config["wandb_watch"],
            watch_log_frequency=config["wandb_watch_log_frequency"],
        )
    raise ValueError(f"Unknown logging backend: {backend}")
