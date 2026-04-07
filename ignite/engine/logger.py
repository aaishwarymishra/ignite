import warnings
from collections.abc import Mapping
from typing import Any

from torch.optim.optimizer import Optimizer

from ignite.engine.engine import Engine
from ignite.engine.events import Events
from ignite.handlers import (
    ClearMLLogger,
    MLflowLogger,
    NeptuneLogger,
    PolyaxonLogger,
    TensorboardLogger,
    VisdomLogger,
    WandBLogger,
    global_step_from_engine,
)
from ignite.handlers.base_logger import BaseLogger

__all__ = [
    "setup_tb_logging",
    "setup_visdom_logging",
    "setup_mlflow_logging",
    "setup_neptune_logging",
    "setup_wandb_logging",
    "setup_plx_logging",
    "setup_clearml_logging",
    "setup_trains_logging",
]


def _setup_logging(
    logger: BaseLogger,
    trainer: Engine,
    optimizers: Optimizer | dict[str, Optimizer] | dict[None, Optimizer] | None,
    evaluators: Engine | dict[str, Engine] | None,
    log_every_iters: int,
) -> None:
    if optimizers is not None:
        if not isinstance(optimizers, (Optimizer, Mapping)):
            raise TypeError("Argument optimizers should be either a single optimizer or a dictionary or optimizers")

    if evaluators is not None:
        if not isinstance(evaluators, (Engine, Mapping)):
            raise TypeError("Argument evaluators should be either a single engine or a dictionary or engines")

    if log_every_iters is None:
        log_every_iters = 1

    logger.attach_output_handler(
        trainer, event_name=Events.ITERATION_COMPLETED(every=log_every_iters), tag="training", metric_names="all"
    )

    if optimizers is not None:
        # Log optimizer parameters
        if isinstance(optimizers, Optimizer):
            optimizers = {None: optimizers}

        for k, optimizer in optimizers.items():
            logger.attach_opt_params_handler(
                trainer, Events.ITERATION_STARTED(every=log_every_iters), optimizer, param_name="lr", tag=k
            )

    if evaluators is not None:
        # Log evaluation metrics
        if isinstance(evaluators, Engine):
            evaluators = {"validation": evaluators}

        event_name = Events.ITERATION_COMPLETED if isinstance(logger, WandBLogger) else None
        gst = global_step_from_engine(trainer, custom_event_name=event_name)
        for k, evaluator in evaluators.items():
            logger.attach_output_handler(
                evaluator, event_name=Events.COMPLETED, tag=k, metric_names="all", global_step_transform=gst
            )


def setup_tb_logging(
    output_path: str,
    trainer: Engine,
    optimizers: Optimizer | dict[str, Optimizer] | None = None,
    evaluators: Engine | dict[str, Engine] | None = None,
    log_every_iters: int = 100,
    **kwargs: Any,
) -> TensorboardLogger:
    """Setup TensorBoard logging on trainer and optional evaluators."""
    logger = TensorboardLogger(log_dir=output_path, **kwargs)
    _setup_logging(logger, trainer, optimizers, evaluators, log_every_iters)
    return logger


def setup_visdom_logging(
    trainer: Engine,
    optimizers: Optimizer | dict[str, Optimizer] | None = None,
    evaluators: Engine | dict[str, Engine] | None = None,
    log_every_iters: int = 100,
    **kwargs: Any,
) -> VisdomLogger:
    """Setup Visdom logging on trainer and optional evaluators."""
    logger = VisdomLogger(**kwargs)
    _setup_logging(logger, trainer, optimizers, evaluators, log_every_iters)
    return logger


def setup_mlflow_logging(
    trainer: Engine,
    optimizers: Optimizer | dict[str, Optimizer] | None = None,
    evaluators: Engine | dict[str, Engine] | None = None,
    log_every_iters: int = 100,
    **kwargs: Any,
) -> MLflowLogger:
    """Setup MLflow logging on trainer and optional evaluators."""
    logger = MLflowLogger(**kwargs)
    _setup_logging(logger, trainer, optimizers, evaluators, log_every_iters)
    return logger


def setup_neptune_logging(
    trainer: Engine,
    optimizers: Optimizer | dict[str, Optimizer] | None = None,
    evaluators: Engine | dict[str, Engine] | None = None,
    log_every_iters: int = 100,
    **kwargs: Any,
) -> NeptuneLogger:
    """Setup Neptune logging on trainer and optional evaluators."""
    logger = NeptuneLogger(**kwargs)
    _setup_logging(logger, trainer, optimizers, evaluators, log_every_iters)
    return logger


def setup_wandb_logging(
    trainer: Engine,
    optimizers: Optimizer | dict[str, Optimizer] | None = None,
    evaluators: Engine | dict[str, Engine] | None = None,
    log_every_iters: int = 100,
    **kwargs: Any,
) -> WandBLogger:
    """Setup Weights and Biases logging on trainer and optional evaluators."""
    logger = WandBLogger(**kwargs)
    _setup_logging(logger, trainer, optimizers, evaluators, log_every_iters)
    return logger


def setup_plx_logging(
    trainer: Engine,
    optimizers: Optimizer | dict[str, Optimizer] | None = None,
    evaluators: Engine | dict[str, Engine] | None = None,
    log_every_iters: int = 100,
    **kwargs: Any,
) -> PolyaxonLogger:
    """Setup Polyaxon logging on trainer and optional evaluators."""
    logger = PolyaxonLogger(**kwargs)
    _setup_logging(logger, trainer, optimizers, evaluators, log_every_iters)
    return logger


def setup_clearml_logging(
    trainer: Engine,
    optimizers: Optimizer | dict[str, Optimizer] | None = None,
    evaluators: Engine | dict[str, Engine] | None = None,
    log_every_iters: int = 100,
    **kwargs: Any,
) -> ClearMLLogger:
    """Setup ClearML logging on trainer and optional evaluators."""
    logger = ClearMLLogger(**kwargs)
    _setup_logging(logger, trainer, optimizers, evaluators, log_every_iters)
    return logger


def setup_trains_logging(
    trainer: Engine,
    optimizers: Optimizer | dict[str, Optimizer] | None = None,
    evaluators: Engine | dict[str, Engine] | None = None,
    log_every_iters: int = 100,
    **kwargs: Any,
) -> ClearMLLogger:
    """``setup_trains_logging`` was renamed to :func:`~ignite.engine.setup_clearml_logging`."""
    warnings.warn("setup_trains_logging was renamed to setup_clearml_logging.")
    return setup_clearml_logging(trainer, optimizers, evaluators, log_every_iters, **kwargs)
