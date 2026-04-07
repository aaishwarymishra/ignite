from unittest.mock import patch

from ignite.engine import (
    setup_clearml_logging,
    setup_mlflow_logging,
    setup_neptune_logging,
    setup_plx_logging,
    setup_tb_logging,
    setup_trains_logging,
    setup_visdom_logging,
    setup_wandb_logging,
)


def test_import_setup_logging_helpers_from_engine_namespace():
    assert callable(setup_tb_logging)
    assert callable(setup_visdom_logging)
    assert callable(setup_mlflow_logging)
    assert callable(setup_neptune_logging)
    assert callable(setup_wandb_logging)
    assert callable(setup_plx_logging)
    assert callable(setup_clearml_logging)
    assert callable(setup_trains_logging)


def test_setup_wandb_logging_resolves_new_module_path():
    from unittest.mock import MagicMock

    with patch("ignite.engine.logger.WandBLogger") as _:
        setup_wandb_logging(MagicMock())
