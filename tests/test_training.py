import os
import shutil
from types import SimpleNamespace

import pytest

from dtu_mlops_project.train_model import main as train_func
from tests import _PATH_DATA


def test_training():
    """Test training of the model."""
    cfg_dict = {
        "hyperparameters": {
            "model_id": "microsoft/deberta-v3-xsmall",
            "lr": 1e-4,
            "batch_size": 32,
            "num_epochs": 3,
            "output_dir": "tests/output",
            "train_log_dir": "tests/logs",
            "weight_decay": 0.01,
            "eval_strategy": "epoch",
            "save_strategy": "epoch",
            "use_wandb": False,
        },
        "dataset": {"train_set_path": "tests/data/tiny_train", "val_set_path": "tests/data/tiny_train"},
    }
    cfg = SimpleNamespace()
    cfg.hyperparameters = SimpleNamespace(**cfg_dict["hyperparameters"])
    cfg.dataset = SimpleNamespace(**cfg_dict["dataset"])

    os.environ["WANDB_MODE"] = "disabled"
    results = train_func(cfg)

    assert os.path.exists("tests/output"), "Output folder not created"
    assert os.path.exists("tests/logs"), "Logs folder not created"
    assert os.path.exists("tests/output/model.safetensors"), "Model weights not saved"
    assert results["eval_accuracy"] > 0.5, "Accuracy too low"

    shutil.rmtree("tests/output", ignore_errors=True)
    shutil.rmtree("tests/logs", ignore_errors=True)
