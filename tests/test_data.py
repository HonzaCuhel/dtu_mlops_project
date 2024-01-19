import os.path

import datasets
import pytest
from datasets import load_from_disk

from tests import _PATH_DATA


def _check_dataset(dataset: datasets.arrow_dataset.Dataset, split: str) -> set:
    """Check dataset."""
    import logging

    logger = logging.getLogger(__name__)
    logger.info(f"Checking {type(dataset)} dataset...")
    labels = set()
    for example in dataset:
        assert isinstance(example["text"], str), f"Invalid input type found in {split} dataset"
        assert isinstance(example["label"], int), f"Invalid label type found in {split} dataset"
        assert example["label"] in [0, 1, 2], f"Invalid label found in {split} dataset"
        labels.add(example["label"])
    return labels


@pytest.mark.skipif(not os.path.exists(_PATH_DATA), reason="Data files not found")
def test_data():
    """Test data."""
    assert os.path.exists(os.path.join(_PATH_DATA, "processed")), "Processed data folder not found"
    assert os.path.exists(os.path.join(_PATH_DATA, "processed", "train")), "Processed train data folder not found"
    assert os.path.exists(os.path.join(_PATH_DATA, "processed", "test")), "Processed test data folder not found"
    assert os.path.exists(os.path.join(_PATH_DATA, "processed", "val")), "Processed val data folder not found"

    train_set = load_from_disk(os.path.join(_PATH_DATA, "processed", "train"))
    val_set = load_from_disk(os.path.join(_PATH_DATA, "processed", "val"))
    test_set = load_from_disk(os.path.join(_PATH_DATA, "processed", "test"))

    labels = _check_dataset(train_set, "train")
    assert labels == set(range(3)), "Missing labels in the train dataset"
    _ = _check_dataset(val_set, "validation")
    _ = _check_dataset(test_set, "test")
