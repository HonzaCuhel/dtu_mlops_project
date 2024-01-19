import os.path

import pytest
from datasets import load_from_disk

from dtu_mlops_project.predict_model import predict
from tests import _PATH_DATA


def test_prediction():
    """Test prediction of the trained model."""
    test_set = load_from_disk(os.path.join(_PATH_DATA, "processed", "test"))
    subset = test_set.shuffle().select(range(10))
    for example in subset:
        prediction = predict(example["text"])
        assert prediction in ["Bearish", "Bullish", "Neutral"], "Unrecognized prediction"
