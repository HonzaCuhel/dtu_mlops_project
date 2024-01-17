import pytest
import os.path
from datasets import load_from_disk
from tests import _PATH_DATA
from dtu_mlops_project.predict_model import predict


def test_prediction():
    test_set = load_from_disk(os.path.join(_PATH_DATA, "processed", "test"))
    subset = test_set.shuffle().select(range(10))
    for example in subset:
        prediction = predict(example["text"])
        assert prediction in ["Bearish", "Bullish", "Neutral"], "Unrecognized prediction"
