from funding_model.predict import predict
from funding_model.manager import load_dataset
from funding_model import __version__ as _version


def test_datapoint_prediction():
    datapoint = load_dataset().iloc[8:9]
    result = predict(datapoint)

    assert result is not None
    assert result['version'] == _version
    assert result['predictions'][0] in [0, 1]


def test_multiple_points_prediction():
    datapoints = load_dataset().iloc[0:10]
    result = predict(datapoints)

    assert result['version'] == _version
    assert result is not None
    assert len(result['predictions']) <= 10
