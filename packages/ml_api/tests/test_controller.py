from funding_model.manager import load_dataset
from funding_model.config import config
from funding_model import __version__ as model_version
import pandas as pd
import json


def test_prediction_endpoint(flask_test_client):
    datapoints = load_dataset().loc[:1000, :]
    sample = datapoints[1:2].to_json(orient='records')
    response = flask_test_client.post(
        'test_prediction', json=json.loads(sample))
    assert response.status_code == 200
    response_json = json.loads(response.data)
    prediction = response_json['predictions']
    version = response_json['version']

    assert prediction is not None
    assert prediction[0] in [0, 1]
    assert version == model_version
