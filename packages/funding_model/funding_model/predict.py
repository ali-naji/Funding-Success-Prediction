from funding_model import __version__ as _version
import pandas as pd
import logging
from funding_model.manager import load_pipeline
logger = logging.getLogger('funding_model')


def predict(data):
    pipeline = load_pipeline()
    predictions = pipeline.predict(pd.DataFrame(data))
    logger.info(f"Making predictions with model version {_version}"
                f"Inputs : {data}"
                f"Predictions : {predictions}")

    return {'predictions': predictions, 'version': _version}
