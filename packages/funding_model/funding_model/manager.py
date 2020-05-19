from funding_model.config import config
import funding_model
from funding_model import __version__ as _version
import pandas as pd
import os
import boto3
import logging
import joblib

logger = logging.getLogger('funding_model')
s3 = boto3.client('s3', aws_access_key_id=os.environ.get(
    'AWS_ACCESS_KEY_ID'), aws_secret_access_key=os.environ.get('AWS_SECRET_ACCESS_KEY'))


def load_dataset(filename='train.csv'):
    ''' Downloads dataset locally if not available, Loads it into memory '''
    save_path = (config.DATASET_DIR / filename).absolute().as_posix()
    if not os.path.exists(config.DATASET_DIR / filename):
        logger.info('Dataset not found. Will download from source')
        s3.download_file('myprivatedatasets',
                         'funding_model/'+filename, save_path)
    df = pd.read_csv(save_path)
    logger.info(f'Dataset loaded successfully from path : {save_path}')
    return df


def load_pipeline(filename=f"{config.PIPELINE_FILENAME}{_version}.pkl"):
    ''' Downloads pipeline locally if not available, loads it into memory '''
    save_path = (config.TRAINED_MODELS_DIR / filename).absolute().as_posix()
    if not os.path.exists(config.TRAINED_MODELS_DIR / filename):
        logger.info("Pipeline not found. Will download from source")
        s3.download_file("mytrainmodels", 'funding_model/'+filename, save_path)
    pipeline = joblib.load(config.TRAINED_MODELS_DIR / filename)
    logger.info(f"Pipeline loaded successfully from path : {save_path}")
    return pipeline


def save_pipeline(pipeline_object, filename=f"{config.PIPELINE_FILENAME}{_version}.pkl") -> None:
    ''' Saves pipeline locally '''
    joblib.dump(pipeline_object, config.TRAINED_MODELS_DIR / filename)
    logger.info(
        f"Pipeline saved successfully to path : {config.TRAINED_MODELS_DIR/filename}")
