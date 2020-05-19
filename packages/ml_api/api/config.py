import pathlib
import logging
import os
import sys
from logging.handlers import TimedRotatingFileHandler

ROOT = pathlib.Path(__file__).resolve().parent.parent
LOG_DIR = PACKAGE_ROOT / 'logs'
LOG_DIR.mkdir(exist_ok=True)
LOG_FILE = LOG_DIR / 'ml_api.log'
FORMATTER = logging.Formatter(
    "%(asctime)s — %(name)s — %(levelname)s —"
    "%(funcName)s:%(lineno)d — %(message)s")


def get_console_handler():
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(FORMATTER)
    handler.setLevel(logging.DEBUG)
    return handler


def get_file_handler():
    handler = TimedRotatingFileHandler(
        LOG_FILE, when='midnight')
    handler.setFormatter(FORMATTER)
    handler.setLevel(logging.WARNING)
    return handler


def get_logger(logger_name):
    logger = logging.getLogger(logger_name)
    logger.addHandler(get_console_handler())
    logger.addHandler(get_file_handler())
    logger.setLevel(logging.DEBUG)
    logger.propagate = False
    return logger


class Config:
    DEBUG = False
    TESTING = False
    CSRF_ENABLED = True
    SECRET_KEY = 'ml-api'
    SERVER_PORT = 5000


class ProductionConfig(Config):
    DEBUG = False
    SERVER_PORT = os.environ.get('PORT', 5000)


class DevelopmentConfig(Config):
    DEVELOPMENT = True
    DEBUG = True


class TestingConfig(Config):
    TESTING = True
