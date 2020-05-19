import logging
import sys
from funding_model.config import config

VERSION_PATH = config.PACKAGE_ROOT / 'VERSION'
with open(VERSION_PATH, 'r') as f:
    __version__ = f.read().strip()

Formatter = logging.Formatter("%(asctime)s — %(name)s — %(levelname)s —" "%(funcName)s:%(lineno)d — %(message)s")

def get_console_handler():
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(Formatter)
    return handler

logger = logging.getLogger('funding_model')
logger.setLevel(logging.DEBUG)
logger.addHandler(get_console_handler())
logger.propagate=False
