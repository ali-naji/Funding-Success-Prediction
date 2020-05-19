from api import config
from api.controller import prediction_app
from flask import Flask

logger = config.get_logger(__name__)


def create_app(configObject):
    flask_app = Flask('ml_api')
    flask_app.config.from_object(configObject)
    flask_app.register_blueprint(prediction_app)

    logger.info("Application Instance Created")

    return flask_app
