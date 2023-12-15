import logging
import os
import pathlib
import sys
from logging.config import fileConfig

from project import api

# define a logging formatter  that specifies the format for log messages, including the timestamp,
# logger name, log level, function name, line number, and the actual log message.
FORMATTER = logging.Formatter(
    "%(asctime)s — %(name)s — %(levelname)s —" "%(funcName)s:%(lineno)d — %(message)s"
)

# Project Directories
ROOT = pathlib.Path(api.__file__).resolve().parent.parent

APP_NAME = 'ml_api'


class Config:
    # tells the app whether to show detailed errors when something goes wrong.
    DEBUG = False
    # tells the app whether it's being tested.
    TESTING = False
    # tells the app whether it's in development, testing, or production.
    #  os.getenv is a function in Python that is used to get the value of an environment variable.
    # it checks if there's an environment variable called FLASK_ENV set, and if not, it will
    # use "production" as the default value.
    ENV = os.getenv("FLASK_ENV", "production")
    # SERVER_PORT and SERVER_HOST tell the app where to listen for web traffic.
    SERVER_PORT = int(os.getenv("SERVER_PORT", 5000))
    SERVER_HOST = os.getenv("SERVER_HOST", "0.0.0.0")
    # tells the app how much detail to include in the logs, which are records of what
    # the app is doing. More detail is helpful when you're developing, but in production,
    # we might only want to log when something goes wrong.
    LOGGING_LEVEL = os.getenv("LOGGING_LEVEL", logging.INFO)
    SHADOW_MODE_ACTIVE = os.getenv('SHADOW_MODE_ACTIVE', True)
    SQLALCHEMY_DATABASE_URI = (
        f"postgresql+psycopg2://{os.getenv('DB_USER')}:"
        f"{os.getenv('DB_PASSWORD')}@{os.getenv('DB_HOST')}/{os.getenv('DB_NAME')}"
    )
    # DB config matches docker container
    DB_USER = os.getenv("DB_USER", "user")
    DB_PASSWORD = os.getenv("DB_PASSWORD", "password")
    DB_PORT = os.getenv("DB_PORT", 6609)
    DB_HOST = os.getenv("DB_HOST", "0.0.0.0")
    DB_NAME = os.getenv("DB_NAME", "ml_api_dev")


class DevelopmentConfig(Config):
    DEBUG = True
    ENV = "development"  # do not use in production!
    LOGGING_LEVEL = logging.DEBUG


class TestingConfig(Config):
    DEBUG = True
    TESTING = True
    LOGGING_LEVEL = logging.DEBUG

    # DB config matches test docker container
    DB_USER = os.getenv("DB_USER", "test_user")
    DB_PASSWORD = os.getenv("DB_PASSWORD", "password")
    DB_PORT = os.getenv("DB_PORT", 6608)
    DB_HOST = os.getenv("DB_HOST", "0.0.0.0")
    DB_NAME = "ml_api_test"
    SQLALCHEMY_DATABASE_URI = (
        f"postgresql+psycopg2://{DB_USER}:"
        f"{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
    )


class ProductionConfig(Config):
    DB_USER = os.getenv("DB_USER", "user")
    DB_PASSWORD = os.getenv("DB_PASSWORD", "password")
    DB_PORT = os.getenv("DB_PORT", 6609)
    DB_HOST = os.getenv("DB_HOST", "database")
    DB_NAME = os.getenv("DB_NAME", "ml_api")
    SQLALCHEMY_DATABASE_URI = (
        f"postgresql+psycopg2://{DB_USER}:"
        f"{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
    )


def get_console_handler():
    """
    A function that creates a console logging handler that outputs logs to sys.stdout (the console).
    """

    # Initializes a new logging.StreamHandler instance, which is a type of logging handler that
    # can send log messages to a stream, such as sys.stdout or sys.stderr. In this case, sys.stdout (standard output)
    # is used, which means the log messages will be printed out to the console.
    console_handler = logging.StreamHandler(sys.stdout)

    # The handler's formatter is set to the FORMATTER variable, which was defined earlier in the code.
    # This formatter specifies the layout of the log messages.
    console_handler.setFormatter(FORMATTER)
    return console_handler


def setup_app_logging(config: Config) -> None:
    """
      A function that sets up application logging using the provided configuration.
      It disables irrelevant loggers to reduce noise and configures logging from a file named
     'gunicorn_logging.conf' located at the ROOT directory.
     """
    _disable_irrelevant_loggers()

    # Use fileConfig() to read a logging configuration file, which is typically a separate
    # file that defines how logging should be handled (e.g., formatting, handlers, log levels).
    # In this case, it's reading from 'gunicorn_logging.conf', which is a configuration file specifically
    # used for setting up logging when we're running a Flask application with Gunicorn, which is
    # a popular WSGI HTTP server for UNIX systems. This file contains settings that define how
    # Gunicorn should handle logging for our application.
    fileConfig(ROOT / 'gunicorn_logging.conf')

    # Create a logger named 'mlapi'. This is the logger that our application will
    # use to output log messages.
    logger = logging.getLogger('mlapi')

    # Sets the logging level of the 'mlapi' logger to the level specified in the config object
    # passed to the function. The logging level determines the severity of messages that will
    # be logged (e.g., DEBUG, INFO, WARNING, ERROR, CRITICAL). This allows you to control which
    # messages are important enough to be recorded.
    logger.setLevel(config.LOGGING_LEVEL)


def _disable_irrelevant_loggers() -> None:
    """
       A helper function that sets the logging level to warning for a list of loggers
       that are not relevant to the application, to prevent them from cluttering the log output.
    """
    for logger_name in (
        "connexion.apis.flask_api",
        "connexion.apis.abstract",
        "connexion.decorators",
        "connexion.operation",
        "connexion.operations",
        "connexion.app",
        "openapi_spec_validator",
    ):
        logging.getLogger(logger_name).level = logging.WARNING
