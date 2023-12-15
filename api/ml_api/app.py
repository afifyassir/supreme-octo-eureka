import logging

import connexion
from sqlalchemy.orm import scoped_session

from project.api import Config
from project.api import setup_metrics
from project.api import init_database

_logger = logging.getLogger('mlapi')


def create_app(
    *, config_object: Config, db_session: scoped_session = None
) -> connexion.App:
    """
        This function is a factory function that sets up and returns a new instance of a web application.
        It takes a config_object which contains configuration settings for the application, and an
        optional db_session which is a scoped session for database interactions.
    """
    # Initialize a connexion.App instance. Connexion is a framework that simplifies the creation
    # of REST APIs. The parameters include:
    # __name__: The name of the application module or package, typically used for Flask applications.
    # debug: A boolean that indicates whether the application is in debug mode, which is taken from the config_object.
    # specification_dir: The directory where the API specifications are located, in this case, "spec/".
    connexion_app = connexion.App(
        __name__, debug=config_object.DEBUG, specification_dir="spec/"
    )

    # Retrieve the Flask App
    flask_app = connexion_app.app

    # The Flask app's configuration is updated with settings from the config_object.
    flask_app.config.from_object(config_object)

    # The init_database function is called to initialize the database with the Flask app and the
    # provided configuration. This step involves setting up the database connection and session management.
    init_database(flask_app, config=config_object, db_session=db_session)

    # The setup_metrics function is called to set up Prometheus monitoring for the application.
    # This allows for tracking various metrics about the application's performance and usage.
    setup_metrics(flask_app)

    # The connexion_app.add_api("api.yaml") call adds the API specification to the Connexion app.
    # The specification defines the endpoints and operations of the API, and Connexion uses this to
    # automatically handle routing and input validation.
    connexion_app.add_api("api.yaml")
    _logger.info("Application instance created")

    return connexion_app
