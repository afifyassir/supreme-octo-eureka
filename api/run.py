import prometheus_client
from werkzeug.middleware.dispatcher import DispatcherMiddleware

from project.api import create_app
from project.api import DevelopmentConfig, setup_app_logging

# This run file is a script used to start a web application. It sets up the application
# and its environment before running it.

# This line creates an instance of DevelopmentConfig, which contains configuration
# settings for the application
_config = DevelopmentConfig()

# The setup_app_logging function is called to configure logging for the application. This
# is important for tracking events, errors, and other significant occurrences during the app's runtime.
setup_app_logging(config=_config)

# The create_app function is called with the configuration object, which initializes
# the application and returns an instance of it.
main_app = create_app(config_object=_config).app

# This block sets up the DispatcherMiddleware, which allows the application to serve both the
# main app and the Prometheus metrics at different endpoints. The main app is served at the root,
# while the Prometheus metrics are available at the /metrics endpoint.
application = DispatcherMiddleware(
        app=main_app.wsgi_app,
        mounts={'/metrics': prometheus_client.make_wsgi_app()}
    )



if __name__ == "__main__":
    main_app.run(port=_config.SERVER_PORT, host=_config.SERVER_HOST)
