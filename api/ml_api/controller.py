import json
import logging
import threading

from flask import request, jsonify, Response, current_app
from prometheus_client import Histogram, Gauge, Info
from regression_model import __version__ as live_version

from project.api import APP_NAME
from project.api import PredictionPersistence, ModelType
from gradient_boosting_model import __version__ as shadow_version
from gradient_boosting_model.predict import make_prediction

_logger = logging.getLogger('mlapi')

# This is a Histogram object from the prometheus_client library. A histogram is used to observe and
# track the distribution of numerical data. PREDICTION_TRACKER is configured to track the distribution
# of house price predictions made by the machine learning model.
# The name parameter assigns a name to the histogram, which is used for identification in Prometheus.
# The documentation parameter provides a description of what the histogram is tracking.
# The labelnames parameter specifies the labels that will be used to differentiate the metrics.
# In this case, it includes the application name, model name, and model version.
PREDICTION_TRACKER = Histogram(
    name='house_price_prediction_dollars',
    documentation='ML Model Prediction on House Price',
    labelnames=['app_name', 'model_name', 'model_version']
)

# A Gauge is another metric type from the prometheus_client that represents a value that can go up and down.
# The PREDICTION_GAUGE is set up to track the current value of house price predictions, which could be useful
# for tracking the minimum and maximum predicted values over time.
# Similar to the histogram, it has a name, documentation, and labelnames.
PREDICTION_GAUGE = Gauge(
    name='house_price_gauge_dollars',
    documentation='ML Model Prediction on House Price for min max calcs',
    labelnames=['app_name', 'model_name', 'model_version']
)

# The method labels is used to set the labels for the gauge with specific values.
# These labels will be attached to the gauge metric whenever it's updated with a new value.
PREDICTION_GAUGE.labels(
                app_name=APP_NAME,
                model_name=ModelType.LASSO.name,
                model_version=live_version)

# The Info metric type is used to provide static information that doesn't change over time.
# MODEL_VERSIONS is an Info object that captures the version details of the models in use.
# It's not used for tracking numerical data but rather for providing metadata about the model versions.
MODEL_VERSIONS = Info(
    'model_version_details',
    'Capture model version information',
)

# The info method sets the information for the MODEL_VERSIONS metric.
MODEL_VERSIONS.info({
    'live_model': ModelType.LASSO.name,
    'live_version': live_version,
    'shadow_model': ModelType.GRADIENT_BOOSTING.name,
    'shadow_version': shadow_version})

# It is a health check endpoint that returns the status of the service, it used to determine
# if the service is running and responsive.
def health():
    # checks if the incoming request is a GET request. Health checks are usually done
    # with GET requests because they are retrieving the status of the service without making any changes.
    if request.method == "GET":

        # create a dictionary with a key "status" and a value "ok", indicating that the
        # service is operational.
        status = {"status": "ok"}

        # log the status dictionary at the debug level. This is useful for debugging purposes
        # to ensure that the health check endpoint is being called and is returning the correct status.
        _logger.debug(status)
        return jsonify(status)

# The main prediction endpoint that handles POST requests with input data for predictions. It logs the
# input data, saves predictions from the live model (LASSO), and optionally from a shadow model
# (Gradient Boosting) if shadow mode is active. It also tracks predictions using the Prometheus
# metrics defined earlier.
def predict():
    if request.method == "POST":
        # Step 1: Extract POST data from request body as JSON.
        json_data = request.get_json()
        # It writes down (logs) the data it received for record-keeping.
        for entry in json_data:
            _logger.info(entry)

        # Step 2a: Get and save live model predictions
        persistence = PredictionPersistence(db_session=current_app.db_session)
        result = persistence.make_save_predictions(
            db_model=ModelType.LASSO, input_data=json_data
        )

        # Step 2b: Get and save shadow predictions asynchronously
        # This is an asynchronous operation, the code below uses Python's threading library to run the
        # shadow model's predictions in a separate thread, which is like having a separate worker
        # doing a task at the same time as the main worker. This is done so that the main process,
        # which handles the Lasso model predictions, doesn't have to wait for the Gradient Boosting
        # model to finish its predictions in other terms, it makes predictions with the shadow model too,
        # but it does this in the background, so it doesn't slow down the main task.

        # Checks if the shadow mode is turned on in the application's configuration. If it's on, it means
        # the service should use the shadow model for predictions.
        # the current_app object is a special context-aware object that proxies the active application
        # instance. It's part of Flask's application context, which allows you to access application-specific
        # data throughout the code without having to pass the application object around.
        if current_app.config.get("SHADOW_MODE_ACTIVE"):
            # Writes a debug message to the log, indicating that the shadow model is being called asynchronously.
            _logger.debug(
                f"Calling shadow model asynchronously: "
                f"{ModelType.GRADIENT_BOOSTING.value}"
            )
            # Creates a new thread. The target parameter is set to persistence.make_save_predictions, which
            # is the function that will run in the new thread. The kwargs parameter provides the arguments
            # for that function: the shadow model and the input data.
            thread = threading.Thread(
                target=persistence.make_save_predictions,
                kwargs={
                    "db_model": ModelType.GRADIENT_BOOSTING,
                    "input_data": json_data,
                },
            )
            thread.start()

        # Step 3: Handle errors
        # If something goes wrong while making predictions with the Lasso model, the function
        # will let the sender know by sending back an error message.
        if result.errors:
            _logger.warning(f"errors during prediction: {result.errors}")
            return Response(json.dumps(result.errors), status=400)

        # Step 4: Monitoring
        #  It keeps track of the predictions made by the Lasso model using two tools: PREDICTION_TRACKER
        #  and PREDICTION_GAUGE. These tools help to see how well the model is doing over time.
        for _prediction in result.predictions:
            # The labels method is used to add metadata to these metrics, such as the application name,
            # model name, and model version. This helps in filtering and querying the metrics data later on.
            # The observe method takes a numerical value (in this case, a prediction value from the Lasso model)
            # and records it in the histogram. The histogram keeps track of all the observed values and organizes
            # them into buckets to create a distribution, which helps in understanding the frequency and range of
            # the prediction values.
            PREDICTION_TRACKER.labels(
                app_name=APP_NAME,
                model_name=ModelType.LASSO.name,
                model_version=live_version).observe(_prediction)

            # The set method updates the gauge with a new value (again, a prediction value from the
            # Lasso model). Unlike histograms, gauges represent a single numerical value that can go
            # up or down over time. By using set, the gauge is updated to reflect the most recent prediction
            # value each time a prediction is made.
            PREDICTION_GAUGE.labels(
                app_name=APP_NAME,
                model_name=ModelType.LASSO.name,
                model_version=live_version).set(_prediction)
        _logger.info(
            f'Prediction results for model: {ModelType.LASSO.name} '
            f'version: {result.model_version} '
            f'Output values: {result.predictions}')

        # Step 5: Prepare prediction response
        return jsonify(
            {
                "predictions": result.predictions,
                "version": result.model_version,
                "errors": result.errors,
            }
        )


def predict_previous():
    if request.method == "POST":
        # Step 1: Extract POST data from request body as JSON
        json_data = request.get_json()

        # Step 2: Access the model prediction function (also validates data)
        result = make_prediction(input_data=json_data)

        # Step 3: Handle errors
        errors = result.get("errors")
        if errors:
            return Response(json.dumps(errors), status=400)

        # Step 4: Split out results
        # Regression model interface has changed
        # so no need to call tolist
        predictions = result.get("predictions")
        version = result.get("version")

        # Step 5: Save predictions
        persistence = PredictionPersistence(db_session=current_app.db_session)
        persistence.save_predictions(
            inputs=json_data,
            model_version=version,
            predictions=predictions,
            db_model=ModelType.GRADIENT_BOOSTING,
        )

        # Step 6: Prepare prediction response
        return jsonify(
            {"predictions": predictions, "version": version, "errors": errors}
        )
