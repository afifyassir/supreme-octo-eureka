# The code in middleware.py is a monitoring middleware that's used to collect metrics
# about the web application's performance. It tracks the number of requests, the latency
# of each request, and other useful data that can help understand how the app is performing
# in real-time. This information is particularly valuable for maintaining the reliability and
# efficiency of the application. By monitoring these metrics, developers can identify and address
# issues, optimize performance, and ensure that the app provides a good user experience.

from flask import request, Flask
from flask.wrappers import Response
from prometheus_client import Counter, Histogram
import time

from project.api import APP_NAME


# A Counter that tracks the total number of HTTP requests made to the application,
# labeled by app_name, method, endpoint, and http_status.
REQUEST_COUNT = Counter(
    name='http_request_count',
    documentation='App Request Count',
    labelnames=['app_name', 'method', 'endpoint', 'http_status']
)

# A Histogram that measures the latency (time taken) of HTTP requests, labeled by app_name and endpoint.
REQUEST_LATENCY = Histogram(
    name='http_request_latency_seconds',
    documentation='Request latency',
    labelnames=['app_name', 'endpoint']
)

# Starts a timer for a request by recording the current time.
# The function's main purpose is to record the start time of an HTTP request that is being
# handled by the Flask application.
# request is a global object provided by Flask that contains all the information about the current
# HTTP request that the server is handling.
# _prometheus_metrics_request_start_time is a custom attribute that is being set on the request object.
# The underscore at the beginning of the attribute name is a convention in Python indicating that
# this attribute is intended for internal (inside the middleware.py file) use and should not
# be accessed directly outside of this context.
# _prometheus_metrics_request_start_time is not a built-in attribute of the request object provided by
# Flask. It is a custom attribute that the middleware script adds to the request object for its own use.
def start_timer() -> None:
    """Get start time of a request."""
    request._prometheus_metrics_request_start_time = time.time()

# Stops the timer for a request and records the request latency in the REQUEST_LATENCY histogram.
def stop_timer(response: Response) -> Response:
    """Get stop time of a request.."""
    request_latency = time.time() - request._prometheus_metrics_request_start_time
    REQUEST_LATENCY.labels(
        app_name=APP_NAME,
        endpoint=request.path).observe(request_latency)
    return response

# Records data about the request, such as the method, endpoint, and HTTP status code, in
# the REQUEST_COUNT counter.
def record_request_data(response: Response) -> Response:
    """
    The function's main goal is to increment a counter metric for monitoring
    purposes every time an HTTP request is processed by the application.

    """

    # labels: This method is used to assign specific labels to the metric. Labels are
    # key-value pairs that provide additional dimensions to the metric, allowing for more
    # detailed and granular monitoring.
    REQUEST_COUNT.labels(
        app_name=APP_NAME,
        # This label captures the HTTP method of the request (e.g., GET, POST).
        method=request.method,
        # This label captures the endpoint that was accessed (e.g., /api/data).
        endpoint=request.path,
        # This label captures the HTTP status code of the response (e.g., 200, 404).
        http_status=response.status_code).inc()
    return response

# Sets up the metrics collection by adding start_timer, record_request_data, and stop_timer as hooks
# to run before and after each request within the Flask app.
def setup_metrics(app: Flask) -> None:
    """
    sets up the necessary hooks for capturing metrics about HTTP requests in a Flask application.
    """

    # app is an instance of the Flask class, which represents the Flask application. The app
    # parameter is passed to the function so that it can register the hooks with the application.

    # app.before_request(start_timer): This line registers the start_timer function to be called
    # before each request is processed. The before_request hook is a feature provided by Flask that
    # allows developers to run specific code before the request hits the actual view function
    # (the code that generates the response).
    app.before_request(start_timer)

    # app.after_request(record_request_data): This line registers the record_request_data function
    # to be called after each request is processed but before the response is sent to the client. T
    # he after_request hook is another feature provided by Flask that allows developers to run specific
    # code after the view function has run.
    app.after_request(record_request_data)

    # app.after_request(stop_timer): This line registers the stop_timer function to be called after
    # the record_request_data function. Flask allows multiple functions to be registered to the same
    # after_request hook, and they will be called in the order they were registered.
    app.after_request(stop_timer)
