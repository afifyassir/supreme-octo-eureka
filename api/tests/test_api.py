import json
import time

import numpy as np
import pytest

from project.api import SECONDARY_VARIABLES_TO_RENAME
from project.api import (
    GradientBoostingModelPredictions,
    LassoModelPredictions,
)
from gradient_boosting_model.processing.data_management import load_dataset

# Checks if the root endpoint / is healthy by asserting a 200 OK status code and the expected response body.
# The decorator @pytest.mark.integration marks the test as an integration test, meaning it tests how different pieces of
# the application interact with each other, in this case, how the API endpoint interacts with
# the machine learning model and the data it receives.
# while the @pytest.mark.integration decorator is not strictly necessary for the code to function, it
# is a best practice that enhances the maintainability, organization, and execution control of the test
# suite. It's part of writing clean, understandable, and manageable code, especially as the size of the
# codebase and the number of tests grow.
@pytest.mark.integration
def test_health_endpoint(client):
    # When
    response = client.get("/")

    # Then
    assert response.status_code == 200
    assert json.loads(response.data) == {"status": "ok"}


@pytest.mark.integration
# The @pytest.mark.parametrize(...) decorator is used for parameterization, which allows running
# the same test function multiple times with different arguments. Here, it's used to test two
# different API endpoints: v1/predictions/regression and v1/predictions/gradient. For each endpoint,
# there's an expected number of predictions that should be returned by the API.4
# The variable expected_no_predictions holds the expected number of predictions the API
# should return after filtering out certain rows. For the regression endpoint, 1451 predictions
# are expected, and for the gradient endpoint, 1457 predictions are expected.
# client is a fixture that provides a test client for the application. It's used to make requests
# to the API endpoints.
# The decorator takes two main parameters:
#  1. A string that specifies the names of the arguments that you want to vary for each test run.
#  These names should match the parameters used in the test function.
#  2. A list of tuples, where each tuple contains a set of values that correspond to the argument names.
#  Each tuple represents a different test case.
# Both square brackets [] and parentheses () can be used to define a list of tuples in Python.
@pytest.mark.parametrize(
    "api_endpoint, expected_no_predictions",
    (
        (
            "v1/predictions/regression",
            # test csv contains 1459 rows
            # we expect 2 rows to be filtered
            1451,
        ),
        (
            "v1/predictions/gradient",
            # we expect 8 rows to be filtered
            1457,
        ),
    ),
)
def test_prediction_endpoint(
    api_endpoint, expected_no_predictions, client, test_inputs_df
):
    # Given
    # The test data is loaded.
    test_inputs_df = load_dataset(file_name="test.csv")

    # If the endpoint being tested is the regression model, the column names in the DataFrame are
    # renamed according to the SECONDARY_VARIABLES_TO_RENAME mapping. This is done to match the
    # expected input format of the regression model.
    if api_endpoint == "v1/predictions/regression":
        test_inputs_df.rename(columns=SECONDARY_VARIABLES_TO_RENAME, inplace=True)

    # A POST request is made to the API endpoint using the client. The test input data is
    # converted to a list of dictionaries (orient="records") and sent as JSON in the request body.
    response = client.post(api_endpoint, json=test_inputs_df.to_dict(orient="records"))

    # Checking that the response status code is 200 OK, indicating a successful request.
    assert response.status_code == 200
    # Parsing the response data from JSON and asserts that there are no errors in the response.
    data = json.loads(response.data)
    assert data["errors"] is None
    # Asserting that the number of predictions returned by the API matches the expected
    # number (expected_no_predictions).
    assert len(data["predictions"]) == expected_no_predictions


# parameterizations allows us to try many combinations of data
# within the same test, see the pytest docs for details:
# https://docs.pytest.org/en/latest/parametrize.html
@pytest.mark.parametrize(
    "field, field_value, index, expected_error",
    (
        (
            "BldgType",
            1,  # expected str
            33,
            {"33": {"BldgType": ["Not a valid string."]}},
        ),
        (
            "GarageArea",  # model feature
            "abc",  # expected float
            45,
            {"45": {"GarageArea": ["Not a valid number."]}},
        ),
        (
            "CentralAir",
            np.nan,  # nan not allowed
            34,
            {"34": {"CentralAir": ["Field may not be null."]}},
        ),
        ("LotArea", "", 2, {"2": {"LotArea": ["Not a valid integer."]}}),
    ),
)
# The function test_prediction_validation is designed to ensure that the machine learning API
# properly validates the input data it receives. It checks that the API returns the correct error
# messages when given invalid data.
# The function is set up to test four different scenarios, each with a different type of validation error:
#    - A string is expected, but an integer is provided (BldgType).
#    - A float is expected, but a string is provided (GarageArea).
#    - A non-null value is expected, but np.nan (a missing value) is provided (CentralAir).
#    - An integer is expected, but an empty string is provided (LotArea).
@pytest.mark.integration
def test_prediction_validation(
    field, field_value, index, expected_error, client, test_inputs_df
):
    # The test function modifies the test_inputs_df DataFrame by inserting the incorrect value
    # at the specified index and field.
    test_inputs_df.loc[index, field] = field_value

    # It then sends a POST request to the /v1/predictions/gradient endpoint with the modified data.
    response = client.post(
        "/v1/predictions/gradient", json=test_inputs_df.to_dict(orient="records")
    )

    # The function checks that the response status code is 400, indicating a client error due to
    # invalid input data.
    assert response.status_code == 400
    # It parses the response data from JSON and compares it to the expected_error to ensure that the
    # API returned the correct validation error message.
    data = json.loads(response.data)
    assert data == expected_error

# The function test_prediction_data_saved is a test case designed to verify that the machine
# learning API correctly saves prediction data to the database
@pytest.mark.integration
def test_prediction_data_saved(client, app, test_inputs_df):

    # Querying the database to get the current count of records in the GradientBoostingModelPredictions
    # and LassoModelPredictions tables.
    initial_gradient_count = app.db_session.query(
        GradientBoostingModelPredictions
    ).count()
    initial_lasso_count = app.db_session.query(LassoModelPredictions).count()

    # The test sends a POST request to the /v1/predictions/regression endpoint of the API using the
    # client. This request includes the test input data (test_inputs_df) converted to JSON format.
    # The API is expected to process this data, make predictions using the regression model, and save
    # the results to the database.
    response = client.post(
        "/v1/predictions/regression", json=test_inputs_df.to_dict(orient="records")
    )

    # Checking that the response status code is 200, indicating that the request was successful.
    assert response.status_code == 200

    # Querying the LassoModelPredictions table again to check if the count has increased by one, which
    # would mean that a new record has been successfully added.
    assert (
        app.db_session.query(LassoModelPredictions).count() == initial_lasso_count + 1
    )

    # Since the GradientBoostingModelPredictions save operation occurs on a separate asynchronous
    # thread, the function waits for 2 seconds using time.sleep(2) to give the operation time to complete.
    # After the pause, it checks that the count of the GradientBoostingModelPredictions table has also
    # increased by one.
    time.sleep(2)
    assert (
        app.db_session.query(GradientBoostingModelPredictions).count()
        == initial_gradient_count + 1
    )
