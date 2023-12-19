import argparse
import os
import time
import typing as t
from random import randint, choice

import pandas as pd
import requests
from gradient_boosting_model.config.core import config
from gradient_boosting_model.processing.data_management import load_dataset

# This line is creating a URL for the local server where the API is running. It uses the os.getenv
# function to get the environment variable DB_HOST, which contain the hostname of the database server.
# If DB_HOST is not set, it defaults to "localhost". The :5000 at the end specifies the port number
# where the server is listening.
LOCAL_URL = f'http://{os.getenv("DB_HOST", "localhost")}:5000'

# This dictionary defines the HTTP headers that will be sent with the requests to the API. It's telling
# the server that the client accepts JSON responses and will be sending JSON-formatted data.
HEADERS = {"Accept": "application/json", "Content-Type": "application/json"}

# This dictionary defines the minimum and maximum values for the LotArea feature of the housing data.
# These values are used to generate random integers within this range.
LOT_AREA_MAP = {"min": 1470, "max": 56600}

# This dictionary defines the range for the FirstFlrSF feature, which represents the size of the
# first floor in square feet.
FIRST_FLR_SF_MAP = {"min": 407, "max": 5095}

# This dictionary defines the range for the SecondFlrSF feature, which represents the size of the
# second floor in square feet.
SECOND_FLR_SF_MAP = {"min": 0, "max": 1862}

# This tuple contains possible values for the BsmtQual feature, which represents the quality of the
# basement. The values are abbreviations for different quality levels, such as 'Good',
# 'Typical/Average', 'Excellent', and 'Fair'.
BSMT_QUAL_VALUES = ('Gd', 'TA', 'Ex', 'Fa')


def _generate_random_int(value: int, value_ranges: t.Mapping) -> int:
    """Generate random integer within a min and max range."""
    random_value = randint(value_ranges["min"], value_ranges["max"])
    return int(random_value)


def _select_random_category(value: str, value_options: t.Sequence) -> str:
    """Select random category given a sequence of categories."""
    random_category = choice(value_options)
    return random_category


def _prepare_inputs(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Prepare input data by removing key rows with NA values."""
    clean_inputs_df = dataframe.dropna(
        subset=config.model_config.features + ["KitchenQual", "LotFrontage"]
    ).copy()

    clean_inputs_df.loc[:, "FirstFlrSF"] = clean_inputs_df["FirstFlrSF"].apply(
        _generate_random_int, value_ranges=FIRST_FLR_SF_MAP
    )
    clean_inputs_df.loc[:, "SecondFlrSF"] = clean_inputs_df["SecondFlrSF"].apply(
        _generate_random_int, value_ranges=SECOND_FLR_SF_MAP
    )
    clean_inputs_df.loc[:, "LotArea"] = clean_inputs_df["LotArea"].apply(
        _generate_random_int, value_ranges=LOT_AREA_MAP
    )

    clean_inputs_df.loc[:, "BsmtQual"] = clean_inputs_df["BsmtQual"].apply(
        _select_random_category, value_options=BSMT_QUAL_VALUES
    )

    return clean_inputs_df


def populate_database(n_predictions: int = 500, anomaly: bool = False) -> None:
    """
    The populate_database function is designed to generate a specified number of random
    predictions using the test data and save them to a database.

     - n_predictions: The number of predictions to generate, with a default value of 500.
     - anomaly: A boolean flag that, when set to True, will generate an outlier in the data.

    Before running this script, ensure that the
    API and Database docker containers are running.
    """

    print(f"Preparing to generate: {n_predictions} predictions.")

    # Load the test dataset
    test_inputs_df = load_dataset(file_name="test.csv")

    # call the _prepare_inputs function to clean the data and generate random values for certain features.
    clean_inputs_df = _prepare_inputs(dataframe=test_inputs_df)

    # If the cleaned data has fewer rows than the desired number of predictions, it prints a message
    # suggesting that the script needs to be extended to handle more predictions.
    if len(clean_inputs_df) < n_predictions:
        print(
            f"If you want {n_predictions} predictions, you need to"
            "extend the script to handle more predictions."
        )
    # If the anomaly flag is True, the function sets extremely low values for certain features to
    # create an outlier in the data.
    if anomaly:
        n_predictions = 1
        clean_inputs_df.loc[:, "FirstFlrSF"] = 1
        clean_inputs_df.loc[:, "LotArea"] = 1
        clean_inputs_df.loc[:, "OverallQual"] = 1
        clean_inputs_df.loc[:, "GrLivArea"] = 1

    # This line replaces any remaining NA values in the dataframe with None.
    clean_inputs_df = clean_inputs_df.where(pd.notnull(clean_inputs_df), None)

    # The function iterates over the rows of the cleaned dataframe and sends each row as a JSON
    # payload in a POST request to the API endpoint defined by LOCAL_URL. It stops once the desired
    # number of predictions has been reached.
    # This line starts a loop that goes through the DataFrame row by row. The iterrows() method returns
    # an iterator yielding index and row data for each row. index is the index of the row, and data
    # is the data in the row as a pandas Series.
    for index, data in clean_inputs_df.iterrows():

        # checks if the current index (which represents the number of predictions processed so far)
        # is greater than the desired number of predictions (n_predictions). If this condition is true,
        # it means the function has already processed the required number of predictions, and it should
        # stop the loop.
        if index > n_predictions:
            if anomaly:
                print('Created 1 anomaly')
            break

        response = requests.post(
            f"{LOCAL_URL}/v1/predictions/regression",
            headers=HEADERS,
            json=[data.to_dict()],
        )

        # After each POST request, it checks for HTTP errors and raises an exception if any are found.
        response.raise_for_status()

        # Every 50 predictions, it prints a message indicating progress.
        if index % 50 == 0:
            print(f"{index} predictions complete")

            # To avoid overloading the server, the function includes a half-second pause after each prediction.
            time.sleep(0.5)

    print("Prediction generation complete.")

# This block of code is used to sets up command-line argument parsing for the script.
if __name__ == "__main__":

    # This line initializes the anomaly variable to False. This variable is used to control
    # whether the script will generate normal predictions or an anomaly (outlier).
    anomaly = False

    # An ArgumentParser object is created from the argparse module. This object is used to handle
    # command-line arguments. The description parameter provides a brief description of what
    # the script does.
    parser = argparse.ArgumentParser(
        description='Send random requests to House Price API.')

    # This line adds the --anomaly argument to the parser. When this argument is specified
    # on the command line(python populate_database.py --anomaly), it will trigger the script to
    # generate an anomaly. The help parameter provides a description of what the argument does.
    parser.add_argument('--anomaly', help="generate unusual inputs")

    # This line parses the command-line arguments provided when the script is run. The parsed
    # arguments are stored in the args variable.
    args = parser.parse_args()

    # This conditional block checks if the --anomaly argument was included in the command
    # line (args.anomaly would be True if it was). If so, it prints a message indicating that
    # unusual inputs will be generated, and sets the anomaly variable to True.
    if args.anomaly:
        print("Generating unusual inputs")
        anomaly = True

    # Finally, this line calls the populate_database function with the desired number of predictions
    # (n_predictions=500) and the anomaly flag set according to the command-line argument.
    populate_database(n_predictions=500, anomaly=anomaly)
