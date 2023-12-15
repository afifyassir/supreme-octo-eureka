import enum
import json
import logging
import typing as t

import numpy as np
import pandas as pd
from regression_model.predict import make_prediction as make_live_prediction
from sqlalchemy.orm.session import Session

from project.api import (
    LassoModelPredictions,
    GradientBoostingModelPredictions,
)
from gradient_boosting_model.predict import make_prediction as make_shadow_prediction

# Set up a logger instance in Python using the logging module.
# This is a method provided by the logging module. It returns a logger instance associated with the
# specified name. If a logger with that name already exists, it will return the existing instance; otherwise,
# it will create a new one.
_logger = logging.getLogger('mlapi')


SECONDARY_VARIABLES_TO_RENAME = {
    "FirstFlrSF": "1stFlrSF",
    "SecondFlrSF": "2ndFlrSF",
    "ThreeSsnPortch": "3SsnPorch",
}

# ModelType is an enumeration that defines constants for different types of models.
# It uses Python's enum module to create an Enum class. Each member of the Enum
# has a name (e.g., LASSO, GRADIENT_BOOSTING) and a value (e.g., "lasso", "gradient_boosting").
# Enumerations are used to create a set of named constants that can make code more readable and less error-prone.
class ModelType(enum.Enum):
    LASSO = "lasso"
    GRADIENT_BOOSTING = "gradient_boosting"

# PredictionResult is a named tuple that is used to store the results of a prediction. Named
# tuples are a subclass of tuples that allow you to access elements by name instead of by index,
# which can make code more readable. In this case, PredictionResult has three fields: errors,
# predictions, and model_version.
class PredictionResult(t.NamedTuple):
    errors: t.Any
    predictions: np.array
    model_version: str

# MODEL_PREDICTION_MAP is a dictionary that maps each ModelType to a corresponding prediction function.
# This allows you to dynamically select the prediction function based on the model type. For example,
# if you have a ModelType.LASSO, you can use MODEL_PREDICTION_MAP[ModelType.LASSO] to get
# the make_live_prediction function.
MODEL_PREDICTION_MAP = {
    ModelType.GRADIENT_BOOSTING: make_shadow_prediction,
    ModelType.LASSO: make_live_prediction,
}

# The PredictionPersistence class is designed to handle the storage of machine learning model
# predictions into a database. It has two main responsibilities: making predictions using
# the specified model and saving those predictions.
class PredictionPersistence:
    def __init__(self, *, db_session: Session, user_id: str = None) -> None:
        self.db_session = db_session
        if not user_id:
            # in reality, here we would use something like a UUID for anonymous users
            # and if we had user logins, we would record the user ID.
            self.user_id = "007"

    def make_save_predictions(
        self, *, db_model: ModelType, input_data: t.List
    ) -> PredictionResult:
        """
        The make_save_predictions function within the PredictionPersistence class
        is responsible for two main tasks: making predictions using a specified
        machine learning model and saving those predictions to a database.
        """
        # Checking the type of model (db_model) for which predictions are to be made.
        # If the model is of type LASSO, it performs a renaming of certain input columns.
        # This is necessary to maintain compatibility with the expected input format of the LASSO model.
        if db_model == ModelType.LASSO:
            live_frame = pd.DataFrame(input_data)
            # The method to_dict is called on the DataFrame to convert it into a dictionary.
            # The orient='records' parameter means that each row of the DataFrame will be converted
            # into a dictionary where the keys are the column names and the values are the row values.
            # The result is a list of dictionaries, with each dictionary representing a single record
            # (row) from the DataFrame.
            input_data = live_frame.rename(
                columns=SECONDARY_VARIABLES_TO_RENAME
            ).to_dict(orient="records")

        # Using a mapping (MODEL_PREDICTION_MAP) to find the correct prediction function
        # based on the model type. This function is called with the provided input_data to
        # generate predictions.
        result = MODEL_PREDICTION_MAP[db_model](input_data=input_data)

        # retrieve any errors from the prediction results. If there are no errors, proceed to
        # format the predictions into a list.
        errors = None
        try:
            errors = result["errors"]
        except KeyError:
            pass

        prediction_result = PredictionResult(
            errors=errors,
            predictions=result.get("predictions").tolist() if not errors else None,
            model_version=result.get("version"),
        )

        # If there are no errors, the function calls the save_predictions method to save the prediction
        # results to the database.
        if prediction_result.errors:
            return prediction_result

        self.save_predictions(
            inputs=input_data, prediction_result=prediction_result, db_model=db_model
        )

        return prediction_result

    def save_predictions(
        self,
        *,
        inputs: t.List,
        prediction_result: PredictionResult,
        db_model: ModelType,
    ) -> None:
        """
        This function creates new database record, which corresponds to either a LassoModelPredictions
        or GradientBoostingModelPredictions object, depending on the model type (db_model) used for
        the prediction.This record includes several pieces of information:
            - user_id: The identifier of the user for whom the prediction was made. This could be a real user ID or a placeholder if the user is anonymous.
            - model_version: The version of the model that generated the prediction, as stored in the PredictionResult.
            - inputs: The input data that was fed into the model, serialized into a JSON format.
            - outputs: The predictions made by the model, also serialized into a JSON format.
        """
        if db_model == db_model.LASSO:
            prediction_data = LassoModelPredictions(
                user_id=self.user_id,
                model_version=prediction_result.model_version,
                inputs=json.dumps(inputs),
                outputs=json.dumps(prediction_result.predictions),
            )
        else:
            prediction_data = GradientBoostingModelPredictions(
                user_id=self.user_id,
                model_version=prediction_result.model_version,
                inputs=json.dumps(inputs),
                outputs=json.dumps(prediction_result.predictions),
            )
        # Add the new record to the database session.
        self.db_session.add(prediction_data)
        # Commit the session in order to save the record to the database.
        self.db_session.commit()
        _logger.debug(f"saved data for model: {db_model}")
