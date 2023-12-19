from unittest import mock
import pytest

from project.api import PredictionPersistence, ModelType

from project.api import (
    GradientBoostingModelPredictions,
    LassoModelPredictions,
)


# The function test_data_access checks the functionality of the PredictionPersistence class, which
# is responsible for saving prediction data to the database.
# The @pytest.mark.parametrize decorator is used to run the test with different parameters. In this
# case, it's testing the persistence of predictions for two different types of models: Gradient Boosting
# and Lasso.
# The parameter model_type will take the values ModelType.GRADIENT_BOOSTING and ModelType.LASSO during the tests.
# The parameter model will take the corresponding SQLAlchemy model classes GradientBoostingModelPredictions
# and LassoModelPredictions.
# test_inputs_df is a fixture that provides the test input data as a pandas DataFrame.
@pytest.mark.parametrize(
    "model_type, model,",
    (
        (ModelType.GRADIENT_BOOSTING, GradientBoostingModelPredictions),
        (ModelType.LASSO, LassoModelPredictions),
    ),
)
def test_data_access(model_type, model, test_inputs_df):

    # The mock_session is a mock object created to simulate the database session. It uses
    # MagicMock is a specific tool from the mock toolbox. It's like a magic wand that can create a
    # fake version of almost anything. When you use it, you get something that looks and behaves
    # like the real thing, but it's actually just pretend.
    mock_session = mock.MagicMock()

    # _persistence is an instance of PredictionPersistence that is initialized with the mock_session.
    _persistence = PredictionPersistence(db_session=mock_session)

    # When
    _persistence.make_save_predictions(
        db_model=model_type, input_data=test_inputs_df.to_dict(orient="records")
    )

    # The test checks that mock_session.commit is called once, which would commit(save) the transaction
    # to the database if it were a real session.
    assert mock_session.commit.call_count == 1

    # It also checks that mock_session.add is called once, which would add a new record to the session.
    assert mock_session.add.call_count == 1

    # It asserts that the object added to the session is an instance of the correct model class.
    assert isinstance(mock_session.add.call_args[0][0], model)
