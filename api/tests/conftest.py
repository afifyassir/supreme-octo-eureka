import os

from unittest import mock
import pytest
from gradient_boosting_model.processing.data_management import load_dataset
from sqlalchemy_utils import create_database, database_exists

from project.api import create_app
from project.api import TestingConfig
from project.api import core


# The _db fixture is responsible for setting up the database for the test session. It
# checks if the test database exists, and if not, it creates one. Then, it configures the
# database engine using the TestingConfig and runs the database migrations to prepare
# the database schema.
@pytest.fixture(scope='session')
def _db():

    # Retrieving the database URL from the TestingConfig class, which should contain the
    # configuration specific to the testing environment.
    db_url = TestingConfig.SQLALCHEMY_DATABASE_URI

    # checking if the database exists using the database_exists function from the sqlalchemy_utils
    # package. If the database does not exist, it creates a new one using the create_database function.
    if not database_exists(db_url):
        create_database(db_url)

    # Creates a database engine using the create_db_engine_from_config function
    # from the core module. This engine is configured according to the TestingConfig and
    # will be used to interact with the database.
    engine = core.create_db_engine_from_config(config=TestingConfig())

    # The evars dictionary is created with the key ALEMBIC_DB_URI set to the database URL. This is
    # used to configure Alembic, a database migration tool. The mock.patch.dict function is used to
    # temporarily set the environment variable ALEMBIC_DB_URI for the duration of the migrations.
    evars = {"ALEMBIC_DB_URI": db_url}

    # In this case, mock.patch.dict() is used to patch os.environ, which is the dictionary that holds
    # the environment variables for the running process. By patching os.environ,
    # we're altering the runtime environment in which the system operates. This is done to
    # ensure that when Alembic runs its migrations, it does so using the test database specified
    # by the db_url rather than any production database that might be configured in the actual environment
    # variables.
    with mock.patch.dict(os.environ, evars):

        # The core.run_migrations() function is called to apply the database schema
        # migrations to the test database. This ensures that the database schema is up to
        # date with the current state of the application's models.
        core.run_migrations()

    # The function yields the database engine. This means that the engine will be available for use
    # in other fixtures or tests until the test session ends.
    yield engine


# The _db_session fixture creates a database session that can be used during testing. This session
# is yielded, which means it will be available for the duration of the test session.
@pytest.fixture(scope='session')
def _db_session(_db):
    """ Create DB session for testing.
    """
    session = core.create_db_session(engine=_db)
    yield session

# The app fixture creates an instance of the Flask application configured for testing.
# It uses the TestingConfig and the test database session. The application context is then
# made available for testing.
# The @pytest.fixture(scope='session') decorator indicates that this function is a pytest fixture
# with a session scope. This means that the fixture will be executed once at the start of the test
# session and will be available to all tests in that session.
@pytest.fixture(scope='session')
def app(_db_session):

    # This creates an instance of the Flask application with the testing configuration and the
    # test database session passed in.
    app = create_app(config_object=TestingConfig(), db_session=_db_session).app

    # This statement activates the application context. The application context is a Flask concept that
    # makes certain variables globally accessible to the Flask application during a request. In the
    # context of testing, it allows the tests to run as if they were handling a request.
    with app.app_context():

        # The yield app statement makes the application instance available to the tests.
        yield app


# The client fixture provides a test client for the Flask application. This client can be used
# to make requests to the application and test its endpoints.
# The client function takes the app fixture as an argument.
@pytest.fixture
def client(app):

    # Creating a test client for the Flask application. This test client provides methods
    # to simulate requests to the application, such as get, post, put, delete, etc.
    # The with statement is used as a context manager to ensure that the test client is properly
    # created and destroyed.
    with app.test_client() as client:

        # The yield client statement makes the test client available to the tests.
        yield client


# The test_inputs_df fixture loads a test dataset from a CSV file. This dataset is presumably
# used to test the gradient boosting model's predictions. The deep=True parameter ensures that
# a deep copy of the data is made, so the original data is not modified during testing.
@pytest.fixture
def test_inputs_df():
    # Load the gradient boosting test dataset which
    # is included in the model package
    test_inputs_df = load_dataset(file_name="test.csv")
    return test_inputs_df.copy(deep=True)
