import logging
import os

import alembic.config
from flask import Flask
from sqlalchemy import create_engine
from sqlalchemy.engine import Engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import scoped_session, sessionmaker
from sqlalchemy_utils import database_exists, create_database

from project.api import Config, ROOT

_logger = logging.getLogger('mlapi')

# The Base object serves as a base class for all SQLAlchemy models in the application.
Base = declarative_base()


def establish_db_engine(*, config: Config) -> Engine:
    """
    This function creates a SQLAlchemy Engine object from a configuration object.
    The Engine is the starting point for any SQLAlchemy application and represents a connection to
    the database.
    """

    # Get the database URL from the configuration
    database_uri = config.SQLALCHEMY_DATABASE_URI

    # Ensure the database exists
    database_exists(database_uri) or create_database(database_uri)

    # Create and return the SQLAlchemy engine
    engine = create_engine(database_uri)

    # Log the successful creation of the database connection
    _logger.info(f"Database connection established with URI: {database_uri}")

    return engine


# The create_db_session function creates a SQLAlchemy Session, which is the primary interface
# for persistence operations.
def create_db_session(*, engine: Engine) -> scoped_session:
    """Broadly speaking, the Session establishes all conversations with the database.

     It represents a “holding zone” for all the objects which you’ve loaded or
     associated with it during its lifespan.
     """
    return scoped_session(sessionmaker(autocommit=False, autoflush=False, bind=engine))



def establish_database_session(*, engine: Engine) -> scoped_session:
    """
    This function sets up a SQLAlchemy Session, which is the primary interface for all database operations.
    The Session serves as a workspace for all the objects loaded into it during its lifespan.
    """
    # Sessionmaker is a factory that constructs new Session objects when called. It's being configured
    # with autocommit=False and autoflush=False to control the behavior of the session.
    # autocommit=False means that the session will not commit changes to the database automatically,
    # we will need to manually call session.commit() to commit changes.
    # autoflush=False means that the session will not automatically flush or synchronize changes to
    # the database (in simple terms, updating the database with the session state) unless
    # session.flush() is called or a query is executed.
    session_factory = sessionmaker(bind=engine, autoflush=False, autocommit=False)

    # Create a scoped session, scoped_session is used to ensure that the session is thread-local,
    # meaning each thread will have its own session to be able to  work with the database independently
    # without interfering with other threads.
    db_session = scoped_session(session_factory)

    return db_session

# The init_database function initializes the database by creating an engine and a session, and attaching
# the session to the Flask application. It also defines a teardown function to remove the session after the
# app context ends.
def init_database(app: Flask, config: Config, db_session=None) -> None:
    """
    This function is responsible for initializing the database by creating an engine and a session,
    and attaching the session to the Flask application. It uses a teardown function  which ensures that resources are properly cleaned up and that subsequent requests will start with a fresh session.

    """

    if not db_session:
        engine = create_db_engine_from_config(config=config)
        db_session = create_db_session(engine=engine)
    # attaching the session to the app object, here we are making the session available globally in the application. This means that anywhere in your code where you have access to the app object, you can also access the database session using app.db_session.
    app.db_session = db_session

    @app.teardown_appcontext
    def shutdown_session(exception=None):
        db_session.remove()

# The run_migrations function runs the database migrations using Alembic. This is typically done when
# the application starts, to ensure the database schema is up-to-date.
def run_migrations():
    """Run the DB migrations prior to the tests."""

    # alembic looks for the migrations in the current
    # directory so we change to the correct directory.
    os.chdir(str(ROOT))
    alembicArgs = ["--raiseerr", "upgrade", "head"]
    alembic.config.main(argv=alembicArgs)
