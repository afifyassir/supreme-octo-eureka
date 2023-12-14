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
def setup_database(app: Flask, config: Config, session=None) -> None:
    """
    This function is responsible for initializing the database by creating an engine and a session,
    and attaching the session to the Flask application. It uses a teardown function  which ensures that resources are properly cleaned up and that subsequent requests will start with a fresh session.

    """

    if session is None:
        engine = establish_db_engine(config=config)
        session = establish_database_session(engine=engine)
    # attaching the session to the app object, here we are making the session available globally in
    # the application. This means that anywhere in our code where we have access to the app object,
    # we can also access the database session using app.db_session.
    app.db_session = session

    # The @app.teardown_appcontext is a decorator that registers the function as a teardown function.
    # The function shutdown_session(exception=None) is defined to remove the session with db_session.remove().
    # This ensures that each request starts with a fresh session and there's no cross-talk between requests.
    @app.teardown_appcontext
    def cleanup(exception=None):
        session.remove()

# The run_migrations function runs the database migrations using Alembic. This is typically done when
# the application starts, to ensure the database schema is up-to-date.
# Database migrations are a safe and organized way to update the database as it grows
def run_migrations():
    """
    This is a utility function that sets up the environment for Alembic and then runs the database migrations
    to the latest version.
    Alembic, which is a lightweight database migration tool for usage with the SQLAlchemy Database Toolkit for Python.
    It's like a version control system for our database schema. While SQLAlchemy lets us build and change
    the database directly, Alembic helps manage and apply those changes in a controlled and organized way,
    especially when multiple people are involved or when we need to maintain a consistent database
    structure across different environments (like development, testing, and production). It ensures
    that our machine learning project's data infrastructure can evolve safely and efficiently.
    """

    # By changing the directory to ROOT, we're ensuring that Alembic has access to all the migration
    # scripts in our project.
    os.chdir(str(ROOT))
    # sets up the arguments for Alembic. These arguments tell Alembic to raise an error if something
    # goes wrong (--raiseerr), to apply the migrations up to the latest version (upgrade), and to use
    # the "head" version, which is the latest revision in the migration scripts.
    alembicArgs = ["--raiseerr", "upgrade", "head"]
    # Run the migrations with the specified arguments.
    alembic.config.main(argv=alembicArgs)
