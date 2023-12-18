import os

from alembic import context
from sqlalchemy import engine_from_config, pool

# Import the models so the changes in them are automatically reflected in the
# generated migrations.
from project.api import models  # noqa
from project.api import DevelopmentConfig as user_config
from project.api import Base

# The line config = context.config is a way of setting up a shortcut to access the configuration
# settings that Alembic uses to manage database migrations. When we run an Alembic command, Alembic
# automatically reads the alembic.ini file and loads its settings into the context.config object.
# This is part of Alembic's initialization process, which happens before any migrations are run.
# The context object is our assistant in the world of database migrations. It holds all
# the necessary information and tools to help us update our database safely and efficiently.
# The context object has access to the configuration details, to the metadata and can operate in two
# modes, online and offline.
config = context.config

# This variable retrieves the database URL from an environment variable named ALEMBIC_DB_URI.
# If it's not set, it falls back to the SQLALCHEMY_DATABASE_URI from the user_config.
database_url = os.environ.get("ALEMBIC_DB_URI", user_config.SQLALCHEMY_DATABASE_URI)

# This sets the main database connection URL in the Alembic configuration to the database_url.
# This line is using the config shortcut we talked about earlier to set a main option for Alembic:
# "sqlalchemy.url": This is the name of the option we're setting. It tells Alembic where to find the database.
# It is a convention to use "sqlalchemy.url" to set the database connection URL for SQLAlchemy.
# database_url: This is the value we're setting for that option. It's the address of the database we
# determined in the previous line.
config.set_main_option("sqlalchemy.url", database_url)

# This is the MetaData object that holds all the declared models in the Base class.
# It's used by Alembic to determine which tables and columns exist and need to be considered for migration.
# Alembic uses the information in the Base class's metadata to determine if there have been any changes to
# the database schema that require migrations, this is done by comparing the current database schema
# (the structure of the database as it exists in the database itself) with the schema defined by our
# models' metadata (the structure of the database as defined in our Python code).
target_metadata = Base.metadata

# This function is used to run migrations without an actual database connection ("offline" mode).
# It's useful for generating SQL scripts that can be applied to the database later.
def run_migrations_offline():
    """
    The run_migrations_offline function is preparing a list of instructions for
    updating the database based on the changes we've made in our code. It doesn't need
    an actual connection to the database to do this, which is why it's called "offline." Later, we
    can take this list and apply the changes to the database when we're ready.
    """
    # Getting the Database Address.
    # The config.get_main_option("sqlalchemy.url") is retrieving the URL that was previously set
    # by the line: database_url = os.environ.get("ALEMBIC_DB_URI", user_config.SQLALCHEMY_DATABASE_URI)
    url = config.get_main_option("sqlalchemy.url")

    # Setting Up the Changes.
    context.configure(
        url=url, target_metadata=target_metadata, literal_binds=True,
    )
    # Preparing to Make Changes.
    with context.begin_transaction():

        # Writing Down the Changes
        context.run_migrations()

# This function runs migrations with a live database connection ("online" mode).
# It creates a connection to the database and applies the migration changes directly.
def run_migrations_online():
    """
    Run migrations in 'online' mode.
    In this scenario we need to create a user_ratings
    and associate a connection with the context.
    """

    # Getting ready, this is like gathering all the instructions and tools we need before starting the work.
    # config.config_ini_section refers to a specific section within the Alembic configuration.
    # By default, it's the [alembic] section in the alembic.ini file, which is like the main chapter
    # that contains general instructions for Alembic operations.
    # config.get_section is a method that retrieves all the settings from a specified section of the
    # configuration  file.
    alembic_config = config.get_section(config.config_ini_section)

    # Creating a connection, the engine_from_config function creates a connection to
    # our database based on the configuration we've gathered.
    # The engine_from_config Function is a function  provided by SQLAlchemy.
    # The engine_from_config function takes configuration settings and uses them to create an "engine".
    # An engine is an object that manages the connection to the database. It's like a control panel
    # that allows us to communicate with the database—sending commands, receiving responses, and
    # performing operations.
    # The alembic_config variable, which you've gathered from the [alembic] section of the alembic.ini
    # file, contains the necessary settings to create this engine. These settings include the database
    # URL, connection parameters, and other options that define how to interact with the database.
    connectable = engine_from_config(
        alembic_config, prefix="sqlalchemy.", poolclass=pool.NullPool,
    )

    # Starting the work by opening the communication with the database.
    with connectable.connect() as connection:

        # context.configure(...) tells Alembic to use the live connection to the database
        # for the upcoming changes.
        context.configure(
            connection=connection, target_metadata=target_metadata,
        )

        # context.run_migrations() is where the actual work happens. Alembic goes through
        # the list of changes (migrations) that need to be made to the database and carries them out
        with context.begin_transaction():
            context.run_migrations()


# So, this code is deciding whether to prepare a list of changes to be made later (offline)
# or to go ahead and make the changes right now (online).
# check to see if Alembic is set to run without a live connection to the database.
# The context object in Alembic knows which mode the system is running in—offline or online—based on
# how the Alembic command is executed. If we run an Alembic command with the --sql flag, it
# tells Alembic to operate in offline mode. In this mode, Alembic doesn't connect to the database;
# instead, it prints the SQL statements that would be executed to the console or a file.
# This is useful for generating migration scripts that we can review or apply manually later.
# If we run an Alembic command without the --sql flag, it operates in online mode. In this mode,
# Alembic connects to the database and applies the migrations directly.
# The context object has a method called is_offline_mode() that checks whether the --sql flag
# was used when the Alembic command was run. Based on this check, the context object determines the mode.
if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
