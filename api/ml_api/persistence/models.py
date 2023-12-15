from sqlalchemy import Column, String, DateTime, Integer
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.sql import func

from project.api import Base

# The LassoModelPredictions class represents a table in a database that is designed to store predictions
# made by a Lasso model.
# This class inherits from Base, which is a declarative base provided by SQLAlchemy. It allows the class
# to be associated with a specific table in a database.
class LassoModelPredictions(Base):

    # This attribute specifies the name of the table in the database that this class will be linked to.
    # __tablename__ is a convention used to define the name of the table in the database that a model
    # class should be mapped to.
    __tablename__ = "regression_model_predictions"

    # This is a column in the table that serves as the primary key. It's of type Integer, which means it
    # will store integer values. The primary_key=True argument indicates that this column will uniquely
    # identify each record in the table
    id = Column(Integer, primary_key=True)

    # This is a column that stores a string of length 36, which is typically used to store UUIDs.
    # The nullable=False argument means that this column cannot be empty; it must have a value
    # for every record.
    user_id = Column(String(36), nullable=False)

    # This column stores the date and time when the prediction was made. It's of type DateTime with
    # timezone support enabled. The server_default=func.now() sets the default value of this column to
    # the current time on the database server when a new record is created, and index=True means that
    # this column will be indexed, which can speed up queries involving this column.
    # The timezone=True parameter indicates that the column will store timezone-aware datetime values.
    # This means that the datetime values will include information about the timezone,
    datetime_captured = Column(
        DateTime(timezone=True), server_default=func.now(), index=True
    )

    # This column also stores a string of length 36 and cannot be empty. It's used to store the
    # version of the model that made the prediction.
    model_version = Column(String(36), nullable=False)

    # This is a column of type JSONB, which is a binary JSON column type provided by PostgreSQL. It allows
    # for storing JSON data in a more efficient binary format. This column will store the inputs to the
    # model that were used to make the prediction.
    inputs = Column(JSONB)

    # This column will store the output of the model.
    outputs = Column(JSONB)

# The GradientBoostingModelPredictions class represents a table in a database that is designed to store predictions
# made by a Gradient Boosting model.
# This class inherits from Base, which is a declarative base provided by SQLAlchemy. It allows the class
# to be associated with a specific table in a database.
class GradientBoostingModelPredictions(Base):

    __tablename__ = "gradient_boosting_model_predictions"
    id = Column(Integer, primary_key=True)
    user_id = Column(String(36), nullable=False)
    datetime_captured = Column(
        DateTime(timezone=True), server_default=func.now(), index=True
    )
    model_version = Column(String(36), nullable=False)
    inputs = Column(JSONB)
    outputs = Column(JSONB)
