# Specifies the base image to use, which is a slim version of Python 3.9.5 based on
# Debian Buster. It’s a lightweight image that contains the minimal packages needed to run Python.
FROM python:3.9.5-slim-buster

# Creates a directory /opt/app in the container’s filesystem to store the application code.
RUN mkdir -p /opt/app
# Copies the requirements directory from your project into the /opt/app/requirements directory
# inside the container.
COPY requirements /opt/app/requirements
# Upgrades pip, the Python package installer, to its latest version inside the container.
RUN pip install --upgrade pip

# The next RUN command is a series of commands connected with && to reduce the number of layers in the image.
# apt-get update -y: Updates the list of available packages and their versions, but it does not install or upgrade any packages.
# apt-get install -y make: Installs the make utility, which is used to control the generation of executables and
# other non-source files of a program from the program’s source files.
# apt-get install -y libffi-dev gcc: Installs the libffi development files and the GNU Compiler Collection
# (gcc). libffi is often used for calling C functions from within Python, and gcc is used for compiling
# C extensions for Python.
# apt-get install -y curl: Installs curl, a tool for transferring data with URLs, which is useful for
# testing APIs.
# apt-get install -y libpq-dev: Installs the development files for the PostgreSQL database, which are
# necessary to compile the psycopg2 package, a PostgreSQL adapter for Python.

RUN apt-get update -y && \
 	apt-get install -y make && \
 	apt-get install -y libffi-dev gcc && \
 	# for swagger
 	apt-get install -y curl && \
 	# for postgres driver
 	apt-get install -y libpq-dev

# Installs the Python packages specified in the requirements.txt file.
RUN pip install -r /opt/app/requirements/requirements.txt
# Sets the PYTHONPATH environment variable to include the /opt/app/ directory. This ensures that Python
# will include the application code in its module search path.
ENV PYTHONPATH "${PYTHONPATH}:/opt/app/"

# Adds the contents of the current directory (where the Dockerfile is located) into the /opt/app directory
# inside the container.
ADD . /opt/app
# Sets the working directory to /opt/app. Any subsequent RUN, CMD, ENTRYPOINT, COPY, and ADD
# instructions will be executed in this directory.
WORKDIR /opt/app
