# For details on Makefiles, see the section notes.
# The name of the Docker image, set to ml_api
NAME=ml_api
# This variable is used to tag the Docker image with a specific version. The command git rev-parse HEAD
# retrieves the latest commit hash from your Git repository, which is then used as the version tag.
# This is a common practice for versioning Docker images, as it allows us to track which version of
# the code corresponds to which Docker image..
VERSION=$(shell git rev-parse HEAD)
# This stands for the Docker repository where our image will be stored. Before we can use this Makefile,
# we need to replace UPDATEME with the actual repository name. This could be a repository on Docker Hub,
# or any other Docker registry we are using..
REPO=UPDATEME
# This is the password for our Docker repository. For security reasons, it is not included in the Makefile
# and must be provided by the user. You will need to replace UPDATEME with the actual password to ensure
# that the Makefile can log in to the Docker repository and push the Docker image.
PASSWORD=UPDATEME

# In a Makefile, targets are usually associated with files. Make uses the presence or absence of a file
# with the same name as the target to determine whether the commands associated with that target need to
# be run. If the file exists and is up to date, Make will not run the commands for that target.
# However, there are times when you want to run commands regardless of whether a file exists or not.
# These are called ‘phony’ targets. By declaring a target as phony, you tell Make that this target is not
# associated with a file; rather, it’s just a name for a set of commands to be executed.
# .PHONY: This is a special built-in target used to declare phony targets. Any target listed
# as a prerequisite to .PHONY is considered phony.
# This is particularly useful for targets that perform actions like cleaning up build directories (clean),
# running tests (test), or, as in your case, pushing Docker images and running services.

.PHONY: run-service-development tag-push-master tag-push-local db-migrations


tag-push-local:
	@echo "+ $@"
	docker login --username $(REPO) --password $(PASSWORD)
	env TARGET=$(VERSION) docker-compose -f docker/docker-compose-ci-candidate.yml build
	docker push $(REPO)/$(NAME):$(VERSION)

tag-push-master:
	@echo "+ $@"
	docker login --username $(REPO) --password $(PASSWORD)
	env TARGET=master docker-compose -f docker/docker-compose-ci-master.yml build
	docker push $(REPO)/$(NAME):master

# start up Flask API service
# @echo "+ $@": This command prints out the name of the target being executed, which in this case
# is run-service-development. The @ symbol at the beginning of the line tells Make not to print the command
# itself before executing it, which is the default behavior. The + is just a character chosen to precede the
# output for clarity, and $@ is a special Make variable that represents the name of the target.

run-service-development:
	@echo "+ $@"
	python run.py

# @echo "+ $@": This command prints the name of the target (run-service-wsgi) before executing the
# following commands. The @ symbol prevents Make from printing the command itself.
# gunicorn: This is the command to run Gunicorn, the WSGI server for running Python web applications.
# --bind 0.0.0.0:5000: This option tells Gunicorn to bind to all IP addresses available on the
# machine (0.0.0.0) and listen for requests on port 5000. This makes your API service accessible from
# other computers.
# --workers=1: This sets the number of worker processes for handling requests. In this case, it’s set
# to 1, which means only one worker will handle all the requests. For production environments, you might
# want to increase this number to handle more traffic.
# --log-config gunicorn_logging.conf: This option specifies a configuration file for logging.
# gunicorn_logging.conf should contain settings for how Gunicorn logs information about the running service.
# --log-level=DEBUG: Sets the logging level to DEBUG, which means Gunicorn will output detailed logs for
# debugging purposes. Other levels include INFO, WARNING, ERROR, and CRITICAL.
# --access-logfile=- and --error-logfile=-: These options set the access and error log files.
# The - means that logs will be printed to the standard output (stdout) and standard error (stderr),
# respectively.
# run:application: This tells Gunicorn to look for the application object in the run module (which is
# typically a Python file named run.py). The application object is the WSGI application that Gunicorn
# will serve.

run-service-wsgi:
	@echo "+ $@"
	gunicorn --bind 0.0.0.0:5000 \
			 --workers=1 \
			 --log-config gunicorn_logging.conf \
             --log-level=DEBUG \
             --access-logfile=- \
             --error-logfile=- \
             run:application

# The db-migrations target is designed to handle database migrations using Alembic,
# a lightweight database migration tool for Python.
# @echo "+ $@": This command prints the name of the target (db-migrations) before executing the
# following commands. The @ symbol prevents Make from printing the command itself.
# PYTHONPATH=.: This sets the PYTHONPATH environment variable to the current directory (denoted by .).
# This is important because it tells Python where to look for modules when running scripts. By setting it
# to the current directory, we ensure that Python can find and import our project’s modules correctly.
# alembic -c alembic.ini upgrade head: This command performs the database migrations.
# alembic: This is the command-line tool provided by Alembic for handling migrations.
# -c alembic.ini: The -c option specifies the configuration file for Alembic, which is alembic.ini in
# this case. This file contains all the necessary settings for Alembic to connect to our database and
# manage migrations.
# upgrade head: This tells Alembic to apply the migrations up to the latest version, referred to as
# head. When you create new migrations, Alembic generates scripts that describe the changes to be made
# to the database schema. The upgrade head command applies all pending migrations in the order they were
# created, updating the database schema to the latest version.

db-migrations:
	@echo "+ $@"
	PYTHONPATH=. alembic -c alembic.ini upgrade head
