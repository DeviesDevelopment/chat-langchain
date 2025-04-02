FROM python:3.11-buster

# Install system packages OpenCV needs
RUN apt-get update && apt-get install -y libgl1 libglib2.0-0

RUN pip install poetry==1.5.1

RUN poetry config virtualenvs.create false

COPY ./pyproject.toml ./poetry.lock* ./

RUN poetry install --no-interaction --no-ansi --no-root --no-directory

COPY ./backend/*.py ./backend/

RUN poetry install  --no-interaction --no-ansi

# Copy entrypoint script
COPY entrypoint.sh .
RUN chmod +x entrypoint.sh

# Run entrypoint script when container starts
CMD ["./entrypoint.sh"]

