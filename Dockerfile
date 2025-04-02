FROM python:3.11-buster

RUN pip install poetry==1.5.1

RUN poetry config virtualenvs.create false

COPY ./pyproject.toml ./poetry.lock* ./

RUN poetry install --no-interaction --no-ansi --no-root --no-directory

COPY ./backend/*.py ./backend/

COPY .env .

RUN poetry install  --no-interaction --no-ansi

# Copy entrypoint script
COPY entrypoint.sh .
RUN chmod +x entrypoint.sh

# Run entrypoint script when container starts
CMD ["./entrypoint.sh"]

