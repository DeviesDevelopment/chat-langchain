FROM python:3.12

RUN pip install poetry==1.8.2

RUN poetry config virtualenvs.create false

COPY ./pyproject.toml ./poetry.lock* ./

RUN poetry install --no-interaction --no-ansi --no-root --no-directory

COPY ./backend/*.py ./backend/

RUN poetry install  --no-interaction --no-ansi

# env variables
ENV OPENAI_API_KEY=""
ENV TARGET_SOURCE_CHUNKS=4
ENV LOAD_WEB_URL="https://www.devies.se/"
ENV LOAD_DIR_PATH=""
ENV PERSIST_DIRECTORY="db"

CMD exec uvicorn --app-dir=backend main:app --host 0.0.0.0 --port 8080
