"""Main entrypoint for the app."""
from typing import Optional, Union
from uuid import UUID

from chain import ChatRequest, answer_chain
from fastapi import FastAPI
from fastapi.responses import PlainTextResponse
from fastapi.middleware.cors import CORSMiddleware
from langserve import add_routes
from pydantic import BaseModel


app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
)


add_routes(
    app,
    answer_chain,
    path="/chat",
    input_type=ChatRequest,
    config_keys=["metadata", "configurable", "tags"],
)


class SendFeedbackBody(BaseModel):
    run_id: UUID
    key: str = "user_score"

    score: Union[float, int, bool, None] = None
    feedback_id: Optional[UUID] = None
    comment: Optional[str] = None


class UpdateFeedbackBody(BaseModel):
    feedback_id: UUID
    score: Union[float, int, bool, None] = None
    comment: Optional[str] = None


@app.get("/health", response_class=PlainTextResponse)
def health_check():
    return "OK"


class GetTraceBody(BaseModel):
    run_id: UUID


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8080)
