import os
import json
import argparse

import uvicorn
from pydantic import BaseModel
from fastapi import FastAPI, Request, HTTPException

from semantic_cache.manager.data_manager.init_schema import *

from semantic_cache.adapter.api import put, get, flush

app = FastAPI()


class CacheData(BaseModel):
    prompt: str
    answer: str


@app.get("/")
async def default():
    return "Semntic cache running"


@app.post("/put")
async def cache_put(cache_data: CacheData):
    put(cache_data.prompt, cache_data.answer)
    return "successfully updated the cache"


def main():
    from starlette.middleware.cors import CORSMiddleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    init_redis_schema()
    init_milvus_schema()

    uvicorn.run(app, port=8000)


if __name__ == "__main__":
    main()
