"""FastAPI service exposing only openslide_crop."""

from __future__ import annotations

import os
import time
from typing import Any

from .schema import TOOL_NAME
from .slide import OpenSlideCropService, SlideError, encode_image


def create_app(service: OpenSlideCropService):
    from fastapi import FastAPI, HTTPException, Request
    from fastapi.responses import JSONResponse

    app = FastAPI(title="Medical SVS OpenSlide Tool")

    @app.middleware("http")
    async def authenticate(request: Request, call_next):
        api_key = os.getenv("OPENSLIDE_TOOL_API_KEY")
        if api_key and request.headers.get("Authorization") != f"Bearer {api_key}":
            return JSONResponse(status_code=401, content={"detail": "invalid API key"})
        return await call_next(request)

    @app.get("/health")
    async def health():
        return {"status": "ok", "tool": TOOL_NAME, "registered_slides": len(service.manifest)}

    @app.post("/execute")
    async def execute(payload: dict[str, Any]):
        started = time.perf_counter()
        try:
            if payload.get("name") != TOOL_NAME:
                raise SlideError(f"unsupported tool: {payload.get('name')}")
            result, image = service.crop(
                str(payload.get("slide_id") or ""), payload.get("arguments") or {}
            )
            return {
                "status": "success",
                "result": result,
                "images": [encode_image(image)],
                "metrics": {"latency_ms": round((time.perf_counter() - started) * 1000, 3)},
            }
        except SlideError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"OpenSlide execution failed: {exc}") from exc

    return app

