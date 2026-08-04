"""VeRL/SGLang client for openslide_crop."""

from __future__ import annotations

import asyncio
import json
from typing import Any, Optional, Tuple
from uuid import uuid4

import aiohttp
from verl.tools.base_tool import BaseTool
from verl.tools.schemas import OpenAIFunctionToolSchema


class OpenSlideCropTool(BaseTool):
    def __init__(self, config: dict, tool_schema: OpenAIFunctionToolSchema):
        super().__init__(config, tool_schema)
        self.base_url = str(config.get("base_url", "http://127.0.0.1:9010")).rstrip("/")
        self.api_key = config.get("api_key") or None
        self.timeout = float(config.get("timeout", 120))
        self.retries = int(config.get("max_retries", 2))
        self.instances: dict[str, str] = {}

    async def create(self, instance_id: Optional[str] = None, **kwargs) -> str:
        slide_id = str(kwargs.get("slide_id") or "")
        if not slide_id:
            raise ValueError("openslide_crop requires slide_id in create_kwargs")
        instance_id = instance_id or str(uuid4())
        self.instances[instance_id] = slide_id
        return instance_id

    async def execute(
        self, instance_id: str, parameters: dict[str, Any], **kwargs
    ) -> Tuple[str, float, dict]:
        slide_id = self.instances.get(instance_id)
        if slide_id is None:
            raise KeyError(f"unknown openslide instance: {instance_id}")
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        payload = {
            "instance_id": instance_id,
            "name": self.name,
            "slide_id": slide_id,
            "arguments": parameters,
        }
        error: Exception | None = None
        for attempt in range(self.retries + 1):
            try:
                timeout = aiohttp.ClientTimeout(total=self.timeout)
                async with aiohttp.ClientSession(timeout=timeout) as session:
                    async with session.post(
                        self.base_url + "/execute", json=payload, headers=headers
                    ) as response:
                        body = await response.text()
                        if response.status >= 400:
                            raise RuntimeError(f"HTTP {response.status}: {body[:1000]}")
                        result = json.loads(body)
                metrics = dict(result.get("metrics") or {})
                metrics.update({"returned_images": result.get("images") or [], "tool": self.name})
                return json.dumps(result["result"], ensure_ascii=False), 0.0, metrics
            except (aiohttp.ClientError, asyncio.TimeoutError, RuntimeError, json.JSONDecodeError) as exc:
                error = exc
                if attempt < self.retries:
                    await asyncio.sleep(min(2**attempt, 5))
        raise RuntimeError(f"openslide_crop failed: {error}") from error

    async def calc_reward(self, instance_id: str, **kwargs) -> float:
        return 0.0

    async def release(self, instance_id: str, **kwargs) -> None:
        self.instances.pop(instance_id, None)

