"""
OpenAI API interface for LLMs
"""

import os
import time
import json
import uuid
import shutil
import asyncio
import logging
import pathlib
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Union

import openai

from openevolve.config import LLMConfig
from openevolve.llm.base import LLMInterface

logger = logging.getLogger(__name__)


def _iso_now() -> str:
    return datetime.now(tz=timezone.utc).isoformat()


def _mkdir_and_clear(p: pathlib.Path) -> None:
    if os.path.isdir(str(p)):
        try: shutil.rmtree(p)
        except: pass
    
    # Recreate the empty directory
    p.mkdir(parents=True, exist_ok=True)


def _build_display_prompt(messages: List[Dict[str, str]]) -> str:
    """
    Render messages into a single plain-text 'prompt' for the left panel.
    We show role headers, this is friendlier for humans than raw JSON.
    """
    chunks = []

    for m in messages:
        role = m.get("role", "user").upper()
        content = m.get("content", "")
        chunks.append(f"### {role}\n{content}\n")
    
    return "\n".join(chunks).rstrip() + "\n"


class OpenAILLM(LLMInterface):
    """LLM interface using OpenAI-compatible APIs"""

    def __init__(
        self,
        model_cfg: Optional[dict] = None,
    ):
        self.model = model_cfg.name
        self.system_message = model_cfg.system_message
        self.temperature = model_cfg.temperature
        self.top_p = model_cfg.top_p
        self.max_tokens = model_cfg.max_tokens
        self.timeout = model_cfg.timeout
        self.retries = model_cfg.retries
        self.retry_delay = model_cfg.retry_delay
        self.api_base = model_cfg.api_base
        self.api_key = model_cfg.api_key
        self.random_seed = getattr(model_cfg, "random_seed", None)
        self.reasoning_effort = getattr(model_cfg, "reasoning_effort", None)

        # Manual mode
        self.manual_mode = bool(getattr(model_cfg, "manual_mode", False))
        manual_dir = getattr(model_cfg, "manual_queue_dir", None)

        if self.manual_mode:
            if manual_dir is None:
                raise ValueError("You should provide `manual_queue_dir` in manual mode")

            self.manual_queue_dir = pathlib.Path(manual_dir)
            _mkdir_and_clear(self.manual_queue_dir)
            logger.info(f"Manual mode is ON. Queue dir: {self.manual_queue_dir}")

        # Set up API client
        # OpenAI client requires max_retries to be int, not None
        max_retries = self.retries if self.retries is not None else 0

        if not self.manual_mode:
            if self.api_base == "https://api.openai.com/v1":  # removing api_base argument
                self.client = openai.OpenAI(
                    api_key=self.api_key,
                    timeout=self.timeout,
                    max_retries=max_retries,
                )
            else:
                self.client = openai.OpenAI(
                    api_key=self.api_key,
                    base_url=self.api_base,
                    timeout=self.timeout,
                    max_retries=max_retries,
                )

        # Only log unique models to reduce duplication
        if not hasattr(logger, "_initialized_models"):
            logger._initialized_models = set()

        if self.model not in logger._initialized_models:
            logger.info(f"Initialized OpenAI LLM with model: {self.model}")
            logger._initialized_models.add(self.model)

    async def generate(self, prompt: str, **kwargs) -> str:
        """Generate text from a prompt"""
        return await self.generate_with_context(
            system_message=self.system_message,
            messages=[{"role": "user", "content": prompt}],
            **kwargs,
        )

    async def generate_with_context(
        self, system_message: str, messages: List[Dict[str, str]], **kwargs
    ) -> str:
        """Generate text using a system message and conversational context"""
        # Prepare messages with system message
        formatted_messages = [{"role": "system", "content": system_message}]
        formatted_messages.extend(messages)

        # Set up generation parameters
        # Define OpenAI reasoning models that require max_completion_tokens
        # These models don't support temperature/top_p and use different parameters
        OPENAI_REASONING_MODEL_PREFIXES = (
            # O-series reasoning models
            "o1-",
            "o1",  # o1, o1-mini, o1-preview
            "o3-",
            "o3",  # o3, o3-mini, o3-pro
            "o4-",  # o4-mini
            # GPT-5 series are also reasoning models
            "gpt-5-",
            "gpt-5",  # gpt-5, gpt-5-mini, gpt-5-nano
            # The GPT OSS series are also reasoning models
            "gpt-oss-120b",
            "gpt-oss-20b",
        )

        # Check if this is an OpenAI reasoning model
        model_lower = str(self.model).lower()
        is_openai_reasoning_model = (
            self.api_base in ["https://api.openai.com/v1", "https://api.eliza.yandex.net/raw/openai/v1"]
            and model_lower.startswith(OPENAI_REASONING_MODEL_PREFIXES)
        )

        if is_openai_reasoning_model and not self.manual_mode:
            # For OpenAI reasoning models
            params = {
                "model": self.model,
                "messages": formatted_messages,
                "max_completion_tokens": kwargs.get("max_tokens", self.max_tokens),
            }
            # Add optional reasoning parameters if provided
            reasoning_effort = kwargs.get("reasoning_effort", self.reasoning_effort)
            if reasoning_effort is not None:
                params["reasoning_effort"] = reasoning_effort
            if "verbosity" in kwargs:
                params["verbosity"] = kwargs["verbosity"]
        else:
            # Standard parameters for all other models
            params = {
                "model": self.model,
                "messages": formatted_messages,
                "temperature": kwargs.get("temperature", self.temperature),
                "top_p": kwargs.get("top_p", self.top_p),
                "max_tokens": kwargs.get("max_tokens", self.max_tokens),
            }

            # Handle reasoning_effort for open source reasoning models.
            reasoning_effort = kwargs.get("reasoning_effort", self.reasoning_effort)
            if reasoning_effort is not None:
                params["reasoning_effort"] = reasoning_effort
        
        # NOTE: seed is ignored in manual_mode on purpose

        # Add seed parameter for reproducibility if configured
        # Skip seed parameter for Google AI Studio endpoint as it doesn't support it
        seed = kwargs.get("seed", self.random_seed)
        if seed is not None and not self.manual_mode:
            if self.api_base == "https://generativelanguage.googleapis.com/v1beta/openai/":
                logger.warning(
                    "Skipping seed parameter as Google AI Studio endpoint doesn't support it. "
                    "Reproducibility may be limited."
                )
            else:
                params["seed"] = seed

        # Attempt the API call with retries
        retries = kwargs.get("retries", self.retries)
        retry_delay = kwargs.get("retry_delay", self.retry_delay)
        timeout = kwargs.get("timeout", self.timeout)

        if self.manual_mode:
            return await self._manual_wait_for_answer(params, timeout=timeout) 

        for attempt in range(retries + 1):
            try:
                response = await asyncio.wait_for(self._call_api(params), timeout=timeout)
                return response
            except asyncio.TimeoutError:
                if attempt < retries:
                    logger.warning(f"Timeout on attempt {attempt + 1}/{retries + 1}. Retrying...")
                    await asyncio.sleep(retry_delay)
                else:
                    logger.error(f"All {retries + 1} attempts failed with timeout")
                    raise
            except Exception as e:
                if attempt < retries:
                    logger.warning(
                        f"Error on attempt {attempt + 1}/{retries + 1}: {str(e)}. Retrying..."
                    )
                    await asyncio.sleep(retry_delay)
                else:
                    logger.error(f"All {retries + 1} attempts failed with error: {str(e)}")
                    raise

    async def _call_api(self, params: Dict[str, Any]) -> str:
        """Make the actual API call"""
        # Use asyncio to run the blocking API call in a thread pool
        
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(
            None, lambda: self.client.chat.completions.create(**params)
        )
        # Logging of system prompt, user message and response content
        logger = logging.getLogger(__name__)
        logger.debug(f"API parameters: {params}")
        logger.debug(f"API response: {response.choices[0].message.content}")
        return response.choices[0].message.content

    async def _manual_wait_for_answer(self, params: Dict[str, Any], timeout: Optional[Union[int, float]]) -> str:
        """
        Manual mode: write a task JSON file and poll for *.answer.json.
        If timeout is provided, we respect it; otherwise we wait indefinitely.
        """
        task_id = str(uuid.uuid4())

        messages = params.get("messages", [])
        display_prompt = _build_display_prompt(messages)

        task_payload = {
            "id": task_id,
            "created_at": _iso_now(),
            "model": params.get("model"),
            "display_prompt": display_prompt,
            "messages": messages,
            "meta": {
                "max_tokens": params.get("max_tokens"),
                "temperature": params.get("temperature"),
                "top_p": params.get("top_p"),
                "reasoning_effort": params.get("reasoning_effort"),
                # seed intentionally omitted in manual mode
            },
        }

        task_path = self.manual_queue_dir / f"{task_id}.json"
        tmp_path = self.manual_queue_dir / f".{task_id}.json.tmp"
        answer_path = self.manual_queue_dir / f"{task_id}.answer.json"

        # Atomic-ish write
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(task_payload, f, ensure_ascii=False, indent=2)
        os.replace(tmp_path, task_path)

        logger.info(f"[manual_mode] Task enqueued: {task_path}")

        # Poll for answer
        start = time.time()
        poll_interval = 1.0

        while True:
            if answer_path.exists():
                try:
                    with open(answer_path, "r", encoding="utf-8") as file:
                        data = json.load(file)
                    
                    answer = data.get("answer", "")
                    logger.info(f"[manual_mode] Answer received for {task_id}")

                    # optional: clean up original task file (leave artifacts if you prefer)
                    return answer
                finally:
                    # Keep files for audit by default; comment next two lines if you want to keep answers.
                    # task_path.unlink(missing_ok=True)
                    # answer_path.unlink(missing_ok=True)
                    pass

            # timeout check
            if timeout is not None and (time.time() - start) > float(timeout):
                raise asyncio.TimeoutError(
                    f"Manual mode timed out after {timeout} seconds waiting for answer of task {task_id}"
                )

            await asyncio.sleep(poll_interval)