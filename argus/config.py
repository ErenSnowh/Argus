"""
config.py
----------
Central configuration for ARGUS. Auto-loads the Gemini API key so every
module (dashboard, CLI, live agent runner) gets it without any manual setup.

Priority:
  1. GOOGLE_API_KEY already set in the environment (user override)
  2. Embedded default key (ships with the repo for public shareability)

Import this module early in any entry point (dashboard, CLI, run_live.py)
and the key is automatically available to the Google GenAI SDK.
"""

from __future__ import annotations

import os

# Load .env file manually from the current directory, package directory, or parent directory
def _load_env():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(base_dir)
    cwd = os.getcwd()
    
    env_paths = [
        os.path.join(cwd, ".env"),
        os.path.join(base_dir, ".env"),
        os.path.join(parent_dir, ".env"),
    ]
    
    seen = set()
    unique_paths = []
    for p in env_paths:
        if p not in seen:
            seen.add(p)
            unique_paths.append(p)
            
    for env_path in unique_paths:
        if os.path.isfile(env_path):
            try:
                with open(env_path, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line or line.startswith("#"):
                            continue
                        if "=" in line:
                            k, v = line.split("=", 1)
                            k = k.strip()
                            v = v.strip().strip("'\"")
                            if k and k not in os.environ:
                                os.environ[k] = v
            except Exception:
                pass

_load_env()

# ---------------------------------------------------------------------------
# API key is loaded from the environment or a .env file.
# ---------------------------------------------------------------------------
_DEFAULT_API_KEY = ""

GOOGLE_API_KEY: str = os.environ.get("GOOGLE_API_KEY") or _DEFAULT_API_KEY

# Push into os.environ so the Google GenAI SDK picks it up automatically.
if GOOGLE_API_KEY:
    os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY


def has_api_key() -> bool:
    """Return True if a valid-looking Gemini API key is available."""
    return bool(GOOGLE_API_KEY and len(GOOGLE_API_KEY) > 10)


# ---------------------------------------------------------------------------
# Monkeypatch Google GenAI SDK to automatically handle 429/Rate Limit & 503
# errors with exponential backoff, and fall back to alternative models if the
# requested model is daily-quota exhausted.
# ---------------------------------------------------------------------------
try:
    import re
    import asyncio
    import sys
    from google.genai.models import AsyncModels

    # Order of fallback models
    ALL_CANDIDATE_MODELS = [
        "gemini-3.1-flash-lite",
        "gemini-2.5-flash-lite",
        "gemini-2.0-flash",
        "gemini-2.5-flash",
        "gemini-flash-latest"
    ]

    _orig_generate_content = AsyncModels.generate_content
    _orig_generate_content_stream = AsyncModels.generate_content_stream

    def _get_current_model(args, kwargs) -> str | None:
        if "model" in kwargs:
            return kwargs["model"]
        if len(args) > 0:
            return args[0]
        return None

    def _set_model(args, kwargs, new_model):
        args_list = list(args)
        if "model" in kwargs:
            kwargs["model"] = new_model
        elif len(args_list) > 0:
            args_list[0] = new_model
        return tuple(args_list), kwargs

    async def _patched_generate_content(self, *args, **kwargs):
        current_model = _get_current_model(args, kwargs)
        max_retries = 5
        base_delay = 5
        
        tried_models = [current_model] if current_model else []
        last_exception = None
        
        while True:
            model_to_use = tried_models[-1] if tried_models else current_model
            curr_args, curr_kwargs = _set_model(args, kwargs, model_to_use)
            
            for attempt in range(1, max_retries + 1):
                try:
                    return await _orig_generate_content(self, *curr_args, **curr_kwargs)
                except Exception as e:
                    last_exception = e
                    err_msg = str(e)
                    
                    is_daily_limit = (
                        "GenerateRequestsPerDay" in err_msg
                        or "limit: 20" in err_msg
                        or "limit: 0" in err_msg
                    )
                    is_rate_limit = (
                        "429" in err_msg
                        or "RESOURCE_EXHAUSTED" in err_msg
                        or "quota" in err_msg.lower()
                        or (hasattr(e, "code") and e.code == 429)
                    )
                    is_transient_503 = (
                        "503" in err_msg
                        or "UNAVAILABLE" in err_msg
                        or (hasattr(e, "code") and e.code == 503)
                    )

                    # If daily quota limit hit, skip remaining retries and fallback
                    if is_daily_limit:
                        sys.stderr.write(
                            f"\n[ARGUS SDK PATCH] Daily quota exhausted for model {model_to_use}. Triggering fallback...\n"
                        )
                        sys.stderr.flush()
                        break

                    if (is_rate_limit or is_transient_503) and attempt < max_retries:
                        delay = base_delay * (2 ** (attempt - 1))
                        match = re.search(r"retry in (\d+(?:\.\d+)?)s", err_msg, re.IGNORECASE)
                        if match:
                            delay = float(match.group(1)) + 2.0
                        else:
                            match_delay = re.search(
                                r"retryDelay[\"']?\s*:\s*[\"']?(\d+)s?", err_msg, re.IGNORECASE
                            )
                            if match_delay:
                                delay = float(match_delay.group(1)) + 2.0
                            elif is_rate_limit:
                                delay = 45.0
                        
                        sys.stderr.write(
                            f"\n[ARGUS SDK PATCH] Rate limit/503 for {model_to_use} (attempt {attempt}/{max_retries}). "
                            f"Sleeping for {delay:.2f}s...\n"
                        )
                        sys.stderr.flush()
                        await asyncio.sleep(delay)
                        continue
                    
                    # If retries exhausted on this model, break loop to fallback
                    break
            
            # Find next model in candidate list to try
            next_model = None
            for candidate in ALL_CANDIDATE_MODELS:
                if candidate not in tried_models:
                    next_model = candidate
                    break
            
            if next_model:
                sys.stderr.write(f"\n[ARGUS SDK PATCH] Falling back from {model_to_use} to {next_model}\n")
                sys.stderr.flush()
                tried_models.append(next_model)
            else:
                # No candidates left, raise original exception
                if last_exception:
                    raise last_exception
                raise Exception("Model fallback failed with no captured exception")

    async def _patched_generate_content_stream(self, *args, **kwargs):
        current_model = _get_current_model(args, kwargs)
        max_retries = 5
        base_delay = 5
        
        tried_models = [current_model] if current_model else []
        last_exception = None
        
        while True:
            model_to_use = tried_models[-1] if tried_models else current_model
            curr_args, curr_kwargs = _set_model(args, kwargs, model_to_use)
            
            for attempt in range(1, max_retries + 1):
                try:
                    return await _orig_generate_content_stream(self, *curr_args, **curr_kwargs)
                except Exception as e:
                    last_exception = e
                    err_msg = str(e)
                    
                    is_daily_limit = (
                        "GenerateRequestsPerDay" in err_msg
                        or "limit: 20" in err_msg
                        or "limit: 0" in err_msg
                    )
                    is_rate_limit = (
                        "429" in err_msg
                        or "RESOURCE_EXHAUSTED" in err_msg
                        or "quota" in err_msg.lower()
                        or (hasattr(e, "code") and e.code == 429)
                    )
                    is_transient_503 = (
                        "503" in err_msg
                        or "UNAVAILABLE" in err_msg
                        or (hasattr(e, "code") and e.code == 503)
                    )

                    if is_daily_limit:
                        sys.stderr.write(
                            f"\n[ARGUS SDK PATCH] Daily quota exhausted for model {model_to_use} during stream. Triggering fallback...\n"
                        )
                        sys.stderr.flush()
                        break

                    if (is_rate_limit or is_transient_503) and attempt < max_retries:
                        delay = base_delay * (2 ** (attempt - 1))
                        match = re.search(r"retry in (\d+(?:\.\d+)?)s", err_msg, re.IGNORECASE)
                        if match:
                            delay = float(match.group(1)) + 2.0
                        else:
                            match_delay = re.search(
                                r"retryDelay[\"']?\s*:\s*[\"']?(\d+)s?", err_msg, re.IGNORECASE
                            )
                            if match_delay:
                                delay = float(match_delay.group(1)) + 2.0
                            elif is_rate_limit:
                                delay = 45.0
                        
                        sys.stderr.write(
                            f"\n[ARGUS SDK PATCH] Rate limit/503 for {model_to_use} on stream (attempt {attempt}/{max_retries}). "
                            f"Sleeping for {delay:.2f}s...\n"
                        )
                        sys.stderr.flush()
                        await asyncio.sleep(delay)
                        continue
                    
                    break
            
            next_model = None
            for candidate in ALL_CANDIDATE_MODELS:
                if candidate not in tried_models:
                    next_model = candidate
                    break
            
            if next_model:
                sys.stderr.write(f"\n[ARGUS SDK PATCH] Falling back stream from {model_to_use} to {next_model}\n")
                sys.stderr.flush()
                tried_models.append(next_model)
            else:
                if last_exception:
                    raise last_exception
                raise Exception("Model fallback failed with no captured exception")

    AsyncModels.generate_content = _patched_generate_content
    AsyncModels.generate_content_stream = _patched_generate_content_stream

except Exception as e:
    pass
