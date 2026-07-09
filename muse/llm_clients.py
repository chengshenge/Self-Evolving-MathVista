from __future__ import annotations

import ast
import base64
import hashlib
import json
import os
import random
import re
import threading
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Iterable, List, Optional, Tuple


_RATE_LIMIT_THREAD_LOCKS: Dict[str, threading.Lock] = {}
_RATE_LIMIT_THREAD_LOCKS_GUARD = threading.Lock()


def image_to_data_url(image_path: str | Path) -> str:
    image_path = Path(image_path)
    suffix = image_path.suffix.lower().lstrip(".") or "png"
    mime = "jpeg" if suffix in {"jpg", "jpeg"} else suffix
    raw = image_path.read_bytes()
    return f"data:image/{mime};base64,{base64.b64encode(raw).decode('utf-8')}"


def _strip_code_fence(text: str) -> str:
    text = text.strip()
    fenced = re.match(r"^```(?:json|python|text)?\s*(.*?)\s*```$", text, flags=re.DOTALL | re.IGNORECASE)
    if fenced:
        return fenced.group(1).strip()
    return text


def _balanced_snippet(text: str) -> Optional[str]:
    starts: List[Tuple[int, str]] = []
    for i, ch in enumerate(text):
        if ch in "[{":
            starts.append((i, ch))
    for start_idx, opener in starts:
        closer = "}" if opener == "{" else "]"
        depth = 0
        in_str = False
        quote = ""
        escape = False
        for j in range(start_idx, len(text)):
            ch = text[j]
            if in_str:
                if escape:
                    escape = False
                elif ch == "\\":
                    escape = True
                elif ch == quote:
                    in_str = False
            else:
                if ch in {'"', "'"}:
                    in_str = True
                    quote = ch
                elif ch == opener:
                    depth += 1
                elif ch == closer:
                    depth -= 1
                    if depth == 0:
                        return text[start_idx : j + 1]
    return None


def extract_json_any(text: Any) -> Any:
    if text is None:
        return {}
    if isinstance(text, (dict, list)):
        return text
    text = str(text).strip()
    if not text:
        return {}

    candidates: List[str] = []
    stripped = _strip_code_fence(text)
    candidates.extend([text, stripped])

    for block in re.findall(r"```(?:json|python|text)?\s*(.*?)\s*```", text, flags=re.DOTALL | re.IGNORECASE):
        candidates.append(block.strip())

    snippet = _balanced_snippet(stripped)
    if snippet:
        candidates.append(snippet)

    seen = set()
    ordered: List[str] = []
    for cand in candidates:
        if cand and cand not in seen:
            seen.add(cand)
            ordered.append(cand)

    for cand in ordered:
        for parser in (json.loads, ast.literal_eval):
            try:
                return parser(cand)
            except Exception:
                pass
    return {}


def extract_json(text: Any) -> Dict[str, Any]:
    parsed = extract_json_any(text)
    if isinstance(parsed, dict):
        return parsed
    if isinstance(parsed, list) and len(parsed) == 1 and isinstance(parsed[0], dict):
        return parsed[0]
    return {}


def _safe_dump(obj: Any) -> Any:
    if obj is None:
        return None
    if hasattr(obj, "model_dump"):
        try:
            return obj.model_dump()
        except Exception:
            pass
    if isinstance(obj, (dict, list, str, int, float, bool)):
        return obj
    try:
        json.dumps(obj)
        return obj
    except Exception:
        return repr(obj)


class OpenAIStyleClient:
    def __init__(self, config):
        self.config = config
        self.client = None
        if config.enabled:
            if self._protocol() == "ANTHROPIC_STYLE":
                self.client = True
            else:
                try:
                    from openai import OpenAI
                except ImportError as e:
                    raise RuntimeError("openai package is required for non-mock mode. Install requirements.txt") from e
                self.client = OpenAI(api_key=config.api_key, base_url=config.base_url, timeout=config.timeout)

    def _ensure_enabled(self) -> None:
        if not self.client:
            raise RuntimeError(
                "Model config is incomplete. Please set BASE_URL / API_KEY / MODEL in .env or use MOCK_MODE=1."
            )

    def _protocol(self) -> str:
        return str(getattr(self.config, "protocol", "OPENAI_STYLE") or "OPENAI_STYLE").strip().upper()

    def _is_gemini_model(self) -> bool:
        haystack = f"{self.config.model} {self.config.base_url}".lower()
        return "gemini" in haystack or "generativelanguage.googleapis.com" in haystack

    def _is_glm_model(self) -> bool:
        haystack = f"{self.config.model} {self.config.base_url}".lower()
        return "glm" in haystack or "bigmodel.cn" in haystack or "api.z.ai" in haystack

    def _env_float(self, *names: str, default: float) -> float:
        for name in names:
            raw = os.getenv(name)
            if raw not in (None, ""):
                try:
                    return max(0.0, float(raw))
                except ValueError:
                    pass
        return default

    def _env_int(self, *names: str, default: int) -> int:
        for name in names:
            raw = os.getenv(name)
            if raw not in (None, ""):
                try:
                    return max(1, int(raw))
                except ValueError:
                    pass
        return default

    def _rate_limit_interval_seconds(self) -> float:
        if not self._is_glm_model():
            return self._env_float("MUSE_LLM_MIN_INTERVAL_SECONDS", default=0.0)
        return self._env_float(
            "MUSE_GLM_MIN_INTERVAL_SECONDS",
            "GLM_MIN_INTERVAL_SECONDS",
            "MUSE_LLM_MIN_INTERVAL_SECONDS",
            default=1.5,
        )

    def _rate_limit_state_path(self) -> Path:
        configured = os.getenv("MUSE_LLM_RATE_LIMIT_STATE")
        if configured:
            return Path(configured)
        key = f"{self.config.base_url}|{self.config.model}|{self.config.api_key[:8]}"
        digest = hashlib.sha256(key.encode("utf-8")).hexdigest()[:16]
        return Path(os.getenv("TMPDIR", "/tmp")) / f"muse_llm_rate_{digest}.state"

    def _rate_limit_thread_lock(self) -> threading.Lock:
        path_key = str(self._rate_limit_state_path())
        with _RATE_LIMIT_THREAD_LOCKS_GUARD:
            lock = _RATE_LIMIT_THREAD_LOCKS.get(path_key)
            if lock is None:
                lock = threading.Lock()
                _RATE_LIMIT_THREAD_LOCKS[path_key] = lock
            return lock

    def _wait_for_global_rate_limit(self) -> None:
        interval = self._rate_limit_interval_seconds()
        if interval <= 0:
            return

        thread_lock = self._rate_limit_thread_lock()
        thread_lock.acquire()
        try:
            try:
                import fcntl
            except ImportError:
                time.sleep(interval)
                return

            path = self._rate_limit_state_path()
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a+", encoding="utf-8") as fh:
                fcntl.flock(fh.fileno(), fcntl.LOCK_EX)
                try:
                    fh.seek(0)
                    raw = fh.read().strip()
                    try:
                        last_started = float(raw) if raw else 0.0
                    except ValueError:
                        last_started = 0.0
                    wait_for = interval - (time.time() - last_started)
                    if wait_for > 0:
                        time.sleep(wait_for)
                    fh.seek(0)
                    fh.truncate()
                    fh.write(str(time.time()))
                    fh.flush()
                    os.fsync(fh.fileno())
                finally:
                    fcntl.flock(fh.fileno(), fcntl.LOCK_UN)
        finally:
            thread_lock.release()

    def _retry_attempts(self) -> int:
        default = 8 if self._is_glm_model() else 3
        return self._env_int("MUSE_LLM_RETRY_ATTEMPTS", default=default)

    def _retry_sleep_seconds(self, attempt: int, exc: Exception) -> float:
        base = self._env_float("MUSE_LLM_RETRY_BASE_SECONDS", default=8.0 if self._is_glm_model() else 1.0)
        cap = self._env_float("MUSE_LLM_RETRY_MAX_SECONDS", default=90.0 if self._is_glm_model() else 20.0)
        sleep_for = min(cap, base * (2 ** attempt))
        if self._is_retryable_error(exc) and self._is_glm_model():
            sleep_for = max(sleep_for, self._rate_limit_interval_seconds() * (attempt + 2))
        return sleep_for + random.uniform(0.0, min(1.0, sleep_for * 0.1))

    def _is_retryable_error(self, exc: Exception) -> bool:
        text = str(exc).lower()
        return any(
            marker in text
            for marker in [
                "429",
                "rate limit",
                "ratelimit",
                "1302",
                "速率限制",
                "too many requests",
                "timeout",
                "api connection",
                "connection error",
                "temporarily unavailable",
                "502",
                "503",
                "504",
            ]
        )

    def _is_reasoning_chat_model(self) -> bool:
        model = (self.config.model or "").strip().lower()
        return model.startswith(("gpt-5", "o1", "o3", "o4"))

    def _instruction_role(self) -> str:
        return "developer" if self._is_reasoning_chat_model() else "system"

    def _message_to_text(self, message: Any) -> str:
        content = getattr(message, "content", message)

        def _list_to_text(parts: List[Any]) -> str:
            out: List[str] = []
            for chunk in parts:
                if isinstance(chunk, dict):
                    if isinstance(chunk.get("text"), str):
                        out.append(chunk["text"])
                        continue
                    if chunk.get("type") == "output_text" and isinstance(chunk.get("text"), str):
                        out.append(chunk["text"])
                        continue
                    out.append(json.dumps(chunk, ensure_ascii=False))
                    continue

                text_attr = getattr(chunk, "text", None)
                if isinstance(text_attr, str):
                    out.append(text_attr)
                    continue

                if hasattr(chunk, "model_dump"):
                    try:
                        dumped = chunk.model_dump()
                        if isinstance(dumped, dict) and isinstance(dumped.get("text"), str):
                            out.append(dumped["text"])
                        else:
                            out.append(json.dumps(dumped, ensure_ascii=False))
                        continue
                    except Exception:
                        pass

                if isinstance(chunk, str):
                    out.append(chunk)
                else:
                    out.append(str(chunk))
            return "\n".join(x for x in out if x)

        if isinstance(content, list):
            text = _list_to_text(content)
        elif isinstance(content, str):
            text = content
        elif content is None:
            text = ""
        else:
            try:
                text = json.dumps(content, ensure_ascii=False)
            except Exception:
                text = str(content)

        if text:
            return text

        # Some SDK shapes may keep parsed structured data outside message.content.
        parsed = getattr(message, "parsed", None)
        if parsed is not None:
            try:
                return json.dumps(parsed, ensure_ascii=False)
            except Exception:
                return str(parsed)

        if hasattr(message, "model_dump"):
            try:
                dumped = message.model_dump()
                parsed = dumped.get("parsed")
                if parsed is not None:
                    return json.dumps(parsed, ensure_ascii=False)
                dumped_content = dumped.get("content")
                if isinstance(dumped_content, str):
                    return dumped_content
                if isinstance(dumped_content, list):
                    return _list_to_text(dumped_content)
            except Exception:
                pass
        return ""

    def _response_meta(self, response: Any, *, response_format: Optional[Dict[str, Any]], reasoning_effort: Optional[str]) -> Dict[str, Any]:
        choice0 = response.choices[0]
        message = choice0.message
        meta: Dict[str, Any] = {
            "id": getattr(response, "id", None),
            "model": getattr(response, "model", None),
            "finish_reason": getattr(choice0, "finish_reason", None),
            "response_format": response_format,
            "reasoning_effort": reasoning_effort,
            "usage": _safe_dump(getattr(response, "usage", None)),
            "message_dump": _safe_dump(message),
        }
        return meta

    def _anthropic_content_part(self, part: Any) -> Dict[str, Any]:
        if isinstance(part, str):
            return {"type": "text", "text": part}
        if not isinstance(part, dict):
            return {"type": "text", "text": str(part)}
        if part.get("type") == "text":
            return {"type": "text", "text": str(part.get("text", ""))}
        if part.get("type") == "image_url":
            url = ((part.get("image_url") or {}).get("url") or "").strip()
            match = re.match(r"^data:(image/[^;]+);base64,(.*)$", url, flags=re.DOTALL)
            if match:
                return {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": match.group(1),
                        "data": match.group(2),
                    },
                }
        return {"type": "text", "text": json.dumps(part, ensure_ascii=False)}

    def _anthropic_messages(self, messages: List[Dict[str, Any]]) -> Tuple[str, List[Dict[str, Any]]]:
        system_parts: List[str] = []
        out: List[Dict[str, Any]] = []
        for message in messages:
            role = message.get("role", "user")
            content = message.get("content", "")
            if role in {"system", "developer"}:
                system_parts.append(content if isinstance(content, str) else json.dumps(content, ensure_ascii=False))
                continue
            anthropic_role = "assistant" if role == "assistant" else "user"
            if isinstance(content, list):
                converted = [self._anthropic_content_part(part) for part in content]
            else:
                converted = [{"type": "text", "text": str(content)}]
            out.append({"role": anthropic_role, "content": converted})
        return "\n\n".join(part for part in system_parts if part), out

    def _anthropic_chat_create(
        self,
        messages: List[Dict[str, Any]],
        *,
        temperature: Optional[float],
        max_tokens: int,
    ):
        try:
            import requests
        except ImportError as e:
            raise RuntimeError("requests package is required for Anthropic-style mode.") from e

        system, anthropic_messages = self._anthropic_messages(messages)
        request: Dict[str, Any] = {
            "model": self.config.model,
            "max_tokens": max_tokens,
            "messages": anthropic_messages,
        }
        if system:
            request["system"] = system
        effective_temperature = self.config.temperature if temperature is None else temperature
        if effective_temperature is not None:
            request["temperature"] = effective_temperature

        url = self.config.base_url.rstrip("/")
        if not url.endswith("/messages"):
            url = f"{url}/messages"
        headers = {
            "Content-Type": "application/json",
            "x-api-key": self.config.api_key,
            "anthropic-version": "2023-06-01",
        }
        last_error: Optional[Exception] = None
        for attempt in range(self._retry_attempts()):
            try:
                self._wait_for_global_rate_limit()
                response = requests.post(url, headers=headers, json=request, timeout=self.config.timeout)
                response.raise_for_status()
                payload = response.json()
                text = "\n".join(
                    block.get("text", "")
                    for block in payload.get("content", [])
                    if isinstance(block, dict) and block.get("type") == "text"
                )
                message = SimpleNamespace(content=text)
                choice = SimpleNamespace(message=message, finish_reason=payload.get("stop_reason"))
                return SimpleNamespace(
                    id=payload.get("id"),
                    model=payload.get("model", self.config.model),
                    choices=[choice],
                    usage=payload.get("usage"),
                )
            except Exception as exc:
                last_error = exc
                if attempt >= self._retry_attempts() - 1 or not self._is_retryable_error(exc):
                    break
                time.sleep(self._retry_sleep_seconds(attempt, exc))
        raise RuntimeError(f"Anthropic request failed after retries: {last_error}")

    def _chat_create(
        self,
        messages: List[Dict[str, Any]],
        *,
        temperature: Optional[float],
        max_tokens: int,
        response_format: Optional[Dict[str, Any]] = None,
        reasoning_effort: Optional[str] = None,
    ):
        if self._protocol() == "ANTHROPIC_STYLE":
            return self._anthropic_chat_create(messages, temperature=temperature, max_tokens=max_tokens)

        request: Dict[str, Any] = {
            "model": self.config.model,
            "messages": messages,
        }

        effective_temperature = self.config.temperature if temperature is None else temperature

        if self._is_reasoning_chat_model():
            request["max_completion_tokens"] = max_tokens
            if reasoning_effort:
                request["reasoning_effort"] = reasoning_effort
        else:
            request["max_tokens"] = max_tokens
            if effective_temperature is not None:
                request["temperature"] = effective_temperature

        if response_format is not None and not (self._is_gemini_model() or self._is_glm_model()):
            request["response_format"] = response_format

        last_error: Optional[Exception] = None
        for attempt in range(self._retry_attempts()):
            try:
                self._wait_for_global_rate_limit()
                return self.client.chat.completions.create(**request)
            except Exception as exc:
                last_error = exc
                if attempt >= self._retry_attempts() - 1 or not self._is_retryable_error(exc):
                    raise
                time.sleep(self._retry_sleep_seconds(attempt, exc))
        raise RuntimeError(f"LLM request failed after retries: {last_error}")

    def _build_messages(self, instruction: str, user_content: Any) -> List[Dict[str, Any]]:
        return [
            {"role": self._instruction_role(), "content": instruction},
            {"role": "user", "content": user_content},
        ]

    def complete_text(
        self,
        system_prompt: str,
        user_prompt: str,
        *,
        temperature: Optional[float] = None,
        max_tokens: int = 1200,
        response_format: Optional[Dict[str, Any]] = None,
        reasoning_effort: Optional[str] = None,
        return_meta: bool = False,
    ):
        self._ensure_enabled()
        response = self._chat_create(
            self._build_messages(system_prompt, user_prompt),
            temperature=temperature,
            max_tokens=max_tokens,
            response_format=response_format,
            reasoning_effort=reasoning_effort,
        )
        message = response.choices[0].message
        text = self._message_to_text(message)
        meta = self._response_meta(response, response_format=response_format, reasoning_effort=reasoning_effort)
        return (text, meta) if return_meta else text

    def complete_json(
        self,
        system_prompt: str,
        user_prompt: str,
        *,
        temperature: Optional[float] = None,
        max_tokens: int = 1200,
        response_format: Optional[Dict[str, Any]] = None,
        reasoning_effort: Optional[str] = None,
        return_raw: bool = False,
        return_meta: bool = False,
    ):
        rf = response_format or {"type": "json_object"}
        text, meta = self.complete_text(
            system_prompt,
            user_prompt + "\n\nReturn a JSON object only.",
            temperature=temperature,
            max_tokens=max_tokens,
            response_format=rf,
            reasoning_effort=reasoning_effort,
            return_meta=True,
        )
        parsed = extract_json(text)
        if return_raw and return_meta:
            return parsed, text, meta
        if return_raw:
            return parsed, text
        if return_meta:
            return parsed, meta
        return parsed

    def complete_multimodal_text(
        self,
        system_prompt: str,
        user_prompt: str,
        image_paths: Iterable[str | Path],
        *,
        temperature: Optional[float] = None,
        max_tokens: int = 1200,
        response_format: Optional[Dict[str, Any]] = None,
        reasoning_effort: Optional[str] = None,
        return_meta: bool = False,
    ):
        self._ensure_enabled()
        content: List[Dict[str, Any]] = [{"type": "text", "text": user_prompt}]
        for image_path in image_paths:
            content.append({"type": "image_url", "image_url": {"url": image_to_data_url(image_path)}})
        response = self._chat_create(
            self._build_messages(system_prompt, content),
            temperature=temperature,
            max_tokens=max_tokens,
            response_format=response_format,
            reasoning_effort=reasoning_effort,
        )
        message = response.choices[0].message
        text = self._message_to_text(message)
        meta = self._response_meta(response, response_format=response_format, reasoning_effort=reasoning_effort)
        return (text, meta) if return_meta else text

    def complete_multimodal_json(
        self,
        system_prompt: str,
        user_prompt: str,
        image_paths: Iterable[str | Path],
        *,
        temperature: Optional[float] = None,
        max_tokens: int = 1200,
        response_format: Optional[Dict[str, Any]] = None,
        reasoning_effort: Optional[str] = None,
        return_raw: bool = False,
        return_meta: bool = False,
    ):
        rf = response_format or {"type": "json_object"}
        text, meta = self.complete_multimodal_text(
            system_prompt,
            user_prompt + "\n\nReturn a JSON object only.",
            image_paths,
            temperature=temperature,
            max_tokens=max_tokens,
            response_format=rf,
            reasoning_effort=reasoning_effort,
            return_meta=True,
        )
        parsed = extract_json(text)
        if return_raw and return_meta:
            return parsed, text, meta
        if return_raw:
            return parsed, text
        if return_meta:
            return parsed, meta
        return parsed
