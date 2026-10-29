"""Extract the real ComfyUI exception without stringifying runtime tensors."""
from __future__ import annotations

from typing import Any
import ast
import re
from pathlib import Path


_MEDIA_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff", ".gif",
                     ".mp4", ".mov", ".webm", ".mkv", ".avi", ".wav", ".mp3", ".flac", ".ogg", ".m4a"}


def large_media_choices(choices: Any) -> bool:
    return isinstance(choices, list) and len(choices) > 24 and all(
        isinstance(value, str) and (value in {"None", "none", ""} or Path(value).suffix.lower() in _MEDIA_EXTENSIONS)
        for value in choices)


def normalize_legacy_media_enum_error(error: Any) -> Any:
    """Replace our old rendered upload catalog with its count, retaining error facts.

    The original message, target, expected/actual values stay byte-identical.
    Full catalog contents remain in the persisted original report/validator.
    Arbitrary application exceptions are never shortened by this rule.
    """
    if not isinstance(error, str) or not error.startswith("enum_value at /"):
        return error

    def replace(match):
        try:
            choices = ast.literal_eval(match.group(1))
        except (ValueError, SyntaxError, RecursionError, MemoryError):
            return match.group(0)
        if not large_media_choices(choices):
            return match.group(0)
        return f"; allowed_count={len(choices)}; complete_upload_catalog_in_original_report=true"

    return re.sub(r"; allowed=(\[[^\n]*?\])(?=;|$)", replace, error)


def execution_error_text(history: dict[str, Any], fallback: str = "") -> str:
    """Keep the exception message exact; traceback remains in history.status.

    ComfyUI execution_error also contains current_inputs with tensor dumps.
    These are runtime values, rather than the exception message.
    """
    messages = history.get("status", {}).get("messages", [])
    for event in reversed(messages if isinstance(messages, list) else []):
        if not isinstance(event, (list, tuple)) or len(event) != 2 or not isinstance(event[1], dict):
            continue
        name, details = event
        if name == "execution_error" and isinstance(details.get("exception_message"), str):
            prefix = f"node {details.get('node_id', '?')} ({details.get('node_type', '?')}): "
            return prefix + str(details.get("exception_type") or "Error") + ": " + details["exception_message"]
    return fallback


def normalize_legacy_execution_error(execution: dict) -> Any:
    """Only replace a proven old str(status.messages) dump, never arbitrary errors."""
    error = execution.get("error")
    history = execution.get("history")
    if isinstance(error, str) and isinstance(history, dict):
        messages = history.get("status", {}).get("messages")
        if isinstance(messages, list) and error == str(messages):
            return execution_error_text(history, error)
    details = legacy_error_details(execution)
    if details is not None:
        return execution_error_text({"status": {"messages": [["execution_error", details]]}}, error)
    return normalize_legacy_media_enum_error(error)


def legacy_error_details(execution: dict) -> dict | None:
    """Recover a persisted Python event-list repr when no history was retained."""
    error = execution.get("error")
    if not isinstance(error, str) or not error.startswith("[["):
        return None
    history = execution.get("history")
    messages = history.get("status", {}).get("messages") if isinstance(history, dict) else None
    if messages is None:
        try:
            messages = ast.literal_eval(error)
        except (ValueError, SyntaxError, RecursionError, MemoryError):
            return None
    if not isinstance(messages, list) or error != str(messages) or not all(
        isinstance(event, list) and len(event) == 2 and isinstance(event[1], dict)
        and event[0] in {"execution_start", "execution_cached", "execution_error", "execution_interrupted", "execution_success"}
        for event in messages):
        return None
    for name, details in reversed(messages):
        if name == "execution_error" and isinstance(details.get("exception_message"), str):
            return {key: value for key, value in details.items() if key not in {"current_inputs", "current_outputs"}}
    return None
