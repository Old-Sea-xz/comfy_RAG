#!/usr/bin/env python3
"""Evaluate one completed case with the official ComfyBench visual rubric."""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path
from types import ModuleType
from typing import Any

import httpx

ROOT = Path(__file__).resolve().parents[1]
BENCH_ROOT = Path(os.getenv("COMFYROUTER_COMFYBENCH_ROOT", str(ROOT.parent / "ComfyBench")))
CASE_CRITERIA_PATH = ROOT / "data" / "comfybench_case_evaluation_criteria.json"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(BENCH_ROOT))

from comfyrouter import config
from comfyrouter.agentic import OpenAICompatibleClient
from comfyrouter.api_serial import serialized_api_request


class VisionStreamProtocolError(RuntimeError):
    pass


def load_case_evaluation_criteria(case_id: str) -> list[str]:
    """Load optional hard quality gates without changing benchmark instructions."""
    if not CASE_CRITERIA_PATH.is_file():
        return []
    payload = json.loads(CASE_CRITERIA_PATH.read_text(encoding="utf-8"))
    criteria = payload.get(str(case_id), [])
    if not isinstance(criteria, list) or not all(
        isinstance(item, str) and item.strip() for item in criteria
    ):
        raise ValueError(f"invalid evaluation criteria for case {case_id}")
    return [item.strip() for item in criteria]


def instruction_with_evaluation_criteria(
        instruction: str, criteria: list[str]) -> str:
    if not criteria:
        return instruction
    checklist = "\n".join(
        f"{index}. {criterion}" for index, criterion in enumerate(criteria, 1)
    )
    return (
        instruction
        + "\n\nCase-specific mandatory evaluation checklist:\n"
        + checklist
        + "\nEvery checklist item is mandatory and overrides the general request "
          "to be tolerant of minor quality differences."
    )


def validate_vision_evaluation_text(text: str) -> tuple[str, str]:
    """Reject a successful HTTP response that violates the evaluator contract."""
    analysis_match = re.search(
        r"<analysis\b[^>]*>(.*?)</analysis>", text, re.IGNORECASE | re.DOTALL,
    )
    judgment_match = re.search(
        r"<judgment\b[^>]*>(.*?)</judgment>", text, re.IGNORECASE | re.DOTALL,
    )
    analysis = analysis_match.group(1).strip() if analysis_match else ""
    judgment = judgment_match.group(1).strip() if judgment_match else ""
    if not analysis:
        raise VisionStreamProtocolError(
            "vision evaluator response is missing non-empty <analysis>"
        )
    if judgment not in {"True", "False"}:
        raise VisionStreamProtocolError(
            "vision evaluator response must contain <judgment>True</judgment> "
            "or <judgment>False</judgment>"
        )
    return analysis, judgment


def read_vision_sse(response: httpx.Response, *, deadline: float,
                    event_logger=None) -> tuple[str, Any]:
    """Consume Responses API SSE incrementally and stop at its terminal event."""
    content_type = response.headers.get("Content-Type", "")
    if "text/event-stream" not in content_type.lower():
        raise VisionStreamProtocolError(
            f"vision API did not honor stream=true: Content-Type={content_type!r}"
        )
    result = OpenAICompatibleClient._read_httpx_sse_completed_response(
        response, deadline=deadline, event_logger=event_logger,
    )
    text = OpenAICompatibleClient._response_output_text(result)
    try:
        validate_vision_evaluation_text(text)
    except VisionStreamProtocolError:
        print(
            "vision_response_status=" + str(result.get("status"))
            + " incomplete_details="
            + json.dumps(result.get("incomplete_details"), ensure_ascii=False)
            + " usage=" + json.dumps(result.get("usage"), ensure_ascii=False),
            flush=True,
        )
        print("vision_raw_response=" + repr(text[:4000]), flush=True)
        raise
    return text, result.get("usage")


def invoke_vision(messages: list[dict[str, Any]]) -> tuple[str, Any]:
    """Send the unmodified official prompt and media through Responses API."""
    config.load_env()
    base_url = os.environ["COMFYROUTER_LLM_BASE_URL"].rstrip("/")
    api_key = config.evaluation_api_key()
    model = config.evaluation_model()
    content: list[dict[str, Any]] = []
    for item in messages[0]["content"]:
        if item["type"] == "text":
            content.append({"type": "input_text", "text": item["text"]})
        elif item["type"] == "image_url":
            content.append({"type": "input_image", "image_url": item["image_url"]["url"]})
    payload = {
        "model": model,
        "temperature": 0,
        "input": [{"role": "user", "content": content}],
        # Responses output tokens include hidden reasoning tokens. A 1024-token
        # cap can finish as `incomplete` before the evaluator emits the two
        # required visible tags, which previously produced empty analysis and
        # judgment fields. Keep a bounded but sufficient evaluation budget.
        "max_output_tokens": min(max(int(os.environ.get(
            "COMFYROUTER_EVALUATOR_MAX_OUTPUT_TOKENS", "4096"
        )), 1024), 8192),
        "stream": True,
    }
    # Gemini gateways may reject the GPT-specific reasoning effort option.
    if not model.casefold().startswith("gemini-"):
        payload["reasoning"] = {"effort": "low"}
    user_agent = os.environ.get(
        "COMFYROUTER_LLM_USER_AGENT",
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 Chrome/140 Safari/537.36",
    )
    headers = {
        "Content-Type": "application/json",
        "Accept": "text/event-stream",
        "Cache-Control": "no-cache",
        "X-Accel-Buffering": "no",
        "User-Agent": user_agent,
    }
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    body = json.dumps(payload).encode("utf-8")
    # Evaluation is allowed more wall-clock time than ordinary Agent calls,
    # while connection establishment remains bounded and retried separately.
    evaluation_timeout = float(os.environ.get("COMFYROUTER_EVALUATOR_TIMEOUT", "300"))
    client = OpenAICompatibleClient(
        base_url=base_url, api_key=api_key, model=model, timeout=120,
    )
    client.timeout = min(max(evaluation_timeout, 120.0), 300.0)
    client.connect_timeout = min(
        max(float(os.environ.get("COMFYROUTER_EVALUATOR_CONNECT_TIMEOUT", "15")), 1.0),
        client.timeout,
    )
    client.header_timeout = min(
        max(float(os.environ.get("COMFYROUTER_EVALUATOR_READ_TIMEOUT", "90")), 1.0),
        client.timeout,
    )
    # Video comparisons upload several reference/result frames. Give their
    # request body its own bounded budget; ordinary Agent uploads retain 30s.
    client.write_timeout = min(max(float(os.environ.get(
        "COMFYROUTER_EVALUATOR_WRITE_TIMEOUT", "120"
    )), 30.0), evaluation_timeout)
    last_error = None
    for attempt in range(1, 4):
        response = None
        try:
            # Keep the official evaluator in the same cross-process API lane
            # as Agent requests. The gate covers headers and the full SSE
            # response, then the retry backoff happens after it is released.
            with serialized_api_request():
                deadline = time.monotonic() + evaluation_timeout
                started = time.monotonic()
                print("vision_headers_wait_start", flush=True)
                response = client._send_httpx(
                    body=body, headers=headers, timeout_seconds=evaluation_timeout,
                )
                print(
                    "vision_headers_received elapsed_seconds="
                    + f"{time.monotonic() - started:.3f}"
                    + " request_id=" + str(response.headers.get("x-request-id")),
                    flush=True,
                )
                response.raise_for_status()
                return read_vision_sse(
                    response, deadline=deadline,
                    event_logger=lambda event, at: print(
                        f"vision_{event} elapsed_seconds={at - started:.3f}", flush=True
                    ),
                )
        except (httpx.TransportError, httpx.HTTPStatusError, TimeoutError,
                json.JSONDecodeError, KeyError, VisionStreamProtocolError) as exc:
            last_error = exc
            status = (
                exc.response.status_code
                if isinstance(exc, httpx.HTTPStatusError) else None
            )
            transient = (
                status in {403, 408, 429, 500, 502, 503, 504, 520, 522, 524}
                or isinstance(exc, (
                    httpx.TransportError, TimeoutError, json.JSONDecodeError,
                    KeyError, VisionStreamProtocolError,
                ))
            )
            print(
                f"vision_transport_attempt={attempt} status=error "
                f"error={type(exc).__name__}: {exc}",
                flush=True,
            )
            if isinstance(exc, httpx.HTTPStatusError):
                try:
                    error_body = exc.response.read().decode("utf-8", errors="replace")[:2000]
                except (httpx.HTTPError, OSError):
                    error_body = "unavailable"
                print(
                    "vision_error_response=" + error_body,
                    flush=True,
                )
            if not transient:
                raise
            if attempt == 3:
                raise
            if isinstance(exc, (httpx.TransportError, TimeoutError)):
                client._reset_http_client(rotate_ipv4=True)
            # Evaluation transport retries use the same fixed three-second
            # backoff as Agent requests.
            time.sleep(3.0)
        finally:
            if response is not None:
                response.close()
    else:
        raise RuntimeError(f"vision evaluation failed after three requests: {last_error}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case-id", required=True)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--run-report", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    # Keep the official evaluator's relative resource paths valid even when
    # this script is invoked directly from another working directory.
    os.chdir(BENCH_ROOT)

    # evaluation.py imports invoke_vision from utils.model, whose unrelated RAG
    # dependencies are not needed for scoring. Supply only the expected symbol.
    model_module = ModuleType("utils.model")
    model_module.invoke_vision = invoke_vision
    sys.modules["utils.model"] = model_module
    comfy_module = ModuleType("utils.comfy")
    comfy_module.execute_prompt = lambda prompt: (_ for _ in ()).throw(
        RuntimeError("execution is supplied by ComfyRouter; official evaluator only")
    )
    sys.modules["utils.comfy"] = comfy_module
    from script import evaluation

    # Speaking-face evaluation needs denser temporal evidence than the
    # benchmark's default handful of frames. Keep the official evaluator and
    # prompt, but provide enough ordered pairs to judge a motion sequence
    # instead of over-weighting an isolated closed-mouth instant.
    if args.case_id == "182":
        official_load_video = evaluation.load_video

        def load_case_182_video(video_path, size_limit=(512, 512), frame_limit=5):
            return official_load_video(
                video_path, size_limit=size_limit, frame_limit=max(frame_limit, 24),
            )

        evaluation.load_video = load_case_182_video

    tasks = json.loads((BENCH_ROOT / "dataset/benchmark/instruction/complete.json").read_text())
    task = dict(tasks[args.case_id])
    original_instruction = task["instruction"]
    evaluation_criteria = load_case_evaluation_criteria(args.case_id)
    task["instruction"] = instruction_with_evaluation_criteria(
        original_instruction, evaluation_criteria,
    )
    task["result"] = str(args.result.resolve())
    # The evaluator is only invoked after ComfyUI reports success and the
    # materialized result is handed to this process. Keep `passed` as the
    # execution/result-availability signal; `resolved` below is the independent
    # semantic judgment for whether the task itself was fully satisfied.
    passed = args.result.is_file()
    own_evaluation = None
    if args.run_report and args.run_report.is_file():
        report = json.loads(args.run_report.read_text(encoding="utf-8"))
        attempts = report.get("attempts", [])
        if attempts:
            latest = attempts[-1]
            execution = latest.get("execution") or {}
            if execution.get("status") == "success":
                passed = True
            else:
                # Composition reports record execution per stage rather than
                # under a single attempt.execution object.
                stages = latest.get("stages") or []
                stage_statuses = [
                    (stage.get("execution") or {}).get("status")
                    for stage in stages
                ]
                passed = bool(stage_statuses) and all(
                    status == "success" for status in stage_statuses
                ) and args.result.is_file()
            own_evaluation = attempts[-1].get("evaluation")
        elif isinstance(report.get("fresh_cold_start"), dict):
            cold = report["fresh_cold_start"]
            passed = cold.get("execution", {}).get("status") == "success"
            own_evaluation = cold.get("evaluation")
        elif isinstance(report.get("cold_start"), dict):
            cold = report["cold_start"]
            passed = cold.get("execution", {}).get("status") == "success"
            own_evaluation = cold.get("evaluation")
    if task["modality"] == "t2i":
        analysis, judgment = evaluation.evaluate_t2i(task)
    elif task["modality"] == "i2i":
        analysis, judgment = evaluation.evaluate_i2i(task)
    elif task["modality"] == "t2v":
        analysis, judgment = evaluation.evaluate_t2v(task)
    elif task["modality"] == "i2v":
        analysis, judgment = evaluation.evaluate_i2v(task)
    elif task["modality"] == "v2v":
        analysis, judgment = evaluation.evaluate_v2v(task)
    else:
        raise ValueError(f"unsupported ComfyBench modality: {task['modality']}")
    resolved = judgment.strip() == "True"
    payload = {
        "schema_version": "comfybench-case-evaluation-1",
        "case_id": args.case_id,
        "category": task["category"],
        "modality": task["modality"],
        "instruction": original_instruction,
        "evaluation_criteria": evaluation_criteria,
        "result": task["result"],
        "comfybench": {"passed": passed, "resolved": resolved,
                       "analysis": analysis, "judgment": judgment},
        "system_evaluation": own_evaluation,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
