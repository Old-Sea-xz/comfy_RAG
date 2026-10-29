"""Shared API transport, generation, evaluation and repair for RAG."""
from __future__ import annotations
import hashlib
import copy
import io
import json
import os
import re
import socket
import time
import codecs
from email.utils import parsedate_to_datetime
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol
from uuid import uuid4
import httpx
import httpcore
from . import config
from .api_serial import serial_api_enabled, serialized_api_request
from .prompt_evidence import compact_agent_payload, compact_prompt_history
from .operation_notebook import notebook_agent_payload
from .prompt_interfaces import compact_interface_context
from .result_protocol import AGENT_PROTOCOL, RESULT_TOOL, result_tool, parse_result_call, strict_compatible
from .types import RequirementSpec
BROAD_MODEL_PROFILES = {'flux', 'flux-dev', 'sdxl', 'sd-generic', 'sd3', 'wan', 'hunyuanvideo', 'animatediff', 'svd', 'utility'}
VIDEO_TIMING_POLICY = 'Treat ordinary requested video durations as nominal, not exact frame-count requirements. Accept small differences explained by frame quantization or interpolation endpoint conventions. For m-fold interpolation of N source frames, (N - 1) * m + 1 output frames at m times the source FPS is valid: its duration differs by less than one source-frame interval. Do not require padding, duplicate terminal frames, a topology repair, or an independent stage repair solely for that difference. Preserve the requested FPS multiplier and all source content. Do not infer exact timing from an ordinary duration request or from previous diagnostic prose. Explicit task requirements for an exact frame count, exact synchronization, or hard duration/frame ceilings still apply. Substantial truncation, wrong playback speed, and missing content remain failures.'

class _IPv4SyncBackend(httpcore.SyncBackend):
    """Resolve only IPv4 addresses and share one timeout across all of them."""

    def __init__(self, rotation_offset: int=0) -> None:
        self.rotation_offset = max(int(rotation_offset), 0)

    def connect_tcp(self, host: str, port: int, timeout: float | None=None, local_address: str | None=None, socket_options=None):
        try:
            addresses = socket.getaddrinfo(host, port, socket.AF_INET, socket.SOCK_STREAM)
        except OSError as exc:
            raise httpcore.ConnectError(str(exc)) from exc
        if not addresses:
            raise httpcore.ConnectError(f'no IPv4 address resolved for {host}')
        unique_addresses = []
        seen: set[str] = set()
        for address in addresses:
            ip = str(address[4][0])
            if ip not in seen:
                seen.add(ip)
                unique_addresses.append(address)
        if unique_addresses:
            offset = self.rotation_offset % len(unique_addresses)
            unique_addresses = unique_addresses[offset:] + unique_addresses[:offset]
        deadline = time.monotonic() + timeout if timeout is not None else None
        last_error: Exception | None = None
        remaining_addresses = len(unique_addresses)
        for address in unique_addresses:
            ip = str(address[4][0])
            remaining = None if deadline is None else deadline - time.monotonic()
            if remaining is not None and remaining <= 0:
                break
            address_timeout = None if remaining is None else remaining / max(remaining_addresses, 1)
            try:
                return super().connect_tcp(ip, port, timeout=address_timeout, local_address=local_address or '0.0.0.0', socket_options=socket_options)
            except (httpcore.ConnectError, httpcore.ConnectTimeout) as exc:
                last_error = exc
            finally:
                remaining_addresses -= 1
        if last_error is not None:
            attempted = [str(address[4][0]) for address in unique_addresses]
            error_type = httpcore.ConnectTimeout if isinstance(last_error, httpcore.ConnectTimeout) else httpcore.ConnectError
            raise error_type(f'IPv4 connection failed for {host}:{port}; attempted={attempted}; last={last_error}') from last_error
        raise httpcore.ConnectTimeout(f'IPv4 connection deadline exceeded for {host}:{port}')

class _IPv4HTTPTransport(httpx.HTTPTransport):
    """httpx transport whose connection pool cannot fall through to IPv6."""

    def __init__(self, *, rotation_offset: int=0, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        if hasattr(self._pool, '_network_backend'):
            self._pool._network_backend = _IPv4SyncBackend(rotation_offset)
_DYNAMIC_PROMPT_KEYS = frozenset({'request', 'instruction', 'instructions', 'current_request', 'request_suffix', 'phase', 'validation_feedback', 'static_validation_feedback', 'execution_feedback', 'execution_result', 'evaluation', 'visual_evaluation', 'quality_feedback', 'feedback', 'candidate', 'candidate_workflow', 'candidate_json', 'edits', 'current_graph', 'graph', 'workflow_api', 'post_edit_graph', 'execution', 'media', 'media_paths', 'output_paths', 'visual_input_order', 'repair_evidence', 'preflight_failures', 'previous_plan', 'operation_notebook', 'notebook_earlier_records', 'current_workflow_policy', 'interface_options_policy', 'interface_scope_policy', 'installed_node_interfaces', 'previous_response', 'contract_correction', 'attempt', 'round', 'turn', 'json_error', 'repair_errors', 'repair_records', 'new_evidence'})

def _is_dynamic_prompt_key(key: str) -> bool:
    """Recognize per-turn evidence without relying on dict iteration order."""
    if key.startswith('_'):
        return True
    if key in _DYNAMIC_PROMPT_KEYS:
        return True
    lowered = key.lower()
    return any((token in lowered for token in ('feedback', 'evaluation', 'execution_result', 'validation_error', 'repair_error', 'attempt_result', 'turn_result')))
_TRACE_PROMPT_KEYS = frozenset({'trace', 'full_trace', 'attempts', 'attempt_history', 'repair_history', 'failure_analyses', 'execution_history', 'workflow_modification_attempts', 'reviewed_failure_analysis', 'base_execution_history'})

def _is_trace_agent(agent_id: str) -> bool:
    lowered = agent_id.lower()
    return any((token in lowered for token in ('evolution', 'failure_analysis', 'evaluation', 'review')))

class AgentOutputError(ValueError):
    pass

class AgentTransportError(AgentOutputError):
    """Retryable API/SSE transport failure; not an Agent contract failure."""
    pass

class AgentProviderError(AgentOutputError):
    """Provider rejected a request; JSON corrections cannot repair this envelope."""

    def __init__(self, agent_id: str, response: dict) -> None:
        self.provider_error = response.get('error') or response.get('incomplete_details') or {}
        details = self.provider_error if isinstance(self.provider_error, dict) else {}
        self.provider_code = str(details.get('code') or details.get('reason') or response.get('status') or 'provider_error')
        self.raw_response = response
        super().__init__(f'LLM provider failure for {agent_id}: {self.provider_code}: {details.get('message', '')}')

class LLMClient(Protocol):
    """供应商无关接口；调用方负责实现真实 API 客户端。"""

    def complete_json(self, agent_id: str, contract: dict[str, Any], payload: dict[str, Any]) -> dict[str, Any]:
        ...

class OpenAICompatibleClient:
    """无第三方依赖的 OpenAI-compatible Responses API JSON 客户端。

    通过 `COMFYROUTER_LLM_BASE_URL`、`COMFYROUTER_LLM_API_KEY` 和
    `COMFYROUTER_LLM_MODEL` 配置；不配置时系统不会自动联网。
    """

    def __init__(self, base_url: str | None=None, api_key: str | None=None, model: str | None=None, timeout: int=120) -> None:
        config.load_env()
        self.base_url = (base_url or os.getenv('COMFYROUTER_LLM_BASE_URL', '')).rstrip('/')
        self.api_key = api_key or os.getenv('COMFYROUTER_LLM_API_KEY', '')
        self.model = model or os.getenv('COMFYROUTER_LLM_MODEL', '')
        self.iteration_model = os.getenv('COMFYROUTER_LLM_ITERATION_MODEL', '').strip() or self.model
        self.evaluation_model = config.evaluation_model(self.iteration_model)
        self.evaluation_api_key = config.evaluation_api_key(self.api_key)
        self.proxy_url = os.getenv('COMFYROUTER_LLM_PROXY_URL', '').strip()
        self.user_agent = os.getenv('COMFYROUTER_LLM_USER_AGENT', 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 Chrome/140 Safari/537.36')
        self.timeout = min(int(os.getenv('COMFYROUTER_LLM_TIMEOUT', str(timeout))), 120)
        self.connect_timeout = min(max(float(os.getenv('COMFYROUTER_LLM_CONNECT_TIMEOUT', '8')), 1.0), float(self.timeout))
        self.header_timeout = min(max(float(os.getenv('COMFYROUTER_LLM_HEADER_TIMEOUT', '90')), 1.0), float(self.timeout))
        self.max_timeout = max(float(os.getenv('COMFYROUTER_LLM_MAX_TIMEOUT', '180')), self.timeout)
        self.timeout_increment = max(float(os.getenv('COMFYROUTER_LLM_TIMEOUT_INCREMENT', '30')), 0.0)
        self.retries = min(max(int(os.getenv('COMFYROUTER_LLM_RETRIES', '2')), 0), 2)
        self.retry_backoff = min(max(float(os.getenv('COMFYROUTER_LLM_RETRY_BACKOFF', '5')), 0.0), 120.0)
        write_setting = os.getenv('COMFYROUTER_LLM_WRITE_TIMEOUT', '').strip()
        self.write_timeout = max(float(write_setting), 1.0) if write_setting else None
        self.max_output_tokens = int(os.getenv('COMFYROUTER_LLM_MAX_OUTPUT_TOKENS', '12000'))
        self.result_protocol = os.getenv('COMFYROUTER_LLM_RESULT_PROTOCOL', 'function_calling').strip().lower()
        if self.result_protocol not in {'function_calling', 'json_schema'}:
            raise ValueError('COMFYROUTER_LLM_RESULT_PROTOCOL must be function_calling or json_schema')
        self.prompt_cache_namespace = os.getenv('COMFYROUTER_LLM_PROMPT_CACHE_NAMESPACE', 'comfyrouter').strip() or 'comfyrouter'
        self._ipv4_rotation_offset = 0
        self.http_client = self._new_http_client()
        self._prompt_histories: dict[str, list[dict[str, Any]]] = {}
        audit_dir = os.getenv('COMFYROUTER_AGENT_AUDIT_DIR', '')
        self.audit_dir = Path(audit_dir) if audit_dir else None
        if self.audit_dir:
            self.audit_dir.mkdir(parents=True, exist_ok=True)
        if not self.base_url or not self.model:
            raise ValueError('LLM client requires COMFYROUTER_LLM_BASE_URL and COMFYROUTER_LLM_MODEL')

    def _retry_delay_seconds(self, attempt: int, response_headers: dict[str, Any] | None=None) -> float:
        """Honor Retry-After when present; otherwise use bounded exponential backoff."""
        response_headers = response_headers or {}
        retry_after = next((str(value) for key, value in response_headers.items() if str(key).casefold() == 'retry-after'), '').strip()
        if retry_after:
            try:
                delay = float(retry_after)
            except ValueError:
                try:
                    retry_at = parsedate_to_datetime(retry_after)
                    if retry_at.tzinfo is None:
                        retry_at = retry_at.replace(tzinfo=timezone.utc)
                    delay = (retry_at - datetime.now(timezone.utc)).total_seconds()
                except (TypeError, ValueError, OverflowError):
                    delay = -1
            if delay >= 0:
                return min(120.0, delay)
        return min(120.0, self.retry_backoff * 2 ** attempt)

    def _new_http_client(self) -> httpx.Client:
        transport = _IPv4HTTPTransport(retries=0, local_address='0.0.0.0', proxy=self.proxy_url or None, rotation_offset=self._ipv4_rotation_offset)
        return httpx.Client(trust_env=False, follow_redirects=True, limits=httpx.Limits(max_connections=8, max_keepalive_connections=2, keepalive_expiry=15.0), transport=transport)

    def _reset_http_client(self, *, rotate_ipv4: bool) -> None:
        """Discard a suspect pool and optionally start with the next A record."""
        old_client = self.http_client
        if rotate_ipv4:
            self._ipv4_rotation_offset += 1
        try:
            old_client.close()
        except Exception:
            pass
        self.http_client = self._new_http_client()

    def _model_for_call(self, agent_id: str, payload: dict[str, Any]) -> tuple[str, str]:
        if agent_id in {'feedback_agent', 'visual_evaluation_agent'}:
            return (self.evaluation_model, 'evaluation')
        return (self.iteration_model, 'generation' if agent_id == 'generation_agent' else 'attribution_iteration')

    def _prompt_cache_key_for(self, agent_id: str, payload: dict[str, Any]) -> str:
        requested_cache_key = payload.get('_prompt_cache_key')
        requested_namespace = payload.get('_prompt_cache_namespace')
        if not requested_namespace:
            for key in ('session_id', 'run_id', 'case_id'):
                candidates = [payload]
                for nested_key in ('full_trace', 'trace', 'report', 'request'):
                    nested = payload.get(nested_key)
                    if isinstance(nested, dict):
                        candidates.append(nested)
                value = next((candidate.get(key) for candidate in candidates if isinstance(candidate.get(key), (str, int)) and str(candidate.get(key)).strip()), None)
                if value is not None:
                    requested_namespace = f'{self.prompt_cache_namespace}:{key}:{value}'
                    break
        if not isinstance(requested_namespace, str) or not requested_namespace.strip():
            requested_namespace = self.prompt_cache_namespace
        if isinstance(requested_cache_key, str) and requested_cache_key.strip():
            return requested_cache_key.strip()
        stage_discriminator = ''
        if agent_id == 'generation_agent':
            implementation = payload.get('implementation_plan')
            if isinstance(implementation, list) and len(implementation) == 1 and isinstance(implementation[0], dict):
                step = implementation[0]
                decomposition = step.get('decomposition_step') or {}
                identity = {'step': decomposition.get('step'), 'operation': step.get('operation'), 'base_path': step.get('base_path')}
                stage_discriminator = ':stage:' + hashlib.sha256(json.dumps(identity, ensure_ascii=False, sort_keys=True).encode('utf-8')).hexdigest()[:16]
        digest = hashlib.sha256(f'{requested_namespace}:{agent_id}{stage_discriminator}'.encode('utf-8')).hexdigest()[:24]
        return f'{requested_namespace}:{agent_id}{stage_discriminator}:{digest}'

    def complete_json(self, agent_id: str, contract: dict[str, Any], payload: dict[str, Any]) -> dict[str, Any]:
        if not serial_api_enabled():
            return self._complete_json(agent_id, contract, payload)
        with serialized_api_request(agent_id=agent_id):
            return self._complete_json(agent_id, contract, payload)

    def _complete_json(self, agent_id: str, contract: dict[str, Any], payload: dict[str, Any]) -> dict[str, Any]:
        from .quality_policy import visual_tolerance_enabled, POLICY_TEXT
        if visual_tolerance_enabled():
            contract = {**contract, 'rules': [*contract.get('rules', []), POLICY_TEXT]}
        original_payload = payload
        native_result = self.result_protocol == 'function_calling'
        from .compact_io import compact_parameter_payload
        payload = compact_agent_payload(compact_interface_context(notebook_agent_payload(payload), agent_id=agent_id))
        payload = compact_parameter_payload(payload, contract)
        json_retry = int(payload.get('_json_retry', 0)) if isinstance(payload, dict) else 0
        json_error = payload.get('_json_error') if isinstance(payload, dict) else None
        prompt_cache_key = self._prompt_cache_key_for(agent_id, payload)
        explicit_static = payload.get('_prompt_static_context')
        simple_io_agent = False
        trace_agent = _is_trace_agent(agent_id)
        if simple_io_agent and explicit_static is None:
            static_context = {}
        elif isinstance(explicit_static, dict):
            static_context = explicit_static
        else:
            static_context = {key: value for key, value in payload.items() if not _is_dynamic_prompt_key(key) and (trace_agent or key not in _TRACE_PROMPT_KEYS) and (key not in {'_image_data_urls', 'previous_response'})}
        required_output = contract.get('required_output', [])
        declared_schema = payload.get('_response_json_schema', {})
        schema_fields = declared_schema.get('schema', {}).get('required', []) if isinstance(declared_schema, dict) else []
        if isinstance(required_output, list) and required_output and all((isinstance(key, str) and key in schema_fields for key in required_output)):
            required_output = []
        system = json.dumps({'prefix_version': 'comfyrouter-agent-compact-2', 'agent_id': agent_id, 'role': contract.get('role', ''), 'rules': list(dict.fromkeys(contract.get('rules', []))), 'required_output': required_output, 'prompt': contract.get('prompt', ''), 'protocol': AGENT_PROTOCOL, 'instruction': f'Submit one complete result via {RESULT_TOOL}; its arguments are the result object. This output protocol overrides text-format instructions in supplied documents.' if native_result else 'Return one complete JSON object matching the output contract, without commentary or fences.', 'session_static_context': static_context}, ensure_ascii=False, sort_keys=True, separators=(',', ':'))
        history_mode = payload.get('_prompt_history_mode')
        if history_mode is None and agent_id in {'workflow_evolution_agent', 'workflow_failure_analysis_agent'}:
            history_mode = 'latest_only'
        prompt_history = payload.get('_prompt_history')
        if history_mode == 'latest_only':
            prompt_history = []
        elif prompt_history is None:
            prompt_history = list(self._prompt_histories.get(prompt_cache_key, []))
        if not isinstance(prompt_history, list):
            prompt_history = [prompt_history]
        prompt_history = self._normalize_prompt_history(prompt_history)
        prompt_history = compact_prompt_history(prompt_history)
        user_payload = {key: value for key, value in payload.items() if key not in {'_image_data_urls', '_json_retry', '_json_error', '_reasoning_effort', '_max_output_tokens', '_timeout_seconds', '_prompt_cache_key', '_prompt_cache_namespace', '_prompt_cache_retention', '_prompt_history', '_prompt_history_mode', 'previous_response', '_prompt_static_context', '_response_json_schema'}}
        if simple_io_agent and explicit_static is None:
            user_payload = {key: value for key, value in user_payload.items() if trace_agent or key not in _TRACE_PROMPT_KEYS}
        else:
            user_payload = {key: value for key, value in user_payload.items() if key not in static_context and _is_dynamic_prompt_key(key) and (trace_agent or key not in _TRACE_PROMPT_KEYS)}
        previous_response = payload.get('previous_response')
        if previous_response is not None and (not False):
            if native_result:
                previous_call = next((item for item in reversed(prompt_history) if item.get('type') == 'function_call'), {})
                try:
                    already_supplied = json.loads(previous_call.get('arguments', 'null')) == previous_response
                except (ValueError, TypeError):
                    already_supplied = False
                if not already_supplied:
                    user_payload['previous_response'] = previous_response
            else:
                previous_message = self._assistant_history_message(previous_response)
                if not prompt_history or prompt_history[-1] != previous_message:
                    prompt_history.append(previous_message)
        if json_error:
            user_payload['contract_correction'] = (f'Correct the previous result by calling {RESULT_TOOL} with valid object arguments. ' if native_result else 'Return a complete valid JSON object without prose or fences. ') + f'Parser error: {json_error}'
        input_text = 'Current request suffix:\n' + json.dumps(user_payload, ensure_ascii=False, sort_keys=True, separators=(',', ':'))
        user_content: list[dict[str, Any]] = [{'type': 'input_text', 'text': input_text}]
        image_urls = payload.get('_image_data_urls', [])
        if isinstance(image_urls, list) and image_urls:
            user_content.extend(({'type': 'input_image', 'image_url': url} for url in image_urls if isinstance(url, str)))
        selected_model, model_route = self._model_for_call(agent_id, payload)
        prompt_cache_enabled = True
        max_output_tokens = payload.get('_max_output_tokens', self.max_output_tokens)
        if not isinstance(max_output_tokens, int) or max_output_tokens < 1:
            max_output_tokens = self.max_output_tokens
        timeout_seconds = payload.get('_timeout_seconds', self.timeout)
        if not isinstance(timeout_seconds, (int, float)) or timeout_seconds <= 0:
            timeout_seconds = self.timeout
        timeout_seconds = min(float(timeout_seconds), self.max_timeout)
        cache_mode = os.getenv('COMFYROUTER_LLM_PROMPT_CACHE_MODE', 'implicit').strip().lower()
        if cache_mode not in {'implicit', 'explicit'}:
            cache_mode = 'explicit'
        stable_content: dict[str, Any] = {'type': 'input_text', 'text': system}
        if cache_mode == 'explicit':
            stable_content['prompt_cache_breakpoint'] = {'mode': 'explicit'}
        developer_message = {'type': 'message', 'role': 'developer', 'content': [stable_content]}
        request_body = {'model': selected_model, 'temperature': 0, 'tools': self._stable_tools(contract.get('tools', [])), 'input': [developer_message] + prompt_history + [{'type': 'message', 'role': 'user', 'content': user_content}], 'max_output_tokens': min(max_output_tokens, self.max_output_tokens)}
        request_body['instructions'] = 'Follow the developer contract for the current request. ' + (f'Submit its result using {RESULT_TOOL}.' if native_result else 'Return its JSON result.')
        response_json_schema = payload.get('_response_json_schema')
        if native_result:
            request_body['tools'] = [result_tool(response_json_schema)]
            request_body['tool_choice'] = {'type': 'function', 'name': RESULT_TOOL}
            request_body['parallel_tool_calls'] = False
        elif isinstance(response_json_schema, dict):
            request_body['text'] = {'format': {'type': 'json_schema', 'name': str(response_json_schema.get('name') or 'agent_response'), 'strict': strict_compatible(response_json_schema.get('schema', {})), 'schema': response_json_schema.get('schema', {})}}
        if prompt_cache_enabled and prompt_cache_key:
            request_body['prompt_cache_key'] = prompt_cache_key
            request_body['prompt_cache_options'] = {'mode': cache_mode, 'ttl': os.getenv('COMFYROUTER_LLM_PROMPT_CACHE_TTL', '30m')}
        reasoning_effort = payload.get('_reasoning_effort')
        if reasoning_effort in {'none', 'minimal', 'low', 'medium', 'high', 'xhigh'}:
            request_body['reasoning'] = {'effort': reasoning_effort}
        if os.getenv('COMFYROUTER_LLM_STREAM', '1') != '0':
            request_body['stream'] = True
        body = json.dumps(request_body).encode('utf-8')
        headers = {'Content-Type': 'application/json', 'User-Agent': self.user_agent}
        if prompt_cache_enabled and prompt_cache_key:
            headers['X-Prompt-Cache-Key'] = prompt_cache_key
        if request_body.get('stream'):
            headers['Accept'] = 'text/event-stream'
        api_key = self.evaluation_api_key if model_route == 'evaluation' else self.api_key
        if api_key:
            headers['Authorization'] = f'Bearer {api_key}'
        audit: dict[str, Any] = {'agent_id': agent_id, 'contract': contract, 'payload': original_payload, 'sent_payload': payload, 'request_options': {'reasoning': request_body.get('reasoning'), 'model': selected_model, 'model_route': model_route, 'result_protocol': self.result_protocol, 'result_tool_strict': request_body['tools'][0]['strict'] if native_result else None, 'timeout_seconds': timeout_seconds, 'connect_timeout_seconds': min(self.connect_timeout, timeout_seconds), 'header_read_timeout_seconds': min(self.header_timeout, timeout_seconds), 'prompt_cache_enabled': prompt_cache_enabled, 'prompt_cache_key': prompt_cache_key, 'prompt_cache_mode': cache_mode, 'history_messages': len(prompt_history), 'history_mode': history_mode or 'append', 'static_context_keys': sorted(static_context), 'dynamic_payload_keys': sorted(user_payload), 'prefix_version': 'comfyrouter-agent-compact-2'}}
        if agent_id == 'direct_semantic_selection_agent':
            audit['request_body'] = request_body
        result = None
        attempts = self.retries + 1
        for attempt in range(attempts):
            attempt_timeout = min(timeout_seconds + self.timeout_increment * attempt, self.max_timeout)
            attempt_header_timeout = min(self.header_timeout * (attempt + 1), attempt_timeout)
            attempt_started_wall = datetime.now(timezone.utc).isoformat()
            attempt_started = time.monotonic()
            attempt_audit: dict[str, Any] = {'attempt': attempt + 1, 'started_at': attempt_started_wall, 'connect_timeout_seconds': min(self.connect_timeout, attempt_timeout), 'header_read_timeout_seconds': attempt_header_timeout, 'total_deadline_seconds': attempt_timeout, 'request_body_bytes': len(body), 'write_timeout_seconds': self._request_write_timeout(body, attempt_timeout)}
            audit.setdefault('transport_attempts', []).append(attempt_audit)
            response = None
            stream_phase = 'response_headers'
            try:
                deadline = time.monotonic() + attempt_timeout
                response = self._send_httpx(body=body, headers=headers, timeout_seconds=attempt_timeout, header_timeout_seconds=attempt_header_timeout)
                try:
                    content_type = response.headers.get('Content-Type', '')
                    audit['response_headers'] = dict(response.headers.items())
                    attempt_audit['http_status_code'] = response.status_code
                    attempt_audit['response_headers'] = dict(response.headers.items())
                    stream_phase = 'response_headers'
                    attempt_audit['phase'] = stream_phase
                    attempt_audit['response_headers_received'] = True
                    attempt_audit['headers_received_at'] = datetime.now(timezone.utc).isoformat()
                    attempt_audit['headers_elapsed_seconds'] = round(time.monotonic() - attempt_started, 3)
                    attempt_audit['request_id'] = response.headers.get('x-request-id')
                    attempt_audit['client_request_id'] = response.headers.get('x-client-request-id')
                    if response.is_error:
                        response.read()
                        attempt_audit['response_body_text'] = response.content.decode('utf-8', errors='replace')
                    response.raise_for_status()
                    if 'text/event-stream' in content_type.lower():
                        stream_lines: list[str] = []
                        attempt_audit['response_stream_lines'] = stream_lines
                        result = self._read_httpx_sse_completed_response(response, deadline=deadline, event_logger=lambda event, _at: attempt_audit.update({'phase': 'sse_stream', 'last_sse_event': event, 'response_completed_received': event == 'response.completed' or attempt_audit.get('response_completed_received', False), 'done_received': event == 'done_received' or attempt_audit.get('done_received', False)}), line_logger=stream_lines.append)
                    else:
                        result = self._read_httpx_completed_json(response)
                finally:
                    if 'response_stream_lines' in attempt_audit:
                        attempt_audit['response_body_text'] = '\n'.join(attempt_audit['response_stream_lines'])
                    elif response is not None:
                        try:
                            raw_body = getattr(response, 'content', None)
                        except Exception:
                            raw_body = None
                        if isinstance(raw_body, bytes):
                            attempt_audit['response_body_text'] = raw_body.decode('utf-8', errors='replace')
                    response.close()
                if isinstance(result, dict) and result.get('status') == 'failed':
                    provider_error = result.get('error') or {}
                    provider_code = provider_error.get('code') if isinstance(provider_error, dict) else None
                    if provider_code in {'server_error', 'internal_error', 'service_unavailable', 'overloaded_error', 'rate_limit_exceeded'}:
                        attempt_audit['status'] = 'provider_retryable_error'
                        attempt_audit['provider_error'] = provider_error
                        attempt_audit['completed_at'] = datetime.now(timezone.utc).isoformat()
                        attempt_audit['elapsed_seconds'] = round(time.monotonic() - attempt_started, 3)
                        audit.setdefault('provider_retry_errors', []).append({'attempt': attempt + 1, 'http_status_code': attempt_audit.get('http_status_code'), 'error': provider_error})
                        if attempt < self.retries:
                            delay = self._retry_delay_seconds(attempt, attempt_audit.get('response_headers'))
                            attempt_audit['retry_delay_seconds'] = delay
                            self._write_audit(audit)
                            time.sleep(delay)
                            continue
                        attempt_audit['status'] = 'provider_error_retries_exhausted'
                        self._write_audit(audit)
                        break
                attempt_audit['status'] = 'success'
                attempt_audit['completed_at'] = datetime.now(timezone.utc).isoformat()
                attempt_audit['elapsed_seconds'] = round(time.monotonic() - attempt_started, 3)
                break
            except (httpx.TransportError, httpx.HTTPStatusError, TimeoutError, json.JSONDecodeError) as exc:
                error_text = str(exc)
                if isinstance(exc, httpx.HTTPStatusError):
                    try:
                        exc.response.read()
                        response_body = exc.response.content.decode('utf-8', errors='replace')
                    except Exception:
                        response_body = ''
                    attempt_audit['http_status_code'] = exc.response.status_code
                    attempt_audit['response_headers'] = dict(exc.response.headers.items())
                    attempt_audit['response_body_text'] = response_body
                    if response_body:
                        audit.setdefault('response_errors', []).append({'status': exc.response.status_code, 'body': response_body})
                        error_text = f'{error_text}: {response_body}'
                        try:
                            provider_response = json.loads(response_body)
                        except ValueError:
                            provider_response = {}
                        if isinstance(provider_response, dict) and isinstance(provider_response.get('error'), dict) and (provider_response['error'].get('code') == 'context_length_exceeded'):
                            audit['provider_error'] = provider_response['error']
                            self._write_audit(audit)
                            raise AgentProviderError(agent_id, provider_response) from exc
                audit.setdefault('request_errors', []).append({'attempt': attempt + 1, 'error': error_text})
                attempt_audit['status'] = 'error'
                attempt_audit['error'] = error_text
                if isinstance(exc, (httpx.ConnectError, httpx.ConnectTimeout)):
                    stream_phase = 'connect'
                elif isinstance(exc, httpx.WriteTimeout):
                    stream_phase = 'request_body'
                elif isinstance(exc, httpx.ReadTimeout):
                    stream_phase = 'response_headers'
                attempt_audit.setdefault('phase', stream_phase)
                attempt_audit.setdefault('response_headers_received', False)
                attempt_audit['completed_at'] = datetime.now(timezone.utc).isoformat()
                attempt_audit['elapsed_seconds'] = round(time.monotonic() - attempt_started, 3)
                status = exc.response.status_code if isinstance(exc, httpx.HTTPStatusError) else None
                transient = status in {408, 429, 500, 502, 503, 504, 524} or isinstance(exc, (httpx.TransportError, TimeoutError, json.JSONDecodeError))
                if not transient or attempt >= self.retries:
                    self._write_audit(audit)
                    raise AgentTransportError(f'LLM request failed for {agent_id}: {error_text}') from exc
                if isinstance(exc, (httpx.TransportError, TimeoutError)):
                    self._reset_http_client(rotate_ipv4=True)
                    attempt_audit['connection_pool_reset'] = True
                    attempt_audit['next_ipv4_rotation_offset'] = self._ipv4_rotation_offset
                    time.sleep(self._retry_delay_seconds(attempt, attempt_audit.get('response_headers')))
        if result is None:
            self._write_audit(audit)
            raise AgentTransportError(f'LLM request failed for {agent_id}: no response')
        audit['response'] = result
        if isinstance(result, dict) and (result.get('error') or result.get('status') in {'failed', 'incomplete'}):
            audit['provider_error'] = result.get('error') or result.get('incomplete_details') or {'status': result.get('status')}
            self._write_audit(audit)
            raise AgentProviderError(agent_id, result)
        try:
            native_history = []
            if native_result:
                audit['raw_content'] = json.dumps(result.get('output', []), ensure_ascii=False)
                parsed, content, native_history = parse_result_call(result)
            else:
                content = self._response_output_text(result)
                audit['raw_content'] = content
                parsed = self._parse_json_object(content)
            audit['raw_content'] = content
        except (KeyError, IndexError, TypeError, ValueError) as exc:
            if not native_result and agent_id == 'workflow_skill_publication_agent' and isinstance(audit.get('raw_content'), str) and audit['raw_content'].lstrip().startswith(('# ', '---')) and all((section.casefold() in audit['raw_content'].casefold() for section in ('When To Use', 'Required Inputs', 'How To Use', 'Do Not Use', 'Verification Evidence'))):
                parsed = {'skill_md': audit['raw_content']}
                audit['accepted_direct_markdown'] = True
            else:
                audit['parse_error'] = str(exc)
                is_evolution = agent_id in {'evolution_planner_agent', 'skill_evolution_agent'}
                audit['evolution_status'] = 'blocked' if is_evolution else None
                audit['block_reason'] = ('agent_invalid_function_call' if native_result else 'agent_non_json_output') if is_evolution else None
                audit['agent_raw_snippet'] = str(audit.get('raw_content', ''))[:4000]
                audit['json_retry'] = json_retry
                self._write_audit(audit)
                if json_retry < 2:
                    retry_payload = dict(payload)
                    retry_payload['_json_retry'] = json_retry + 1
                    retry_payload['_json_error'] = str(exc)
                    retry_payload['_invalid_json_response'] = str(audit.get('raw_content', ''))[:4000]
                    retry_payload['_json_correction'] = f'Correct the invalid result using {RESULT_TOOL} with complete JSON object arguments.' if native_result else 'Correct the invalid response; return one JSON object without Markdown, prose, or tool calls.'
                    return self._complete_json(agent_id, contract, retry_payload)
                error = AgentOutputError(f'LLM returned invalid function_call result for {agent_id}' if native_result else f'LLM returned non-JSON output for {agent_id}')
                error.raw_response = str(audit.get('raw_content', ''))[:4000]
                error.parse_error = str(exc)
                raise error from exc
        if not isinstance(parsed, dict):
            raise AgentOutputError(f'LLM output for {agent_id} is not an object')
        current_user_message = {'role': 'user', 'content': user_content}
        next_history = list(prompt_history)
        if not next_history or next_history[-1] != current_user_message:
            next_history.append(current_user_message)
        if native_result:
            next_history.extend(native_history)
        else:
            next_history.append(self._assistant_history_message(parsed, raw_text=content))
        if history_mode != 'latest_only':
            self._prompt_histories[prompt_cache_key] = next_history
        usage = result.get('usage', {}) if isinstance(result, dict) else {}
        usage_details = usage.get('input_tokens_details') or usage.get('prompt_tokens_details') or {}
        input_tokens = usage.get('input_tokens', usage.get('prompt_tokens'))
        cached_tokens = usage_details.get('cached_tokens', 0)
        try:
            cache_hit_ratio = float(cached_tokens) / float(input_tokens) if input_tokens else 0.0
        except (TypeError, ValueError, ZeroDivisionError):
            cache_hit_ratio = 0.0
        audit['cache_metrics'] = {'input_tokens': input_tokens, 'cached_tokens': cached_tokens, 'cache_write_tokens': usage_details.get('cache_write_tokens', 0), 'cache_hit_ratio': cache_hit_ratio}
        audit['parsed'] = parsed
        self._write_audit(audit)
        return parsed

    def _request_write_timeout(self, body: bytes, timeout_seconds: float) -> float:
        override = getattr(self, 'write_timeout', None)
        upload_budget = float(override) if override is not None else timeout_seconds if len(body) >= 1024 * 1024 else 30.0
        return min(upload_budget, timeout_seconds)

    def _send_httpx(self, *, body: bytes, headers: dict[str, str], timeout_seconds: float, header_timeout_seconds: float | None=None) -> httpx.Response:
        """Open one streamed Responses request with bounded connection phases."""
        read_timeout = header_timeout_seconds or self.header_timeout
        timeout = httpx.Timeout(connect=min(self.connect_timeout, timeout_seconds), read=min(read_timeout, timeout_seconds), write=self._request_write_timeout(body, timeout_seconds), pool=min(self.connect_timeout, timeout_seconds))
        request = self.http_client.build_request('POST', self.base_url + '/responses', headers=headers, content=body, timeout=timeout)
        return self.http_client.send(request, stream=True)

    @staticmethod
    def _stable_tools(value: Any) -> list[dict[str, Any]]:
        """Canonicalize tool object keys without changing tool list order."""
        if not isinstance(value, list):
            return []
        try:
            normalized = json.loads(json.dumps(value, ensure_ascii=False, sort_keys=True))
        except (TypeError, ValueError):
            return []
        return [item for item in normalized if isinstance(item, dict)]

    @staticmethod
    def _assistant_history_message(value: Any, *, raw_text: str | None=None) -> dict[str, Any]:
        """Encode an earlier Agent result, preserving raw provider text when available."""
        return {'role': 'assistant', 'content': [{'type': 'output_text', 'text': raw_text if isinstance(raw_text, str) else json.dumps(value, ensure_ascii=False, sort_keys=True)}]}

    @classmethod
    def _normalize_prompt_history(cls, history: list[Any]) -> list[dict[str, Any]]:
        """Accept new message history and old evidence-shaped history safely."""
        normalized: list[dict[str, Any]] = []
        for item in history:
            if isinstance(item, dict) and item.get('type') in {'function_call', 'function_call_output', 'reasoning'}:
                normalized.append(item)
            elif isinstance(item, dict) and item.get('role') == 'user':
                normalized.append(item)
            elif isinstance(item, dict) and item.get('role') == 'assistant':
                content = item.get('content')
                if isinstance(content, str):
                    normalized.append({'role': 'assistant', 'content': [{'type': 'output_text', 'text': content}]})
                else:
                    normalized.append(item)
            elif isinstance(item, dict) and 'response' in item:
                normalized.append(cls._assistant_history_message(item['response']))
            elif isinstance(item, dict) and 'previous_response' in item:
                normalized.append(cls._assistant_history_message(item['previous_response']))
            else:
                normalized.append({'role': 'user', 'content': json.dumps(item, ensure_ascii=False, sort_keys=True)})
        return normalized

    @staticmethod
    def _read_httpx_sse_completed_response(response: httpx.Response, deadline: float | None=None, event_logger: Any | None=None, line_logger: Any | None=None) -> dict[str, Any]:
        """Consume an httpx Responses stream until its terminal SSE event."""
        output_text: list[str] = []
        completed_response: dict[str, Any] | None = None
        first_chunk = True
        last_chunk_at = time.monotonic()
        for raw_line in response.iter_lines():
            if line_logger is not None:
                line_logger(raw_line)
            now = time.monotonic()
            if event_logger is not None:
                event_logger('first_sse_chunk' if first_chunk else 'sse_chunk', now)
            first_chunk = False
            last_chunk_at = now
            if deadline is not None and time.monotonic() >= deadline:
                raise TimeoutError('SSE response exceeded the request deadline')
            line = raw_line.strip()
            if not line:
                continue
            if line == 'data: [DONE]':
                if event_logger is not None:
                    event_logger('done_received', last_chunk_at)
                break
            if not line.startswith('data: '):
                continue
            event = json.loads(line[6:])
            event_type = event.get('type')
            if event_type == 'response.output_text.delta':
                output_text.append(str(event.get('delta', '')))
            if event_type in {'response.completed', 'response.failed', 'response.incomplete'}:
                if event_logger is not None:
                    event_logger(event_type, last_chunk_at)
                completed = event.get('response')
                if isinstance(completed, dict):
                    completed_response = completed
                continue
        if completed_response is not None:
            return completed_response
        if output_text:
            return {'output_text': ''.join(output_text)}
        raise json.JSONDecodeError('SSE stream ended without a terminal response', '', 0)

    @staticmethod
    def _read_httpx_completed_json(response: httpx.Response) -> dict[str, Any]:
        """Read one bounded non-streaming httpx response as a JSON object."""
        response.read()
        parsed = response.json()
        if not isinstance(parsed, dict):
            raise json.JSONDecodeError('JSON root is not an object', response.text, 0)
        return parsed

    @staticmethod
    def _read_sse_completed_response(response: Any, deadline: float | None=None) -> dict[str, Any]:
        """Read Responses SSE until a terminal event instead of waiting for EOF."""
        output_text: list[str] = []
        while True:
            if deadline is not None:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError('SSE response exceeded the request deadline')
                sock = getattr(getattr(getattr(response, 'fp', None), 'raw', None), '_sock', None)
                if sock is not None:
                    sock.settimeout(max(0.1, remaining))
            try:
                raw_line = response.readline()
            except TimeoutError as exc:
                raise TimeoutError('SSE response exceeded the request deadline') from exc
            if not raw_line:
                break
            line = raw_line.decode('utf-8', errors='replace').strip()
            if not line:
                continue
            if line == 'data: [DONE]':
                break
            if not line.startswith('data: '):
                continue
            event = json.loads(line[6:])
            event_type = event.get('type')
            if event_type == 'response.output_text.delta':
                output_text.append(str(event.get('delta', '')))
            if event_type in {'response.completed', 'response.failed', 'response.incomplete'}:
                completed = event.get('response')
                if isinstance(completed, dict):
                    return completed
                break
        if output_text:
            return {'output_text': ''.join(output_text)}
        raise json.JSONDecodeError('SSE stream ended without a terminal response', '', 0)

    @staticmethod
    def _read_completed_json(response: Any) -> dict[str, Any]:
        """Parse a complete JSON response without waiting for connection close.

        Some compatible gateways keep an HTTP connection alive after sending a
        finished JSON object. ``response.read()`` then waits until socket
        timeout even though the model output is already available. Read one
        transport chunk at a time and return as soon as the JSON document is
        complete.
        """
        chunks: list[bytes] = []
        text_chunks: list[str] = []
        utf8_decoder = codecs.getincrementaldecoder('utf-8')()
        decoder = json.JSONDecoder()
        reader = getattr(response, 'read1', None)
        while True:
            chunk = reader(65536) if callable(reader) else response.read(65536)
            if not chunk:
                break
            chunks.append(chunk)
            text_chunks.append(utf8_decoder.decode(chunk, final=False))
            text = ''.join(text_chunks)
            try:
                parsed, end = decoder.raw_decode(text.lstrip())
                if text.lstrip()[end:].strip() == '':
                    if not isinstance(parsed, dict):
                        raise json.JSONDecodeError('JSON root is not an object', text, 0)
                    return parsed
            except json.JSONDecodeError:
                pass
        text_chunks.append(utf8_decoder.decode(b'', final=True))
        return json.loads(''.join(text_chunks))

    @staticmethod
    def _response_output_text(response: dict[str, Any]) -> str:
        direct = response.get('output_text')
        if isinstance(direct, str) and direct.strip():
            return direct
        for item in response.get('output', []):
            if not isinstance(item, dict) or item.get('type') != 'message':
                continue
            for content in item.get('content', []):
                if isinstance(content, dict) and content.get('type') == 'output_text' and isinstance(content.get('text'), str) and content['text'].strip():
                    return content['text']
        if isinstance(direct, str):
            return direct
        raise KeyError('Responses API output_text is missing')

    def _write_audit(self, record: dict[str, Any]) -> None:
        if not self.audit_dir:
            return
        stamp = uuid4().hex
        (self.audit_dir / f'{stamp}.json').write_text(json.dumps(record, ensure_ascii=False, indent=2, default=str), encoding='utf-8')

    @staticmethod
    def _parse_json_object(content: Any) -> dict[str, Any]:
        if not isinstance(content, str):
            raise json.JSONDecodeError('content is not text', '', 0)
        candidates = [content.strip()]
        fenced = re.search('```(?:json)?\\s*(\\{.*\\})\\s*```', content, re.DOTALL | re.IGNORECASE)
        if fenced:
            candidates.append(fenced.group(1).strip())
        start = content.find('{')
        if start >= 0:
            candidates.append(content[start:].strip())
        last_error = None
        parsed = None
        for candidate in candidates:
            try:
                parsed, end = json.JSONDecoder().raw_decode(candidate)
                if candidate[end:].strip() and candidate is candidates[0]:
                    continue
                break
            except json.JSONDecodeError as exc:
                last_error = exc
        if parsed is None:
            repaired = re.sub(',\\s*([}\\]])', '\\1', candidates[-1])
            try:
                parsed = json.loads(repaired)
            except json.JSONDecodeError:
                raise last_error or json.JSONDecodeError('invalid JSON', content, 0)
        if not isinstance(parsed, dict):
            raise json.JSONDecodeError('JSON root is not object', content, 0)
        return parsed

@dataclass(frozen=True)
class AgentDecision:
    task_family: str
    subworkflow_sequence: tuple[str, ...]
    capabilities: tuple[str, ...]
    priorities: tuple[str, ...]
    constraints: tuple[str, ...]
    explicit_model: str | None
    confidence: str
    proposal_ids: tuple[str, ...] = ()
    reason: str = ''

class AgentOrchestrator:
    """Plan, evaluate and repair the workflows selected by RAG."""

    def __init__(self, client: LLMClient, store=None, max_corrections: int=2) -> None:
        self.client = client
        self.store = None
        self.max_corrections = max_corrections
        self.image_max_dimension = max(int(os.getenv('COMFYROUTER_LLM_IMAGE_MAX_DIMENSION', '768')), 0)
        payload = json.loads(config.AGENT_SKILLS.read_text(encoding='utf-8'))
        self.skills = payload['agents']
        self.skills['generation_agent'].setdefault('rules', []).append('The physical leaf base_template.json is always the starting graph. If that template plus prompt/conditioning edits is sufficient, return an empty skill_ids list and the smallest justified edits; never select a Skill merely because it is available.')
        self.skills['generation_agent'].setdefault('rules', []).append('Inspect the actual graph before writing exclusions: when a true negative-conditioning input/path exists, use it for concise artifact and confusable-action exclusions; when the selected template has no negative-conditioning interface, do not invent a negative node or parameter and express necessary exclusions only in the existing positive conditioning fields.')
        self.skills['feedback_agent'].setdefault('rules', []).append('Treat clearly visible facial distortion, misplaced or merged eyes, malformed mouth/nose, severe hand or finger deformation, or unmistakably broken limb proportions as material output defects: mark output/task_alignment fail or partial, list the concrete defect, and score below 90. Do not dismiss these as ordinary model imperfections when the instruction requests high quality or a human subject.')
        self.skills['feedback_agent'].setdefault('rules', []).append('Evaluate whole-image plausibility, not keyword presence alone. Check whether the subject, action, objects, contacts, perspective, scale, depth, lighting, shadows, background, and framing form one physically and semantically coherent scene. For an action request, verify that the depicted pose actually supports the action and that interacted objects are attached, held, released, or positioned plausibly. Missing body parts are only one possible defect: fused or floating objects, impossible contacts, broken perspective, inconsistent scale, duplicated or cut-off elements, contradictory shadows, and collage-like local distortions are also material image-reasoning failures. If any visible global incoherence materially harms the requested result, mark output/task_alignment fail or partial and score below 90 even when the requested nouns are present.')
        self.skills['feedback_agent'].setdefault('rules', []).append('Apply action-mechanics and structural-causality checks to the specific scene. A sports action must show a physically readable phase and follow-through, plausible joint orientation, contact/release state, object trajectory, and target relationship; for a basketball jump-shot release, an extended shooting arm without a readable wrist snap/flexion and separated guide-hand role is not enough. Key equipment must also be structurally plausible: a primary hoop must have an explainable backboard/support or crop, and multiple hoops must occupy a coherent court/depth arrangement rather than appear duplicated or floating. Treat failures of these mechanics or structures as material whole-image incoherence and score below 90.')
        self.skills['generation_agent'].setdefault('rules', []).append('For every multi-stage media handoff (image->image, image->video, video->video, or fan-in), inspect handoff_geometry_context before editing the current graph. An unplanned_template_hint consumer canvas is a mutable Base default, not a task resolution constraint. Preserve source content and aspect ratio; the consumer may adapt its own dimensions when it is planned. Do not block or distort the producer solely to match an unplanned square Base. Compute r_up=W_up/H_up and r_down=W_down/H_down. Make the producer canvas match the direct consumer canvas when possible. If they differ, explicitly choose producer resizing, contain/letterbox padding, or a justified crop whose estimated retained region preserves all required content. Never accept an implicit center/cover crop that can remove a head, feet, hands, text, logo, boundary object, or required composition. Prompts for bounded subjects must request visible safety margin on every edge, and the handoff result itself must be checked for complete framing. This policy is global and applies regardless of leaf, model, modality, or case.')

    @staticmethod
    def _normalize_mask_construction_fields(step: dict[str, Any]) -> None:
        """Normalize typed mask intent while retaining compatibility with old traces."""
        if str(step.get('output_type') or '').casefold() != 'mask':
            step['target_presence'] = None
            step['mask_construction'] = None
            step['geometry_requirement'] = None
            step['protected_regions'] = []
            return
        text = str(step.get('description') or '').casefold()
        construction = step.get('mask_construction')
        if construction not in {'segment_visible', 'prospective_placement', 'replacement_envelope', 'unknown'}:
            if any((marker in text for marker in ('prospective', 'placement footprint', 'placement region', 'future footprint', '占位', '放置区域'))):
                construction = 'prospective_placement'
            elif any((marker in text for marker in ('replacement envelope', 'complete replacement footprint', 'minimum enclosing', '替换包络', '完整替换区域'))):
                construction = 'replacement_envelope'
            else:
                construction = 'segment_visible'
        step['mask_construction'] = construction
        presence = step.get('target_presence')
        if presence not in {'present', 'absent', 'mixed', 'unknown'}:
            presence = 'absent' if construction == 'prospective_placement' else 'present'
        step['target_presence'] = presence
        geometry = step.get('geometry_requirement')
        if geometry not in {'source_topology', 'single_regular', 'composite_or_freeform', 'infer_from_visual'}:
            if construction == 'segment_visible':
                geometry = 'source_topology'
            elif any((marker in text for marker in ('multi-part', 'multipart', 'nonrectangular', 'non-rectangular', 'composite', 'curved', '多部分', '非矩形', '复合'))):
                geometry = 'composite_or_freeform'
            elif any((marker in text for marker in ('rectangle', 'rectangular', 'circle', 'ellipse', 'bounded horizontal', '矩形', '圆形', '椭圆'))):
                geometry = 'single_regular'
            else:
                geometry = 'infer_from_visual'
        step['geometry_requirement'] = geometry
        protected = step.get('protected_regions')
        if not isinstance(protected, list) or not all((isinstance(item, str) and item.strip() for item in protected)):
            protected = []
        step['protected_regions'] = list(dict.fromkeys((item.strip() for item in protected)))

    @staticmethod
    def _validate_runtime_research_bindings(spec: RequirementSpec, implementation_plan: list[dict[str, Any]]) -> None:
        """Compatibility hook; model/Base verification is deferred by policy."""
        return None

    def generate_graph_edit_plan(self, spec: RequirementSpec, implementation_plan: list[dict[str, Any]], graph: dict[str, Any] | None=None, repair_evidence: dict[str, Any] | None=None, allow_active_skills: bool=False, parameter_only: bool=False, allow_topology_repair: bool=False, instruction: str='', overall_instruction: str='', handoff_geometry_context: dict[str, Any] | None=None, handoff_contract_context: dict[str, Any] | None=None, visual_context_paths: list[str] | None=None) -> dict[str, Any]:
        """Ask the generation Agent for a typed, reviewable graph-edit plan.

        This method plans edits only. Applying edits still requires static graph
        validation and runtime execution before any workflow Skill can evolve.
        """
        self._validate_runtime_research_bindings(spec, implementation_plan)
        workflow_skills = []
        skill_folders = {}
        workflow_selection_metadata = []
        knowledge_steps = []
        knowledge_steps = list(implementation_plan)
        allowed_skill_ids = {item['skill_id'] for item in workflow_skills}
        task_model_knowledge = []
        base_workflow_experience = {'required': False, 'entries': []}
        request_payload: dict[str, Any] = {'video_timing_policy': VIDEO_TIMING_POLICY, 'requirement_spec': {'modality': spec.modality, 'task_family': spec.task_family, 'operations': list(spec.operations), 'capabilities': list(spec.capabilities), 'priorities': list(spec.priorities), 'constraints': list(spec.constraints), 'explicit_model': spec.explicit_model}, 'instruction': instruction, 'implementation_plan': implementation_plan, 'available_workflow_skills': workflow_skills, 'cold_start': not workflow_skills, 'knowledge_fallback': bool(task_model_knowledge), 'task_model_knowledge': task_model_knowledge, 'base_workflow_experience': base_workflow_experience, 'experience_policy': "Experience is owned by one workflow, never by the leaf as a shared pool. Read base_workflow_experience only when assessing or selecting Base. Read each candidate Skill's workflow_experience only while assessing that same Skill, and after selection use only the selected workflow's experience. Never transfer a rule between Base and a Skill or between Skills. Experience is reusable conditional guidance, not a case template: do not copy concrete files, subjects, scenes, seeds, or exact training prompts. Apply a rule only when its documented trigger and current observable evidence match.", 'skill_selection_policy': "Select Base and every supplied Workflow Skill from the typed operation and mask-construction contract; Base has no priority. Read every supplied Workflow Skill's instructions independently. A Workflow Skill supplies its own complete replacement graph: selecting it does not edit the Base topology, even in parameter-only planning. Do not reject a matching Skill because the Base lacks nodes that are present in the Skill's workflow_template. For every supplied Skill, return exactly one skill_assessments row explaining whether its SKILL.md triggers, required inputs, exclusions, and scope match the current request and failure trace. Select the single best complete matching workflow in skill_ids; if none match, return skill_ids=[] and base_decision.use_base=true. Do not select a Skill merely because it is available or historically successful. Do not merge complete Skills with Base or each other. A selected Skill's workflow_template is the candidate branch to bind and validate; the Base JSON remains unchanged.", 'incremental_skill_selection': {'required': True, 'per_skill_reading': 'Read each supplied SKILL.md before deciding.', 'decision_values': ['use_increment', 'use_base'], 'use_increment_when': "The exact leaf/profile matches and the Skill's documented trigger is supported by the current input, execution trace, and evaluation result.", 'use_base_when': "No supplied trigger matches, the required handoff is absent from both the supplied upstream_handoff_contract and runtime artifacts, or the task is outside the Skill's documented scope. A missing loader in current_graph alone is not evidence that a declared upstream handoff is absent when the selected Skill is a complete replacement graph.", 'output_requirement': 'Assess every supplied Skill exactly once in skill_assessments. Return zero or one selected skill id. An empty list requires base_decision.use_base=true and explicitly means use Base.'}}
        required_stage_input_types = {str(value).casefold() for step in implementation_plan if isinstance(step, dict) for value in (step.get('decomposition_step') or {}).get('input_types', [(step.get('decomposition_step') or {}).get('input_type')]) if value}
        source_visuals = [str(path) for path in list(visual_context_paths or []) if isinstance(path, (str, Path)) and Path(path).is_file() and (Path(path).suffix.casefold() in {'.png', '.jpg', '.jpeg', '.webp', '.bmp'})][:4] if not required_stage_input_types or 'image' in required_stage_input_types else []
        if source_visuals:
            request_payload['source_visual_order'] = [{'position': index + 1, 'role': 'source_input', 'path': path} for index, path in enumerate(source_visuals)]
            request_payload.update(self._image_evidence_payload(source_visuals))
        if overall_instruction and overall_instruction != instruction:
            request_payload['overall_instruction'] = overall_instruction
            request_payload['stage_scope_policy'] = 'Edit only the atomic stage described by instruction. Use overall_instruction only to preserve global numeric and handoff constraints such as final duration, FPS, dimensions, required upstream/downstream media contracts, and visible outcomes that an upstream stage already established. A downstream generative stage must not reintroduce content that an upstream stage removed or replaced.'
        if handoff_geometry_context:
            request_payload['handoff_geometry_context'] = handoff_geometry_context
            request_payload['handoff_geometry_policy'] = "This is direct-stage geometry evidence. An unplanned_template_hint consumer is a mutable Base default, not a mandatory output size: preserve the producer's source composition and adapt that consumer when it is planned, unless the task explicitly fixes its size. Use the supplied node IDs, width, height, aspect ratios, crop behavior, and cover-crop risk estimates. For a producer, prefer set_param edits that align its output canvas to each direct consumer. If exact alignment is unavailable, preserve the whole required subject with contain/letterbox or explicit safe margins. Include prompt edits for framing when the graph exposes text conditioning, but do not treat prompt text alone as a substitute for editable width and height. Never silently rely on center crop."
        if handoff_contract_context:
            request_payload['upstream_handoff_contract'] = copy.deepcopy(handoff_contract_context)
            request_payload['handoff_contract_policy'] = "Treat upstream_handoff_contract as authoritative media-boundary evidence. A required MASK produced by an earlier stage satisfies a Skill's input precondition when its SKILL.md requires an aligned IMAGE+MASK handoff, even if current_graph/Base has no LoadImageMask. After selecting a complete Skill, the runtime adapter binds the real artifact and static validation checks the selected graph's loader. If no selected Skill or Base can consume a required MASK, report a structural handoff failure and choose topology/leaf repair; do not turn it into a prompt-only repair."
        generated_mask_nodes: list[str] = []
        required_mask_skill_ids: set[str] = set()
        if graph is not None:
            request_payload['current_graph'] = graph
            if repair_evidence is not None:
                request_payload['base_graph'] = graph
            request_payload['temporal_parameter_context'] = self._temporal_parameter_context(graph, overall_instruction or instruction)
            generated_mask_nodes = [str(node_id) for node_id, node in graph.items() if isinstance(node, dict) and node.get('class_type') in {'GroundingDinoSAMSegment (segment anything)', 'SAM3_VideoTrack'}]
            mask_prompt_bindings = [{'node_id': str(node_id), 'param': 'prompt' if node.get('class_type') == 'GroundingDinoSAMSegment (segment anything)' else 'text'} for node_id, node in graph.items() if isinstance(node, dict) and (node.get('class_type') == 'GroundingDinoSAMSegment (segment anything)' and node.get('inputs', {}).get('prompt') == 'TARGET_MASK_PROMPT' or (node.get('class_type') == 'CLIPTextEncode' and node.get('inputs', {}).get('text') == 'TARGET_MASK_PROMPT'))]
            if generated_mask_nodes:
                request_payload['generated_mask_nodes'] = generated_mask_nodes
                request_payload['mask_prompt_policy'] = "This Base generates its own mask. The segmentation text prompt identifies the concrete visible subject or region from the request; never request an external mask. For removal tasks, put the removed subject name only in the segmentation prompt. Positive inpainting prompts must describe the replacement background and must not repeat the removed subject name, even in a negative phrase such as 'no car'. Apply this policy on initial generation and every repair round."
            if mask_prompt_bindings:
                request_payload['required_mask_prompt_bindings'] = mask_prompt_bindings
                request_payload['mask_prompt_binding_policy'] = "For every required_mask_prompt_binding, return a set_param edit that replaces TARGET_MASK_PROMPT with the concrete visible subject or region to segment from this request. Do not leave the placeholder unchanged. For SAM3 CLIPTextEncode.text, use one short positive whole-object noun phrase such as 'hatchback car on the road'. Do not describe colors, materials, object parts, protected regions, exclusions, or editing instructions; SAM3 tracks the whole object and the downstream mask controls the edit boundary."
        operations = {str(step.get('operation', '')).casefold() for step in implementation_plan if isinstance(step, dict)}
        if any((operation == 'geometry_editing/extension' for operation in operations)):
            request_payload['image_extension_prompt_policy'] = 'Describe only the new content that must continue across each extended edge, grounded in visible objects that cross that boundary. Do not enumerate protected content that is already complete inside the source image. For a downward portrait extension, if the face, head, neck, and shoulders are above the bottom boundary, omit them from positive conditioning and describe the single continuous chest, lower torso, waist, garment edges, and adjacent backdrop below the crop. Write a direct positive description of the final visible composition, not operation instructions or a negated list of unwanted content. Require one spatially continuous subject through positive final-state language. Follow the current source image rather than assuming portrait-specific content for other extension requests.'
        if any((operation.endswith('/content_cloning') for operation in operations)):
            negative_text_bindings = []
            for node_id, node in (graph or {}).items():
                if not isinstance(node, dict) or node.get('class_type') != 'InstructPixToPixConditioning':
                    continue
                source = (node.get('inputs') or {}).get('negative')
                if not isinstance(source, list) or not source:
                    continue
                source_id = str(source[0])
                source_node = (graph or {}).get(source_id)
                if isinstance(source_node, dict) and source_node.get('class_type') == 'CLIPTextEncode' and ('text' in (source_node.get('inputs') or {})):
                    negative_text_bindings.append({'node_id': source_id, 'param': 'text', 'consumer_node_id': str(node_id)})
            request_payload['content_cloning_prompt_policy'] = "Write positive conditioning as a direct description of the final visible image. When overall_instruction or an upstream stage requires an object or attribute to remain absent, do not mention that excluded concept in positive conditioning, including negated phrases such as 'no X', 'free of X', or 'without X'. Derive the excluded concept from the supplied request; never reuse a concrete subject from examples or experience. If content_cloning_negative_text_bindings is non-empty, return a set_param edit for each binding that places the dynamically derived excluded concept and useful close synonyms in negative conditioning, together with generic quality defects. Preserve all unrelated scene content in positive conditioning."
            request_payload['content_cloning_negative_text_bindings'] = negative_text_bindings
        if any((operation.endswith('/removal') for operation in operations)):
            has_external_mask = any((isinstance(node, dict) and node.get('class_type') == 'LoadImageMask' for node in (graph or {}).values()))
            if generated_mask_nodes or has_external_mask:
                request_payload['object_removal_prompt_policy'] = 'This removal graph is mask-conditioned. Inspect source_visual_order when supplied and describe the concrete final scene content that should occupy the selected region: its surface or material, geometry, texture, perspective, lighting, and shadows. Do not repeat the removed object in the positive prompt when the mask or segmentation prompt already identifies it. Preserve unrelated content.'
        if any((operation.endswith('/replacement') for operation in operations)):
            request_payload['object_replacement_prompt_policy'] = "Write positive conditioning as a direct visual description of the final filled region. Describe the new object's concrete shape, material, pose/orientation, scale, perspective, lighting, reflections, contact shadow, and surrounding continuity. Do not use action wording such as replace, remove, instead of, former, or where the old object was, and do not name or describe the old object in positive conditioning. The external mask already defines what is edited; mentioning the old object can regenerate its semantics. If the graph uses ConditioningZeroOut for negative conditioning, do not claim that an old-object negative text exists. Preserve unrelated content through the final-state description and mask boundary."
        if any((operation == 'identity_editing/clothing' for operation in operations)):
            request_payload['clothing_replacement_prompt_policy'] = 'Describe the requested final garment without inventing an open, zipped, unzipped, pullover, or layered construction that the instruction does not specify. When an upstream edit-region mask is a replacement envelope, do not preserve or mention an underlayer that lies inside that editable envelope if the requested garment may cover it. Preserve only explicitly protected body parts, accessories, and scene content outside the garment footprint. For a complete garment category-and-color replacement using FluxGuidance, prefer a moderate value around 7 instead of retaining an extreme value such as 30; keep existing values for localized topology-preserving attribute edits unless evidence requires a change.'
        if any((operation.endswith('/seam_refinement') for operation in operations)):
            request_payload['seam_refinement_prompt_policy'] = "Positive text conditioning describes only the desired final surface: its observed material, color, fine texture, lighting, and continuity, while preserving nearby objects. Do not describe the location or geometry of the old boundary, the former object, or a circular/linear seam. A phrase like 'where the removed object met the surface' can draw that outline back into the repaired image. Keep the editable region defined solely by the external mask, not by a verbal description of its shape."
        if any((operation == 'mask_processing/generation' for operation in operations)):
            for step in implementation_plan:
                if not isinstance(step, dict) or str(step.get('operation') or '').casefold() != 'mask_processing/generation':
                    continue
                decomposition = step.setdefault('decomposition_step', {})
                decomposition.setdefault('output_type', 'mask')
                self._normalize_mask_construction_fields(decomposition)
            declared_mask_roles = {str((step.get('decomposition_step') or {}).get('mask_role') or '').strip() for step in implementation_plan if isinstance(step, dict) and str(step.get('operation') or '').casefold() == 'mask_processing/generation'} - {''}
            declared_mask_role = next(iter(declared_mask_roles)) if len(declared_mask_roles) == 1 else None
            request_payload['mask_output_contract'] = {'mask_role': declared_mask_role or 'explicit_contract_required', 'white_means': 'selected_region', 'black_means': 'everything_else', 'policy': 'Inspect the source visual, overall instruction, declared mask_role, and typed mask-construction contract. Only segment_visible and replacement_envelope may set a GroundingDINO prompt, and that prompt must name a concrete visible source target. prospective_placement must construct geometry from visible anchors instead. For edit_region, white must cover the pixels whose requested state may change; do not expand an attribute edit to the complete entity merely because the entity contains those pixels. Include holes, gaps, or support pixels only when a separate explicit footprint contract authorizes their reconstruction. Tune prompt, threshold, and bounded growth from visual mask evidence when required changed pixels are missing. For preserve_region, white must select the concrete foreground subject that remains unchanged while its complement is edited. Never silently change one role into the other and never segment the generic word background.'}
            mask_contracts = [{key: decomposition.get(key) for key in ('target_presence', 'mask_construction', 'geometry_requirement', 'protected_regions')} for step in implementation_plan if isinstance(step, dict) for decomposition in [step.get('decomposition_step') or {}] if str(step.get('operation') or '').casefold() == 'mask_processing/generation']
            if len(mask_contracts) == 1:
                request_payload['mask_construction_contract'] = mask_contracts[0]
                request_payload['mask_construction_policy'] = 'Treat this typed contract as authoritative. prospective_placement with an absent target makes semantic detection of the requested target and whole-support segmentation ineligible; select a placement Workflow Skill and infer its normalized geometry from the current source visual and visible anchors. single_regular requires the smallest safe regular footprint. composite_or_freeform requires a supplied complete Skill that can combine a prospective outer footprint with explicit protected-region subtraction. segment_visible uses Base only when the concrete target is already visible. replacement_envelope starts from a correctly detected visible source target. For replacement_envelope, the GroundingDINO prompt may and must name multiple concrete visible source regions when their union forms the final editable footprint. If the requested replacement closes or covers an opening in the source object, include the visible underlayer or interior region inside that opening unless it is explicitly listed in protected_regions. For example, replacing an open jacket with a hoodie requires the jacket and visible torso shirt in the mask. Use the smallest bounded GrowMask expansion that closes residual narrow seams in this union while keeping protected face and hands black; for an approximately 800-pixel garment portrait, 12 to 18 pixels is a reasonable starting range. A protected subtraction represents only protected_regions; it must begin at the actual boundary where editable source pixels end and preserved content begins, and must not subtract source pixels that the stage description explicitly includes in white. Never repair one construction mode by silently using another.'
                mask_contract = mask_contracts[0]

                def prospective_skill_kind(skill: dict[str, Any]) -> str | None:
                    node_types = {str(node.get('class_type') or '') for node in (skill.get('workflow_template') or {}).values() if isinstance(node, dict)}
                    if 'MaskComposite' in node_types and 'CreateShapeMask' in node_types:
                        return 'composite_or_freeform'
                    if 'MaskRectAreaAdvanced' in node_types and (not any(('GroundingDino' in value for value in node_types))):
                        return 'single_regular'
                    return None
                if mask_contract.get('mask_construction') == 'prospective_placement':
                    geometry = mask_contract.get('geometry_requirement')
                    eligible = []
                    workflow_skills = eligible
                    allowed_skill_ids = {str(item['skill_id']) for item in workflow_skills}
                    required_mask_skill_ids = set(allowed_skill_ids)
                    request_payload['available_workflow_skills'] = workflow_skills
                    request_payload['cold_start'] = not workflow_skills
                elif mask_contract.get('mask_construction') in {'segment_visible', 'replacement_envelope'}:
                    workflow_skills = [skill for skill in workflow_skills if prospective_skill_kind(skill) is None]
                    allowed_skill_ids = {str(item['skill_id']) for item in workflow_skills}
                    request_payload['available_workflow_skills'] = workflow_skills
                    request_payload['cold_start'] = not workflow_skills
            downstream_consumers = handoff_contract_context.get('downstream_consumers', []) if isinstance(handoff_contract_context, dict) else []
            replacement_consumers = [row for row in downstream_consumers if isinstance(row, dict) and str(row.get('operation') or '').casefold().endswith('/replacement')]
            if replacement_consumers:
                request_payload['mask_output_contract']['replacement_footprint'] = {'enabled': True, 'downstream_consumers': replacement_consumers, 'policy': "This mask is consumed by a complete replacement stage. Select the final editable footprint, not only disconnected semantic pixels of the old object. White may include holes, gaps, and reconstructible support-surface pixels enclosed by the intended replacement. Preserve the support object's perimeter, protected foreground occluders, and unrelated objects. A correct regular envelope inside a protected boundary is not disqualified merely because it includes reconstructible interior support pixels. If prior detection selected the support object instead of the source target, correct the concrete segmentation prompt before geometric regularization, then reassess every available geometry Skill against the corrected candidate. Compare every geometry option documented by a matching Skill: do not reject the entire Skill merely because one option would cross a protected boundary. Prefer the smallest boundary-safe option and smallest margin. Boundary proximity is an exclusion only when every supported geometry necessarily covers protected content; uncertainty before execution is not proof that all options fail."}
            replacement_footprint = request_payload['mask_output_contract'].get('replacement_footprint')
            if not replacement_footprint:

                def regularizes_source_topology(skill: dict[str, Any]) -> bool:
                    return any((isinstance(node, dict) and str(node.get('class_type') or '') == 'ReplacementMaskGeometry' for node in (skill.get('workflow_template') or {}).values()))
                excluded_geometry_skill_ids = sorted((str(skill.get('skill_id')) for skill in workflow_skills if regularizes_source_topology(skill)))
                if excluded_geometry_skill_ids:
                    workflow_skills = [skill for skill in workflow_skills if not regularizes_source_topology(skill)]
                    allowed_skill_ids = {str(item['skill_id']) for item in workflow_skills}
                    request_payload['available_workflow_skills'] = workflow_skills
                    request_payload['cold_start'] = not workflow_skills
                request_payload['mask_output_contract']['source_topology'] = {'mode': 'preserve', 'excluded_workflow_skill_ids': excluded_geometry_skill_ids, 'policy': 'No explicit complete-replacement footprint contract exists. Preserve the topology of the requested source region. Do not convert a semantic mask into a circle, rectangle, convex envelope, bounding box, or other regular geometric footprint. Use Base or another workflow whose mask follows the requested visible pixels.'}
        if repair_evidence is not None:
            request_payload['repair_evidence'] = repair_evidence
            if isinstance(repair_evidence.get('operation_notebook'), list):
                request_payload['operation_notebook'] = copy.deepcopy(repair_evidence['operation_notebook'])
                request_payload['repair_evidence'] = {key: value for key, value in repair_evidence.items() if key != 'operation_notebook'}
            repair_visual_inputs = repair_evidence.get('repair_visual_inputs', [])
            if isinstance(repair_visual_inputs, list):
                visual_rows = [row for row in repair_visual_inputs if isinstance(row, dict) and isinstance(row.get('path'), str) and Path(row['path']).is_file()][:6]
                if visual_rows:
                    request_payload['repair_visual_order'] = [{key: row[key] for key in ('position', 'role', 'stage_id', 'path') if key in row} for row in visual_rows]
                    request_payload.update(self._image_evidence_payload([row['path'] for row in visual_rows]))
                    request_payload['conditioning_attribution_policy'] = 'Inspect the attached original input, effective mask, and failed output together. Before selecting an edit, separately assess text_conditioning, source_image_conditioning, mask_conditioning, negative_conditioning, and sampler_conditioning. If the unwanted object persists in the same category, location, or silhouette after its name is absent from the positive prompt, do not attribute the failure to text again; prioritize source-image leakage, mask shape/boundary cues, negative-conditioning capability, or sampler strength.'
            diagnostic_outputs = repair_evidence.get('diagnostic_outputs', [])
            if isinstance(diagnostic_outputs, list):
                diagnostic_paths = [str(path) for path in diagnostic_outputs if isinstance(path, (str, Path)) and Path(path).is_file()][:4]
                if diagnostic_paths:
                    request_payload['diagnostic_visual_order'] = [{'position': index + 1, 'role': 'effective_mask', 'path': path} for index, path in enumerate(diagnostic_paths)]
                    request_payload.update(self._image_evidence_payload(diagnostic_paths))
                    request_payload['mask_diagnosis_policy'] = 'The attached effective-mask images are the exact processed masks consumed by the downstream edit. Diagnose visible target coverage, false positives, boundary growth, and inversion before changing prompt, threshold, or expand parameters.'
            request_payload['repair_priority_policy'] = 'If execution or static validation reports a concrete node/input error, repair that exact path before any quality-oriented prompt or sampler tuning. For enum_value or a ComfyUI node_errors allowed-value list, choose only a declared compatible value and emit set_param for that node/input. A prompt-only edit does not repair an API, schema, type, or enum error.'
            request_payload['stale_plan_policy'] = "current_graph_edit_contract is authoritative. rejected_previous_edits are negative examples and must not be copied. A previous-plan node is editable only when it is present in editable_node_ids or is introduced by add_node in the new transaction. Never invent a set_param input absent from that node's declared_inputs."
            request_payload['iteration_diversity_policy'] = 'Use operation_notebook as the record of prior operations. Continue from current_graph, preserving prior parameter changes unless explicitly revised. If a prior prompt-only repair was followed by the same semantic failure, another prompt-only repair is forbidden. Change a different causal channel, select a contributing upstream stage, or explicitly state that the parameter-only graph cannot express the required repair.'
        metadata_selection = None
        metadata_base_workflow = None
        forbid_prompt_only = bool(repair_evidence and graph is not None and self._latest_repair_was_prompt_only(repair_evidence.get('attempt_history', []), graph))
        if forbid_prompt_only:
            request_payload['prompt_only_repair_forbidden'] = True
        require_edit = bool(repair_evidence and self._repair_evidence_failed(repair_evidence))
        single_skill_ids = {str(item['skill_id']) for item in workflow_skills}
        parameter_only_skill = parameter_only
        if allow_topology_repair:
            request_payload['topology_repair_policy'] = 'Structured diagnostics require a controlled topology repair. Keep the selected model and stage boundary fixed; return only the smallest add_node/remove_node/connect/disconnect/set_param edits needed to satisfy the reported input, mask handoff, or connection contract.'
            request_payload['external_handoff_loader_contract'] = {'binding_rule': 'Persisted upstream media can only be rebound through a discoverable external loader. A free downstream typed socket is not a file handoff endpoint.', 'IMAGE': ['LoadImage.image'], 'MASK': ['LoadImageMask.image', 'LoadVideo.file', 'VHS_LoadVideo.video'], 'VIDEO': ['LoadVideo.file', 'VHS_LoadVideo.video'], 'AUDIO': ['LoadAudio.audio', 'LoadAudio.audio_file', 'LoadAudio.file'], 'repair_rule': 'When the required loader is absent, add the compatible loader with its placeholder filename and connect its typed output to the consumer in the same transaction. For a white-selects raster MASK use LoadImageMask with channel=red and MASK_INPUT.png; the alpha channel of an opaque mask image does not encode its white/black selection. For a per-frame video MASK use MASK_INPUT.mp4 and mark the video loader _meta.comfyrouter_adapter=external_mask_video_loader, then decode frames and convert their red channel to MASK. Keep the source VIDEO loader separate. Connect the MASK to a typed mask-aware edit or a composite that preserves the original outside the edit region; an unused loader does not satisfy the task. Do not merely disconnect the existing source and leave a free socket.'}
            request_payload['output_schema'] = {'skill_ids': [], 'edits': [{'action': 'add_node | remove_node | connect | disconnect | set_param', 'reason': 'short evidence-backed reason', 'node_id': 'required for add_node/remove_node', 'node': {'class_type': 'required for add_node', 'inputs': {}}, 'source': ['source node id', 0], 'destination': ['destination node id', 'input name']}], 'rationale': 'one concise summary', 'decision_trace': {'evidence': [], 'hypotheses': [], 'expected_observations': [], 'selected_strategy': ''}, 'workflow_composition_assessment': {'decision': 'keep_current | propose_composition', 'reason': 'short reason', 'evidence': [], 'proposed_stages': []}}
        if parameter_only_skill:
            request_payload['execution_policy'] = 'An exact execution-verified single Workflow Skill is bound as current_graph. Reuse its topology and return only set_param edits for declared node inputs (input files, prompts, duration, sampler/output parameters). Do not add, remove, connect, or disconnect nodes. Use full graph edits only for multi-Skill/composition routes.'
        lean_parameter_plan = parameter_only
        generation_contract = self.skills['generation_agent']
        if allow_topology_repair:
            generation_contract = copy.deepcopy(generation_contract)
            generation_contract.setdefault('rules', []).extend(['For persisted media handoff, a free downstream socket is insufficient. Return add_node plus connect (and any required disconnect/remove_node) so the graph exposes one supported external loader.', 'Every edit uses action plus reason. add_node requires node_id and node={class_type,inputs}; connect/disconnect require source=[node_id,output_index] and destination=[node_id,input_name].', 'workflow_composition_assessment.decision must be exactly keep_current or propose_composition; for an in-place consumer handoff repair use keep_current.'])
        if lean_parameter_plan:
            request_payload.update({'planning_mode': 'parameter_repair_values_only' if repair_evidence is not None else 'parameter_values_only', 'output_schema': {'skill_ids': ['zero or more supplied workflow Skill ids'], 'edits': [{'op': 'set_param', 'node_id': 'existing node id', 'param': 'declared input name', 'value': 'inferred parameter value', 'reason': 'short parameter-specific reason'}], 'rationale': 'one concise summary'}, '_max_output_tokens': 3000})
            generation_contract = {'version': 1, 'role': 'Repair parameter values for an already selected workflow graph.' if repair_evidence is not None else 'Infer parameter values for an already selected workflow graph.', 'required_output': ['skill_ids', 'edits', 'rationale'], 'rules': ['The workflow, model, topology, and stage order are already selected. Infer only values for existing declared node inputs.', 'Return only set_param edits. Do not add, remove, connect, or disconnect nodes.', 'For repairs, use repair_evidence, preflight_failures, allowed_enums, and previous_plan as authoritative diagnostics; do not repeat their analysis in the output.', 'Use the supplied temporal_parameter_context for duration and frame parameters, and keep coupled timing inputs consistent.', 'Return the smallest sufficient edit list and a one-sentence rationale. Do not return decision_trace or workflow_composition_assessment.']}
        for key in ('available_workflow_skills', 'skill_selection_policy', 'incremental_skill_selection', 'cold_start'):
            request_payload.pop(key, None)
        if not task_model_knowledge:
            request_payload.pop('task_model_knowledge', None)
            request_payload.pop('knowledge_fallback', None)
        has_skill_experience = any((item.get('workflow_experience', {}).get('available') for item in workflow_skills))
        if not base_workflow_experience.get('entries'):
            request_payload.pop('base_workflow_experience', None)
        if not base_workflow_experience.get('entries') and (not has_skill_experience):
            request_payload.pop('experience_policy', None)
        if not request_payload.get('available_workflow_compositions'):
            request_payload.pop('available_workflow_compositions', None)
        compact_parameter_output = bool(lean_parameter_plan and isinstance(graph, dict) and (not workflow_skills or metadata_selection is not None))
        if compact_parameter_output:
            from .generation_output import named_parameter_slots, parameter_prompt_context
            from .parameter_generation import parameter_response_schema
            registry = json.loads(config.NODE_REGISTRY.read_text(encoding='utf-8')) if config.NODE_REGISTRY.is_file() else {}
            generation_slots = named_parameter_slots(graph, registry)
            generation_contract = {'role': 'Fill prompts and parameter values in the selected workflow.', 'required_output': ['parameters'], 'rules': ['Return only changed prompt and parameter values as parameters={name:value}; use {} if no change is needed. Names come from editable_parameters.', 'Use existing declared node inputs; preserve workflow, model, topology and stage boundaries.', 'Use supplied timing, geometry, workflow instructions, experience and exact repair diagnostics. Keep coupled parameters consistent; do not repeat previous invalid edits.', 'Use operation_notebook and the latest failure when iterating. Do not output nodes, edits, actions, reasons, rationale, decision_trace, composition assessments, or workflow/Skill decisions; the runner already owns that metadata.']}
            request_payload['_response_json_schema'] = parameter_response_schema(generation_slots, sparse_object=True)
            request_payload['editable_parameters'] = parameter_prompt_context(generation_slots, graph_supplied=True)
            request_payload.pop('output_schema', None)
            request_payload.pop('skill_selection_policy', None)
            request_payload.pop('incremental_skill_selection', None)
            if request_payload.get('base_graph') == request_payload.get('current_graph'):
                request_payload.pop('base_graph', None)
        generation_payload = dict(request_payload)
        raw = None
        for generation_attempt in range(self.max_corrections + 1):
            raw = self.client.complete_json('generation_agent', generation_contract, generation_payload)
            if isinstance(raw, dict) and raw.get('status') == 'blocked':
                raise AgentOutputError('generation blocked: ' + str(raw.get('reason') or 'unspecified required input'))
            parameter_response = raw
            try:
                if compact_parameter_output:
                    from .generation_output import expand_parameter_edits
                    raw = expand_parameter_edits(raw, generation_slots, metadata_selection)
                raw = self._normalize_generation_plan(raw)
                if metadata_selection and raw['skill_ids'] != metadata_selection['skill_ids']:
                    raise AgentOutputError('generation cannot change the workflow selected from metadata.json')
                self._validate_generation_plan(raw, allowed_skill_ids)
                self._validate_bound_profile_plan(implementation_plan, graph, raw)
                if forbid_prompt_only and self._is_prompt_only_edits(graph, raw.get('edits', [])):
                    raise AgentOutputError('the previous prompt-only repair did not resolve the failure; another prompt-only repair is forbidden')
                if parameter_only_skill:
                    topology_ops = {str(edit.get('action')) for edit in raw.get('edits', []) if isinstance(edit, dict)} - {'set_param'}
                    if topology_ops:
                        raise AgentOutputError('single Workflow Skill is parameter-only; topology edits are not allowed: ' + ', '.join(sorted(topology_ops)))
                selected_skill = next((item for item in workflow_skills if item['skill_id'] in raw['skill_ids']), None)
                selected_template = selected_skill['workflow_template'] if selected_skill else None
                reuse_current_skill_graph = bool(repair_evidence is not None and isinstance(graph, dict) and isinstance(selected_template, dict) and self._same_graph_topology(graph, selected_template))
                active_graph = (graph if reuse_current_skill_graph else selected_template) if selected_skill else graph
                if selected_skill and str(selected_skill.get('skill_id')) == 'protected-placement-footprint':
                    self._validate_protected_placement_geometry(active_graph, raw.get('edits', []))
                active_mask_prompt_bindings = [item for item in mask_prompt_bindings if isinstance(active_graph.get(item['node_id']), dict) and (item['param'] == 'prompt' and active_graph[item['node_id']].get('class_type') == 'GroundingDinoSAMSegment (segment anything)' or (item['param'] == 'text' and active_graph[item['node_id']].get('class_type') == 'CLIPTextEncode'))] if isinstance(active_graph, dict) else []
                if graph is not None and active_mask_prompt_bindings:
                    bound = {(str(edit.get('target', edit.get('node_id', ''))), str(edit.get('param', edit.get('key', '')))) for edit in raw.get('edits', []) if isinstance(edit, dict) and edit.get('action') == 'set_param' and isinstance(edit.get('value'), str) and edit.get('value', '').strip() and (edit.get('value') != 'TARGET_MASK_PROMPT')}
                    missing = [item for item in active_mask_prompt_bindings if (item['node_id'], item['param']) not in bound]
                    if missing:
                        raise AgentOutputError('generation plan must bind every TARGET_MASK_PROMPT: ' + ', '.join((f'{item['node_id']}.{item['param']}' for item in missing)))
                break
            except AgentProviderError:
                raise
            except AgentOutputError as exc:
                if generation_attempt >= self.max_corrections:
                    raise
                generation_payload['previous_response'] = raw
                if compact_parameter_output:
                    generation_payload['previous_response'] = parameter_response
                    generation_payload['contract_correction'] = f'Parameter validation failed: {exc}. Return corrected parameters={{name:value}} only, using editable_parameters. Keep suitable values and the current workflow. Use operation_notebook and the latest exact diagnostic.'
                    generation_payload['attempt'] = generation_attempt + 1
                    continue
                generation_payload['contract_correction'] = f'The previous generation plan failed validation: {exc}. Repair only the invalid graph-edit fields and return the complete plan again. Canonical edits are: set_param={{op,node_id,param,value}}; add_node={{op,node_id,class_type,inputs}}; remove_node={{op,node_id}}; connect/disconnect={{op,source_node_id,source_slot,target_node_id,target_input}}. Every referenced node_id must exist in current_graph unless that same ordered plan first adds it. Every param/target_input must be declared by that node schema. Do not use aliases such as node, target objects, from/to, source/destination, parameter, or path. Select only supplied workflow Skill ids.'
                if required_mask_skill_ids:
                    generation_payload['contract_correction'] += ' The typed mask contract is prospective_placement and a compatible complete placement Skill is available, so Base is ineligible. Select exactly one of: ' + ', '.join(sorted(required_mask_skill_ids)) + '.'
                if parameter_only_skill:
                    generation_payload['contract_correction'] += ' This is a single Skill route: remove every add_node/remove_node/connect/disconnect operation and keep only set_param edits on the bound Skill graph.'
                generation_payload['attempt'] = generation_attempt + 1
        assert raw is not None
        composition_requested = raw['workflow_composition_assessment']['decision'] == 'propose_composition'
        if require_edit and (not raw['edits']) and (not composition_requested) and (not raw['skill_ids']):
            correction_payload = dict(request_payload)
            correction_payload['previous_response'] = raw
            correction_payload['edit_correction'] = 'The prior run failed quality evaluation. Return at least one changed prompt or parameter value in parameters={name:value}, using the current diagnostics and operation_notebook.' if compact_parameter_output else 'The prior run failed the quality gate. An empty edits list is not allowed for a repair plan. Keep your existing evidence assessment, then return at least one smallest typed graph edit with a concrete reason and a decision_trace.'
            raw = self.client.complete_json('generation_agent', generation_contract, correction_payload)
            if compact_parameter_output:
                raw = expand_parameter_edits(raw, generation_slots, metadata_selection)
            raw = self._normalize_generation_plan(raw)
            self._validate_generation_plan(raw, allowed_skill_ids, require_edit=True)
            self._validate_bound_profile_plan(implementation_plan, graph, raw)
            if forbid_prompt_only and self._is_prompt_only_edits(graph, raw.get('edits', [])):
                raise AgentOutputError('the previous prompt-only repair did not resolve the failure; another prompt-only repair is forbidden')
        elif require_edit:
            self._validate_generation_plan(raw, allowed_skill_ids, require_edit=not bool(raw['skill_ids']))
            self._validate_bound_profile_plan(implementation_plan, graph, raw)
        selected_ids = [str(skill_id) for skill_id in raw['skill_ids']]
        selected_workflow = metadata_base_workflow
        selected_workflow_source = 'metadata_selected_base' if metadata_base_workflow else None
        selection_warning = None
        if len(selected_ids) > 1 and len(implementation_plan) == 1:
            raise AgentOutputError('single-operation execution may select exactly one incremental Workflow Skill; compare metadata.json records, then return one skill_id or [] for Base')
        if selected_ids:
            selected = next((item for item in workflow_skills if str(item.get('skill_id')) == selected_ids[0]), None)
            if not isinstance(selected, dict) or not isinstance(selected.get('workflow_template'), dict):
                raise AgentOutputError(f'selected Workflow Skill {selected_ids[0]!r} has no usable workflow_template')
            selected_template = selected['workflow_template']
            if repair_evidence is not None and isinstance(graph, dict) and self._same_graph_topology(graph, selected_template):
                selected_workflow = copy.deepcopy(graph)
                selected_workflow_source = 'current_selected_skill_graph'
            else:
                selected_workflow = copy.deepcopy(selected_template)
                selected_workflow_source = 'selected_skill.workflow_template'
        return {'skill_ids': selected_ids, 'repair_blocked_reason': str(raw.get('reason') or raw.get('rationale') or '') if raw.get('status') == 'blocked' else None, 'workflow_selection': metadata_selection, 'skill_assessments': metadata_selection['skill_assessments'] if metadata_selection else raw.get('skill_assessments', []), 'base_decision': metadata_selection['base_decision'] if metadata_selection else raw.get('base_decision', {'use_base': not selected_ids, 'reason': 'No Workflow Skills were available for assessment.'}), 'edits': raw['edits'], 'rationale': raw['rationale'], 'decision_trace': raw['decision_trace'], 'workflow_composition_assessment': raw['workflow_composition_assessment'], 'requires_composition_replan': raw['workflow_composition_assessment']['decision'] == 'propose_composition', 'selected_workflow': selected_workflow, 'selected_workflow_source': selected_workflow_source, 'selected_workflow_skill_ids': selected_ids, 'selection_warning': selection_warning, 'status': 'agent_plan_pending_graph_validation'}

    def review_candidate_prompt_semantics(self, *, instruction: str, positive_prompts: list[dict[str, str]], plan: dict[str, Any], failure_analysis: dict[str, Any]) -> dict[str, Any]:
        """Verify positive prompts still describe the requested transformation."""
        contract = {'role': 'workflow candidate semantic preflight', 'rules': ['Judge only whether each positive prompt supports the original user request.', 'Reject unrelated inherited subjects, styles, objects, or actions even when the graph is API-valid.', 'Do not reject ordinary quality words or implementation details that do not contradict the request.', 'This is a text-to-text audit. Never discuss file accessibility, tools, execution ability, or perform the edit.', 'Every conflict must copy node_id, input, and text exactly from positive_prompts; never invent identifiers.', 'Return status=pass or reject and exact node/input conflicts.'], 'required_output': {'status': 'pass|reject', 'conflicts': [{'node_id': 'string', 'input': 'string', 'text': 'string', 'reason': 'string'}], 'reason': 'string'}}
        payload = {'instruction': instruction, 'positive_prompts': positive_prompts, 'candidate_plan': plan, 'failure_analysis': failure_analysis}
        previous = None
        last_error = ''
        allowed_prompts = {(item['node_id'], item['input'], item['text']) for item in positive_prompts if all((isinstance(item.get(key), str) for key in ('node_id', 'input', 'text')))}
        for attempt in range(self.max_corrections + 1):
            request = copy.deepcopy(payload)
            if last_error:
                request['previous_response'] = previous
                request['contract_correction'] = f'The previous response was rejected: {last_error}. Return one JSON object with status exactly pass or reject, conflicts as an array of objects with node_id/input/text/reason strings, and a reason string. Do not change the semantic judgment merely to satisfy the schema.'
            raw = self.client.complete_json('candidate_semantic_preflight_agent', contract, request)
            previous = raw
            try:
                if not isinstance(raw, dict):
                    raise AgentOutputError('semantic preflight output must be an object')
                normalized = copy.deepcopy(raw)
                status = str(normalized.get('status') or '').strip().casefold()
                status_aliases = {'passed': 'pass', 'approved': 'pass', 'consistent': 'pass', 'failed': 'reject', 'rejected': 'reject', 'inconsistent': 'reject'}
                status = status_aliases.get(status, status)
                normalized['status'] = status
                if status not in {'pass', 'reject'}:
                    raise AgentOutputError('status must be pass or reject')
                conflicts = normalized.get('conflicts')
                if conflicts is None and status == 'pass':
                    conflicts = []
                    normalized['conflicts'] = conflicts
                if not isinstance(conflicts, list):
                    raise AgentOutputError('conflicts must be an array')
                reason = normalized.get('reason', normalized.get('rationale'))
                if not isinstance(reason, str) or not reason.strip():
                    raise AgentOutputError('reason must be a non-empty string')
                normalized['reason'] = reason.strip()
                for item in conflicts:
                    if not isinstance(item, dict) or not all((isinstance(item.get(key), str) and item.get(key).strip() for key in ('node_id', 'input', 'text', 'reason'))):
                        raise AgentOutputError('every conflict requires node_id/input/text/reason strings')
                    identity = (item['node_id'], item['input'], item['text'])
                    if identity not in allowed_prompts:
                        raise AgentOutputError(f'conflict must reference an exact supplied positive prompt; invented conflict={identity!r}')
                    if any((token in item['reason'].casefold() for token in ('file is not accessible', 'file inaccessible', 'no image-editing tool', 'no video-editing tool', 'cannot access the file', 'unable to edit'))):
                        raise AgentOutputError('semantic audit must compare prompt meaning only; file/tool availability is irrelevant')
                if status == 'reject' and (not conflicts):
                    raise AgentOutputError('status=reject requires at least one exact prompt conflict')
                return normalized
            except AgentProviderError:
                raise
            except AgentOutputError as exc:
                last_error = str(exc)
                if attempt >= self.max_corrections:
                    raise AgentOutputError(f'candidate semantic preflight contract failed after {attempt + 1} outputs: {last_error}') from exc
        raise AgentOutputError('candidate semantic preflight contract failure')

    def analyze_workflow_failure(self, spec: RequirementSpec, implementation_plan: list[dict[str, Any]], trace: dict[str, Any], instruction: str='') -> dict[str, Any]:
        """Return one JSON-only diagnosis without changing the workflow.

        The runner calls this after each failed execution/preflight event.  The
        returned records are accumulated and later supplied to
        ``evolve_workflow_from_trace`` as the evidence summary.
        """
        if not isinstance(trace, dict):
            raise AgentOutputError('failure analysis requires a trace object')
        leaf_context = []
        for index, step in enumerate(implementation_plan, 1):
            operation = str(step.get('operation', ''))
            modality = str(step.get('modality') or spec.modality)
            profile = str(step.get('selected_profile') or step.get('routed_profile') or '')
            leaf = {'stage_id': f'stage_{index:02d}', 'modality': modality, 'task_family': str(step.get('task_family', '')), 'operation': operation, 'profile': profile, 'base_path': str(step.get('base_path') or ''), 'leaf_key': f'{modality}/{operation}/{profile}'}
            base_path = Path(leaf['base_path'])
            workflow_path = base_path / 'workflow_api.json'
            if workflow_path.is_file():
                try:
                    base_graph = json.loads(workflow_path.read_text(encoding='utf-8'))
                except (OSError, json.JSONDecodeError):
                    base_graph = None
                if isinstance(base_graph, dict):
                    external_inputs = []
                    for node_id, node in base_graph.items():
                        if not isinstance(node, dict):
                            continue
                        class_type = str(node.get('class_type', ''))
                        if class_type not in {'LoadImage', 'LoadImageMask', 'LoadVideo', 'VHS_LoadVideo', 'LoadAudio'}:
                            continue
                        literal_inputs = {key: value for key, value in (node.get('inputs') or {}).items() if isinstance(value, str)}
                        external_inputs.append({'node_id': str(node_id), 'class_type': class_type, 'literal_inputs': literal_inputs})
                    leaf['base_workflow_path'] = str(workflow_path)
                    leaf['base_external_input_nodes'] = external_inputs
            leaf_context.append(leaf)
        contract = {'version': 1, 'role': 'Self-evolution Agent: analyze one failed workflow attempt; do not edit the graph.', 'video_timing_policy': VIDEO_TIMING_POLICY, 'required_output': ['failure_class', 'root_causes', 'evidence', 'next_action', 'affected_nodes', 'affected_leaves', 'comparison', 'confidence'], 'rules': ['Inspect only the supplied attempt and its comparison context.', 'Do not return graph edits, workflow JSON, or a repair plan.', 'failure_class must be parameter, workflow, handoff, execution, quality, or transport.', 'Use execution only when the trace proves the workflow was submitted to the executor and failed there. A dependency-binding or handoff failure before stage submission is handoff (or workflow when the graph itself lacks the required interface), not execution.', 'next_action must be parameter_retry, evolve_workflow, or summarize_only.', 'root_causes, evidence, and affected_nodes must each be a JSON array, even when grouping observed facts and hypotheses.', 'affected_leaves must be a non-empty JSON array when leaf_context is non-empty. Use only exact leaf identities supplied in leaf_context; never invent a profile or path.', 'Each affected_leaves item must contain stage_id, modality, task_family, operation, profile, base_path, node_ids, problem_type, owner, leaf_change_required, aggregation_key, and evidence.', 'owner must be leaf_base, composition_adapter, routing_or_decomposition, runner, or model_capability. leaf_change_required must be a JSON boolean.', 'Use routing_or_decomposition only when the selected business operation, stage boundary, or dependency plan is semantically wrong for the original task. A graph-edit schema error, pending dependency placeholder, validation-order issue, or handoff binding defect is composition_adapter or runner, never routing_or_decomposition.', 'A leaf can be affected without owning the defect. Set leaf_change_required=false when the evidence points to composition_adapter, routing_or_decomposition, or runner.', 'Compare base_external_input_nodes with the attempt/preflight graph. A Base placeholder that exists on disk but was rebound or lost before handoff discovery is adapter/Runner evidence, not proof that the leaf Base lacks the input contract.', 'aggregation_key must be <leaf_key>:<problem_type>, using the exact leaf_key from leaf_context. node_ids and evidence must be JSON arrays.', 'When graph_edit_application is present, treat it as authoritative Runner evidence: use failed_edit_number, failure_phase, validation, later_edits_applied, and partial_graph to distinguish a final invalid plan from a transient invalid intermediate graph.', 'Separate observed evidence from hypotheses and identify changes from prior attempts.', 'For quality failures, explicitly assess whether the selected backbone/model family and conditioning mechanism can express the requested transformation. Distinguish a missing node from an architecture capability limit (for example SDXL + AnimateDiff global img2img/IPAdapter does not necessarily provide spatial or directional relighting). When the backbone cannot express the requirement, mark owner=model_capability and recommend a new workflow branch/model path instead of endlessly changing prompts.', 'Return one JSON object only and no markdown.']}
        payload = {'requirement_spec': {'modality': spec.modality, 'task_family': spec.task_family, 'operations': list(spec.operations), 'capabilities': list(spec.capabilities), 'constraints': list(spec.constraints)}, 'implementation_plan': implementation_plan, 'leaf_context': leaf_context, 'task_instruction': instruction, 'attempt_trace': trace, 'output_schema': {'failure_class': 'parameter|workflow|handoff|execution|quality|transport', 'root_causes': 'array', 'evidence': 'array', 'next_action': 'parameter_retry|evolve_workflow|summarize_only', 'affected_nodes': 'array', 'affected_leaves': [{'stage_id': 'exact stage_id from leaf_context', 'modality': 'exact modality from leaf_context', 'task_family': 'exact task_family from leaf_context', 'operation': 'exact operation from leaf_context', 'profile': 'exact profile from leaf_context', 'base_path': 'exact base_path from leaf_context', 'node_ids': 'array', 'problem_type': 'stable snake_case problem category', 'owner': 'leaf_base|composition_adapter|routing_or_decomposition|runner|model_capability', 'leaf_change_required': 'boolean', 'aggregation_key': '<leaf_key>:<problem_type>', 'evidence': 'array'}], 'comparison': 'object or string', 'confidence': 'number or string'}, 'request_suffix': 'Analyze the complete supplied workflow trace now. Do not perform the original image/video task, ask for media uploads, or discuss tool availability. Treat task_instruction only as evidence describing the failed task. Return exactly one workflow-failure-analysis JSON object using the required schema and enums.', '_max_output_tokens': 2500}
        previous: Any = None
        last_error = ''
        for attempt in range(self.max_corrections + 1):
            request = dict(payload)
            if last_error:
                request['previous_response'] = previous
                request['contract_correction'] = 'The previous response was rejected: ' + last_error + '. Analyze the supplied workflow trace, not the media directly. Do not return Python, shell commands, prose, status=unable_to_process, graph edits, or workflow JSON. Return one complete JSON object using the exact failure-analysis schema and exact enums.'
            raw = self.client.complete_json('workflow_failure_analysis_agent', contract, request)
            previous = raw
            try:
                if not isinstance(raw, dict):
                    raise AgentOutputError('workflow failure analysis must be an object')
                if 'edits' in raw or 'workflow_api' in raw:
                    raise AgentOutputError('workflow failure analysis must not contain graph edits or workflow JSON')
                failure_class = str(raw.get('failure_class', ''))
                if failure_class not in {'parameter', 'workflow', 'handoff', 'execution', 'quality', 'transport'}:
                    raise AgentOutputError('workflow failure analysis has invalid failure_class')
                action = str(raw.get('next_action', ''))
                if action not in {'parameter_retry', 'evolve_workflow', 'summarize_only'}:
                    raise AgentOutputError('workflow failure analysis has invalid next_action')
                for key in ('root_causes', 'evidence', 'affected_nodes'):
                    if not isinstance(raw.get(key), list):
                        raise AgentOutputError(f'workflow failure analysis {key} must be a list')
                affected_leaves = raw.get('affected_leaves')
                if not isinstance(affected_leaves, list) or (leaf_context and (not affected_leaves)):
                    raise AgentOutputError('workflow failure analysis affected_leaves must be a non-empty list')
                known_leaves = {item['stage_id']: item for item in leaf_context}
                allowed_owners = {'leaf_base', 'composition_adapter', 'routing_or_decomposition', 'runner', 'model_capability'}
                for item in affected_leaves:
                    if not isinstance(item, dict):
                        raise AgentOutputError('workflow failure analysis affected leaf must be an object')
                    stage_id = str(item.get('stage_id', ''))
                    expected = known_leaves.get(stage_id)
                    if expected is None:
                        raise AgentOutputError(f'workflow failure analysis affected leaf has unknown stage_id {stage_id!r}')
                    for key in ('modality', 'task_family', 'operation', 'profile', 'base_path'):
                        if str(item.get(key, '')) != expected[key]:
                            raise AgentOutputError(f'workflow failure analysis affected leaf {stage_id} has invalid {key}')
                    for key in ('node_ids', 'evidence'):
                        if not isinstance(item.get(key), list):
                            raise AgentOutputError(f'workflow failure analysis affected leaf {stage_id} {key} must be a list')
                    problem_type = str(item.get('problem_type', '')).strip()
                    if not problem_type:
                        raise AgentOutputError(f'workflow failure analysis affected leaf {stage_id} problem_type is required')
                    if item.get('owner') not in allowed_owners:
                        raise AgentOutputError(f'workflow failure analysis affected leaf {stage_id} has invalid owner')
                    if not isinstance(item.get('leaf_change_required'), bool):
                        raise AgentOutputError(f'workflow failure analysis affected leaf {stage_id} leaf_change_required must be boolean')
                    expected_key = f'{expected['leaf_key']}:{problem_type}'
                    if item.get('aggregation_key') != expected_key:
                        raise AgentOutputError(f'workflow failure analysis affected leaf {stage_id} aggregation_key must be {expected_key}')
                if not isinstance(raw.get('comparison'), (dict, str)):
                    raise AgentOutputError('workflow failure analysis comparison must be an object or string')
                if not isinstance(raw.get('confidence'), (int, float, str)):
                    raise AgentOutputError('workflow failure analysis confidence is required')
                return raw
            except AgentProviderError:
                raise
            except AgentOutputError as exc:
                last_error = str(exc)
                if attempt >= self.max_corrections:
                    raise AgentOutputError(f'workflow failure analysis contract failed after {attempt + 1} outputs: {last_error}') from exc
        raise AgentOutputError(last_error)

    def plan_capability_retrieval(self, *, spec: RequirementSpec, implementation_plan: list[dict[str, Any]], trace: dict[str, Any], failure_analysis: dict[str, Any], current_graph: dict[str, Any], instruction: str='', retrieval_feedback: dict[str, Any] | None=None) -> dict[str, Any]:
        """Let the self-evolution Agent define the live-registry query.

        The runner deliberately does not translate failure prose into node
        keywords.  It only executes and validates this structured plan.
        """
        contract = self._workflow_evolution_contract()
        retrieval_trace = {'selected_leaf_target': copy.deepcopy(trace.get('selected_leaf_target')), 'reviewed_failure_analysis': copy.deepcopy(failure_analysis), 'base_execution_history': copy.deepcopy(trace.get('base_execution_history')), 'current_leaf_asset_recheck': copy.deepcopy(trace.get('current_leaf_asset_recheck'))}
        static_context = self._workflow_evolution_static_context(spec=spec, implementation_plan=implementation_plan, graph=current_graph, trace=retrieval_trace, instruction=instruction, workflow_skills=trace.get('_workflow_evolution_workflow_skills') or [])
        payload = {'phase': 'capability_retrieval_planning', 'request_suffix': 'Using the latest failure analysis, create the local live-registry retrieval plan. Treat capability_defects_for_retrieval as the only authoritative retrieval scope. Ignore excluded_non_capability_or_other_leaf_defects when choosing capabilities or terms. Separate independent capability gaps. Do not search for composition-adapter, routing, or handoff defects as custom nodes. Choose terms likely to occur in installed ComfyUI class names, display names, categories, or descriptions. Return exactly: {"retrieval_required":true,"retrieval_requests":[{"capability":"...","local_search_terms":["..."],"required_input_types":["IMAGE"],"required_output_types":["IMAGE"],"desired_role":"image_transform","reason":"..."}],"retrieval_reason":"..."}. The runner will query live /object_info with these exact terms; it will not invent substitute keywords. Before node retrieval, assess whether the current backbone/model family can express the requested capability. If it cannot, state architecture_limit in retrieval_reason and include a model/backbone branch request; do not assume a missing node alone can overcome the limitation.', 'failure_analysis': failure_analysis, 'retrieval_feedback': retrieval_feedback, '_prompt_static_context': static_context, '_prompt_cache_key': trace.get('_workflow_evolution_prompt_cache_key'), '_max_output_tokens': 2600}
        previous = None
        last_error = ''
        for attempt in range(self.max_corrections + 1):
            request = dict(payload)
            if last_error:
                request['previous_response'] = previous
                request['contract_correction'] = last_error
            raw = self.client.complete_json('workflow_evolution_agent', contract, request)
            previous = raw
            try:
                if not isinstance(raw, dict):
                    raise AgentOutputError('capability retrieval plan must be an object')
                required = raw.get('retrieval_required')
                requests = raw.get('retrieval_requests')
                reason = raw.get('retrieval_reason')
                if not isinstance(required, bool):
                    raise AgentOutputError('retrieval_required must be a boolean')
                if not isinstance(requests, list):
                    raise AgentOutputError('retrieval_requests must be an array')
                normalized = []
                for index, item in enumerate(requests):
                    if not isinstance(item, dict):
                        raise AgentOutputError(f'retrieval_requests[{index}] must be an object')
                    capability = item.get('capability')
                    terms = item.get('local_search_terms')
                    inputs = item.get('required_input_types')
                    outputs = item.get('required_output_types')
                    role = item.get('desired_role')
                    item_reason = item.get('reason')
                    if not isinstance(capability, str) or not capability.strip():
                        raise AgentOutputError(f'retrieval_requests[{index}].capability is required')
                    if not self._string_list(terms):
                        raise AgentOutputError(f'retrieval_requests[{index}].local_search_terms must be non-empty')
                    if not self._string_list(inputs, allow_empty=True):
                        raise AgentOutputError(f'retrieval_requests[{index}].required_input_types must be a string array')
                    if not self._string_list(outputs):
                        raise AgentOutputError(f'retrieval_requests[{index}].required_output_types must be non-empty')
                    if not isinstance(role, str) or not role.strip():
                        raise AgentOutputError(f'retrieval_requests[{index}].desired_role is required')
                    if not isinstance(item_reason, str) or not item_reason.strip():
                        raise AgentOutputError(f'retrieval_requests[{index}].reason is required')
                    normalized.append({'capability': capability.strip(), 'local_search_terms': list(dict.fromkeys((term.strip() for term in terms if term.strip()))), 'required_input_types': sorted({value.strip().upper() for value in inputs if value.strip()}), 'required_output_types': sorted({value.strip().upper() for value in outputs if value.strip()}), 'desired_role': role.strip(), 'reason': item_reason.strip()})
                if required and (not normalized or not isinstance(reason, str) or (not reason.strip())):
                    raise AgentOutputError('retrieval_required=true requires retrieval_requests and retrieval_reason')
                if not required and normalized:
                    raise AgentOutputError('retrieval_requests must be empty when retrieval_required=false')
                return {'retrieval_required': required, 'retrieval_requests': normalized, 'retrieval_reason': str(reason or '').strip()}
            except AgentProviderError:
                raise
            except AgentOutputError as exc:
                last_error = str(exc)
                if attempt >= self.max_corrections:
                    raise
        raise AgentOutputError(last_error)

    def review_workflow_failure_analysis(self, spec: RequirementSpec, implementation_plan: list[dict[str, Any]], trace: dict[str, Any], draft: dict[str, Any], instruction: str='') -> dict[str, Any]:
        """Independently audit a failure-analysis draft against the full trace.

        This is deliberately a separate reviewer call.  It must verify route,
        exact physical leaf, owner, problem category, and whether Base changes
        are actually supported by graph/interface evidence.  It does not edit
        or execute a workflow.
        """
        if not isinstance(draft, dict):
            raise AgentOutputError('failure-analysis review requires a draft object')
        contract = {'version': 1, 'role': 'Independent reviewer of a workflow failure analysis; reject unsupported attribution.', 'required_output': ['verdict', 'findings', 'verified_analysis', 'confidence'], 'rules': ['Read the complete trace, implementation plan, Base graph context, handoff records, static validation, and all attempts.', 'Do not trust the draft merely because it is schema-valid.', 'First verify the task family and operation route. If wrong, withhold leaf attribution and mark routing_correction.', 'Use the exact physical leaf and base_path from implementation_plan; never invent a sibling profile.', 'Distinguish leaf_base/model_capability from composition_adapter, routing_or_decomposition, and runner.', 'Use routing_or_decomposition only for a semantically wrong task route, operation, stage boundary, or dependency plan. Repair edit-schema errors, pending dependency placeholders, validation-order problems, and runtime handoff binding belong to composition_adapter or runner.', 'A persisted external IMAGE/MASK/VIDEO placeholder that is lost or rebound before handoff discovery is adapter/runner evidence, not a Base defect.', 'A quality failure may contain multiple independent leaf/problem groups; preserve separate problem_type values instead of merging them.', 'verified_analysis must use the same JSON shape as the draft and must contain exact affected_leaves identities when implementation_plan is non-empty.', 'Return exactly one JSON object and no markdown.']}
        payload = {'task_instruction': instruction, 'requirement_spec': {'modality': spec.modality, 'task_family': spec.task_family, 'operations': list(spec.operations), 'capabilities': list(spec.capabilities), 'constraints': list(spec.constraints)}, 'implementation_plan': implementation_plan, 'full_trace': trace, 'draft_analysis': draft, 'output_schema': {'verdict': 'correct|partially_correct|incorrect', 'findings': [{'severity': 'error|warning|confirmed', 'claim': '...', 'evidence': ['...']}], 'verified_analysis': 'corrected failure-analysis object; no graph edits', 'confidence': 'high|medium|low'}, 'request_suffix': 'Review the supplied draft against the complete workflow trace now. Do not execute the original media request, ask for uploads, or discuss tool availability. Treat task_instruction only as evidence. Return exactly one independent-review JSON object using the required schema.', '_max_output_tokens': 3500}
        expected_leaves = {f'stage_{index:02d}': {'modality': str(step.get('modality') or spec.modality), 'task_family': str(step.get('task_family') or ''), 'operation': str(step.get('operation') or ''), 'profile': str(step.get('selected_profile') or step.get('routed_profile') or ''), 'base_path': str(step.get('base_path') or '')} for index, step in enumerate(implementation_plan, 1)}
        previous: Any = None
        last_error = ''
        for attempt in range(self.max_corrections + 1):
            request = dict(payload)
            if last_error:
                request['previous_response'] = previous
                request['contract_correction'] = 'The previous review was rejected: ' + last_error + '. Return the complete review again. verified_analysis must retain the exact failure-analysis schema, required enums, arrays, confidence, and exact physical leaf identities from implementation_plan. Do not omit fields when correcting the draft.'
            raw = self.client.complete_json('workflow_failure_analysis_review_agent', contract, request)
            previous = raw
            try:
                if not isinstance(raw, dict):
                    raise AgentOutputError('workflow failure analysis review must be an object')
                if raw.get('verdict') not in {'correct', 'partially_correct', 'incorrect'}:
                    raise AgentOutputError('workflow failure analysis review verdict is invalid')
                if not isinstance(raw.get('findings'), list):
                    raise AgentOutputError('workflow failure analysis review findings must be a list')
                verified = raw.get('verified_analysis')
                if not isinstance(verified, dict):
                    raise AgentOutputError('workflow failure analysis review verified_analysis must be an object')
                if verified.get('failure_class') not in {'parameter', 'workflow', 'handoff', 'execution', 'quality', 'transport'}:
                    raise AgentOutputError('workflow failure analysis review verified_analysis has invalid failure_class')
                if verified.get('next_action') not in {'parameter_retry', 'evolve_workflow', 'summarize_only'}:
                    raise AgentOutputError('workflow failure analysis review verified_analysis has invalid next_action')
                for key in ('root_causes', 'evidence', 'affected_nodes', 'affected_leaves'):
                    if not isinstance(verified.get(key), list):
                        raise AgentOutputError(f'workflow failure analysis review verified_analysis {key} must be a list')
                if expected_leaves and (not verified['affected_leaves']):
                    raise AgentOutputError('workflow failure analysis review verified_analysis affected_leaves must be non-empty')
                for item in verified['affected_leaves']:
                    if not isinstance(item, dict):
                        raise AgentOutputError('workflow failure analysis review affected leaf must be an object')
                    stage_id = str(item.get('stage_id') or '')
                    expected = expected_leaves.get(stage_id)
                    if expected is None:
                        raise AgentOutputError(f'workflow failure analysis review affected leaf has unknown stage_id {stage_id!r}')
                    for key, value in expected.items():
                        if str(item.get(key) or '') != value:
                            raise AgentOutputError(f'workflow failure analysis review affected leaf {stage_id} has invalid {key}')
                    if not isinstance(item.get('node_ids'), list) or not isinstance(item.get('evidence'), list):
                        raise AgentOutputError(f'workflow failure analysis review affected leaf {stage_id} arrays are invalid')
                    if item.get('owner') not in {'leaf_base', 'composition_adapter', 'routing_or_decomposition', 'runner', 'model_capability'}:
                        raise AgentOutputError(f'workflow failure analysis review affected leaf {stage_id} owner is invalid')
                    if not isinstance(item.get('leaf_change_required'), bool):
                        raise AgentOutputError(f'workflow failure analysis review affected leaf {stage_id} leaf_change_required must be boolean')
                    if not str(item.get('problem_type') or '').strip() or not str(item.get('aggregation_key') or '').strip():
                        raise AgentOutputError(f'workflow failure analysis review affected leaf {stage_id} category is incomplete')
                if 'comparison' not in verified:
                    raise AgentOutputError('workflow failure analysis review verified_analysis comparison is required')
                if not isinstance(verified.get('confidence'), (int, float, str)):
                    raise AgentOutputError('workflow failure analysis review verified_analysis confidence is required')
                if not isinstance(raw.get('confidence'), (int, float, str)):
                    raise AgentOutputError('workflow failure analysis review confidence is required')
                return raw
            except AgentProviderError:
                raise
            except AgentOutputError as exc:
                last_error = str(exc)
                if attempt >= self.max_corrections:
                    raise AgentOutputError('workflow failure analysis review contract failure: ' + last_error) from exc
        raise AgentOutputError('workflow failure analysis review failed')

    @staticmethod
    def _temporal_parameter_context(graph: dict[str, Any], instruction: str) -> dict[str, Any]:
        """Give the execution Agent explicit timing equations and coupled inputs."""
        duration_match = re.search('(?:for\\s+)?(\\d+(?:\\.\\d+)?)\\s*(?:-\\s*)?(?:seconds?|secs?|s\\b|秒)', instruction, re.IGNORECASE)
        duration = float(duration_match.group(1)) if duration_match else None
        timing_keys = {'fps', 'frame_rate', 'force_rate', 'length', 'num_frames', 'frame_load_cap', 'skip_first_frames', 'select_every_nth', 'start_index', 'transition_start_index', 'transitioning_frames'}
        parameters = []
        loaders = []
        output_fps = None
        for node_id, node in graph.items():
            if not isinstance(node, dict) or not isinstance(node.get('inputs'), dict):
                continue
            class_type = str(node.get('class_type', ''))
            inputs = node['inputs']
            values = {key: inputs[key] for key in timing_keys if key in inputs}
            if values:
                parameters.append({'node_id': str(node_id), 'class_type': class_type, 'inputs': values})
            if class_type in {'VHS_VideoCombine', 'CreateVideo', 'SaveVideo'}:
                candidate = inputs.get('frame_rate', inputs.get('fps'))
                if isinstance(candidate, (int, float)) and candidate > 0:
                    output_fps = float(candidate)
            if class_type in {'VHS_LoadVideo', 'LoadVideo'}:
                loaders.append({'node_id': str(node_id), 'frame_load_cap': inputs.get('frame_load_cap'), 'force_rate': inputs.get('force_rate'), 'select_every_nth': inputs.get('select_every_nth')})
        return {'requested_duration_seconds': duration, 'output_fps': output_fps, 'derived_target_output_frames': round(duration * output_fps) if duration is not None and output_fps is not None else None, 'timing_parameters': parameters, 'video_loaders': loaders, 'required_checks': ['Trace produced frame count through every temporal node.', 'Verify every referenced index is below each corresponding input length.', 'Verify paired temporal inputs have compatible frame ranges.', 'Verify final_frames / output_fps equals requested duration.']}

    @staticmethod
    def _diagnosis_graph(graph: Any) -> dict[str, Any]:
        """Keep runtime wiring and causal parameters without resending the full graph."""
        if not isinstance(graph, dict):
            return {}
        diagnostic_keys = {'image', 'mask', 'video', 'channel', 'text', 'prompt', 'negative', 'seed', 'steps', 'cfg', 'guidance', 'denoise', 'strength', 'width', 'height', 'batch_size', 'frames', 'num_frames', 'length', 'fps', 'frame_rate', 'frame_load_cap', 'force_rate', 'select_every_nth', 'ckpt_name', 'vae_name', 'clip_name', 'model_name', 'control_net_name', 'lora_name', 'sampler_name', 'scheduler'}
        compact: dict[str, Any] = {}
        for node_id, node in graph.items():
            if not isinstance(node, dict):
                continue
            inputs = node.get('inputs')
            kept_inputs: dict[str, Any] = {}
            if isinstance(inputs, dict):
                for name, value in inputs.items():
                    is_edge = isinstance(value, list) and len(value) == 2 and isinstance(value[0], (str, int)) and isinstance(value[1], int)
                    lowered = str(name).casefold()
                    if is_edge or lowered in diagnostic_keys or any((token in lowered for token in ('prompt', 'text', 'mask', 'image', 'video', 'model', 'checkpoint', 'lora', 'control', 'sampler', 'scheduler', 'guidance', 'denoise', 'strength', 'seed', 'step', 'width', 'height', 'frame', 'fps', 'rate'))):
                        kept_inputs[str(name)] = value
            compact[str(node_id)] = {'class_type': str(node.get('class_type') or ''), 'inputs': kept_inputs}
        return compact

    def diagnose_composition_failure(self, *, instruction: str, stages: list[dict[str, Any]], final_evaluation: dict[str, Any], media_paths: list[str], visual_manifest: list[dict[str, Any]] | None=None, include_repair_decision: bool=False) -> dict[str, Any]:
        """Let the execution Agent select the earliest faulty composition stage."""
        if not stages:
            raise AgentOutputError('composition diagnosis requires stages')
        stage_ids = [str(stage.get('stage_id') or '') for stage in stages]
        if any((not stage_id for stage_id in stage_ids)) or len(set(stage_ids)) != len(stage_ids):
            raise AgentOutputError('composition diagnosis requires unique stage ids')
        repair_rules = ['In the same response, classify the failure, state the next repair action, and decide whether public external evidence is actually needed. Do not repeat the visual diagnosis in a second Agent call.', 'Use parameter_retry when the supplied graph can express the repair through existing prompts or values. Use topology_repair only when the evidence proves that nodes, connections, or input contracts are missing.', 'Request web search only when model or node behavior remains uncertain after inspecting the graph and all attached media. Return at most three precise, public, non-sensitive queries.'] if include_repair_decision else []
        contract = {'role': 'Execution Agent: locate the earliest faulty stage in a composed workflow.', 'video_timing_policy': VIDEO_TIMING_POLICY, 'rules': ['Inspect the supplied stage result images or sampled video frames in declared visual_input_order.', 'Treat each ordered_stages[].executed_graph and handoff as the authoritative graph actually submitted for this attempt. They override Base graphs, preflight graphs, and every graph mentioned in prior conversation history. Cite the exact runtime loader value before claiming a handoff mismatch.', 'Walk stages strictly from first to last and assess each stage against only its atomic instruction.', 'Select exactly one earliest_failing_stage_id as the primary repair. Defer downstream visual judgments that depend on a faulty upstream input, but still inspect every supplied executed graph, decoded media metadata, and execution error for independently provable contract violations.', 'A downstream result inherited from an already-faulty upstream input is not evidence that the downstream stage itself should be edited.', 'Return independent_stage_repairs for additional later stages with errors provable even if the upstream input were corrected. Examples include a prompt that contradicts the atomic instruction, wrong media interfaces, or independently measured frame/duration violations. Each entry must cite graph node ids, decoded metadata, or a concrete execution error, the violated contract, and why fixing the upstream input would not fix this error. Unchanged downstream content alone is insufficient. Use an empty list when there is no such evidence. Never reopen passed stages or speculate about unseen outputs.', 'If all earlier stages satisfy their atomic instructions, select the first later stage whose own transformation fails or damages the valid upstream result.', 'Set repair_stage_id to the same earliest failing stage. A stage whose own output passed its atomic contract is frozen and must not be selected merely because a later stage consumed it. Only select an earlier stage when its own output is directly shown to be wrong; downstream artifacts or a later-stage quality failure are not evidence against a passed upstream stage.', 'For localized edits, inspect original_input, effective_mask, and failed_output together. Assess text, source-image leakage, mask silhouette/boundary, negative conditioning, and sampler conditioning independently; a correct target mask can still contribute a strong shape prior.', 'Use the effective mask to separate retained source from generated error. Uncovered black pixels may explain unchanged fragments, but cannot cause new unwanted content inside a white editable region. If most of the target is white and that region is visibly regenerated with the wrong semantics, treat the handoff as working and diagnose downstream conditioning before editing the mask stage.', 'If an unwanted object remains in the same location or silhouette while the positive prompt no longer names it, text_conditioning cannot be the primary attribution.', 'Use execution errors when media is unavailable. Do not invent unseen results or propose edits here.', *repair_rules, 'Return one JSON object only.'], 'required_output': ['earliest_failing_stage_id', 'repair_stage_id', 'contributing_stage_ids', 'stage_assessments', 'conditioning_attribution', 'rationale', 'evidence', 'independent_stage_repairs'] + (['failure_class', 'root_causes', 'next_action', 'repair_focus', 'needs_search', 'search_queries'] if include_repair_decision else [])}
        visual_order = visual_manifest or [{'position': index + 1, 'role': 'stage_output', 'path': path} for index, path in enumerate(media_paths)]
        compact_stages = []
        for stage in stages:
            compact_stage = dict(stage)
            compact_stage['executed_graph'] = self._diagnosis_graph(stage.get('executed_graph', {}))
            compact_stages.append(compact_stage)
        payload = {'full_instruction': instruction, 'ordered_stages': compact_stages, 'final_comfybench_evaluation': final_evaluation, 'visual_input_order': visual_order, 'output_schema': {'earliest_failing_stage_id': 'one supplied stage_id', 'repair_stage_id': 'earliest failing stage or a causally contributing earlier stage', 'contributing_stage_ids': ['zero or more supplied stage ids at or before the failure'], 'stage_assessments': [{'stage_id': 'supplied stage_id', 'status': 'pass|fail|unverifiable|deferred_due_to_upstream_failure', 'observations': ['visible or execution evidence'], 'reason': 'atomic-stage judgment'}], 'conditioning_attribution': {key: {'likelihood': 'likely|possible|unlikely|unknown|not_applicable', 'evidence': ['specific visual, graph, or attempt-history evidence'], 'counterfactual': 'what observation would support or refute this cause'} for key in ('text_conditioning', 'source_image_conditioning', 'mask_conditioning', 'negative_conditioning', 'sampler_conditioning')}, 'rationale': 'why this is the earliest stage that should be edited', 'evidence': ['concise evidence tied to supplied output/frame positions'], 'independent_stage_repairs': [{'stage_id': 'a supplied stage strictly after the earliest failure, never a passed stage', 'evidence_basis': 'graph_contract|decoded_media_contract|execution_error', 'node_ids': ['executed graph node ids; required for graph_contract'], 'contract': 'specific violated atomic instruction or mandatory media/interface constraint', 'upstream_independence': 'why this error persists with a corrected upstream input', 'evidence': ['concrete findings from the cited graph, metadata, or execution error'], 'failure_class': 'quality|execution|handoff|parameter|capability|composition', 'root_causes': ['independently supported cause'], 'next_action': 'parameter_retry|topology_repair', 'repair_focus': ['specific changes for this stage only']}]}, **self._image_evidence_payload(media_paths), '_prompt_history_mode': 'latest_only', '_max_output_tokens': 4000 if include_repair_decision else 2000, '_timeout_seconds': 120}
        if include_repair_decision:
            payload['output_schema'].update({'failure_class': 'quality|execution|handoff|parameter|capability|routing|composition', 'root_causes': ['concise evidence-backed causal finding'], 'next_action': 'parameter_retry|topology_repair|composition_repair|routing_correction|stop', 'repair_focus': ['ordered concrete changes for the selected repair stage'], 'needs_search': 'boolean', 'search_queries': ['zero to three queries; empty when needs_search is false']})
        from .compact_io import compact_diagnosis_request, expand_conditioning_attribution
        contract, payload = compact_diagnosis_request(contract, payload)
        previous = None
        last_error = ''
        for attempt in range(self.max_corrections + 1):
            request = dict(payload)
            if last_error:
                request['previous_response'] = previous
                request['contract_correction'] = last_error
            raw = self.client.complete_json('execution_diagnosis_agent', contract, request)
            raw = expand_conditioning_attribution(raw)
            try:
                raw = self._normalize_ordered_composition_diagnosis(raw, stage_ids)
                self._validate_composition_diagnosis(raw, stage_ids, require_repair_decision=include_repair_decision)
                self._validate_independent_repair_evidence(raw, stages)
                return raw
            except AgentProviderError:
                raise
            except AgentOutputError as exc:
                previous = raw
                last_error = str(exc)
                if attempt >= self.max_corrections:
                    raise
        raise AgentOutputError('composition diagnosis failed')

    @staticmethod
    def _normalize_ordered_composition_diagnosis(raw: Any, stage_ids: list[str]) -> Any:
        """Keep a proven earliest failure when later judgments exceed review scope.

        The audited model response remains intact. This narrows unsupported
        downstream judgments to deferred, preserving independent failures.
        All other schema and evidence checks still run on the normalized copy.
        """
        if not isinstance(raw, dict):
            return raw
        selected = raw.get('earliest_failing_stage_id')
        rows = raw.get('stage_assessments')
        if selected not in stage_ids or not isinstance(rows, list):
            return raw
        by_id = {r.get('stage_id'): r for r in rows if isinstance(r, dict)}
        if len(rows) != len(stage_ids) or len(by_id) != len(stage_ids) or set(by_id) != set(stage_ids):
            return raw
        index = stage_ids.index(selected)
        if by_id[selected].get('status') != 'fail' or any((by_id[s].get('status') != 'pass' for s in stage_ids[:index])):
            return raw
        result = copy.deepcopy(raw)
        independent_ids = {row.get('stage_id') for row in raw.get('independent_stage_repairs', []) if isinstance(row, dict) and isinstance(row.get('stage_id'), str)} if isinstance(raw.get('independent_stage_repairs', []), list) else set()
        changes = []
        for row in result['stage_assessments']:
            if row['stage_id'] in independent_ids:
                continue
            if row['stage_id'] in stage_ids[index + 1:] and row.get('status') in {'pass', 'fail'}:
                changes.append({'stage_id': row['stage_id'], 'original_status': row['status'], 'original_reason': row.get('reason')})
                row['status'] = 'deferred_due_to_upstream_failure'
        if changes:
            result['diagnosis_protocol_normalizations'] = changes
        return result

    @staticmethod
    def _validate_composition_diagnosis(raw: Any, stage_ids: list[str], require_repair_decision: bool=False) -> None:
        if not isinstance(raw, dict):
            raise AgentOutputError('composition diagnosis must be an object')
        selected = str(raw.get('earliest_failing_stage_id') or '')
        if selected not in stage_ids:
            raise AgentOutputError('earliest_failing_stage_id must name one supplied stage')
        assessments = raw.get('stage_assessments')
        if not isinstance(assessments, list) or len(assessments) != len(stage_ids):
            raise AgentOutputError('stage_assessments must cover every supplied stage once')
        by_id = {}
        allowed = {'pass', 'fail', 'unverifiable', 'deferred_due_to_upstream_failure'}
        aliases = {'passed': 'pass', 'success': 'pass', 'successful': 'pass', 'correct': 'pass', 'failed': 'fail', 'failure': 'fail', 'quality_failed': 'fail', 'incorrect': 'fail', 'wrong': 'fail', 'unknown': 'unverifiable', 'cannot_verify': 'unverifiable', 'not_verifiable': 'unverifiable', 'blocked_by_upstream': 'deferred_due_to_upstream_failure', 'upstream_failed': 'deferred_due_to_upstream_failure', 'deferred': 'deferred_due_to_upstream_failure'}
        for row in assessments:
            if not isinstance(row, dict):
                raise AgentOutputError('every stage assessment must be an object')
            stage_id = str(row.get('stage_id') or '')
            if stage_id not in stage_ids or stage_id in by_id:
                raise AgentOutputError('stage assessment ids must be unique supplied ids')
            status = str(row.get('status') or '').strip().casefold()
            row['status'] = aliases.get(status, status)
            if row.get('status') not in allowed:
                raise AgentOutputError('invalid stage assessment status')
            observations = row.get('observations')
            if not isinstance(observations, list) or not all((isinstance(x, str) for x in observations)):
                raise AgentOutputError('stage assessment observations must be strings')
            if not isinstance(row.get('reason'), str) or not row['reason'].strip():
                raise AgentOutputError('stage assessment reason is required')
            by_id[stage_id] = row
        selected_index = stage_ids.index(selected)
        if by_id[selected]['status'] != 'fail':
            raise AgentOutputError('selected earliest stage must have fail status')
        for earlier in stage_ids[:selected_index]:
            if by_id[earlier]['status'] != 'pass':
                raise AgentOutputError('every stage before the selected stage must pass')
        independent = raw.get('independent_stage_repairs', [])
        if not isinstance(independent, list):
            raise AgentOutputError('independent_stage_repairs must be a list')
        independent_ids = set()
        for repair in independent:
            if not isinstance(repair, dict):
                raise AgentOutputError('independent stage repair must be an object')
            stage_id = repair.get('stage_id')
            if not isinstance(stage_id, str) or stage_id not in stage_ids[selected_index + 1:] or stage_id in independent_ids:
                raise AgentOutputError('independent repairs must name unique stages after the earliest failure')
            if by_id[stage_id]['status'] == 'pass':
                raise AgentOutputError('independent repair cannot reopen a passed stage')
            independent_ids.add(stage_id)
            if repair.get('evidence_basis') not in {'graph_contract', 'decoded_media_contract', 'execution_error'}:
                raise AgentOutputError('independent repair requires a graph, decoded media, or execution evidence basis')
            for key in ('contract', 'upstream_independence'):
                if not isinstance(repair.get(key), str) or not repair[key].strip():
                    raise AgentOutputError(f'independent repair requires {key}')
            for key in ('evidence', 'root_causes', 'repair_focus'):
                if not AgentOrchestrator._string_list(repair.get(key), allow_empty=False):
                    raise AgentOutputError(f'independent repair requires {key}')
            if repair.get('failure_class') not in {'quality', 'execution', 'handoff', 'parameter', 'capability', 'composition'}:
                raise AgentOutputError('invalid independent repair failure_class')
            if repair.get('next_action') not in {'parameter_retry', 'topology_repair'}:
                raise AgentOutputError('independent repair must specify parameter_retry or topology_repair')
        for later in stage_ids[selected_index + 1:]:
            if by_id[later]['status'] == 'fail' and later in independent_ids:
                continue
            if by_id[later]['status'] not in {'deferred_due_to_upstream_failure', 'unverifiable'}:
                raise AgentOutputError('stages after the earliest failure must be deferred or unverifiable')
        repair_stage_id = str(raw.get('repair_stage_id') or '')
        if repair_stage_id not in stage_ids[:selected_index + 1]:
            raise AgentOutputError('repair_stage_id must name the failure or a causally contributing earlier stage')
        contributors = raw.get('contributing_stage_ids')
        if not isinstance(contributors, list) or any((not isinstance(stage_id, str) or stage_id not in stage_ids[:selected_index + 1] for stage_id in contributors)):
            raise AgentOutputError('contributing_stage_ids must contain supplied stages at or before the failure')
        attribution = raw.get('conditioning_attribution')
        attribution_keys = {'text_conditioning', 'source_image_conditioning', 'mask_conditioning', 'negative_conditioning', 'sampler_conditioning'}
        likelihoods = {'likely', 'possible', 'unlikely', 'unknown', 'not_applicable'}
        if not isinstance(attribution, dict) or set(attribution) != attribution_keys:
            raise AgentOutputError('conditioning_attribution must cover all five conditioning channels')
        for key, row in attribution.items():
            if not isinstance(row, dict) or row.get('likelihood') not in likelihoods or (not AgentOrchestrator._string_list(row.get('evidence'), allow_empty=True)) or (not isinstance(row.get('counterfactual'), str)) or (not row['counterfactual'].strip()):
                raise AgentOutputError(f'invalid conditioning attribution for {key}')
        if not isinstance(raw.get('rationale'), str) or not raw['rationale'].strip():
            raise AgentOutputError('composition diagnosis rationale is required')
        if not AgentOrchestrator._string_list(raw.get('evidence'), allow_empty=False):
            raise AgentOutputError('composition diagnosis evidence is required')
        if require_repair_decision:
            if raw.get('failure_class') not in {'quality', 'execution', 'handoff', 'parameter', 'capability', 'routing', 'composition'}:
                raise AgentOutputError('invalid combined diagnosis failure_class')
            if not AgentOrchestrator._string_list(raw.get('root_causes'), allow_empty=False):
                raise AgentOutputError('combined diagnosis root_causes are required')
            if raw.get('next_action') not in {'parameter_retry', 'topology_repair', 'composition_repair', 'routing_correction', 'stop'}:
                raise AgentOutputError('invalid combined diagnosis next_action')
            if not AgentOrchestrator._string_list(raw.get('repair_focus'), allow_empty=False):
                raise AgentOutputError('combined diagnosis repair_focus is required')
            if not isinstance(raw.get('needs_search'), bool):
                raise AgentOutputError('combined diagnosis needs_search must be boolean')
            queries = raw.get('search_queries')
            if not AgentOrchestrator._string_list(queries, allow_empty=True) or len(queries) > 3:
                raise AgentOutputError('combined diagnosis search_queries must contain at most three strings')
            if raw['needs_search'] != bool(queries):
                raise AgentOutputError('combined diagnosis search_queries must be non-empty exactly when needs_search is true')

    @staticmethod
    def _validate_independent_repair_evidence(raw: dict, stages: list[dict]) -> None:
        """Anchor additional repairs to actual per-stage evidence, never just a final judgment."""
        by_id = {stage['stage_id']: stage for stage in stages}
        for repair in raw.get('independent_stage_repairs', []):
            stage = by_id[repair['stage_id']]
            basis = repair['evidence_basis']
            if basis == 'graph_contract':
                node_ids = repair.get('node_ids')
                graph = stage.get('executed_graph') or {}
                if not AgentOrchestrator._string_list(node_ids, allow_empty=False) or any((node_id not in graph for node_id in node_ids)):
                    raise AgentOutputError('independent graph repair must cite existing executed graph node ids')
            elif basis == 'decoded_media_contract':
                if not stage.get('video_metadata'):
                    raise AgentOutputError('independent media repair requires decoded stage metadata')
            elif basis == 'execution_error':
                execution = stage.get('execution') or {}
                if execution.get('status') in {None, 'success', 'reused'} or not (execution.get('error') or execution.get('node_errors')):
                    raise AgentOutputError('independent execution repair requires a concrete stage execution error')

    @staticmethod
    def _is_prompt_only_edits(graph: dict[str, Any], edits: list[dict[str, Any]]) -> bool:
        if not edits:
            return False
        for edit in edits:
            if not isinstance(edit, dict) or edit.get('action') != 'set_param':
                return False
            node_id = str(edit.get('node_id', edit.get('target', '')))
            class_type = str(graph.get(node_id, {}).get('class_type', ''))
            parameter = str(edit.get('param', edit.get('input', edit.get('key', ''))))
            if 'textencode' not in class_type.casefold() or parameter not in {'text', 'prompt'}:
                return False
        return True

    @classmethod
    def _latest_repair_was_prompt_only(cls, attempt_history: Any, graph: dict[str, Any]) -> bool:
        if not isinstance(attempt_history, list):
            return False
        for attempt in reversed(attempt_history):
            if not isinstance(attempt, dict):
                continue
            containers = [attempt] + [stage for stage in attempt.get('stages', []) if isinstance(stage, dict)]
            for container in reversed(containers):
                records = container.get('repair_generation_plans', [])
                if not isinstance(records, list):
                    continue
                for record in reversed(records):
                    plan = record.get('plan', {}) if isinstance(record, dict) else {}
                    edits = plan.get('edits', []) if isinstance(plan, dict) else []
                    if edits:
                        return cls._is_prompt_only_edits(graph, edits)
        return False

    @staticmethod
    def _repair_evidence_failed(evidence: dict[str, Any]) -> bool:
        if evidence.get('repair_allowed') is False:
            return False
        feedback = evidence.get('feedback', {})
        assessment = evidence.get('output_assessment', feedback.get('output_assessment', {}))
        return isinstance(assessment, dict) and assessment.get('status') in {'fail', 'partial'} or feedback.get('completion_score', 100) < 90

    @staticmethod
    def _normalize_generation_plan(raw: Any) -> Any:
        if not isinstance(raw, dict):
            return raw
        result = dict(raw)
        score = result.get('completion_score')
        if isinstance(score, float) and 0 <= score <= 1:
            result['completion_score'] = round(score * 100)
        if isinstance(result.get('evidence_summary'), str):
            result['evidence_summary'] = [result['evidence_summary']]
        result.setdefault('skill_ids', [])
        assessments = result.get('skill_assessments')
        if isinstance(assessments, dict):
            result['skill_assessments'] = list(assessments.values())
        rationale = result.get('rationale')
        if isinstance(rationale, dict):
            result['rationale'] = json.dumps(rationale, ensure_ascii=False, sort_keys=True)
        trace = result.get('decision_trace')
        if isinstance(trace, str):
            result['decision_trace'] = {'summary': trace}
        if not isinstance(result.get('decision_trace'), dict):
            result['decision_trace'] = {}
        trace = result['decision_trace']
        if 'hypotheses' not in trace and isinstance(trace.get('competing_hypotheses'), list):
            trace['hypotheses'] = [item.get('hypothesis', json.dumps(item, ensure_ascii=False, sort_keys=True)) if isinstance(item, dict) else item for item in trace['competing_hypotheses']]
        evidence = trace.get('evidence', [])
        if isinstance(evidence, dict):
            trace['evidence'] = [json.dumps({key: value}, ensure_ascii=False, sort_keys=True) for key, value in evidence.items()]
        elif isinstance(evidence, list):
            trace['evidence'] = [item if isinstance(item, str) else json.dumps(item, ensure_ascii=False, sort_keys=True) for item in evidence]
        elif isinstance(evidence, str):
            trace['evidence'] = [evidence]
        else:
            trace['evidence'] = []
        for key in ('hypotheses', 'expected_observations'):
            value = trace.get(key)
            if isinstance(value, str):
                trace[key] = [value]
            elif isinstance(value, list):
                trace[key] = [item if isinstance(item, str) else json.dumps(item, ensure_ascii=False, sort_keys=True) for item in value]
            else:
                trace[key] = []
        if not isinstance(trace.get('selected_strategy'), str):
            trace['selected_strategy'] = json.dumps(trace['selected_strategy'], ensure_ascii=False, sort_keys=True) if trace.get('selected_strategy') is not None else ''
        result['decision_trace'].setdefault('evidence', [])
        result['decision_trace'].setdefault('hypotheses', [])
        result['decision_trace'].setdefault('selected_strategy', '')
        result['decision_trace'].setdefault('expected_observations', [])
        composition = result.get('workflow_composition_assessment')
        if not isinstance(composition, dict):
            composition = {}
        composition_evidence = composition.get('evidence', [])
        if isinstance(composition_evidence, str):
            composition_evidence = [composition_evidence]
        result['workflow_composition_assessment'] = {'decision': composition.get('decision', 'keep_current'), 'reason': composition.get('reason', 'Current workflow remains the candidate.'), 'evidence': composition_evidence, 'proposed_stages': composition.get('proposed_stages', composition.get('ordered_stages', []))}
        edits = result.get('edits')
        if isinstance(edits, list):
            normalized = []
            for edit in edits:
                if not isinstance(edit, dict):
                    normalized.append(edit)
                    continue
                item = dict(edit)
                if 'action' not in item and 'operation' in item:
                    item['action'] = item['operation']
                if 'action' not in item and 'op' in item:
                    item['action'] = item['op']
                action = item.get('action')
                if action == 'add_node' and 'node' not in item and isinstance(item.get('class_type'), str):
                    item['node'] = {'class_type': item['class_type'], 'inputs': copy.deepcopy(item.get('inputs', {}))}
                if action in {'connect', 'disconnect'}:
                    if 'source' not in item and item.get('source_node_id') is not None:
                        item['source'] = [str(item['source_node_id']), int(item.get('source_output_index', item.get('source_slot', 0)))]
                    if 'destination' not in item and item.get('target_node_id') is not None:
                        target_input = item.get('target_input', item.get('target_input_name'))
                        if target_input is not None:
                            item['destination'] = [str(item['target_node_id']), str(target_input)]
                if 'target' not in item and 'node_id' in item:
                    item['target'] = str(item['node_id'])
                if 'key' not in item and 'param' in item:
                    item['key'] = item['param']
                if 'key' not in item and 'input_name' in item:
                    item['key'] = item['input_name']
                if 'key' not in item and isinstance(item.get('path'), str):
                    path = item['path']
                    item['key'] = path.removeprefix('inputs.')
                item.setdefault('reason', 'No per-edit reason supplied by the generation Agent.')
                normalized.append(item)
            result['edits'] = normalized
        return result

    @staticmethod
    def _same_graph_topology(left: dict[str, Any], right: dict[str, Any]) -> bool:
        """Compare node classes and connections while ignoring parameter values."""

        def signature(graph: dict[str, Any]) -> tuple[Any, ...]:
            node_ids = {str(node_id) for node_id in graph}
            rows = []
            for node_id, node in graph.items():
                if not isinstance(node, dict):
                    return ()
                connections = []
                for input_name, value in (node.get('inputs') or {}).items():
                    if isinstance(value, list) and len(value) == 2 and (str(value[0]) in node_ids) and isinstance(value[1], int):
                        connections.append((str(input_name), str(value[0]), value[1]))
                rows.append((str(node_id), str(node.get('class_type') or ''), tuple(sorted(connections))))
            return tuple(sorted(rows))
        return bool(left) and signature(left) == signature(right)

    @staticmethod
    def _validate_protected_placement_geometry(graph: dict[str, Any], edits: list[dict[str, Any]]) -> None:
        """Reject subtraction geometry that erases both required side bands."""
        effective_inputs = {str(node_id): copy.deepcopy(node.get('inputs') or {}) for node_id, node in graph.items() if isinstance(node, dict)}
        for edit in edits:
            if not isinstance(edit, dict) or edit.get('action') != 'set_param':
                continue
            node_id = str(edit.get('target', edit.get('node_id', '')))
            param = str(edit.get('param', edit.get('key', '')))
            if node_id in effective_inputs and param:
                effective_inputs[node_id][param] = edit.get('value')
        for node in graph.values():
            if not isinstance(node, dict) or node.get('class_type') != 'MaskComposite':
                continue
            inputs = node.get('inputs') or {}
            if inputs.get('operation') != 'subtract':
                continue
            outer_ref = inputs.get('destination')
            protected_ref = inputs.get('source')
            if not (isinstance(outer_ref, list) and outer_ref and isinstance(protected_ref, list) and protected_ref):
                continue
            outer = effective_inputs.get(str(outer_ref[0]), {})
            protected = effective_inputs.get(str(protected_ref[0]), {})
            outer_width = outer.get('shape_width')
            protected_width = protected.get('shape_width')
            if isinstance(outer_width, (int, float)) and isinstance(protected_width, (int, float)) and (protected_width >= outer_width):
                raise AgentOutputError(f'protected-placement-footprint requires the protected inner shape to be narrower than the outer footprint so required side regions survive subtraction; got protected width {protected_width} and outer width {outer_width}')

    @staticmethod
    def _validate_generation_plan(raw: Any, allowed_skill_ids: set[str], require_edit: bool=False) -> None:
        if not isinstance(raw, dict):
            raise AgentOutputError('generation agent output must be an object')
        if not AgentOrchestrator._string_list(raw.get('skill_ids'), allow_empty=True):
            raise AgentOutputError('generation agent skill_ids must be a string list')
        unknown_skills = set(raw['skill_ids']) - allowed_skill_ids
        if unknown_skills:
            raise AgentOutputError('generation agent selected unavailable workflow Skills: ' + ', '.join(sorted(unknown_skills)))
        if allowed_skill_ids:
            assessments = raw.get('skill_assessments')
            if not isinstance(assessments, list):
                raise AgentOutputError('generation agent skill_assessments must be a list')
            assessed_ids: list[str] = []
            decisions: dict[str, str] = {}
            for assessment in assessments:
                if not isinstance(assessment, dict):
                    raise AgentOutputError('every skill assessment must be an object')
                skill_id = assessment.get('skill_id')
                decision = assessment.get('decision')
                if not isinstance(skill_id, str) or skill_id not in allowed_skill_ids:
                    raise AgentOutputError('skill assessment references an unavailable workflow Skill')
                if decision not in {'match', 'no_match'}:
                    raise AgentOutputError('skill assessment decision must be match or no_match')
                for key in ('trigger_matches', 'missing_requirements', 'exclusion_matches'):
                    if not AgentOrchestrator._string_list(assessment.get(key), allow_empty=True):
                        raise AgentOutputError(f'skill assessment {key} must be a string list')
                if not isinstance(assessment.get('reason'), str) or not assessment['reason'].strip():
                    raise AgentOutputError('every skill assessment requires a reason')
                assessed_ids.append(skill_id)
                decisions[skill_id] = decision
            if len(assessed_ids) != len(set(assessed_ids)):
                raise AgentOutputError('every supplied workflow Skill must be assessed exactly once')
            if set(assessed_ids) != allowed_skill_ids:
                missing = allowed_skill_ids - set(assessed_ids)
                extra = set(assessed_ids) - allowed_skill_ids
                details = []
                if missing:
                    details.append('missing=' + ','.join(sorted(missing)))
                if extra:
                    details.append('extra=' + ','.join(sorted(extra)))
                raise AgentOutputError('skill_assessments must cover every supplied workflow Skill exactly once: ' + '; '.join(details))
            selected = raw['skill_ids']
            if selected and decisions.get(selected[0]) != 'match':
                raise AgentOutputError('selected Workflow Skill must be assessed as match')
            base_decision = raw.get('base_decision')
            if not isinstance(base_decision, dict) or not isinstance(base_decision.get('use_base'), bool) or (not isinstance(base_decision.get('reason'), str)) or (not base_decision['reason'].strip()):
                raise AgentOutputError('generation agent base_decision is invalid')
            if selected and base_decision['use_base']:
                raise AgentOutputError('base_decision.use_base must be false when a Skill is selected')
            if not selected and (not base_decision['use_base']):
                raise AgentOutputError('base_decision.use_base must be true when no Skill is selected')
        if not isinstance(raw.get('rationale'), str):
            raise AgentOutputError('generation agent rationale must be a string')
        edits = raw.get('edits')
        if not isinstance(edits, list):
            raise AgentOutputError('generation agent edits must be a list')
        composition = raw.get('workflow_composition_assessment', {})
        composition_requested = isinstance(composition, dict) and composition.get('decision') == 'propose_composition'
        if require_edit and (not edits) and (not composition_requested):
            raise AgentOutputError('generation agent must propose at least one edit after a failed quality gate')
        trace = raw.get('decision_trace')
        if not isinstance(trace, dict) or not AgentOrchestrator._string_list(trace.get('evidence'), allow_empty=True) or (not AgentOrchestrator._string_list(trace.get('hypotheses'), allow_empty=True)) or (not isinstance(trace.get('selected_strategy'), str)) or (not AgentOrchestrator._string_list(trace.get('expected_observations'), allow_empty=True)):
            raise AgentOutputError('generation agent decision_trace is invalid')
        valid_actions = {'add_node', 'remove_node', 'connect', 'disconnect', 'set_param'}
        for edit in edits:
            if not isinstance(edit, dict) or edit.get('action') not in valid_actions:
                raise AgentOutputError('generation agent returned an invalid graph edit')
            if not isinstance(edit.get('reason'), str):
                raise AgentOutputError('every graph edit requires a reason')
        if not isinstance(composition, dict) or composition.get('decision') not in {'keep_current', 'propose_composition'} or (not isinstance(composition.get('reason'), str)) or (not AgentOrchestrator._string_list(composition.get('evidence'), allow_empty=True)) or (not isinstance(composition.get('proposed_stages'), list)):
            raise AgentOutputError('generation agent workflow_composition_assessment is invalid')
        if composition_requested:
            if not composition['proposed_stages'] or not all((isinstance(stage, dict) and isinstance(stage.get('operation'), str) and isinstance(stage.get('purpose'), str) for stage in composition['proposed_stages'])):
                raise AgentOutputError('composition proposal requires typed stages with operation and purpose')

    @staticmethod
    def _validate_bound_profile_plan(implementation_plan: list[dict[str, Any]], graph: dict[str, Any] | None, raw: dict[str, Any]) -> None:
        """保证图编辑仍绑定到具体模型研究包，而不是旧模型族策略。"""
        for step in implementation_plan:
            profile = str(step.get('selected_profile', ''))
            base_path = Path(str(step.get('base_path') or ''))
            exact_physical_base = graph is not None and base_path.is_dir() and (base_path.name == 'base') and (base_path.parent.name.casefold() == profile.casefold()) and any(((base_path / name).is_file() for name in ('workflow_api.json', 'base_template.json')))
            if not profile or (profile.casefold() in BROAD_MODEL_PROFILES and (not exact_physical_base)):
                raise AgentOutputError('图编辑计划包含宽泛或空模型 profile；必须绑定具体研究包')
            if not str(step.get('base_path') or ''):
                raise AgentOutputError(f'{profile} 缺少作用域 Base 路径，不能生成图编辑计划')

    def evaluate_run_feedback(self, run_id: str, spec: RequirementSpec, implementation_plan: list[dict[str, Any]], evidence: dict[str, Any], instruction: str='', media_paths: list[str] | None=None) -> dict[str, Any]:
        """Use one multimodal feedback Agent for output and run assessment."""
        payload = {'run_id': run_id, 'requirement_spec': {'modality': spec.modality, 'task_family': spec.task_family, 'operations': list(spec.operations), 'capabilities': list(spec.capabilities), 'priorities': list(spec.priorities), 'constraints': list(spec.constraints)}, 'implementation_plan': implementation_plan, 'evidence': evidence, 'instruction': instruction, 'dimension_guidance': 'Dimensions are extensible; report only evidence-backed ones.', 'duration_acceptance': 'For requested short-video duration, accept encoded duration within max(0.25 seconds, 10 percent of the requested duration).'}
        if media_paths:
            payload.update(self._image_evidence_payload(media_paths))
        score_correction_requested = False
        last_error = ''
        previous: dict[str, Any] | None = None
        for attempt in range(2):
            raw: Any = None
            request_payload = dict(payload)
            if last_error:
                request_payload['contract_correction'] = 'Confirm only completion_score as an integer from 0 through 100, based on your existing visual assessment. Preserve and do not reconsider the rest of your feedback. Return a JSON object containing completion_score; unchanged fields may be omitted.'
                request_payload['previous_response'] = previous
            try:
                raw = self.client.complete_json('feedback_agent', self.skills['feedback_agent'], request_payload)
                merged = self._merge_objects(previous or {}, raw) if previous else raw
                merged = self._normalize_feedback(merged)
                if not self._feedback_score_is_integer(merged):
                    if not score_correction_requested:
                        score_correction_requested = True
                        last_error = 'completion_score is not an integer from 0 to 100'
                        previous = merged
                        continue
                    merged = self._normalize_feedback_score(merged)
                    score = merged.get('completion_score')
                    if isinstance(score, float) and 0 <= score <= 100:
                        merged['completion_score'] = round(score)
                self._validate_feedback(merged)
                merged['source'] = 'feedback_agent'
                merged['attempt'] = attempt + 1
                return merged
            except AgentProviderError:
                raise
            except AgentOutputError as exc:
                last_error = str(exc)
                if last_error.startswith('LLM request failed'):
                    break
                if isinstance(raw, dict):
                    previous = self._merge_objects(previous or {}, raw)
                if score_correction_requested:
                    break
        fallback = self._fallback_feedback(evidence, last_error)
        self._validate_feedback(fallback)
        return fallback

    @staticmethod
    def _normalize_feedback(raw: Any) -> Any:
        """Accept useful feedback shapes without binding the model to nine names."""
        if not isinstance(raw, dict):
            return raw
        result = dict(raw)
        result = AgentOrchestrator._normalize_feedback_score(result)
        overall_aliases = {'pass': 'completed', 'passed': 'completed', 'success': 'completed', 'complete': 'completed', 'fail': 'failed', 'failed': 'failed', 'error': 'failed', 'incomplete': 'partial'}
        overall = result.get('overall_status')
        if isinstance(overall, str):
            result['overall_status'] = overall_aliases.get(overall.lower().strip(), overall)
        aliases = {'routing': 'task_classification', 'model_fit': 'model_selection', 'graph_validity': 'workflow_schema', 'graph': 'workflow_schema'}
        dimensions = result.get('dimensions', {})
        if isinstance(dimensions, dict):
            normalized = {}
            for name, value in dimensions.items():
                name = aliases.get(name, name)
                if isinstance(value, str):
                    value = {'status': value}
                if isinstance(value, dict):
                    item = dict(value)
                    if 'repair_suggestion' in item and 'repair_suggestions' not in item:
                        item['repair_suggestions'] = item.pop('repair_suggestion')
                    for key in ('evidence', 'errors', 'repair_suggestions'):
                        if isinstance(item.get(key), str):
                            item[key] = [item[key]]
                        item.setdefault(key, [])
                    normalized[name] = item
            result['dimensions'] = normalized
        assessment = result.get('output_assessment')
        if not isinstance(assessment, dict):
            output = result.get('dimensions', {}).get('output', {}) if isinstance(result.get('dimensions'), dict) else {}
            alignment = result.get('dimensions', {}).get('task_alignment', {}) if isinstance(result.get('dimensions'), dict) else {}
            assessment = {'status': output.get('status', alignment.get('status', 'unverifiable')), 'task_alignment': result.get('task_alignment', ''), 'issues': output.get('errors', []) + alignment.get('errors', []), 'evidence': output.get('evidence', []) + alignment.get('evidence', [])}
        if isinstance(assessment, dict):
            assessment = dict(assessment)
            status = assessment.get('status')
            if isinstance(status, str):
                assessment['status'] = {'passed': 'pass', 'failed': 'fail', 'success': 'pass'}.get(status.lower().strip(), status.lower().strip())
            if assessment.get('status') not in {'pass', 'fail', 'partial', 'unverifiable'}:
                assessment['status'] = 'partial'
            for key in ('issues', 'evidence'):
                if isinstance(assessment.get(key), str):
                    assessment[key] = [assessment[key]]
                assessment.setdefault(key, [])
                if isinstance(assessment[key], list):
                    assessment[key] = [AgentOrchestrator._feedback_text(item, key) for item in assessment[key]]
            assessment.setdefault('task_alignment', str(result.get('task_alignment', '')))
        result['output_assessment'] = assessment
        if result.get('overall_status') not in {'completed', 'partial', 'failed', 'unverifiable'}:
            assessment_status = assessment.get('status') if isinstance(assessment, dict) else 'unverifiable'
            result['overall_status'] = {'pass': 'completed', 'fail': 'failed'}.get(assessment_status, 'partial')
        result.setdefault('dimensions', {})
        result.setdefault('blocking_dimensions', [])
        result.setdefault('evidence_summary', [])
        if isinstance(result.get('evidence_summary'), str):
            result['evidence_summary'] = [result['evidence_summary']]
        return result

    @staticmethod
    def _feedback_text(item: Any, field: str) -> str:
        """Retain structured visual observations in the string-list contract."""
        if isinstance(item, str):
            return item
        if isinstance(item, dict):
            for key in ('issue', 'message', 'evidence', 'description'):
                if isinstance(item.get(key), str):
                    dimension = item.get('dimension')
                    return f'{dimension}: {item[key]}' if isinstance(dimension, str) else item[key]
            return json.dumps(item, ensure_ascii=False, sort_keys=True)
        return str(item)

    @staticmethod
    def _feedback_score_is_integer(raw: Any) -> bool:
        return isinstance(raw, dict) and isinstance(raw.get('completion_score'), int) and (not isinstance(raw.get('completion_score'), bool)) and (0 <= raw['completion_score'] <= 100)

    @staticmethod
    def _normalize_feedback_score(raw: dict[str, Any]) -> dict[str, Any]:
        result = dict(raw)
        score = result.get('completion_score')
        if isinstance(score, str):
            try:
                score = float(score.strip())
            except ValueError:
                return result
        if isinstance(score, (int, float)) and (not isinstance(score, bool)):
            if 0 <= score <= 1 and score != int(score):
                score *= 100
            if isinstance(score, float) and score.is_integer():
                result['completion_score'] = int(score)
            elif isinstance(score, int):
                result['completion_score'] = score
            elif isinstance(score, float) and 0 <= score <= 100:
                result['completion_score'] = score
        return result

    def evaluate_visual_output(self, run_id: str, instruction: str, output_paths: list[str], execution: dict[str, Any], extra: dict[str, Any] | None=None, reference_paths: list[str] | None=None) -> dict[str, Any]:
        """Ask a vision-capable model to assess actual output pixels."""
        reference_urls = self._image_data_urls(reference_paths or [])
        image_urls = self._image_data_urls(output_paths[:4])
        if not image_urls:
            return {'status': 'unverifiable', 'completion_score': 0, 'task_alignment': 'unverifiable', 'issues': ['output image unavailable'], 'evidence': [], 'source': 'visual_evaluation_agent'}
        visual_contract = {'role': 'Evaluate actual output-image pixels against the instruction.', 'rules': ['Return exactly one JSON object with the requested fields.', 'status must be pass, fail, partial, or unverifiable.', 'completion_score must be an integer from 0 through 100.', 'issues and evidence must be arrays of concise strings.', 'Evaluate the supplied image, not the execution status.', 'When reference images are supplied first, compare subject, composition, identity, and requested changes against them; do not judge the output in isolation.'], 'required_output': ['status', 'completion_score', 'task_alignment', 'issues', 'evidence']}
        raw = self.client.complete_json('visual_evaluation_agent', visual_contract, {'run_id': run_id, 'instruction': instruction, 'execution': execution, 'extra': extra or {}, 'visual_input_order': {'reference_images_first': len(reference_urls), 'output_images_after': len(image_urls)}, '_image_data_urls': reference_urls + image_urls, 'output_schema': {'status': 'pass|fail|partial|unverifiable', 'completion_score': 'integer 0..100', 'task_alignment': 'string', 'issues': ['string'], 'evidence': ['string']}})
        required = {'status', 'completion_score', 'task_alignment', 'issues', 'evidence'}
        if set(raw) < required or raw.get('status') not in {'pass', 'fail', 'partial', 'unverifiable'} or (not isinstance(raw.get('completion_score'), int)) or (not 0 <= raw['completion_score'] <= 100) or (not self._string_list(raw.get('issues'), allow_empty=True)) or (not self._string_list(raw.get('evidence'), allow_empty=True)):
            raise AgentOutputError('invalid visual evaluation output')
        raw['source'] = 'visual_evaluation_agent'
        return raw

    def decide_repair_search(self, *, instruction: str, graph: dict[str, Any], execution: dict[str, Any], official_evaluation: dict[str, Any], attempt_history: list[dict[str, Any]], media_paths: list[str] | None=None, visual_manifest: list[dict[str, Any]] | None=None) -> dict[str, Any]:
        """Decide whether a failed attempt needs external parameter documentation."""
        contract = {'role': 'Diagnose a failed ComfyBench attempt and decide whether web research is necessary.', 'rules': ['Use the task instruction, attached original input and output media, official evaluation, execution evidence, current graph, and prior attempts together.', 'When visual_input_order is supplied, inspect every attached image before attributing the failure. For localized edits, compare original_input, effective_mask, and final stage_output directly; never claim that mask coverage or polarity is unknown when the effective mask image is attached.', 'Distinguish retained source pixels from content generated inside the editable mask. Black holes or uncovered boundaries can explain retained source pixels only; they cannot explain newly generated semantically wrong content across a white editable region. When the intended target is substantially masked and visibly regenerated with the wrong object or texture, prioritize downstream text, actual negative conditioning, source-image conditioning, guidance, and sampler settings over regenerating the upstream mask.', 'Identify the concrete mismatch before proposing a repair focus.', 'Decide autonomously whether public external evidence would materially reduce repair uncertainty. Search may cover model or node documentation, parameter behavior, recommended settings, workflow usage, or other facts needed to repair this concrete failure.', 'A visible abnormality with uncertain sampler/CFG/denoise/steps or model-specific parameter interaction is one important example, not an exclusive condition.', 'For localized edits, explicitly compare five causal channels: text conditioning, source-image conditioning, mask shape/coverage/boundary conditioning, actual negative-conditioning capability, and sampler/denoise/guidance conditioning.', 'Treat prior attempts as counterfactual evidence. If removing an object name from the positive prompt does not remove the object and it persists at the same location or silhouette, do not keep text as the primary cause and do not recommend another prompt-only repair.', 'Do not search when the supplied task, graph, execution error, and evaluations already establish the repair with high confidence; explain the decision either way.', 'Return at most three concise, public, non-sensitive search queries.', 'Return one JSON object only.'], 'required_output': ['failure_class', 'root_causes', 'next_action', 'diagnosis', 'conditioning_attribution', 'repair_focus', 'needs_search', 'search_queries', 'evidence']}
        payload = {'instruction': instruction, 'current_graph': graph, 'execution': execution, 'official_comfybench_evaluation': official_evaluation, 'attempt_history': attempt_history, 'visual_input_order': visual_manifest or [], 'output_schema': {'failure_class': 'quality|execution|handoff|parameter|capability|routing|composition', 'root_causes': ['concise evidence-backed causal finding'], 'next_action': 'parameter_retry|topology_repair|composition_repair|routing_correction|stop', 'diagnosis': 'concrete failure explanation', 'conditioning_attribution': {key: {'likelihood': 'likely|possible|unlikely|unknown|not_applicable', 'evidence': ['specific graph, evaluation, or attempt-history evidence'], 'counterfactual': 'observation that would support or refute this cause'} for key in ('text_conditioning', 'source_image_conditioning', 'mask_conditioning', 'negative_conditioning', 'sampler_conditioning')}, 'repair_focus': ['ordered prompt or parameter changes'], 'needs_search': 'boolean', 'search_queries': ['string; maximum three'], 'evidence': ['specific input/output/evaluator evidence']}, **self._image_evidence_payload(media_paths or [])}
        from .compact_io import compact_diagnosis_request, expand_conditioning_attribution
        contract, payload = compact_diagnosis_request(contract, payload)
        previous = None
        correction = ''
        for attempt in range(self.max_corrections + 1):
            request = dict(payload)
            if correction:
                request['previous_response'] = previous
                request['contract_correction'] = correction
            raw = self.client.complete_json('generation_agent', contract, request)
            raw = expand_conditioning_attribution(raw)
            attribution = raw.get('conditioning_attribution', {}) if isinstance(raw, dict) else {}
            attribution_keys = {'text_conditioning', 'source_image_conditioning', 'mask_conditioning', 'negative_conditioning', 'sampler_conditioning'}
            attribution_valid = isinstance(attribution, dict) and set(attribution) == attribution_keys and all((isinstance(row, dict) and row.get('likelihood') in {'likely', 'possible', 'unlikely', 'unknown', 'not_applicable'} and self._string_list(row.get('evidence'), allow_empty=True) and isinstance(row.get('counterfactual'), str) and row['counterfactual'].strip() for row in attribution.values()))
            valid = isinstance(raw, dict) and raw.get('failure_class') in {'quality', 'execution', 'handoff', 'parameter', 'capability', 'routing', 'composition'} and self._string_list(raw.get('root_causes'), allow_empty=False) and (raw.get('next_action') in {'parameter_retry', 'topology_repair', 'composition_repair', 'routing_correction', 'stop'}) and isinstance(raw.get('diagnosis'), str) and raw['diagnosis'].strip() and attribution_valid and isinstance(raw.get('repair_focus'), list) and all((isinstance(item, str) for item in raw['repair_focus'])) and isinstance(raw.get('needs_search'), bool) and isinstance(raw.get('search_queries'), list) and (len(raw['search_queries']) <= 3) and all((isinstance(item, str) and item.strip() for item in raw['search_queries'])) and isinstance(raw.get('evidence'), list) and all((isinstance(item, str) for item in raw['evidence'])) and (not raw['needs_search'] or bool(raw['search_queries']))
            if valid:
                return raw
            previous = raw
            correction = 'Return every required field with the declared enum values and types, including failure_class, root_causes, next_action, and all five conditioning_attribution channels; needs_search=true requires 1-3 queries.'
            if attempt >= self.max_corrections:
                raise AgentOutputError('invalid repair-search decision output')
        raise AgentOutputError('repair-search decision failed')

    def evaluate_video_output(self, run_id: str, instruction: str, result_frame_paths: list[str], result_metadata: dict[str, Any], execution: dict[str, Any], reference_frame_paths: list[str] | None=None, reference_metadata: dict[str, Any] | None=None, extra: dict[str, Any] | None=None) -> dict[str, Any]:
        """Evaluate sampled video frames, matching ComfyBench's evidence shape.

        The model receives reference frames first (for i2v/v2v), followed by
        temporally ordered result frames.  A score remains this project's
        stricter quality signal; ComfyBench itself returns only True/False.
        """
        reference_urls = self._image_data_urls(reference_frame_paths or [])
        result_urls = self._image_data_urls(result_frame_paths)
        if not result_urls:
            return {'status': 'unverifiable', 'completion_score': 0, 'task_alignment': 'unverifiable', 'issues': ['video frames unavailable'], 'evidence': [], 'source': 'visual_evaluation_agent'}
        visual_contract = {'role': 'Evaluate temporally ordered sampled video frames against the instruction.', 'video_timing_policy': VIDEO_TIMING_POLICY, 'rules': ['Return exactly one JSON object with the requested fields.', 'status must be pass, fail, partial, or unverifiable.', 'completion_score must be an integer from 0 through 100.', 'Assess instruction alignment, frame-to-frame consistency, and visible artifacts.', 'For i2v/v2v, compare the result against the reference frames supplied first.'], 'required_output': ['status', 'completion_score', 'task_alignment', 'issues', 'evidence']}
        raw = self.client.complete_json('visual_evaluation_agent', visual_contract, {'run_id': run_id, 'instruction': instruction, 'execution': execution, 'extra': extra or {}, 'reference_metadata': reference_metadata or {}, 'result_metadata': result_metadata, 'frame_order': {'reference_frames_first': len(reference_urls), 'result_frames_after': len(result_urls)}, '_image_data_urls': reference_urls + result_urls, 'output_schema': {'status': 'pass|fail|partial|unverifiable', 'completion_score': 'integer 0..100', 'task_alignment': 'string', 'issues': ['string'], 'evidence': ['string']}})
        self._validate_visual_evaluation(raw)
        raw['source'] = 'visual_evaluation_agent'
        return raw

    def _image_evidence_payload(self, paths: list[str]) -> dict:
        """Keep thumbnail and source pixel coordinates explicit for repairs."""
        import base64
        from PIL import Image
        readable = [str(path) for path in paths if Path(path).is_file()]
        urls = self._image_data_urls(readable)
        manifest = []
        for position, (path, url) in enumerate(zip(readable, urls), 1):
            try:
                with Image.open(path) as original:
                    source_width, source_height = original.size
                with Image.open(io.BytesIO(base64.b64decode(url.split(',', 1)[1]))) as sent:
                    view_width, view_height = sent.size
                manifest.append({'position': position, 'path': path, 'source_width': source_width, 'source_height': source_height, 'view_width': view_width, 'view_height': view_height, 'view_to_source_x': source_width / view_width, 'view_to_source_y': source_height / view_height})
            except (OSError, ValueError):
                continue
        return {'_image_data_urls': urls, 'visual_coordinate_manifest': manifest, 'visual_coordinate_policy': 'Attached images may be resized thumbnails. All graph crop, placement and mask coordinates refer to the pixel dimensions of the image entering that node. Convert visual view coordinates to source pixels using view_to_source_x/y; for an intermediate resized or cropped node, then apply its scale and origin. Do not copy thumbnail pixel coordinates directly into a source-image crop. Report the coordinate space and conversion with each spatial repair.'}

    def _image_data_urls(self, paths: list[str]) -> list[str]:
        import base64
        urls = []
        for raw_path in paths:
            try:
                image_bytes = Path(raw_path).read_bytes()
            except OSError:
                continue
            if self.image_max_dimension:
                try:
                    from PIL import Image
                    with Image.open(io.BytesIO(image_bytes)) as image:
                        if max(image.size) > self.image_max_dimension:
                            image.thumbnail((self.image_max_dimension, self.image_max_dimension), Image.Resampling.LANCZOS)
                            resized = io.BytesIO()
                            image.save(resized, format='PNG', optimize=True)
                            image_bytes = resized.getvalue()
                except (ImportError, OSError, ValueError):
                    pass
            encoded = base64.b64encode(image_bytes).decode('ascii')
            urls.append('data:image/png;base64,' + encoded)
        return urls

    @staticmethod
    def _validate_visual_evaluation(raw: Any) -> None:
        required = {'status', 'completion_score', 'task_alignment', 'issues', 'evidence'}
        if set(raw) < required or raw.get('status') not in {'pass', 'fail', 'partial', 'unverifiable'} or (not isinstance(raw.get('completion_score'), int)) or (not 0 <= raw['completion_score'] <= 100) or (not AgentOrchestrator._string_list(raw.get('issues'), allow_empty=True)) or (not AgentOrchestrator._string_list(raw.get('evidence'), allow_empty=True)):
            raise AgentOutputError('invalid visual evaluation output')

    @staticmethod
    def _fallback_feedback(evidence: dict[str, Any], error: str) -> dict[str, Any]:
        execution = evidence.get('execution', {}) if isinstance(evidence, dict) else {}
        output = evidence.get('output', {}) if isinstance(evidence, dict) else {}
        alignment = evidence.get('task_alignment', {}) if isinstance(evidence, dict) else {}
        visual = evidence.get('visual_evaluation', {}) if isinstance(evidence, dict) else {}
        execution_ok = execution.get('status') == 'success'
        alignment_ok = alignment.get('passed') is True
        if visual.get('status') in {'pass', 'fail', 'partial'}:
            alignment_ok = visual.get('status') == 'pass'
        status = 'completed' if execution_ok and alignment_ok else 'failed' if execution_ok else 'unverifiable'
        statuses = {'task_classification': 'pass', 'model_selection': 'pass', 'workflow_schema': 'pass', 'interface_compatibility': 'pass', 'dependencies': 'pass', 'execution': 'pass' if execution_ok else 'fail', 'output': 'pass' if alignment_ok else 'fail', 'task_alignment': 'pass' if alignment_ok else 'fail', 'resource_constraints': 'pass'}
        dimensions = {}
        for name, dim_status in statuses.items():
            errors = []
            suggestions = []
            if name in {'output', 'task_alignment'} and (not alignment_ok):
                errors.append(str(visual.get('task_alignment') or output.get('visual_observation') or alignment.get('reason') or 'output did not satisfy task'))
                suggestions.append('inspect the graph, execution evidence, and visual evidence for the failing output')
            dimensions[name] = {'status': dim_status, 'score': 100 if dim_status == 'pass' else 0, 'evidence': ['local execution/output evidence'], 'errors': errors, 'repair_suggestions': suggestions}
        score = 100 if status == 'completed' else 40 if execution_ok else 0
        return {'overall_status': status, 'completion_score': score, 'output_assessment': {'status': 'pass' if alignment_ok else 'fail', 'task_alignment': str(visual.get('task_alignment', '')), 'issues': list(visual.get('issues', [])), 'evidence': list(visual.get('evidence', []))}, 'dimensions': dimensions, 'blocking_dimensions': [name for name in ('output', 'task_alignment') if statuses[name] == 'fail'], 'evidence_summary': ['Feedback model response failed contract validation: ' + error], 'source': 'local_contract_fallback', 'attempt': 3}

    @staticmethod
    def _merge_objects(base: dict[str, Any], patch: dict[str, Any]) -> dict[str, Any]:
        """Merge a partial contract correction without inventing missing values."""
        result = dict(base)
        for key, value in patch.items():
            if isinstance(result.get(key), dict) and isinstance(value, dict):
                result[key] = AgentOrchestrator._merge_objects(result[key], value)
            else:
                result[key] = value
        return result

    @staticmethod
    def _validate_feedback(raw: Any) -> None:
        if not isinstance(raw, dict):
            raise AgentOutputError('feedback output must be an object')
        score = raw.get('completion_score')
        if not isinstance(score, int) or not 0 <= score <= 100:
            raise AgentOutputError('completion_score must be an integer from 0 to 100')
        assessment = raw.get('output_assessment')
        if not isinstance(assessment, dict) or assessment.get('status') not in {'pass', 'fail', 'partial', 'unverifiable'} or (not isinstance(assessment.get('task_alignment'), str)) or (not AgentOrchestrator._string_list(assessment.get('issues'), allow_empty=True)) or (not AgentOrchestrator._string_list(assessment.get('evidence'), allow_empty=True)):
            raise AgentOutputError('feedback output_assessment is invalid')

    @staticmethod
    def _string_list(value: Any, allow_empty: bool=False) -> bool:
        return isinstance(value, list) and all((isinstance(item, str) for item in value)) and (allow_empty or bool(value))

    @staticmethod
    def _workflow_evolution_contract() -> dict[str, Any]:
        """One stable contract for the complete self-evolution conversation."""
        return {'version': 2, 'role': 'Self-evolution Agent: reassess current leaf assets, then repair from complete trace evidence.', 'required_output': ['For phase=current_leaf_asset_reassessment: decision, skill_id, rationale, evidence, retrieval_required, retrieval_requests, retrieval_reason.', 'For phase=capability_retrieval_planning: retrieval_required, retrieval_requests, retrieval_reason.', 'For phase=workflow_evolution: skill_ids, edits, rationale, decision_trace, workflow_composition_assessment, candidate_kind, root_cause, promotion_target.'], 'rules': ['Treat the original requirement, complete initial trace, confirmed leaf attribution, current Base, and supplied Skills as fixed background evidence.', 'Obey the phase in the latest dynamic request suffix; do not perform a different phase.', 'For capability_retrieval_planning, the Agent must define the exact local registry search terms, desired role, and typed input/output contract for each independent capability gap.', 'Do not turn composition-adapter, routing, or handoff ownership defects into custom-node retrieval requests.', 'For current_leaf_asset_reassessment, return only decision, skill_id, rationale, evidence, retrieval_required, retrieval_requests, and retrieval_reason.', 'The reassessment decision must be try_current_base, try_existing_skill, or continue_evolution.', 'Use try_current_base only when the current on-disk Base directly addresses the confirmed failure.', 'Use try_existing_skill only when exactly one supplied Skill directly matches the failure; return its exact skill_id.', 'Use continue_evolution when neither current asset is likely to solve the confirmed failure.', 'When continue_evolution needs node/model/interface/workflow-example evidence not already supplied, set retrieval_required=true and provide one retrieval_requests object per independent capability gap.', 'For every retrieval request, choose concrete local_search_terms, required_input_types, required_output_types, desired_role, and a trace-grounded reason. The runner must execute these terms and must not invent the primary search vocabulary.', 'Set retrieval_required=false only when supplied live interfaces and workflow evidence already support a safe executable candidate.', 'A selected Base or Skill still requires static validation, actual execution, and official evaluation.', 'For workflow_evolution, compare all attempts, diagnostics, execution results, and evaluations in the fixed trace plus appended feedback.', 'When capability_retrieval supplies candidate nodes, treat scores and recommendations only as a bounded evidence shortlist, never as an automatic selection.', 'Dynamically choose the smallest sufficient node or node combination by comparing the exact failure semantics, current graph boundary, typed inputs/outputs, required extra models and parameters, integration cost, and verified workflow JSON connections.', 'In decision_trace, state which shortlisted candidates were selected, which plausible alternatives were rejected, and why the selected topology addresses this case rather than merely matching an interface type.', 'You may add_node, remove_node, connect, disconnect, or set_param when evidence justifies it.', 'Repair input/output contracts and MASK/IMAGE/VIDEO handoffs explicitly; do not hide topology repair in prompt text.', 'The original Base is never overwritten. If it has succeeded before, topology changes require a new workflow branch and workflow_skill promotion target.', 'A successful execution is not a task-quality pass.', 'Use canonical edit shapes exactly: add_node={action,node_id,node,reason}; remove_node={action,target,reason}; connect/disconnect={action,source:[node,slot],destination:[node,input],reason}; set_param={action,target,key,value,reason}.', 'For workflow_evolution, candidate_kind must be existing_workflow_revision or new_workflow_branch, and promotion_target must be base or workflow_skill.', 'Return one JSON object only and no markdown.']}

    @staticmethod
    def _workflow_evolution_static_context(*, spec: RequirementSpec, implementation_plan: list[dict[str, Any]], graph: dict[str, Any], trace: dict[str, Any], instruction: str, workflow_skills: list[dict[str, Any]] | None=None) -> dict[str, Any]:
        """Build the immutable prefix shared by reassessment and all repairs."""
        existing = trace.get('_workflow_evolution_static_context')
        if isinstance(existing, dict):
            return existing
        static_context = {'original_user_requirement': instruction, 'requirement_spec': {'modality': spec.modality, 'task_family': spec.task_family, 'operations': list(spec.operations), 'capabilities': list(spec.capabilities), 'priorities': list(spec.priorities), 'constraints': list(spec.constraints), 'explicit_model': spec.explicit_model}, 'implementation_plan': copy.deepcopy(implementation_plan), 'initial_current_graph': copy.deepcopy(graph), 'initial_failure_analysis': copy.deepcopy(trace.get('reviewed_failure_analysis') or trace.get('failure_analysis')), 'selected_leaf_target': copy.deepcopy(trace.get('selected_leaf_target')), 'current_leaf_asset_recheck': copy.deepcopy(trace.get('current_leaf_asset_recheck')), 'current_leaf_workflow_skills': copy.deepcopy(workflow_skills or []), 'evolution_policy': {'ordinary_retry_contract': 'parameter_and_prompt_only', 'base_protection': 'Never overwrite a previously successful Base; publish conditional or topology changes as a workflow Skill.', 'execution_required_after_candidate': True, 'promotion_rule': 'Promote only after static validation, execution, and official quality pass.'}}
        trace['_workflow_evolution_static_context'] = static_context
        trace['_workflow_evolution_prompt_cache_key'] = 'workflow-evolution:' + hashlib.sha256(json.dumps(static_context, ensure_ascii=False, sort_keys=True).encode('utf-8')).hexdigest()[:32]
        return static_context
