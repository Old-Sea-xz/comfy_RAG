"""Dataset-neutral inputs, policies and dependencies for the case executor."""
from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Iterator


@dataclass(frozen=True)
class ExecutionPolicy:
    """Experiment rules; the dataset name never chooses execution behavior."""

    name: str = "standard"
    frozen_graph: bool = False
    planning_attempt_limit: int | None = None
    disable_self_evolution: bool = False
    disable_search: bool = False
    mask_preview_advisory: bool | None = None

    def __post_init__(self):
        if self.planning_attempt_limit is not None and self.planning_attempt_limit < 1:
            raise ValueError("planning_attempt_limit must be positive")

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class ExecutionHooks:
    """Injected dependencies, scoped to one execution rather than module mutation."""

    evaluator: Callable[..., dict] | None = None
    resource_loader: Callable[..., list] | None = None
    agent_factory: Callable[..., Any] | None = None
    executor_factory: Callable[..., Any] | None = None
    implementation_provider: Callable[..., tuple] | None = None
    base_provider: Callable[..., dict] | None = None
    composition_observer: Callable[[list], None] | None = None
    output_observer: Callable[[Path], None] | None = None
    before_submit: Callable[[dict], None] | None = None
    evaluation_context_observer: Callable[[dict], None] | None = None


@dataclass(frozen=True)
class CaseExecutionRequest:
    case_id: str
    case: dict
    output_root: Path
    max_attempts: int = 3
    vram_limit_gib: float = 110.0
    report_root: Path | None = None
    audit_root: Path | None = None
    resource_roots: tuple[Path, ...] | None = None
    runtime_params: tuple[str, ...] = ()
    stage_params: tuple[str, ...] = ()
    routing_correction: dict | None = None
    reuse_report: Path | None = None
    resume_report: Path | None = None
    interrupted_report: Path | None = None
    regenerate_on_resume: bool = False

    def __post_init__(self):
        if self.regenerate_on_resume and not self.resume_report:
            raise ValueError("regeneration requires an evaluated resume report")
        if sum(bool(p) for p in (self.reuse_report, self.resume_report, self.interrupted_report)) > 1:
            raise ValueError("planning restart and evaluated checkpoint continuation are exclusive")
        if not self.case_id:
            raise ValueError("case_id is required")
        if not isinstance(self.case.get("instruction"), str) or not self.case["instruction"].strip():
            raise ValueError("case requires a nonempty instruction")
        if self.case.get("modality") not in {"t2i", "t2v", "i2i", "i2v", "v2v"}:
            raise ValueError("case has an invalid modality")
        if not 1 <= self.max_attempts <= 3:
            raise ValueError("max_attempts must be between 1 and 3")
        if self.vram_limit_gib <= 0:
            raise ValueError("vram_limit_gib must be positive")


@dataclass(frozen=True)
class ExecutionContext:
    request: CaseExecutionRequest | None
    hooks: ExecutionHooks
    policy: ExecutionPolicy


_ACTIVE: ContextVar[ExecutionContext | None] = ContextVar("case_execution_context", default=None)


def current_execution() -> ExecutionContext | None:
    return _ACTIVE.get()


@contextmanager
def execution_context(request: CaseExecutionRequest | None = None, *,
                      hooks: ExecutionHooks | None = None,
                      policy: ExecutionPolicy | None = None) -> Iterator[ExecutionContext]:
    """Also scopes checkpoint continuations; always restore after exceptions."""
    context = ExecutionContext(request, hooks or ExecutionHooks(), policy or ExecutionPolicy())
    token = _ACTIVE.set(context)
    try:
        yield context
    finally:
        _ACTIVE.reset(token)
