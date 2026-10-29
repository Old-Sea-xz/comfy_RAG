from __future__ import annotations
import enum
from dataclasses import dataclass, field
from typing import Any, Optional

@dataclass(frozen=True)
class RequirementSpec:
    """自然语言需求与图实现之间的显式中间表示。"""
    modality: str
    task_family: str
    operations: tuple[str, ...] = ()
    capabilities: tuple[str, ...] = ()
    priorities: tuple[str, ...] = ()
    constraints: tuple[str, ...] = ()
    explicit_model: Optional[str] = None
