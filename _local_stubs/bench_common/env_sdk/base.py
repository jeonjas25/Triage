"""
Local stub for bench_common.env_sdk.base.

The real implementation is provided by the swecc-mesocosm SDK on the platform.
This stub lets run_local.py work without installing the full SDK.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class StepResult:
    observation: Any
    reward: float
    terminated: bool
    truncated: bool
    info: dict = field(default_factory=dict)


class BaseEnv:
    def reset(self, seed: int | None = None, **params: Any) -> Any:
        raise NotImplementedError

    def step(self, action: Any) -> StepResult:
        raise NotImplementedError
