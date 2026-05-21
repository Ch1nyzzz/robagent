"""Drop-in entry point so the runner can call `agent.v1.base.run_task`."""
from .workflow import run_task

__all__ = ["run_task"]
