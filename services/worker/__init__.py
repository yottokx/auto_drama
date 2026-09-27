"""Local worker process; persistence is owned by the coordinator."""

from .client import WorkerClient

__all__ = ["WorkerClient"]
