"""Local worker process; persistence is owned by the coordinator."""

__all__ = ["WorkerClient"]


def __getattr__(name):
    # Dedicated media children import worker-owned libraries without installing
    # supervisor HTTP/Pydantic dependencies or importing other media runtimes.
    if name == "WorkerClient":
        from .client import WorkerClient

        return WorkerClient
    raise AttributeError(name)
