"""Puerto de cola de trabajos. Adaptador en proceso (hilo) para el MVP; para producción, implementar el
mismo Protocol con RQ/Celery y mover `run_job` a un worker independiente."""
from __future__ import annotations

import threading
import traceback
from typing import Any, Callable, Protocol

from .storage import Storage


class JobQueue(Protocol):
    def enqueue(self, analysis_id: str, fn: Callable[[], dict[str, Any]]) -> None: ...


class InProcessQueue:
    """Ejecuta el trabajo en un hilo daemon y persiste estado/informe a través de Storage."""

    def __init__(self, storage: Storage):
        self.storage = storage

    def enqueue(self, analysis_id: str, fn: Callable[[], dict[str, Any]]) -> None:
        def _run():
            try:
                self.storage.write_status(analysis_id, "running")
                report = fn()
                self.storage.write_report(analysis_id, report)
                self.storage.write_status(analysis_id, "done")
            except Exception as e:  # noqa: BLE001
                (self.storage.dir(analysis_id) / "error.log").write_text(traceback.format_exc())
                self.storage.write_status(analysis_id, "error", str(e))

        self.storage.write_status(analysis_id, "queued")
        threading.Thread(target=_run, daemon=True).start()
