"""Puerto de almacenamiento y adaptador local. Para S3/GCS: implementar el mismo Protocol."""
from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol


class Storage(Protocol):
    def new_analysis(self) -> tuple[str, Path]: ...
    def dir(self, analysis_id: str) -> Path: ...
    def write_status(self, analysis_id: str, status: str, error: str | None = None) -> None: ...
    def read_status(self, analysis_id: str) -> dict[str, Any]: ...
    def write_report(self, analysis_id: str, report: dict[str, Any]) -> None: ...
    def read_report(self, analysis_id: str) -> dict[str, Any]: ...
    def list_analyses(self, limit: int) -> list[dict[str, Any]]: ...


class LocalStorage:
    def __init__(self, root: Path):
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)

    def new_analysis(self) -> tuple[str, Path]:
        analysis_id = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S") + "-" + uuid.uuid4().hex[:8]
        d = self.root / analysis_id
        (d / "figures").mkdir(parents=True, exist_ok=True)
        return analysis_id, d

    def dir(self, analysis_id: str) -> Path:
        if not analysis_id.replace("-", "").isalnum():
            raise FileNotFoundError(analysis_id)
        d = self.root / analysis_id
        if not d.exists():
            raise FileNotFoundError(analysis_id)
        return d

    def write_status(self, analysis_id: str, status: str, error: str | None = None) -> None:
        (self.dir(analysis_id) / "status.json").write_text(json.dumps({"status": status, "error": error, "updated_at": datetime.now(timezone.utc).isoformat()}))

    def read_status(self, analysis_id: str) -> dict[str, Any]:
        return json.loads((self.dir(analysis_id) / "status.json").read_text())

    def write_report(self, analysis_id: str, report: dict[str, Any]) -> None:
        (self.dir(analysis_id) / "report.json").write_text(json.dumps(report, indent=2, default=str, ensure_ascii=False))

    def read_report(self, analysis_id: str) -> dict[str, Any]:
        return json.loads((self.dir(analysis_id) / "report.json").read_text())

    def list_analyses(self, limit: int) -> list[dict[str, Any]]:
        out = []
        for d in sorted(self.root.iterdir(), reverse=True)[:limit]:
            if (d / "status.json").exists():
                s = json.loads((d / "status.json").read_text())
                out.append({"analysis_id": d.name, "status": s["status"], "updated_at": s.get("updated_at")})
        return out
