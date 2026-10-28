"""Pruebas de integración de la API con el bundle publicado y un CSV de ejemplo.
Ejecutar desde la raíz: python -m pytest -q tests
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import pandas as pd
import pytest
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from app.backend.core.config import get_settings  # noqa: E402
from app.backend.main import create_app  # noqa: E402

SETTINGS = get_settings()
BUNDLE = SETTINGS.resolve(SETTINGS.model.bundle_path)
pytestmark = pytest.mark.skipif(not BUNDLE.exists(), reason="no hay bundle exportado (scripts/06_export_bundle.py)")


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    settings = SETTINGS.model_copy(deep=True)
    settings.app.storage_dir = str(tmp_path_factory.mktemp("storage"))
    with TestClient(create_app(settings)) as c:
        yield c


@pytest.fixture(scope="module")
def sample_csv(tmp_path_factory) -> Path:
    src = SETTINGS.resolve(SETTINGS.model.pipeline_config).parent.parent / "data/examples/sample_features.csv"
    if src.exists():
        return src
    df = pd.read_csv(SETTINGS.resolve("data/raw/machinelearning_data_EEG.csv")).head(2)
    p = tmp_path_factory.mktemp("in") / "sample.csv"
    df.drop(columns=[c for c in df.columns if c.startswith("specific") or c.startswith("Unnamed")]).to_csv(p, index=False)
    return p


def _wait(client: TestClient, aid: str, timeout: float = 120.0) -> dict:
    t0 = time.time()
    while time.time() - t0 < timeout:
        s = client.get(f"{SETTINGS.app.api_prefix}/analyses/{aid}/status").json()
        if s["status"] in ("done", "error"):
            return s
        time.sleep(0.5)
    raise TimeoutError(aid)


def test_meta_endpoints(client):
    p = SETTINGS.app.api_prefix
    assert client.get(f"{p}/health").json()["status"] == "ok"
    card = client.get(f"{p}/model").json()
    assert card["n_features"] > 0 and 0 < card["train_prevalence"] < 1
    cfg = client.get(f"{p}/config/report").json()
    tiers = sorted(cfg["tiers"], key=lambda t: t["min"])
    assert tiers[0]["min"] == 0 and tiers[-1]["max"] == 1
    assert all(a["max"] == b["min"] for a, b in zip(tiers, tiers[1:]))


def test_rejects_unknown_extension(client):
    r = client.post(f"{SETTINGS.app.api_prefix}/analyses", files={"file": ("x.xyz", b"abc")})
    assert r.status_code == 400


def test_feature_csv_analysis_end_to_end(client, sample_csv):
    p = SETTINGS.app.api_prefix
    with open(sample_csv, "rb") as f:
        r = client.post(f"{p}/analyses", files={"file": (sample_csv.name, f, "text/csv")}, data={"pretest_prevalence": "0.3", "patient_ref": "TEST-1"})
    assert r.status_code == 202, r.text
    aid = r.json()["analysis_id"]
    s = _wait(client, aid)
    assert s["status"] == "done", s
    rep = client.get(f"{p}/analyses/{aid}").json()
    assert rep["quality"]["n_records"] >= 1
    for res in rep["results"]:
        assert 0 <= res["p_calibrated"] <= 1 and 0 <= res["p_posterior"] <= 1
        assert res["tier"]["min"] <= res["p_posterior"] <= res["tier"]["max"]
        assert set(res["band_relative_power"]) == set(res["band_reference"])
        assert abs(sum(res["band_relative_power"].values()) - 1) < 1e-6
    assert client.get(rep["pdf_url"]).headers["content-type"] == "application/pdf"
    first_fig = next(iter(rep["figures"].values()))
    assert client.get(first_fig).status_code == 200


def test_posterior_moves_with_pretest(client, sample_csv):
    p = SETTINGS.app.api_prefix
    posts = {}
    for pre in ("0.1", "0.6"):
        with open(sample_csv, "rb") as f:
            aid = client.post(f"{p}/analyses", files={"file": (sample_csv.name, f, "text/csv")}, data={"pretest_prevalence": pre}).json()["analysis_id"]
        assert _wait(client, aid)["status"] == "done"
        posts[pre] = client.get(f"{p}/analyses/{aid}").json()["results"][0]["p_posterior"]
    assert posts["0.1"] < posts["0.6"]
