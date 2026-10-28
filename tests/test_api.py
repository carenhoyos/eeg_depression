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
P = SETTINGS.app.api_prefix
pytestmark = pytest.mark.skipif(not BUNDLE.exists(), reason="no hay bundle exportado (scripts/06_export_bundle.py)")


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    settings = SETTINGS.model_copy(deep=True)
    tmp = tmp_path_factory.mktemp("storage")
    settings.app.storage_dir = str(tmp)
    settings.database.url = f"sqlite:///{tmp}/test.db"
    with TestClient(create_app(settings)) as c:
        yield c


@pytest.fixture(scope="module")
def auth(client) -> dict:
    a = SETTINGS.auth
    r = client.post(f"{P}/auth/login", json={"username": a.bootstrap_username, "password": a.bootstrap_password})
    assert r.status_code == 200, r.text
    return {"Authorization": f"Bearer {r.json()['access_token']}"}


@pytest.fixture(scope="module")
def sample_csv(tmp_path_factory) -> Path:
    src = ROOT / "data/examples/sample_features.csv"
    if src.exists():
        return src
    df = pd.read_csv(SETTINGS.resolve("data/raw/machinelearning_data_EEG.csv")).head(2)
    p = tmp_path_factory.mktemp("in") / "sample.csv"
    df.drop(columns=[c for c in df.columns if c.startswith("specific") or c.startswith("Unnamed")]).to_csv(p, index=False)
    return p


def _wait(client: TestClient, aid: str, auth: dict, timeout: float = 180.0) -> dict:
    t0 = time.time()
    while time.time() - t0 < timeout:
        s = client.get(f"{P}/analyses/{aid}/status", headers=auth).json()
        if s["status"] in ("done", "error"):
            return s
        time.sleep(0.5)
    raise TimeoutError(aid)


def test_meta_endpoints(client):
    assert client.get(f"{P}/health").json()["status"] == "ok"
    card = client.get(f"{P}/model").json()
    assert card["n_features"] > 0 and 0 < card["train_prevalence"] < 1
    cfg = client.get(f"{P}/config/report").json()
    tiers = sorted(cfg["tiers"], key=lambda t: t["min"])
    assert tiers[0]["min"] == 0 and tiers[-1]["max"] == 1
    assert all(a["max"] == b["min"] for a, b in zip(tiers, tiers[1:]))
    assert "eyes_closed" in cfg["conditions"]


def test_auth_required(client):
    assert client.get(f"{P}/patients").status_code == 401
    assert client.post(f"{P}/auth/login", json={"username": "x", "password": "y"}).status_code == 401


def test_patients_crud(client, auth):
    r = client.post(f"{P}/patients", json={"pseudonym": "PX-1", "age": 34, "sex": "F"}, headers=auth)
    assert r.status_code == 201, r.text
    pid = r.json()["id"]
    assert any(p["id"] == pid for p in client.get(f"{P}/patients", headers=auth).json())
    assert client.patch(f"{P}/patients/{pid}", json={"medication": "sertralina"}, headers=auth).json()["medication"] == "sertralina"
    assert client.get(f"{P}/patients/{pid}/analyses", headers=auth).json() == []
    assert client.get(f"{P}/patients/99999", headers=auth).status_code == 404


def test_rejects_unknown_extension(client, auth):
    r = client.post(f"{P}/analyses", files=[("files", ("x.xyz", b"abc"))], headers=auth)
    assert r.status_code == 400


def test_feature_csv_analysis_end_to_end(client, auth, sample_csv):
    pid = client.post(f"{P}/patients", json={"pseudonym": "PX-2"}, headers=auth).json()["id"]
    with open(sample_csv, "rb") as f:
        r = client.post(f"{P}/analyses", files=[("files", (sample_csv.name, f, "text/csv"))],
                        data={"pretest_prevalence": "0.3", "patient_id": str(pid), "condition": "eyes_closed"}, headers=auth)
    assert r.status_code == 202, r.text
    aid = r.json()["analysis_id"]
    assert _wait(client, aid, auth)["status"] == "done"
    rep = client.get(f"{P}/analyses/{aid}", headers=auth).json()
    assert rep["quality"]["n_records"] >= 1 and rep["recording"]["patient_ref"] == "PX-2"
    for res in rep["results"]:
        assert 0 <= res["p_calibrated"] <= 1 and 0 <= res["p_posterior"] <= 1
        assert res["tier"]["min"] <= res["p_posterior"] <= res["tier"]["max"]
        assert res["preliminary_assessment"] and res["tier"]["name"] in res["preliminary_assessment"]
        assert abs(sum(res["band_relative_power"].values()) - 1) < 1e-6
    assert client.get(rep["pdf_url"], headers=auth).headers["content-type"] == "application/pdf"
    assert client.get(next(iter(rep["figures"].values())), headers=auth).status_code == 200
    hist = client.get(f"{P}/patients/{pid}/analyses", headers=auth).json()
    assert hist and hist[0]["status"] == "done" and hist[0]["tier_code"] == rep["results"][0]["tier"]["code"]
    assert client.get(f"{P}/analyses/{aid}").status_code == 401


def test_posterior_moves_with_pretest(client, auth, sample_csv):
    posts = {}
    for pre in ("0.1", "0.6"):
        with open(sample_csv, "rb") as f:
            aid = client.post(f"{P}/analyses", files=[("files", (sample_csv.name, f, "text/csv"))], data={"pretest_prevalence": pre}, headers=auth).json()["analysis_id"]
        assert _wait(client, aid, auth)["status"] == "done"
        posts[pre] = client.get(f"{P}/analyses/{aid}", headers=auth).json()["results"][0]["p_posterior"]
    assert posts["0.1"] < posts["0.6"]
