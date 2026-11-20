"""Operator scoping + incident-id contracts the FaultLab fixes rely on.

Findings fixed in the same change (FaultLab / Sidebar / CommandCenter):
  * D — an operator's browser used to fetch /stations/{id}/models and
    /stations/{id}/drift/latest for EVERY station in the roster, 403ing
    on the stations they cannot access. The frontend now scopes those
    fetches to the operator's own station; these tests pin the RBAC the
    scoping relies on (cross-station reads 403, own-station reads 200).
  * F — incident ids are stored lower-case hex (INC-<hex>) and the
    lookup is case-sensitive, so the FaultLab "Review Incident" button
    must render the id exactly as stored. A direct GET with the
    upper-cased display form 404s; this test pins that contract.
  * C — a second anomalous report for a station with an open incident
    UPDATES that incident (refreshing updated_at) instead of opening a
    duplicate. The FaultLab "merged" message depends on that refresh.
"""
from fastapi.testclient import TestClient

from backend.app.main import app
from backend.app.auth.security import create_access_token
from backend.app.storage.database import (
    init_db, get_db, create_or_update_incident, list_incidents,
)


def _admin_client():
    init_db()
    client = TestClient(app)
    headers = {"Authorization": "Bearer " + create_access_token(
        {"sub": "admin", "role": "admin"})}
    return client, headers


def _make_station(client, headers, sid):
    resp = client.post("/api/v1/admin/stations", headers=headers, json={
        "station_id": sid, "station_name": f"{sid} Scope Test Station",
        "username": sid.lower().replace("-", ""), "password": "test-password",
        "latitude": 13.34, "longitude": 74.74, "elevation": 27.0,
    })
    assert resp.status_code in (200, 201, 409), resp.text


def _operator_headers(sid):
    return {"Authorization": "Bearer " + create_access_token(
        {"sub": sid.lower().replace("-", ""), "role": "station_operator",
         "station_id": sid})}


def test_operator_cannot_read_other_station_models():
    client, headers = _admin_client()
    _make_station(client, headers, "SCOPE-A")
    _make_station(client, headers, "SCOPE-B")
    op = _operator_headers("SCOPE-A")
    resp = client.get("/api/v1/stations/SCOPE-B/models", headers=op)
    assert resp.status_code == 403, resp.text
    resp = client.get("/api/v1/stations/SCOPE-A/models", headers=op)
    assert resp.status_code == 200, resp.text


def test_operator_cannot_read_other_station_drift_latest():
    client, headers = _admin_client()
    _make_station(client, headers, "SCOPE-A")
    _make_station(client, headers, "SCOPE-B")
    op = _operator_headers("SCOPE-A")
    resp = client.get("/api/v1/stations/SCOPE-B/drift/latest", headers=op)
    assert resp.status_code == 403, resp.text
    resp = client.get("/api/v1/stations/SCOPE-A/drift/latest", headers=op)
    assert resp.status_code == 200, resp.text


def test_incident_id_lookup_is_case_sensitive():
    client, headers = _admin_client()
    _make_station(client, headers, "SCOPE-C")
    incident_id = "INC-scope0cabcdef1234567890abcdef12"
    with get_db() as conn:
        cur = conn.cursor()
        cur.execute(
            "INSERT INTO incidents (id, station_id, station_name, variable,"
            " severity, fault_risk, quality_state, status, created_at, updated_at)"
            " VALUES (?, ?, ?, 'temp:THERMAL_SPIKE', 'high', 0.9,"
            " 'LOCALIZED_ANOMALY', 'open', '2026-11-20T00:00:00+00:00',"
            " '2026-11-20T00:00:00+00:00')",
            (incident_id, "SCOPE-C", "SCOPE-C Scope Test Station"))
        conn.commit()
    resp = client.get(f"/api/v1/incidents/{incident_id}", headers=headers)
    assert resp.status_code == 200, resp.text
    assert resp.json()["incident"]["id"] == incident_id
    # The upper-cased form is what a button with text-transform:
    # uppercase displays — it must NOT resolve.
    resp = client.get(f"/api/v1/incidents/{incident_id.upper()}", headers=headers)
    assert resp.status_code == 404, resp.text


def test_second_report_updates_open_incident_instead_of_duplicating():
    client, headers = _admin_client()
    _make_station(client, headers, "SCOPE-D")
    first = create_or_update_incident({
        "station_id": "SCOPE-D", "station_name": "SCOPE-D Scope Test Station",
        "variable": "temp:THERMAL_SPIKE", "severity": "high",
        "fault_risk": 0.9, "quality_state": "LOCALIZED_ANOMALY",
        "explanation": "first spike report",
    })
    assert first is not None and first["status"] == "open"
    second = create_or_update_incident({
        "station_id": "SCOPE-D", "station_name": "SCOPE-D Scope Test Station",
        "variable": "temp:THERMAL_SPIKE", "severity": "high",
        "fault_risk": 0.95, "quality_state": "LOCALIZED_ANOMALY",
        "explanation": "second spike report",
    })
    assert second["id"] == first["id"], "second report opened a duplicate incident"
    assert second["explanation"] == "second spike report"
    assert second["updated_at"] >= first["updated_at"]
    assert len([i for i in list_incidents(station_id="SCOPE-D", status="open")]) == 1
