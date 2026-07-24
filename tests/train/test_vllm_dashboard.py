"""
uv run --isolated --extra dev pytest tests/train/test_vllm_dashboard.py
"""

import re
from pathlib import Path

from skyrl.train.utils import vllm_dashboard
from skyrl.train.utils.vllm_dashboard import build_dashboard, dashboard_json

_JSON_PATH = Path(vllm_dashboard.__file__).with_suffix(".json")

# Every metric referenced in a panel must be one the vLLM RayPrometheusStatLogger
# exports, so nothing here queries a series that never shows up on Ray's agents.
_METRIC_RE = re.compile(r"\bray_vllm_[a-z0-9_]+")


def _panels(dashboard):
    return [p for p in dashboard["panels"] if p["type"] != "row"]


def test_committed_json_matches_builder():
    """The checked-in JSON is regenerated output; run the module to refresh it."""
    assert _JSON_PATH.read_text() == dashboard_json(), f"{_JSON_PATH.name} is stale"


def test_every_target_queries_a_ray_vllm_metric():
    for panel in _panels(build_dashboard()):
        for target in panel["targets"]:
            assert _METRIC_RE.search(target["expr"]), f"{panel['title']} target has no ray_vllm_ metric"


def test_datasource_uid_is_applied_everywhere():
    dashboard = build_dashboard(datasource_uid="my-prom")
    uids = {p["datasource"]["uid"] for p in _panels(dashboard)}
    uids |= {t["datasource"]["uid"] for p in _panels(dashboard) for t in p["targets"]}
    assert uids == {"my-prom"}


def test_panel_ids_and_grid_are_well_formed():
    panels = build_dashboard()["panels"]
    ids = [p["id"] for p in panels]
    assert len(ids) == len(set(ids)), "panel ids must be unique"
    for panel in _panels(build_dashboard()):
        assert panel["gridPos"]["x"] + panel["gridPos"]["w"] <= 24
        assert {t["refId"] for t in panel["targets"]}.issuperset({"A"})
