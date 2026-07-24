"""Build the SkyRL vLLM Grafana dashboard from Prometheus queries.

The dashboard visualizes the ``ray_vllm_*`` series that vLLM's
``RayPrometheusStatLogger`` exports through Ray's per-node metrics agents when
``generator.inference_engine.enable_ray_prometheus_stats=true``. It groups
throughput, latency, engine-queue, and KV-cache panels the same way
``VLLMMetricsScraper`` groups the curated wandb subset.

``build_dashboard`` returns the Grafana dashboard dict. Panels share one
timeseries template so a query is described by its title, PromQL, and legends
only. Run this module to print the JSON for import into Grafana:

    uv run --isolated --extra dev python -m skyrl.train.utils.vllm_dashboard
"""

import json
from dataclasses import dataclass, field
from typing import Any, Dict, List

DASHBOARD_TITLE = "SkyRL vLLM Dashboard"
DASHBOARD_UID = "skyrl-vllm"
# UID of the Prometheus data source Grafana queries. Override for a different
# instance; the pasted reference dashboard used "PBFA97CFB590B2093".
DEFAULT_DATASOURCE_UID = "PBFA97CFB590B2093"

# 24-column Grafana grid, two panels per content row.
_GRID_WIDTH = 24
_PANEL_WIDTH = _GRID_WIDTH // 2
_PANEL_HEIGHT = 8
_ROW_HEIGHT = 1

# refId letters Grafana assigns to a panel's targets in order.
_REF_IDS = "ABCDEFGH"


@dataclass
class Target:
    """One PromQL query rendered as a series in a panel."""

    expr: str
    legend: str


@dataclass
class Panel:
    """A timeseries panel and its queries."""

    title: str
    targets: List[Target]
    unit: str = ""


@dataclass
class Row:
    """A collapsible section header and the panels under it."""

    title: str
    panels: List[Panel] = field(default_factory=list)


def _mean(metric: str) -> str:
    """Rate-of-sum over rate-of-count, clamped so an idle window reads 0 not NaN."""
    return f"sum(rate({metric}_sum[5m]))\n" f"  / clamp_min(sum(rate({metric}_count[5m])), 1)"


def _quantile(metric: str, q: float) -> str:
    return f"histogram_quantile({q}, sum by (le)(rate({metric}_bucket[5m])))"


def _ratio(numerator: str, denominator: str) -> str:
    return f"sum(rate({numerator}[5m]))\n  / clamp_min(sum(rate({denominator}[5m])), 1)"


def _rollup(gauge: str) -> List[Target]:
    return [
        Target(f"avg({gauge})", "Mean"),
        Target(f"min({gauge})", "Min"),
        Target(f"max({gauge})", "Max"),
        Target(f"sum({gauge})", "Total"),
    ]


def dashboard_rows() -> List[Row]:
    """The panel layout. Each Row becomes a header plus its content panels."""
    return [
        Row(
            "Throughput",
            [
                Panel(
                    "Generated Tokens /s",
                    [Target("sum(rate(ray_vllm_generation_tokens_total[5m]))", "Total")],
                ),
                Panel(
                    "Prompt Tokens /s",
                    [Target("sum(rate(ray_vllm_request_prompt_tokens_sum[5m]))", "Total")],
                ),
            ],
        ),
        Row(
            "Latency",
            [
                Panel("TPOT (s)", [Target(_mean("ray_vllm_request_time_per_output_token_seconds"), "Mean")]),
                Panel("TTFT (s)", [Target(_mean("ray_vllm_time_to_first_token_seconds"), "Mean")]),
                Panel(
                    "Request Decode Time (s)",
                    [
                        Target(_mean("ray_vllm_request_decode_time_seconds"), "Mean"),
                        Target(_quantile("ray_vllm_request_decode_time_seconds", 0.99), "P99"),
                    ],
                ),
                Panel(
                    "Request Prefill Time (s)",
                    [
                        Target(_mean("ray_vllm_request_prefill_time_seconds"), "Mean"),
                        Target(_quantile("ray_vllm_request_prefill_time_seconds", 0.99), "P99"),
                    ],
                ),
                Panel(
                    "Prefill vs Decode (s)",
                    [
                        Target(_mean("ray_vllm_request_prefill_time_seconds"), "Prefill"),
                        Target(_mean("ray_vllm_request_decode_time_seconds"), "Decode"),
                    ],
                ),
            ],
        ),
        Row(
            "Engine",
            [
                Panel("num_requests_waiting", _rollup("ray_vllm_num_requests_waiting")),
                Panel(
                    "Total Preemptions",
                    [Target("sum(increase(ray_vllm_num_preemptions_total[5m]))", "Total Preemptions (in last 5 min)")],
                ),
                Panel("num_requests_running", _rollup("ray_vllm_num_requests_running")),
            ],
        ),
        Row(
            "KV Cache",
            [
                Panel(
                    "KV Cache Usage %",
                    [
                        Target("avg(ray_vllm_kv_cache_usage_perc)", "Avg KV Cache %"),
                        Target("min(ray_vllm_kv_cache_usage_perc)", "Min KV Cache %"),
                        Target("max(ray_vllm_kv_cache_usage_perc)", "Max KV Cache %"),
                    ],
                ),
                Panel(
                    "GPU Prefix Cache Hit Rate",
                    [Target(_ratio("ray_vllm_prefix_cache_hits_total", "ray_vllm_prefix_cache_queries_total"), "Mean")],
                ),
                Panel(
                    "External Prefix Cache Hit Rate",
                    [
                        Target(
                            _ratio(
                                "ray_vllm_external_prefix_cache_hits_total",
                                "ray_vllm_external_prefix_cache_queries_total",
                            ),
                            "Avg (last 5 min)",
                        )
                    ],
                ),
            ],
        ),
    ]


def _datasource(uid: str) -> Dict[str, str]:
    return {"type": "prometheus", "uid": uid}


def _timeseries_field_config() -> Dict[str, Any]:
    """The fieldConfig every timeseries panel shares in the reference dashboard."""
    return {
        "defaults": {
            "color": {"mode": "palette-classic"},
            "custom": {
                "axisBorderShow": False,
                "axisCenteredZero": False,
                "axisColorMode": "text",
                "axisLabel": "",
                "axisPlacement": "auto",
                "barAlignment": 0,
                "drawStyle": "line",
                "fillOpacity": 0,
                "gradientMode": "none",
                "hideFrom": {"legend": False, "tooltip": False, "viz": False},
                "insertNulls": False,
                "lineInterpolation": "linear",
                "lineWidth": 1,
                "pointSize": 5,
                "scaleDistribution": {"type": "linear"},
                "showPoints": "auto",
                "spanNulls": False,
                "stacking": {"group": "A", "mode": "none"},
                "thresholdsStyle": {"mode": "off"},
            },
            "mappings": [],
            "thresholds": {
                "mode": "absolute",
                "steps": [{"color": "green", "value": None}, {"color": "red", "value": 80}],
            },
            "unit": "",
        },
        "overrides": [],
    }


def _panel_json(panel: Panel, panel_id: int, x: int, y: int, uid: str) -> Dict[str, Any]:
    ds = _datasource(uid)
    field_config = _timeseries_field_config()
    field_config["defaults"]["unit"] = panel.unit
    targets = [
        {
            "datasource": ds,
            "editorMode": "code",
            "expr": t.expr,
            "instant": False,
            "legendFormat": t.legend,
            "range": True,
            "refId": _REF_IDS[i],
        }
        for i, t in enumerate(panel.targets)
    ]
    return {
        "datasource": ds,
        "fieldConfig": field_config,
        "gridPos": {"h": _PANEL_HEIGHT, "w": _PANEL_WIDTH, "x": x, "y": y},
        "id": panel_id,
        "options": {
            "legend": {"calcs": [], "displayMode": "list", "placement": "bottom", "showLegend": True},
            "tooltip": {"mode": "single", "sort": "none"},
        },
        "targets": targets,
        "title": panel.title,
        "type": "timeseries",
    }


def _row_json(title: str, panel_id: int, y: int) -> Dict[str, Any]:
    return {
        "collapsed": False,
        "gridPos": {"h": _ROW_HEIGHT, "w": _GRID_WIDTH, "x": 0, "y": y},
        "id": panel_id,
        "panels": [],
        "title": title,
        "type": "row",
    }


def build_panels(rows: List[Row], uid: str) -> List[Dict[str, Any]]:
    """Flatten rows into Grafana panels with sequential ids and auto grid layout."""
    panels: List[Dict[str, Any]] = []
    panel_id = 1
    y = 0
    for row in rows:
        panels.append(_row_json(row.title, panel_id, y))
        panel_id += 1
        y += _ROW_HEIGHT
        for col, panel in enumerate(row.panels):
            x = (col % 2) * _PANEL_WIDTH
            if col > 0 and col % 2 == 0:
                y += _PANEL_HEIGHT
            panels.append(_panel_json(panel, panel_id, x, y, uid))
            panel_id += 1
        y += _PANEL_HEIGHT
    return panels


def build_dashboard(datasource_uid: str = DEFAULT_DATASOURCE_UID) -> Dict[str, Any]:
    """Return the Grafana dashboard dict for the SkyRL vLLM metrics."""
    return {
        "annotations": {
            "list": [
                {
                    "builtIn": 1,
                    "datasource": {"type": "grafana", "uid": "-- Grafana --"},
                    "enable": True,
                    "hide": True,
                    "iconColor": "rgba(0, 211, 255, 1)",
                    "name": "Annotations & Alerts",
                    "type": "dashboard",
                }
            ]
        },
        "editable": True,
        "fiscalYearStartMonth": 0,
        "graphTooltip": 0,
        "id": None,
        "links": [],
        "panels": build_panels(dashboard_rows(), datasource_uid),
        "refresh": "",
        "schemaVersion": 39,
        "tags": [],
        "templating": {"list": []},
        "time": {"from": "now-2d", "to": "now"},
        "timepicker": {},
        "timezone": "browser",
        "title": DASHBOARD_TITLE,
        "uid": DASHBOARD_UID,
        "version": 1,
        "weekStart": "",
    }


def dashboard_json(datasource_uid: str = DEFAULT_DATASOURCE_UID) -> str:
    """Return the dashboard as the indented JSON Grafana import expects."""
    return json.dumps(build_dashboard(datasource_uid), indent=2) + "\n"


if __name__ == "__main__":
    print(dashboard_json(), end="")
