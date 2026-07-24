"""Build the SkyRL training Grafana dashboard from Prometheus queries.

The dashboard is ordered outcome first, root cause last, so a training run that
starves the trainer reads top to bottom: is the trainer waiting, is the rollout
buffer draining, and why (trajectory time, turn growth, engine saturation, KV
pressure, preemptions).

Trainer-side panels query the SkyRL metrics that reach Ray's agents through the
wandb-to-Prometheus dual write; engine-side panels query the ``ray_vllm_*``
series that vLLM's RayPrometheusStatLogger exports. Engine panels draw one thin
line per replica with a bold mean overlay so per-replica spread is visible.

``build_dashboard`` returns the Grafana dashboard dict. Run the module to print
the JSON for import into Grafana:

    uv run --isolated --extra dev python -m skyrl.train.utils.training_dashboard
"""

import json
from dataclasses import dataclass, field
from typing import Any, Dict, List

DASHBOARD_TITLE = "SkyRL Training Dashboard"
DASHBOARD_UID = "skyrl-training"
# UID of the Prometheus data source Grafana queries. Override for a different
# instance; the pasted reference dashboard used "PBFA97CFB590B2093".
DEFAULT_DATASOURCE_UID = "PBFA97CFB590B2093"

# Ray metrics label that distinguishes one vLLM engine replica from another.
# Per-engine panels group by this to draw a line per replica. Verify against a
# live scrape; Ray tags custom metrics with WorkerId and vLLM adds model_name.
_ENGINE_LABEL = "WorkerId"

# SkyRL trainer-side series names. A wandb key ``a/b`` dual-writes to
# ``ray_skyrl_a_b``; direct gauges register as ``ray_skyrl_<name>``. The phase
# and buffer gauges need #1923/#1924 on main; the trajectory-time and
# tokens-per-turn keys need the #1931 wandb dual write.
_M_PHASE = "ray_skyrl_training_phase"  # {phase="..."}, active phase is 1.
_M_BUFFER_QSIZE = "ray_skyrl_gen_buffer_qsize"
_M_BUFFER_MAXSIZE = "ray_skyrl_gen_buffer_maxsize"
_M_MINI_BATCH = "ray_skyrl_mini_batch_size"
_M_TOKENS_PER_TURN = "ray_skyrl_generate_tokens_per_turn"  # append _mean/_max.
# Trajectory time split, pre-aggregated per band and stat. env_setup is a
# not-yet-merged follow-up, so only these bands exist on main.
_TRAJ_TIME = "ray_skyrl_generate_trajectory_time_{band}_{stat}"
_TRAJ_BANDS = ("llm", "env", "other")

_GRID_WIDTH = 24
_FULL = 24
_HALF = 12
_PANEL_HEIGHT = 8
_ROW_HEIGHT = 1

# refId letters Grafana assigns to a panel's targets in order.
_REF_IDS = "ABCDEFGHIJKLMNOP"


@dataclass
class Target:
    """One PromQL query rendered as a series in a panel."""

    expr: str
    legend: str
    axis: str = "left"  # "left" or "right" for a second Y axis.
    emphasize: bool = False  # Draw thick, for a mean overlaid on per-replica lines.
    dashed: bool = False  # Draw dashed, for a constant reference line.


@dataclass
class Panel:
    """A timeseries panel and its queries."""

    title: str
    targets: List[Target]
    unit: str = ""
    right_unit: str = ""
    stack: bool = False
    draw_style: str = "line"  # "line" or "bars".
    fill_opacity: int = 0
    width: int = _HALF


@dataclass
class Row:
    """A section header and the panels under it."""

    title: str
    panels: List[Panel] = field(default_factory=list)


def _mean(metric: str) -> str:
    """Rate-of-sum over rate-of-count, clamped so an idle window reads 0 not NaN."""
    return f"sum(rate({metric}_sum[5m]))\n  / clamp_min(sum(rate({metric}_count[5m])), 1)"


def _mean_by_engine(metric: str) -> str:
    """Per-replica mean latency, one series per engine."""
    return f"sum by ({_ENGINE_LABEL})(rate({metric}_sum[5m]))\n  / clamp_min(sum by ({_ENGINE_LABEL})(rate({metric}_count[5m])), 1)"


def _quantile(metric: str, q: float) -> str:
    return f"histogram_quantile({q}, sum by (le)(rate({metric}_bucket[5m])))"


def _ratio(hits: str, queries: str) -> str:
    return f"sum(rate({hits}[5m]))\n  / clamp_min(sum(rate({queries}[5m])), 1)"


def _ratio_by_engine(hits: str, queries: str) -> str:
    return (
        f"sum by ({_ENGINE_LABEL})(rate({hits}[5m]))\n"
        f"  / clamp_min(sum by ({_ENGINE_LABEL})(rate({queries}[5m])), 1)"
    )


def _combined_ratio(hits: List[str], queries: List[str]) -> str:
    """Hit rate over several counters. rate() applies per counter, then they sum."""
    h = " + ".join(f"sum(rate({m}[5m]))" for m in hits)
    q = " + ".join(f"sum(rate({m}[5m]))" for m in queries)
    return f"({h})\n  / clamp_min({q}, 1)"


def _engine_spread(gauge: str) -> List[Target]:
    """One thin line per replica plus a bold mean, for a gauge."""
    return [
        Target(f"{gauge}", "{{" + _ENGINE_LABEL + "}}"),
        Target(f"avg({gauge})", "Mean", emphasize=True),
    ]


def _starvation_row() -> Row:
    return Row(
        "Trainer starvation",
        [
            Panel(
                "Training Phase",
                [Target(f"max by (phase)({_M_PHASE})", "{{phase}}")],
                unit="none",
                stack=True,
                fill_opacity=60,
                width=_FULL,
            ),
            Panel(
                "Rollout Buffer Depth (groups)",
                [
                    Target(f"max({_M_BUFFER_QSIZE})", "Groups ready to consume"),
                    Target(f"max({_M_MINI_BATCH})", "Mini-batch needed", dashed=True),
                    Target(f"max({_M_BUFFER_MAXSIZE})", "Buffer cap", dashed=True),
                ],
                unit="none",
                fill_opacity=20,
                width=_FULL,
            ),
        ],
    )


def _trajectory_row() -> Row:
    bands = [Target(f"avg({_TRAJ_TIME.format(band=b, stat='mean')})", b) for b in _TRAJ_BANDS]
    return Row(
        "Rollout trajectory",
        [
            Panel(
                "Trajectory Time by Phase (s)",
                bands,
                unit="s",
                stack=True,
                fill_opacity=60,
                width=_FULL,
            ),
            Panel(
                "Trajectory Time (s)",
                [
                    Target(f"avg({_TRAJ_TIME.format(band='completion', stat='mean')})", "Mean"),
                    Target(f"max({_TRAJ_TIME.format(band='completion', stat='p90')})", "P90"),
                    Target(f"max({_TRAJ_TIME.format(band='completion', stat='max')})", "Max", emphasize=True),
                ],
                unit="s",
            ),
            Panel(
                "Tokens per Turn",
                [
                    Target(f"avg({_M_TOKENS_PER_TURN}_mean)", "Mean"),
                    Target(f"max({_M_TOKENS_PER_TURN}_max)", "Max"),
                ],
                unit="none",
            ),
        ],
    )


def _throughput_row() -> Row:
    return Row(
        "Throughput",
        [
            Panel(
                "Generated Tokens /s",
                [Target("sum(rate(ray_vllm_generation_tokens_total[5m]))", "Total")],
                unit="none",
            ),
            Panel(
                "Prompt Tokens /s",
                [Target("sum(rate(ray_vllm_request_prompt_tokens_sum[5m]))", "Total")],
                unit="none",
            ),
            Panel(
                "Tokens per Request",
                [
                    Target(_mean("ray_vllm_request_prompt_tokens"), "Prompt (mean)"),
                    Target(_mean("ray_vllm_request_generation_tokens"), "Generated (mean)", axis="right"),
                ],
                unit="none",
                right_unit="none",
            ),
        ],
    )


def _latency_row() -> Row:
    return Row(
        "Latency",
        [
            Panel(
                "TPOT (s) per engine",
                [
                    Target(
                        _mean_by_engine("ray_vllm_request_time_per_output_token_seconds"), "{{" + _ENGINE_LABEL + "}}"
                    ),
                    Target(_mean("ray_vllm_request_time_per_output_token_seconds"), "Mean", emphasize=True),
                ],
                unit="s",
            ),
            Panel(
                "TTFT (s) per engine",
                [
                    Target(_mean_by_engine("ray_vllm_time_to_first_token_seconds"), "{{" + _ENGINE_LABEL + "}}"),
                    Target(_mean("ray_vllm_time_to_first_token_seconds"), "Mean", emphasize=True),
                ],
                unit="s",
            ),
            Panel(
                "Prefill vs Decode (s)",
                [
                    Target(_mean("ray_vllm_request_prefill_time_seconds"), "Prefill mean"),
                    Target(_quantile("ray_vllm_request_prefill_time_seconds", 0.99), "Prefill p99"),
                    Target(_mean("ray_vllm_request_decode_time_seconds"), "Decode mean"),
                    Target(_quantile("ray_vllm_request_decode_time_seconds", 0.99), "Decode p99"),
                ],
                unit="s",
            ),
            Panel(
                "Request Inference Time (s)",
                [
                    Target(_mean("ray_vllm_request_inference_time_seconds"), "Mean"),
                    Target(_quantile("ray_vllm_request_inference_time_seconds", 0.99), "P99 (tail)", emphasize=True),
                ],
                unit="s",
            ),
        ],
    )


def _engine_row() -> Row:
    return Row(
        "Engine saturation",
        [
            Panel(
                "Scheduler",
                [
                    Target("avg(ray_vllm_num_requests_running)", "Running (mean)"),
                    Target("avg(ray_vllm_num_requests_waiting)", "Waiting (mean)"),
                    Target("sum(rate(ray_vllm_num_preemptions_total[5m]))", "Preemptions /s", axis="right"),
                ],
                unit="none",
                right_unit="none",
                width=_FULL,
            ),
            Panel(
                "Finished Reason /s",
                [Target("sum by (finished_reason)(rate(ray_vllm_request_success_total[5m]))", "{{finished_reason}}")],
                unit="none",
                stack=True,
                fill_opacity=60,
            ),
            Panel(
                "Iteration Tokens (per step)",
                [Target(_mean("ray_vllm_iteration_tokens_total"), "Mean")],
                unit="none",
            ),
        ],
    )


def _kv_cache_row() -> Row:
    return Row(
        "KV Cache",
        [
            Panel(
                "GPU KV Cache Usage % per engine", _engine_spread("ray_vllm_kv_cache_usage_perc"), unit="percentunit"
            ),
            Panel(
                "Prefix Cache Reuse",
                [
                    Target(
                        _ratio_by_engine("ray_vllm_prefix_cache_hits_total", "ray_vllm_prefix_cache_queries_total"),
                        "{{" + _ENGINE_LABEL + "}}",
                    ),
                    Target(
                        _ratio("ray_vllm_prefix_cache_hits_total", "ray_vllm_prefix_cache_queries_total"),
                        "GPU-local (mean)",
                        emphasize=True,
                    ),
                    Target(
                        _combined_ratio(
                            ["ray_vllm_prefix_cache_hits_total", "ray_vllm_external_prefix_cache_hits_total"],
                            ["ray_vllm_prefix_cache_queries_total", "ray_vllm_external_prefix_cache_queries_total"],
                        ),
                        "Combined (mean)",
                        emphasize=True,
                    ),
                ],
                unit="percentunit",
            ),
        ],
    )


def dashboard_rows() -> List[Row]:
    """Panel layout, ordered outcome first (trainer) then root cause (engine)."""
    return [
        _starvation_row(),
        _trajectory_row(),
        _throughput_row(),
        _latency_row(),
        _engine_row(),
        _kv_cache_row(),
    ]


def _datasource(uid: str) -> Dict[str, str]:
    return {"type": "prometheus", "uid": uid}


def _field_config(panel: Panel) -> Dict[str, Any]:
    """The fieldConfig shared by timeseries panels, adjusted for this panel."""
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
                "drawStyle": "bars" if panel.draw_style == "bars" else "line",
                "fillOpacity": panel.fill_opacity,
                "gradientMode": "none",
                "hideFrom": {"legend": False, "tooltip": False, "viz": False},
                "insertNulls": False,
                "lineInterpolation": "linear",
                "lineWidth": 1,
                "pointSize": 5,
                "scaleDistribution": {"type": "linear"},
                "showPoints": "auto",
                "spanNulls": False,
                "stacking": {"group": "A", "mode": "normal" if panel.stack else "none"},
                "thresholdsStyle": {"mode": "off"},
            },
            "mappings": [],
            "thresholds": {"mode": "absolute", "steps": [{"color": "green", "value": None}]},
            "unit": panel.unit,
        },
        "overrides": _overrides(panel),
    }


def _overrides(panel: Panel) -> List[Dict[str, Any]]:
    """Per-series overrides for bold means, dashed reference lines, right axis."""
    overrides: List[Dict[str, Any]] = []
    for target in panel.targets:
        props: List[Dict[str, Any]] = []
        if target.emphasize:
            props.append({"id": "custom.lineWidth", "value": 3})
        if target.dashed:
            props.append({"id": "custom.lineStyle", "value": {"fill": "dash", "dash": [10, 10]}})
        if target.axis == "right":
            props.append({"id": "custom.axisPlacement", "value": "right"})
            if panel.right_unit:
                props.append({"id": "unit", "value": panel.right_unit})
        if props:
            overrides.append({"matcher": {"id": "byName", "options": target.legend}, "properties": props})
    return overrides


def _panel_json(panel: Panel, panel_id: int, x: int, y: int, uid: str) -> Dict[str, Any]:
    ds = _datasource(uid)
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
        "fieldConfig": _field_config(panel),
        "gridPos": {"h": _PANEL_HEIGHT, "w": panel.width, "x": x, "y": y},
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
    """Flatten rows into Grafana panels, packing panels into 24-column rows."""
    panels: List[Dict[str, Any]] = []
    panel_id = 1
    y = 0
    for row in rows:
        panels.append(_row_json(row.title, panel_id, y))
        panel_id += 1
        y += _ROW_HEIGHT
        x = 0
        for panel in row.panels:
            if x + panel.width > _GRID_WIDTH:
                x = 0
                y += _PANEL_HEIGHT
            panels.append(_panel_json(panel, panel_id, x, y, uid))
            panel_id += 1
            x += panel.width
        y += _PANEL_HEIGHT
    return panels


def build_dashboard(datasource_uid: str = DEFAULT_DATASOURCE_UID) -> Dict[str, Any]:
    """Return the Grafana dashboard dict for the SkyRL training run."""
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
