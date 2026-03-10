"""
KPI dashboard module for lap time simulation results.

Provides Plotly-based chart builders and KPI summary tables that consume
SimulationResult objects.
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from ..simulation.lap_time_solver import SimulationResult


def build_kpi_dataframe(results: List[SimulationResult]) -> pd.DataFrame:
    """Build a KPI summary DataFrame from one or more SimulationResult objects."""
    rows = []
    for r in results:
        rows.append({
            "Label":            f"{r.mode.name} | {r.setup_name}",
            "Lap Time (s)":     round(r.lap_time, 3),
            "V avg (km/h)":     round(r.avg_speed_kmh, 1),
            "V max (km/h)":     round(r.max_speed_kmh, 1),
            "Peak lat g":       round(r.peak_lat_g, 3),
            "Peak accel g":     round(r.peak_accel_g, 3),
            "Peak brake g":     round(r.peak_brake_g, 3),
            "WOT (%)": round(r.time_wot_pct, 1),
            "Braking (%)": round(r.time_braking_pct, 1),
            "Fuel (L)":         round(r.fuel_total_l, 3),
            "T tyre final (C)": round(r.final_tyre_temp_c, 1),
            "P tyre hot (bar)": round(r.final_tyre_pressure_bar, 2),
        })
    return pd.DataFrame(rows).set_index("Label")


def compare_lap_times(results: List[SimulationResult]) -> pd.DataFrame:
    """Build a compact lap time comparison table with delta to fastest run."""
    base_time = min(r.lap_time for r in results)
    rows = []
    for r in results:
        delta = r.lap_time - base_time
        rows.append({
            "Run": f"{r.mode.name} | {r.setup_name}",
            "Lap (s)": round(r.lap_time, 3),
            "Delta (s)": f"+{delta:.3f}" if delta > 0 else "REF",
            "V avg (km/h)": round(r.avg_speed_kmh, 1),
        })
    return pd.DataFrame(rows)


def plot_gg_diagram(
    result: SimulationResult,
    title: Optional[str] = None,
    height: int = 500,
) -> go.Figure:
    """Plot GG (traction circle) diagram: ax_long vs ay_lat."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=result.ay_lat_g,
        y=result.ax_long_g,
        mode="markers",
        marker=dict(
            size=4,
            color=result.v_kmh,
            colorscale="Viridis",
            showscale=True,
            colorbar=dict(title="Speed (km/h)"),
        ),
        name="GG",
        hovertemplate="ay: %{x:.3f} g<br>ax: %{y:.3f} g<br>v: %{marker.color:.1f} km/h<extra></extra>",
    ))
    fig.update_layout(
        title=title or f"GG Diagram — {result.setup_name}",
        xaxis_title="Lateral acceleration (g)",
        yaxis_title="Longitudinal acceleration (g)",
        height=height,
        xaxis=dict(zeroline=True),
        yaxis=dict(zeroline=True),
    )
    return fig


def plot_speed_vs_distance(
    results: List[SimulationResult],
    height: int = 350,
) -> go.Figure:
    """Overlay speed traces for multiple simulation results."""
    fig = go.Figure()
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]
    for i, r in enumerate(results):
        fig.add_trace(go.Scatter(
            x=r.distance, y=r.v_kmh, mode="lines",
            name=f"{r.mode.name} | {r.setup_name} ({r.lap_time:.2f}s)",
            line=dict(width=2, color=colors[i % len(colors)]),
        ))
    fig.update_layout(
        title="Speed vs Distance",
        xaxis_title="Distance (m)",
        yaxis_title="Speed (km/h)",
        height=height,
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    return fig


def plot_channels_vs_distance(
    result: SimulationResult,
    channels: Optional[List[str]] = None,
    height: int = 600,
) -> go.Figure:
    """Plot multiple telemetry channels vs distance in stacked subplots."""
    _CHANNEL_META = {
        "v_kmh":            ("Speed",         "km/h",  "#1f77b4"),
        "throttle_pct":     ("Throttle",       "%",      "#2ca02c"),
        "brake_pct":        ("Brake",          "%",      "#d62728"),
        "steering_deg":     ("Steering",       "deg",   "#ff7f0e"),
        "gear":             ("Gear",           "-",      "#9467bd"),
        "rpm":              ("RPM",            "rpm",   "#8c564b"),
        "ax_long_g":        ("ax long",        "g",      "#e377c2"),
        "ay_lat_g":         ("ay lat",         "g",      "#17becf"),
        "temp_tyre_c":      ("Tyre Temp",      "\u00b0C",  "#bcbd22"),
        "tyre_pressure_bar":("Tyre Press",     "bar",   "#7f7f7f"),
        "fuel_used_l":      ("Fuel",           "L",      "#aec7e8"),
    }
    if channels is None:
        channels = ["v_kmh", "throttle_pct", "brake_pct", "gear", "rpm", "temp_tyre_c"]

    n_ch = len(channels)
    fig = make_subplots(
        rows=n_ch, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        subplot_titles=[_CHANNEL_META.get(c, (c,))[0] for c in channels],
    )
    for row, ch in enumerate(channels, start=1):
        data = getattr(result, ch, None)
        if data is None:
            continue
        label, unit, color = _CHANNEL_META.get(ch, (ch, "-", "gray"))
        fig.add_trace(
            go.Scatter(x=result.distance, y=data, mode="lines",
                       name=label, line=dict(color=color, width=1.5), showlegend=False),
            row=row, col=1,
        )
        fig.update_yaxes(title_text=unit, row=row, col=1)
    fig.update_xaxes(title_text="Distance (m)", row=n_ch, col=1)
    fig.update_layout(
        title=f"Telemetry — {result.setup_name} | Lap {result.lap_time:.3f}s",
        height=max(height, n_ch * 120),
    )
    return fig
