"""
Track plotter module for lap time simulation.

Provides Plotly-based functions to visualise circuit geometry and
overlay simulation channel data as a colour map on the track map.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import plotly.graph_objects as go

from ..simulation.lap_time_solver import SimulationResult


def _color_channel_on_track(
    x: np.ndarray,
    y: np.ndarray,
    channel: np.ndarray,
    colorscale: str = "Viridis",
    name: str = "Channel",
    channel_unit: str = "-",
) -> go.Scatter:
    """Colour track centreline by a scalar channel value."""
    return go.Scatter(
        x=x, y=y, mode="markers",
        marker=dict(
            size=4, color=channel, colorscale=colorscale,
            showscale=True, colorbar=dict(title=f"{name} ({channel_unit})"),
        ),
        name=name,
        hovertemplate=(
            f"{name}: %{{marker.color:.2f}} {channel_unit}<br>"
            "x: %{x:.1f} m<br>y: %{y:.1f} m<extra></extra>"
        ),
    )


def plot_track_map(
    circuit,
    result: Optional[SimulationResult] = None,
    color_channel: str = "v_kmh",
    show_boundaries: bool = True,
    title: Optional[str] = None,
    height: int = 600,
) -> go.Figure:
    """Plot circuit map with optional telemetry colour overlay."""
    x_c =  -(circuit.centerline_y - circuit.centerline_y[0])
    y_c =    circuit.centerline_x  - circuit.centerline_x[0]

    fig = go.Figure()

    if show_boundaries and hasattr(circuit, 'left_boundary_x'):
        lx = -(circuit.left_boundary_y  - circuit.centerline_y[0])
        ly =   circuit.left_boundary_x  - circuit.centerline_x[0]
        rx = -(circuit.right_boundary_y - circuit.centerline_y[0])
        ry =   circuit.right_boundary_x - circuit.centerline_x[0]
        fig.add_trace(go.Scatter(x=lx, y=ly, mode="lines",
            line=dict(color="rgba(0,180,0,0.4)", width=1, dash="dot"),
            name="Left boundary"))
        fig.add_trace(go.Scatter(x=rx, y=ry, mode="lines",
            line=dict(color="rgba(220,0,0,0.4)", width=1, dash="dot"),
            name="Right boundary"))

    if result is not None:
        channel_data = getattr(result, color_channel, None)
        _UNITS  = {"v_kmh": "km/h", "ax_long_g": "g", "ay_lat_g": "g",
                   "gear": "-", "throttle_pct": "%", "brake_pct": "%",
                   "rpm": "rpm", "temp_tyre_c": "\u00b0C"}
        _SCALES = {"v_kmh": "Viridis", "ax_long_g": "RdYlGn",
                   "ay_lat_g": "RdBu", "gear": "Plasma",
                   "throttle_pct": "Greens", "brake_pct": "Reds",
                   "temp_tyre_c": "Hot"}
        if channel_data is not None and len(channel_data) == len(x_c):
            fig.add_trace(_color_channel_on_track(
                x=x_c, y=y_c, channel=channel_data,
                colorscale=_SCALES.get(color_channel, "Viridis"),
                name=color_channel,
                channel_unit=_UNITS.get(color_channel, "-"),
            ))
        else:
            fig.add_trace(go.Scatter(x=x_c, y=y_c, mode="lines",
                line=dict(color="steelblue", width=2), name="Centreline"))
    else:
        fig.add_trace(go.Scatter(x=x_c, y=y_c, mode="lines",
            line=dict(color="steelblue", width=2), name="Centreline"))

    fig.add_trace(go.Scatter(x=[x_c[0]], y=[y_c[0]], mode="markers",
        marker=dict(symbol="x", size=14, color="orange"), name="Start/Finish"))
    fig.update_layout(
        title=title or "Track Map", xaxis_title="x (m)", yaxis_title="y (m)",
        height=height, yaxis=dict(scaleanchor="x", scaleratio=1),
    )
    return fig


def plot_curvature_profile(
    circuit,
    result: Optional[SimulationResult] = None,
    height: int = 300,
) -> go.Figure:
    """Plot track curvature vs cumulative distance."""
    x, y = circuit.centerline_x, circuit.centerline_y
    ds = np.zeros(len(x))
    ds[1:] = np.sqrt(np.diff(x)**2 + np.diff(y)**2)
    s = np.cumsum(ds)
    dx, dy   = np.gradient(x), np.gradient(y)
    ddx, ddy = np.gradient(dx), np.gradient(dy)
    kappa = (dx * ddy - dy * ddx) / (dx**2 + dy**2 + 1e-9)**1.5

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=s, y=kappa, mode="lines", name="Curvature (1/m)",
        line=dict(color="steelblue", width=1.5),
        fill="tozeroy", fillcolor="rgba(70,130,180,0.15)"))

    if result is not None:
        fig.add_trace(go.Scatter(
            x=result.distance, y=result.v_kmh / result.v_kmh.max(),
            mode="lines", name="Speed (norm.)",
            line=dict(color="orange", width=1.5, dash="dot"), yaxis="y2",
        ))
        fig.update_layout(yaxis2=dict(
            title="Speed (norm.)", overlaying="y", side="right", range=[0, 1.1]
        ))

    fig.update_layout(
        title="Curvature Profile", xaxis_title="Distance (m)",
        yaxis_title="Curvature (1/m)", height=height,
    )
    return fig
