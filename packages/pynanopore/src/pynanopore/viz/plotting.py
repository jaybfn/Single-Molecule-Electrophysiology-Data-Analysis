"""Plotly visualization helpers (optional dependency)."""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy.ndimage import gaussian_filter1d


def _require_plotly():
    try:
        import plotly.graph_objects as go
    except ImportError as exc:
        raise ImportError(
            "plotly is required for visualization. Install with: pip install 'pynanopore[viz]'"
        ) from exc
    return go


class Plotting:
    """Plot ion-current traces and detected events."""

    @staticmethod
    def plot_data(
        data_time: NDArray[np.floating],
        data_chunk: NDArray[np.floating],
        events_data: list,
        sigma: float = 1.5,
        *,
        std_multiplier: float = 0.25,
        threshold_multiplier: float = 1.5,
    ) -> Any:
        go = _require_plotly()
        data_time_list = data_time.tolist()
        data_chunk_list = data_chunk.tolist()
        smoothed = gaussian_filter1d(data_chunk, sigma=sigma)

        fig = go.Figure()
        fig.add_trace(
            go.Scatter(x=data_time_list, y=smoothed, mode="lines", name="Smoothed Signal")
        )
        fig.add_trace(go.Scatter(x=data_time_list, y=data_chunk_list, mode="lines", name="Signal"))

        mean = float(np.mean(data_chunk))
        std_dev = float(np.std(data_chunk))
        fig.add_hline(y=mean, line=dict(color="red", width=2), name="Mean")
        fig.add_hline(
            y=mean - std_multiplier * std_dev,
            line=dict(color="blue", width=2, dash="dash"),
            name=f"{std_multiplier}x Std Dev",
        )
        fig.add_hline(
            y=mean - threshold_multiplier * std_dev,
            line=dict(color="green", width=2, dash="dot"),
            name=f"{threshold_multiplier}x Std Dev",
        )

        time_to_idx = {t: i for i, t in enumerate(data_time_list)}
        for event in events_data:
            if hasattr(event, "to_dict"):
                event = event.to_dict()
            start_time = event["start_time"]
            end_time = event["end_time"]
            if start_time in time_to_idx and end_time in time_to_idx:
                fig.add_trace(
                    go.Scatter(
                        x=[start_time, end_time],
                        y=[
                            data_chunk_list[time_to_idx[start_time]],
                            data_chunk_list[time_to_idx[end_time]],
                        ],
                        mode="markers",
                        marker=dict(color="white", size=10),
                        showlegend=False,
                    )
                )

        fig.update_layout(
            title="Data with Events",
            xaxis_title="Time (s)",
            yaxis_title="Current (pA)",
        )
        return fig

    @staticmethod
    def plot_data_series(
        data_time: NDArray[np.floating],
        data_chunk: NDArray[np.floating],
        sigma: float = 1.5,
        *,
        std_multiplier: float = 0.25,
        threshold_multiplier: float = 1.5,
    ) -> Any:
        go = _require_plotly()
        data_time_list = data_time.tolist()
        data_chunk_list = data_chunk.tolist()
        smoothed = gaussian_filter1d(data_chunk, sigma=sigma)

        fig = go.Figure()
        fig.add_trace(
            go.Scatter(x=data_time_list, y=smoothed, mode="lines", name="Smoothed Signal")
        )
        fig.add_trace(go.Scatter(x=data_time_list, y=data_chunk_list, mode="lines", name="Signal"))

        mean = float(np.mean(data_chunk))
        std_dev = float(np.std(data_chunk))
        fig.add_hline(y=mean, line=dict(color="red", width=2), name="Mean")
        fig.add_hline(
            y=mean - std_multiplier * std_dev,
            line=dict(color="blue", width=2, dash="dash"),
            name=f"{std_multiplier}x Std Dev",
        )
        fig.add_hline(
            y=mean - threshold_multiplier * std_dev,
            line=dict(color="green", width=2, dash="dot"),
            name=f"{threshold_multiplier}x Std Dev",
        )
        fig.update_layout(
            title="Ion Current Trace",
            xaxis_title="Time (s)",
            yaxis_title="Current (pA)",
        )
        return fig


def plot_pulse_shape(
    time: NDArray[np.floating],
    current: NDArray[np.floating],
    pulse,
    *,
    show_smoothed: bool = False,
    sigma: float = 1.5,
    title: str = "Ion current with pulse-shape idealization",
) -> Any:
    """
    Plot raw current with idealized rectangular pulses (screenshot-style).

    - Gray/blue raw trace
    - Black stepwise idealization
    - Red markers at rising edges (event start)
    - Blue markers at falling edges (event end)
    - Horizontal guides for open-pore and mean blocked levels
    """
    go = _require_plotly()
    fig = go.Figure()

    # Plotly/Pydantic JSON responses require plain Python lists, not ndarrays
    time_list = np.asarray(time, dtype=float).tolist()
    current_list = np.asarray(current, dtype=float).tolist()
    pulse_time = np.asarray(pulse.time, dtype=float).tolist()
    pulse_ideal = np.asarray(pulse.idealized, dtype=float).tolist()

    fig.add_trace(
        go.Scatter(
            x=time_list,
            y=current_list,
            mode="lines",
            name="Raw signal",
            line=dict(color="rgba(80,80,80,0.7)", width=1),
        )
    )
    if show_smoothed:
        fig.add_trace(
            go.Scatter(
                x=time_list,
                y=gaussian_filter1d(current, sigma=sigma).tolist(),
                mode="lines",
                name="Smoothed",
                line=dict(color="rgba(120,120,180,0.8)", width=1),
            )
        )

    fig.add_trace(
        go.Scatter(
            x=pulse_time,
            y=pulse_ideal,
            mode="lines",
            name="Pulse shape",
            line=dict(color="black", width=2, shape="hv"),
        )
    )

    rt, ry = pulse.rising_edges
    ft, fy = pulse.falling_edges
    if len(rt):
        fig.add_trace(
            go.Scatter(
                x=np.asarray(rt, dtype=float).tolist(),
                y=np.asarray(ry, dtype=float).tolist(),
                mode="markers",
                name="Rising edge",
                marker=dict(color="red", size=9, symbol="line-ns-open"),
            )
        )
    if len(ft):
        fig.add_trace(
            go.Scatter(
                x=np.asarray(ft, dtype=float).tolist(),
                y=np.asarray(fy, dtype=float).tolist(),
                mode="markers",
                name="Falling edge",
                marker=dict(color="blue", size=9, symbol="line-ns-open"),
            )
        )

    if pulse.events:
        open_lvl = float(np.median([e.i0 for e in pulse.events]))
        blocked_lvl = float(np.median([e.blockade_mean for e in pulse.events]))
        fig.add_hline(
            y=open_lvl,
            line=dict(color="lightskyblue", width=1.5, dash="dash"),
            annotation_text="Open pore (I0)",
        )
        fig.add_hline(
            y=blocked_lvl,
            line=dict(color="lightskyblue", width=1.5, dash="dot"),
            annotation_text="Blocked level",
        )

    fig.update_layout(
        title=title,
        xaxis_title="Time (s)",
        yaxis_title="Current (pA)",
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    return fig


def plot_dwelltime_histogram(fitter, fit_type=None) -> Any:
    go = _require_plotly()
    fig = go.Figure()
    fig.add_trace(go.Bar(x=fitter.bin_centers, y=fitter.hist, name="Histogram"))
    if fit_type is not None:
        fig.add_trace(
            go.Scatter(
                x=fitter.bin_centers,
                y=fitter.fitted_curve(fit_type),
                mode="lines",
                name=f"{str(fit_type).title()} Exponential",
            )
        )
        title = "Histogram with Exponential Fit"
    else:
        title = "Dwell Time Histogram"
    fig.update_layout(title=title, xaxis_title="Dwell Time (s)", yaxis_title="Density")
    return fig


def plot_psd(frequencies, power_spectrum, *, fitter=None, max_freq: float = 50000.0) -> Any:
    go = _require_plotly()
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=frequencies,
            y=power_spectrum,
            mode="lines",
            line=dict(color="rgb(0,0,255)"),
            name="PSD",
        )
    )
    if fitter is not None and fitter.filtered_frequencies is not None:
        fig.add_trace(
            go.Scatter(
                x=fitter.filtered_frequencies,
                y=fitter.fitted_curve(),
                mode="lines",
                line=dict(color="rgb(0,225,0)"),
                name="Lorentzian fit",
            )
        )
        x_end = min(max_freq, fitter.max_frequency)
    else:
        x_end = max_freq

    fig.update_layout(
        title="Power Spectrum",
        xaxis=dict(
            type="log",
            title="Frequency (Hz)",
            range=[np.log10(max(float(frequencies[0]), 1e-3)), np.log10(x_end)],
        ),
        yaxis=dict(type="log", title="Power Spectrum (pA^2/Hz)"),
        hovermode="closest",
        template="plotly_white",
    )
    return fig
