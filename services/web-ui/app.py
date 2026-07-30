"""Streamlit UI that calls the Pynanopore gateway API."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import httpx
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st

GATEWAY_URL = os.getenv("GATEWAY_URL", "http://localhost:8000")
TIMEOUT = float(os.getenv("HTTP_TIMEOUT_S", "120"))


def _example_csv_path() -> Path:
    """Resolve example CSV for Docker (/app/data) or local repo checkout."""
    if env := os.getenv("EXAMPLE_CSV_PATH"):
        return Path(env)
    here = Path(__file__).resolve()
    for parent in (here.parent, *here.parents):
        candidate = parent / "data" / "test.csv"
        if candidate.is_file():
            return candidate
    return here.parent / "data" / "test.csv"


EXAMPLE_CSV_PATH = _example_csv_path()


class MemoryUpload:
    """File-like upload wrapper for example datasets."""

    def __init__(self, name: str, data: bytes):
        self.name = name
        self._data = data

    def getvalue(self) -> bytes:
        return self._data


def gateway_health() -> dict:
    with httpx.Client(timeout=5.0) as client:
        resp = client.get(f"{GATEWAY_URL}/health")
        resp.raise_for_status()
        return resp.json()


def preview_file(uploaded, *, max_points: int = 20000) -> dict:
    files = {"file": (uploaded.name, uploaded.getvalue(), "application/octet-stream")}
    with httpx.Client(timeout=TIMEOUT) as client:
        resp = client.post(
            f"{GATEWAY_URL}/v1/preview",
            files=files,
            params={"max_points": max_points},
        )
        if resp.status_code >= 400:
            raise RuntimeError(resp.text)
        return resp.json()


def detect_file(uploaded, **params) -> dict:
    files = {"file": (uploaded.name, uploaded.getvalue(), "application/octet-stream")}
    with httpx.Client(timeout=TIMEOUT) as client:
        resp = client.post(f"{GATEWAY_URL}/v1/detect", files=files, params=params)
        if resp.status_code >= 400:
            raise RuntimeError(resp.text)
        return resp.json()


def dwelltime(events: list[dict], **params) -> dict:
    payload = {"events": events, **params}
    with httpx.Client(timeout=TIMEOUT) as client:
        resp = client.post(f"{GATEWAY_URL}/v1/dwelltime", json=payload)
        if resp.status_code >= 400:
            raise RuntimeError(resp.text)
        return resp.json()


def psd_upload(uploaded, **params) -> dict:
    files = {"file": (uploaded.name, uploaded.getvalue(), "application/octet-stream")}
    with httpx.Client(timeout=TIMEOUT) as client:
        resp = client.post(f"{GATEWAY_URL}/v1/psd/upload", files=files, params=params)
        if resp.status_code >= 400:
            raise RuntimeError(resp.text)
        return resp.json()


def psd_from_preview(current: list[float], fs: float, **params) -> dict:
    payload = {"current": current, "fs": fs, **params}
    with httpx.Client(timeout=TIMEOUT) as client:
        resp = client.post(f"{GATEWAY_URL}/v1/psd", json=payload)
        if resp.status_code >= 400:
            raise RuntimeError(resp.text)
        return resp.json()


def render_plotly(plot_dict: dict) -> go.Figure:
    fig = pio.from_json(json.dumps(plot_dict))
    st.plotly_chart(fig, use_container_width=True)
    return fig


def download_figure(fig: go.Figure, stem: str) -> None:
    c1, c2, c3 = st.columns(3)
    html = fig.to_html(include_plotlyjs="cdn", full_html=True)
    c1.download_button(
        "Download plot HTML",
        data=html,
        file_name=f"{stem}.html",
        mime="text/html",
        key=f"dl-html-{stem}",
    )
    c2.download_button(
        "Download plot JSON",
        data=fig.to_json(),
        file_name=f"{stem}.plotly.json",
        mime="application/json",
        key=f"dl-json-{stem}",
    )
    try:
        png = fig.to_image(format="png", scale=2)
        c3.download_button(
            "Download plot PNG",
            data=png,
            file_name=f"{stem}.png",
            mime="image/png",
            key=f"dl-png-{stem}",
        )
    except Exception:  # noqa: BLE001
        c3.caption("PNG needs `kaleido` (optional).")


def overview_figure(
    time: list[float],
    current: list[float],
    t_start: float,
    t_end: float,
) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=time,
            y=current,
            mode="lines",
            name="Full trace",
            line=dict(color="rgba(80,80,80,0.75)", width=1),
        )
    )
    fig.add_vrect(
        x0=t_start,
        x1=t_end,
        fillcolor="rgba(30,136,229,0.18)",
        line_width=0,
        annotation_text="Analysis window",
        annotation_position="top left",
    )
    fig.update_layout(
        title="Select analysis region (drag the slider below)",
        xaxis_title="Time (s)",
        yaxis_title="Current",
        height=320,
        margin=dict(l=40, r=20, t=50, b=40),
    )
    return fig


def detection_work_signal(
    current: np.ndarray,
    sample_rate: float,
    *,
    direction: str,
    baseline: str,
    baseline_window: float,
    baseline_percentile: float = 90.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Approximate event-service residual (events downward) for live preview."""
    raw = np.asarray(current, dtype=float)
    if baseline == "median" and sample_rate > 0:
        win = max(3, int(baseline_window * sample_rate))
        if win % 2 == 0:
            win += 1
        s = pd.Series(raw)
        bl = s.rolling(win, center=True, min_periods=1).median().to_numpy()
    elif baseline == "percentile" and sample_rate > 0:
        win = max(3, int(max(baseline_window, 0.5) * sample_rate))
        s = pd.Series(raw)
        bl = (
            s.rolling(win, center=True, min_periods=1)
            .quantile(baseline_percentile / 100.0)
            .to_numpy(dtype=float)
        )
        if np.isnan(bl).any():
            bl = np.where(np.isnan(bl), float(np.percentile(raw, baseline_percentile)), bl)
    elif baseline == "constant":
        bl = np.full_like(raw, float(np.median(raw)))
    else:
        bl = np.full_like(raw, float(np.mean(raw)))
    residual = raw - bl
    work = residual if direction == "down" else -residual
    return work, bl


def threshold_preview_figure(
    time: list[float],
    current: list[float],
    sample_rate: float,
    *,
    std_multiplier: float,
    threshold_multiplier: float,
    direction: str,
    baseline: str,
    baseline_window: float,
    baseline_percentile: float = 90.0,
) -> go.Figure:
    t = np.asarray(time, dtype=float)
    c = np.asarray(current, dtype=float)
    work, _bl = detection_work_signal(
        c,
        sample_rate,
        direction=direction,
        baseline=baseline,
        baseline_window=baseline_window,
        baseline_percentile=baseline_percentile,
    )
    mean = float(np.mean(work))
    std = float(np.std(work)) or 1e-12
    entry = mean - std_multiplier * std
    deep = mean - threshold_multiplier * std

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=t.tolist(),
            y=work.tolist(),
            mode="lines",
            name="Detection signal",
            line=dict(color="rgba(60,60,60,0.85)", width=1),
        )
    )
    fig.add_hline(y=mean, line=dict(color="red", width=2), annotation_text="Mean")
    fig.add_hline(
        y=entry,
        line=dict(color="blue", width=2, dash="dash"),
        annotation_text=f"Entry ({std_multiplier}×σ)",
    )
    fig.add_hline(
        y=deep,
        line=dict(color="green", width=2, dash="dot"),
        annotation_text=f"Deep ({threshold_multiplier}×σ)",
    )
    fig.update_layout(
        title="Live threshold preview (detection space — events downward)",
        xaxis_title="Time (s)",
        yaxis_title="Residual current",
        height=420,
        legend=dict(orientation="h"),
    )
    return fig


def ensure_preview(uploaded) -> dict | None:
    """Load/cached gateway preview for the current upload identity."""
    key = f"{uploaded.name}:{len(uploaded.getvalue())}"
    if st.session_state.get("preview_key") != key:
        with st.spinner("Loading trace preview..."):
            try:
                preview = preview_file(uploaded)
                st.session_state["preview"] = preview
                st.session_state["preview_key"] = key
            except Exception as exc:  # noqa: BLE001
                st.error(f"Preview failed: {exc}")
                st.session_state.pop("preview", None)
                st.session_state.pop("preview_key", None)
                return None
    return st.session_state.get("preview")


def main() -> None:
    st.set_page_config(
        page_title="Pynanopore Analysis",
        page_icon="🔬",
        layout="wide",
    )
    st.title("Pynanopore — Single-Molecule Electrophysiology")
    st.caption(f"Gateway: `{GATEWAY_URL}`")

    with st.expander("First analysis (quick start)", expanded=False):
        st.markdown(
            """
1. **Upload** an ABF/CSV recording, or click **Use example CSV** in the sidebar.
2. Inspect the **full trace**, then set the **analysis window** slider to the region of interest.
3. Tune **direction**, **baseline**, and threshold multipliers on the live preview.
4. Click **Run event detection** (only the selected window is analyzed), then dwell / PSD.
5. **Export** events CSV, fit JSON, and plot HTML/PNG from each tab.

See also `docs/first_analysis.md` in the repository.
            """
        )

    with st.sidebar:
        st.header("Connection")
        try:
            health = gateway_health()
            st.success(f"Gateway: {health.get('status', 'unknown')}")
            with st.expander("Service health"):
                st.json(health.get("services", {}))
        except Exception as exc:  # noqa: BLE001
            st.error(f"Gateway unreachable: {exc}")

        st.subheader("Recording")
        if st.button("Use example CSV", use_container_width=True):
            if EXAMPLE_CSV_PATH.is_file():
                st.session_state["upload"] = MemoryUpload(
                    EXAMPLE_CSV_PATH.name, EXAMPLE_CSV_PATH.read_bytes()
                )
                st.session_state.pop("detect_result", None)
                st.session_state.pop("stats_result", None)
                st.session_state.pop("psd_result", None)
                st.session_state.pop("preview", None)
                st.session_state.pop("preview_key", None)
                st.session_state.pop("analysis_window", None)
            else:
                st.warning(f"Example not found: {EXAMPLE_CSV_PATH}")

        uploaded_file = st.file_uploader("Upload ABF or CSV", type=["abf", "csv"])
        if uploaded_file is not None:
            st.session_state["upload"] = MemoryUpload(
                uploaded_file.name, uploaded_file.getvalue()
            )
            # New file → clear analysis caches
            if st.session_state.get("upload_name") != uploaded_file.name:
                st.session_state["upload_name"] = uploaded_file.name
                st.session_state.pop("detect_result", None)
                st.session_state.pop("stats_result", None)
                st.session_state.pop("psd_result", None)
                st.session_state.pop("analysis_window", None)

        uploaded = st.session_state.get("upload")
        if uploaded:
            size_mb = len(uploaded.getvalue()) / (1024 * 1024)
            st.caption(f"`{uploaded.name}` · {size_mb:.2f} MB")
            if st.button("Clear session", use_container_width=True):
                for k in (
                    "upload",
                    "upload_name",
                    "preview",
                    "preview_key",
                    "detect_result",
                    "stats_result",
                    "psd_result",
                    "analysis_window",
                ):
                    st.session_state.pop(k, None)
                st.rerun()

        st.subheader("Detection")
        direction = st.selectbox("Event direction", ["down", "up"], index=0)
        baseline = st.selectbox(
            "Baseline", ["none", "median", "constant", "percentile"], index=0
        )
        baseline_window = st.number_input(
            "Baseline window (s)", value=0.05, min_value=0.001, step=0.01
        )
        baseline_percentile = st.number_input(
            "Baseline percentile",
            value=90.0,
            min_value=0.0,
            max_value=100.0,
            step=5.0,
            help="For percentile baseline: ~90 for down events, ~10 for up",
        )
        std_mult = st.number_input("Std multiplier", value=0.25, min_value=0.0, step=0.05)
        thr_mult = st.number_input("Threshold multiplier", value=1.5, min_value=0.0, step=0.1)
        interval = st.number_input("Chunk interval (s)", value=5.0, min_value=0.1, step=0.5)
        overlap = st.number_input("Chunk overlap (s)", value=0.0, min_value=0.0, step=0.1)
        show_pulse = st.checkbox("Show pulse-shape idealization", value=True)
        analyze_levels = st.checkbox("Multi-level conductance analysis", value=True)
        auto_preview = st.checkbox("Live threshold preview", value=True)

    # Tabs first so dwell-time / PSD are always visible (not buried under plots)
    event_tab, stats_tab, psd_tab = st.tabs(
        ["Event Detection", "Statistical Analysis", "Power Spectrum"]
    )

    preview = None
    t_start = t_end = None
    region_time: list[float] = []
    region_current: list[float] = []

    if uploaded:
        preview = ensure_preview(uploaded)
        if preview:
            t_min = float(preview.get("t_min", preview["time"][0]))
            t_max = float(preview.get("t_max", preview["time"][-1]))
            if t_max > t_min:
                if st.session_state.get("analysis_window") is None:
                    st.session_state["analysis_window"] = (t_min, t_max)
                prev_win = st.session_state["analysis_window"]
                win_start = min(max(prev_win[0], t_min), t_max)
                win_end = max(min(prev_win[1], t_max), t_min)
                if win_end <= win_start:
                    win_start, win_end = t_min, t_max
                # Defaults used until Event tab renders the slider
                t_start, t_end = float(win_start), float(win_end)

    with event_tab:
        if not uploaded:
            st.info("Upload a recording (or use the example CSV) in the sidebar to begin.")
        elif not preview:
            st.error("Could not load a preview for this file.")
        else:
            t_min = float(preview.get("t_min", preview["time"][0]))
            t_max = float(preview.get("t_max", preview["time"][-1]))
            if t_max <= t_min:
                st.error("Preview has an invalid time axis.")
            else:
                with st.expander("1. Select analysis region", expanded=True):
                    st.plotly_chart(
                        overview_figure(
                            preview["time"], preview["current"], t_start, t_end
                        ),
                        use_container_width=True,
                    )
                    span = t_max - t_min
                    step = max(span / 1000.0, 1e-6)
                    selected = st.slider(
                        "Analysis window (seconds)",
                        min_value=float(t_min),
                        max_value=float(t_max),
                        value=(float(t_start), float(t_end)),
                        step=float(step),
                        help="Only this time range is sent to detection / PSD analysis.",
                    )
                    t_start, t_end = float(selected[0]), float(selected[1])
                    st.session_state["analysis_window"] = (t_start, t_end)
                    c_a, c_b, c_c = st.columns(3)
                    c_a.metric("Window start (s)", f"{t_start:.4f}")
                    c_b.metric("Window end (s)", f"{t_end:.4f}")
                    c_c.metric("Window length (s)", f"{(t_end - t_start):.4f}")
                    if st.button("Use full recording"):
                        st.session_state["analysis_window"] = (t_min, t_max)
                        st.rerun()

                t_arr = np.asarray(preview["time"], dtype=float)
                c_arr = np.asarray(preview["current"], dtype=float)
                mask = (t_arr >= t_start) & (t_arr <= t_end)
                if not np.any(mask):
                    st.warning("No preview samples fall inside the selected window.")
                else:
                    region_time = t_arr[mask].tolist()
                    region_current = c_arr[mask].tolist()

                    st.subheader("2. Tune thresholds on the selected region")
                    if auto_preview:
                        fig_prev = threshold_preview_figure(
                            region_time,
                            region_current,
                            preview["sample_rate"],
                            std_multiplier=std_mult,
                            threshold_multiplier=thr_mult,
                            direction=direction,
                            baseline=baseline,
                            baseline_window=baseline_window,
                            baseline_percentile=baseline_percentile,
                        )
                        st.plotly_chart(fig_prev, use_container_width=True)
                        st.caption(
                            f"Region preview points: {len(region_time):,} · "
                            f"full file: {preview['n_points_total']:,} samples · "
                            f"fs={preview['sample_rate']:.2f} Hz"
                        )

                    run = st.button("Run event detection", type="primary")
                    if run:
                        size_mb = len(uploaded.getvalue()) / (1024 * 1024)
                        with st.status("Running event detection…", expanded=True) as status:
                            try:
                                st.write(
                                    f"Window [{t_start:.4f}, {t_end:.4f}] s · "
                                    f"uploading `{uploaded.name}` ({size_mb:.2f} MB)…"
                                )
                                st.write("Detecting events in the selected region…")
                                result = detect_file(
                                    uploaded,
                                    std_multiplier=std_mult,
                                    threshold_multiplier=thr_mult,
                                    interval_length=interval,
                                    overlap=overlap,
                                    direction=direction,
                                    baseline=baseline,
                                    baseline_window=baseline_window,
                                    baseline_percentile=baseline_percentile,
                                    include_plot=False,
                                    include_pulse_plot=show_pulse,
                                    analyze_levels=analyze_levels,
                                    t_start=t_start,
                                    t_end=t_end,
                                )
                                st.session_state["detect_result"] = result
                                st.write(f"Done — {result['n_events']} events.")
                                status.update(label="Detection complete", state="complete")
                            except Exception as exc:  # noqa: BLE001
                                status.update(label="Detection failed", state="error")
                                st.error(str(exc))
                                st.info(
                                    "Tip: narrow the analysis window, disable pulse plot, or raise "
                                    "`HTTP_TIMEOUT_S` / `MAX_UPLOAD_MB` if the request timed out."
                                )

                    result = st.session_state.get("detect_result")
                    if result:
                        c1, c2, c3, c4 = st.columns(4)
                        c1.metric("Events detected", result["n_events"])
                        c2.metric("Sample rate (Hz)", f"{result['sample_rate']:.2f}")
                        c3.metric(
                            "Analyzed duration (s)", round(result.get("duration_s", 0), 3)
                        )
                        if result.get("window_start_s") is not None:
                            c4.metric(
                                "Window (s)",
                                f"{result['window_start_s']:.3f}–"
                                f"{result.get('window_end_s', 0):.3f}",
                            )

                        if result.get("events"):
                            events_df = pd.DataFrame(result["events"])
                            st.dataframe(events_df, use_container_width=True)
                            st.download_button(
                                "Download events CSV",
                                data=events_df.to_csv(index=False),
                                file_name="events.csv",
                                mime="text/csv",
                                key="dl-events-csv",
                            )
                            st.download_button(
                                "Download detect JSON",
                                data=json.dumps(
                                    {
                                        "n_events": result["n_events"],
                                        "sample_rate": result["sample_rate"],
                                        "duration_s": result.get("duration_s"),
                                        "window_start_s": result.get("window_start_s"),
                                        "window_end_s": result.get("window_end_s"),
                                        "events": result["events"],
                                    },
                                    indent=2,
                                ),
                                file_name="detect_result.json",
                                mime="application/json",
                                key="dl-detect-json",
                            )

                        fig = None
                        if result.get("levels_plot"):
                            st.markdown("**Multi-level conductance overlay**")
                            fig = render_plotly(result["levels_plot"])
                            download_figure(fig, "levels_plot")
                        if result.get("pulse_plot"):
                            st.markdown("**Pulse-shape idealization**")
                            fig = render_plotly(result["pulse_plot"])
                            download_figure(fig, "events_plot")
                        elif result.get("plot") and not result.get("levels_plot"):
                            fig = render_plotly(result["plot"])
                            download_figure(fig, "events_plot")

    with stats_tab:
        result = st.session_state.get("detect_result")
        if not result or not result.get("events"):
            st.warning("Run event detection first (Event Detection tab).")
        else:
            fit_type = st.selectbox("Fit type", ["single", "double", "auto"])
            method = st.selectbox("Method", ["mle", "histogram"], index=0)
            binning = st.selectbox("Binning", ["linear", "log"], index=0)
            bins = st.number_input("Bins", value=50, min_value=5, max_value=2000)
            if st.button("Fit dwell times"):
                with st.status("Fitting dwell times…", expanded=True) as status:
                    try:
                        st.write(f"Sending {len(result['events'])} events to stats-service…")
                        stats = dwelltime(
                            result["events"],
                            bins=int(bins),
                            fit_type=fit_type,
                            method=method,
                            binning=binning,
                            include_plot=True,
                        )
                        st.session_state["stats_result"] = stats
                        status.update(label="Fit complete", state="complete")
                    except Exception as exc:  # noqa: BLE001
                        status.update(label="Fit failed", state="error")
                        st.error(str(exc))

            stats = st.session_state.get("stats_result")
            if stats:
                st.write(
                    f"Fit: **{stats.get('fit_type')}** via **{stats.get('method')}** "
                    f"(AIC={stats.get('aic')}, BIC={stats.get('bic')})"
                )
                st.dataframe(pd.DataFrame([stats["parameters"]]), use_container_width=True)
                fit_payload: dict[str, Any] = {
                    "fit_type": stats.get("fit_type"),
                    "method": stats.get("method"),
                    "parameters": stats.get("parameters"),
                    "log_likelihood": stats.get("log_likelihood"),
                    "aic": stats.get("aic"),
                    "bic": stats.get("bic"),
                    "n_events": stats.get("n_events"),
                    "n_events_used": stats.get("n_events_used"),
                    "model_comparison": stats.get("model_comparison"),
                }
                st.download_button(
                    "Download fit JSON",
                    data=json.dumps(fit_payload, indent=2),
                    file_name="dwelltime_fit.json",
                    mime="application/json",
                    key="dl-dwell-json",
                )
                if stats.get("model_comparison"):
                    st.write("Model comparison (MLE)")
                    st.json(stats["model_comparison"])
                if stats.get("plot"):
                    fig = render_plotly(stats["plot"])
                    download_figure(fig, "dwelltime_plot")

    with psd_tab:
        if not uploaded:
            st.info("Upload a recording first.")
        else:
            # Prefer window chosen in Event tab (session); fall back to full preview
            win = st.session_state.get("analysis_window")
            if win is not None:
                t_start, t_end = float(win[0]), float(win[1])
            fit = st.checkbox("Fit model", value=True)
            fit_model = st.selectbox(
                "Fit model type",
                [
                    "lorentzian",
                    "composite",
                    "lorentzian_white",
                    "double_lorentzian",
                    "none",
                ],
                index=0,
            )
            window = st.selectbox(
                "Welch window", ["hamming", "hann", "blackman", "flattop"], index=0
            )
            scaling = st.selectbox("Scaling", ["spectrum", "density"], index=0)
            nperseg = st.number_input("nperseg (0=auto)", value=0, min_value=0, step=256)
            if win is not None:
                st.caption(f"Using analysis window [{t_start:.4f}, {t_end:.4f}] s")
            if st.button("Compute PSD"):
                with st.status("Computing PSD…", expanded=True) as status:
                    try:
                        detect_result = st.session_state.get("detect_result")
                        psd_kwargs = {
                            "fit": fit and fit_model != "none",
                            "fit_model": fit_model,
                            "include_plot": True,
                            "window": window,
                            "scaling": scaling,
                        }
                        if nperseg > 0:
                            psd_kwargs["nperseg"] = int(nperseg)
                        if detect_result and detect_result.get("preview_current"):
                            st.write("Using current from the analyzed detection window…")
                            psd_result = psd_from_preview(
                                detect_result["preview_current"],
                                detect_result["sample_rate"],
                                **psd_kwargs,
                            )
                        elif t_start is not None and t_end is not None:
                            st.write(
                                f"Uploading `{uploaded.name}` "
                                f"(window [{t_start:.4f}, {t_end:.4f}] s)…"
                            )
                            psd_result = psd_upload(
                                uploaded,
                                t_start=t_start,
                                t_end=t_end,
                                **psd_kwargs,
                            )
                        else:
                            st.write(f"Uploading `{uploaded.name}` (full file)…")
                            psd_result = psd_upload(uploaded, **psd_kwargs)
                        st.session_state["psd_result"] = psd_result
                        status.update(label="PSD complete", state="complete")
                    except Exception as exc:  # noqa: BLE001
                        status.update(label="PSD failed", state="error")
                        st.error(str(exc))

            psd_result = st.session_state.get("psd_result")
            if psd_result:
                cols = st.columns(5)
                cols[0].metric(
                    "S0",
                    round(psd_result["S0"], 4)
                    if psd_result.get("S0") is not None
                    else "—",
                )
                cols[1].metric(
                    "fc (Hz)",
                    round(psd_result["fc"], 2)
                    if psd_result.get("fc") is not None
                    else "—",
                )
                cols[2].metric(
                    "A",
                    round(psd_result["A"], 6) if psd_result.get("A") is not None else "—",
                )
                cols[3].metric(
                    "alpha",
                    round(psd_result["alpha"], 3)
                    if psd_result.get("alpha") is not None
                    else "—",
                )
                cols[4].metric(
                    "N",
                    round(psd_result["N"], 6) if psd_result.get("N") is not None else "—",
                )
                psd_payload = {
                    "fs": psd_result.get("fs"),
                    "fit_model": psd_result.get("fit_model"),
                    "S0": psd_result.get("S0"),
                    "fc": psd_result.get("fc"),
                    "A": psd_result.get("A"),
                    "alpha": psd_result.get("alpha"),
                    "N": psd_result.get("N"),
                    "diagnostics": psd_result.get("diagnostics"),
                    "n_frequencies": psd_result.get("n_frequencies"),
                }
                st.download_button(
                    "Download PSD fit JSON",
                    data=json.dumps(psd_payload, indent=2),
                    file_name="psd_fit.json",
                    mime="application/json",
                    key="dl-psd-json",
                )
                if psd_result.get("frequencies") and psd_result.get("power_spectrum"):
                    spec_df = pd.DataFrame(
                        {
                            "frequency_hz": psd_result["frequencies"],
                            "power": psd_result["power_spectrum"],
                        }
                    )
                    st.download_button(
                        "Download PSD CSV",
                        data=spec_df.to_csv(index=False),
                        file_name="psd_spectrum.csv",
                        mime="text/csv",
                        key="dl-psd-csv",
                    )
                if psd_result.get("diagnostics"):
                    st.caption(
                        f"R²(log)={psd_result['diagnostics'].get('r2_log')}  "
                        f"RMSE(log)={psd_result['diagnostics'].get('rmse_log')}"
                    )
                if psd_result.get("plot"):
                    fig = render_plotly(psd_result["plot"])
                    download_figure(fig, "psd_plot")


if __name__ == "__main__":
    main()
