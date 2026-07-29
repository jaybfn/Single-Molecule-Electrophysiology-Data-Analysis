"""Streamlit UI that calls the Pynanopore gateway API."""

from __future__ import annotations

import json
import os

import httpx
import pandas as pd
import plotly.io as pio
import streamlit as st

GATEWAY_URL = os.getenv("GATEWAY_URL", "http://localhost:8000")
TIMEOUT = float(os.getenv("HTTP_TIMEOUT_S", "120"))


def gateway_health() -> dict:
    with httpx.Client(timeout=5.0) as client:
        resp = client.get(f"{GATEWAY_URL}/health")
        resp.raise_for_status()
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


def render_plotly(plot_dict: dict) -> None:
    fig = pio.from_json(json.dumps(plot_dict))
    st.plotly_chart(fig, use_container_width=True)


def main() -> None:
    st.set_page_config(
        page_title="Pynanopore Analysis",
        page_icon="🔬",
        layout="wide",
    )
    st.title("Pynanopore — Single-Molecule Electrophysiology")
    st.caption(f"Gateway: `{GATEWAY_URL}`")

    with st.sidebar:
        st.header("Connection")
        try:
            health = gateway_health()
            st.success(f"Gateway: {health.get('status', 'unknown')}")
            st.json(health.get("services", {}))
        except Exception as exc:  # noqa: BLE001
            st.error(f"Gateway unreachable: {exc}")

        uploaded = st.file_uploader("Upload ABF or CSV", type=["abf", "csv"])
        st.subheader("Detection")
        direction = st.selectbox("Event direction", ["down", "up"], index=0)
        baseline = st.selectbox("Baseline", ["none", "median", "constant"], index=0)
        baseline_window = st.number_input(
            "Median window (s)", value=0.05, min_value=0.001, step=0.01
        )
        std_mult = st.number_input("Std multiplier", value=0.25, min_value=0.0, step=0.05)
        thr_mult = st.number_input("Threshold multiplier", value=1.5, min_value=0.0, step=0.1)
        interval = st.number_input("Chunk interval (s)", value=5.0, min_value=0.1, step=0.5)
        overlap = st.number_input("Chunk overlap (s)", value=0.0, min_value=0.0, step=0.1)
        show_pulse = st.checkbox("Show pulse-shape idealization", value=True)

    if not uploaded:
        st.info("Upload a recording to begin analysis.")
        return

    event_tab, stats_tab, psd_tab = st.tabs(
        ["Event Detection", "Statistical Analysis", "Power Spectrum"]
    )

    with event_tab:
        if st.button("Run event detection", type="primary"):
            with st.spinner("Detecting events via event-service..."):
                try:
                    result = detect_file(
                        uploaded,
                        std_multiplier=std_mult,
                        threshold_multiplier=thr_mult,
                        interval_length=interval,
                        overlap=overlap,
                        direction=direction,
                        baseline=baseline,
                        baseline_window=baseline_window,
                        include_plot=False,
                        include_pulse_plot=show_pulse,
                    )
                    st.session_state["detect_result"] = result
                except Exception as exc:  # noqa: BLE001
                    st.error(str(exc))

        result = st.session_state.get("detect_result")
        if result:
            c1, c2, c3 = st.columns(3)
            c1.metric("Events detected", result["n_events"])
            c2.metric("Sample rate (Hz)", result["sample_rate"])
            c3.metric("Duration (s)", round(result.get("duration_s", 0), 3))
            if result.get("pulse_plot"):
                st.markdown("**Pulse-shape idealization**")
                render_plotly(result["pulse_plot"])
            elif result.get("plot"):
                render_plotly(result["plot"])
            if result.get("events"):
                st.dataframe(pd.DataFrame(result["events"]), use_container_width=True)

    with stats_tab:
        result = st.session_state.get("detect_result")
        if not result or not result.get("events"):
            st.warning("Run event detection first.")
        else:
            fit_type = st.selectbox("Fit type", ["single", "double", "auto"])
            method = st.selectbox("Method", ["mle", "histogram"], index=0)
            binning = st.selectbox("Binning", ["linear", "log"], index=0)
            bins = st.number_input("Bins", value=50, min_value=5, max_value=2000)
            if st.button("Fit dwell times"):
                with st.spinner("Fitting via stats-service..."):
                    try:
                        stats = dwelltime(
                            result["events"],
                            bins=int(bins),
                            fit_type=fit_type,
                            method=method,
                            binning=binning,
                            include_plot=True,
                        )
                        st.session_state["stats_result"] = stats
                    except Exception as exc:  # noqa: BLE001
                        st.error(str(exc))

            stats = st.session_state.get("stats_result")
            if stats:
                st.write(
                    f"Fit: **{stats.get('fit_type')}** via **{stats.get('method')}** "
                    f"(AIC={stats.get('aic')}, BIC={stats.get('bic')})"
                )
                st.dataframe(pd.DataFrame([stats["parameters"]]), use_container_width=True)
                if stats.get("model_comparison"):
                    st.write("Model comparison (MLE)")
                    st.json(stats["model_comparison"])
                if stats.get("plot"):
                    render_plotly(stats["plot"])

    with psd_tab:
        fit = st.checkbox("Fit model", value=True)
        fit_model = st.selectbox("Fit model type", ["lorentzian", "composite", "none"], index=0)
        window = st.selectbox("Welch window", ["hamming", "hann", "blackman", "flattop"], index=0)
        scaling = st.selectbox("Scaling", ["spectrum", "density"], index=0)
        nperseg = st.number_input("nperseg (0=auto)", value=0, min_value=0, step=256)
        if st.button("Compute PSD"):
            with st.spinner("Computing PSD via psd-service..."):
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
                        psd_result = psd_from_preview(
                            detect_result["preview_current"],
                            detect_result["sample_rate"],
                            **psd_kwargs,
                        )
                    else:
                        psd_result = psd_upload(uploaded, **psd_kwargs)
                    st.session_state["psd_result"] = psd_result
                except Exception as exc:  # noqa: BLE001
                    st.error(str(exc))

        psd_result = st.session_state.get("psd_result")
        if psd_result:
            cols = st.columns(4)
            cols[0].metric(
                "S0",
                round(psd_result["S0"], 4) if psd_result.get("S0") is not None else "—",
            )
            cols[1].metric(
                "fc (Hz)",
                round(psd_result["fc"], 2) if psd_result.get("fc") is not None else "—",
            )
            cols[2].metric(
                "A",
                round(psd_result["A"], 6) if psd_result.get("A") is not None else "—",
            )
            cols[3].metric(
                "alpha",
                round(psd_result["alpha"], 3) if psd_result.get("alpha") is not None else "—",
            )
            if psd_result.get("diagnostics"):
                st.caption(
                    f"R²(log)={psd_result['diagnostics'].get('r2_log')}  "
                    f"RMSE(log)={psd_result['diagnostics'].get('rmse_log')}"
                )
            if psd_result.get("plot"):
                render_plotly(psd_result["plot"])


if __name__ == "__main__":
    main()
