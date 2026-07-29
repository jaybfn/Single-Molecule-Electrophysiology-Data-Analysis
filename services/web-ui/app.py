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
        std_mult = st.number_input("Std multiplier", value=0.25, min_value=0.0, step=0.05)
        thr_mult = st.number_input("Threshold multiplier", value=1.5, min_value=0.0, step=0.1)
        interval = st.number_input("Chunk interval (s)", value=5.0, min_value=0.1, step=0.5)

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
                        include_plot=True,
                    )
                    st.session_state["detect_result"] = result
                except Exception as exc:  # noqa: BLE001
                    st.error(str(exc))

        result = st.session_state.get("detect_result")
        if result:
            st.metric("Events detected", result["n_events"])
            st.metric("Sample rate (Hz)", result["sample_rate"])
            if result.get("plot"):
                render_plotly(result["plot"])
            if result.get("events"):
                st.dataframe(pd.DataFrame(result["events"]), use_container_width=True)

    with stats_tab:
        result = st.session_state.get("detect_result")
        if not result or not result.get("events"):
            st.warning("Run event detection first.")
        else:
            fit_type = st.selectbox("Fit type", ["single", "double"])
            bins = st.number_input("Bins", value=50, min_value=5, max_value=2000)
            if st.button("Fit dwell times"):
                with st.spinner("Fitting via stats-service..."):
                    try:
                        stats = dwelltime(
                            result["events"],
                            bins=int(bins),
                            fit_type=fit_type,
                            include_plot=True,
                        )
                        st.session_state["stats_result"] = stats
                    except Exception as exc:  # noqa: BLE001
                        st.error(str(exc))

            stats = st.session_state.get("stats_result")
            if stats:
                st.write("Fit parameters")
                st.dataframe(pd.DataFrame([stats["parameters"]]), use_container_width=True)
                if stats.get("plot"):
                    render_plotly(stats["plot"])

    with psd_tab:
        fit = st.checkbox("Fit Lorentzian", value=True)
        if st.button("Compute PSD"):
            with st.spinner("Computing PSD via psd-service..."):
                try:
                    detect_result = st.session_state.get("detect_result")
                    if detect_result and detect_result.get("preview_current"):
                        psd_result = psd_from_preview(
                            detect_result["preview_current"],
                            detect_result["sample_rate"],
                            fit=fit,
                            include_plot=True,
                        )
                    else:
                        psd_result = psd_upload(uploaded, fit=fit, include_plot=True)
                    st.session_state["psd_result"] = psd_result
                except Exception as exc:  # noqa: BLE001
                    st.error(str(exc))

        psd_result = st.session_state.get("psd_result")
        if psd_result:
            cols = st.columns(2)
            cols[0].metric(
                "S0 (pA²/Hz)",
                round(psd_result["S0"], 4) if psd_result.get("S0") is not None else "—",
            )
            cols[1].metric(
                "fc (Hz)",
                round(psd_result["fc"], 2) if psd_result.get("fc") is not None else "—",
            )
            if psd_result.get("plot"):
                render_plotly(psd_result["plot"])


if __name__ == "__main__":
    main()
