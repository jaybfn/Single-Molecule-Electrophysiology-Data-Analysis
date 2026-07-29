"""Example: downsample and plot an ABF file with Streamlit (standalone)."""

from __future__ import annotations

import tempfile

import numpy as np
import plotly.graph_objects as go
import pyabf
import streamlit as st


def downsample_data(data: np.ndarray, factor: int) -> np.ndarray:
    usable = len(data) - (len(data) % factor)
    return np.mean(data[:usable].reshape(-1, factor), axis=1)


def main() -> None:
    st.title("ABF Downsampler (example)")
    uploaded = st.file_uploader("Choose an ABF file", type=["abf"])
    if uploaded is None:
        return

    with tempfile.NamedTemporaryFile(delete=False, suffix=".abf") as tmp:
        tmp.write(uploaded.getvalue())
        tmp.flush()
        abf = pyabf.ABF(tmp.name)

    factor = st.slider("Downsampling factor", 1, 100, 10)
    fig = go.Figure()
    for sweep in abf.sweepList:
        abf.setSweep(sweep)
        down = downsample_data(np.asarray(abf.sweepY), factor)
        fig.add_trace(
            go.Scatter(
                x=abf.sweepX[: len(down)] * factor,
                y=down,
                mode="lines",
                name=f"Sweep {sweep}",
            )
        )
    fig.update_layout(
        title="Downsampled Signal",
        xaxis_title="Time (s)",
        yaxis_title="Signal",
    )
    st.plotly_chart(fig)


if __name__ == "__main__":
    main()
