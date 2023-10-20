import streamlit as st
import plotly.graph_objects as go
import pyabf
import numpy as np
import tempfile

# Function to downsample data
def downsample_data(data, factor):
    """
    Downsamples a dataset by averaging every 'factor' points.
    """
    downsampled = np.mean(data[:len(data) - len(data) % factor].reshape(-1, factor), axis=1)
    return downsampled

def main():
    st.title('ABF File Data Downsampler')

    # Upload the ABF file
    uploaded_file = st.file_uploader("Choose an ABF file", type=['abf'])
    if uploaded_file is not None:
        # Create a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".abf") as tmp:
            # Write the uploaded file's content to the temporary file
            tmp.write(uploaded_file.getvalue())
            # Ensure the file is written
            tmp.flush()

            # Use the temporary file's path for pyabf
            abf = pyabf.ABF(tmp.name)
        # The factor by which to reduce the data
        downsampling_factor = st.slider('Set Downsampling Factor', min_value=1, max_value=100, value=10)
        # Prepare a Plotly figure
        fig = go.Figure()

        # Iterate over each sweep
        for sweepNumber in abf.sweepList:
            abf.setSweep(sweepNumber)
            sweep_data = abf.sweepY

            # Downsample the sweep data
            downsampled_sweep_data = downsample_data(sweep_data, downsampling_factor)

            # Add the downsampled data for this sweep to the plot
            fig.add_trace(go.Scatter(
                x=abf.sweepX[:len(downsampled_sweep_data)] * downsampling_factor,
                y=downsampled_sweep_data,
                mode='lines',
                name=f"Sweep {sweepNumber}"
            ))

        # Enhance plot with labels and title
        fig.update_layout(title='Downsampled Signal from Each Sweep', xaxis_title='Time (s)', yaxis_title='Signal (downsampled)')

        # Display the figure in Streamlit
        st.plotly_chart(fig)

if __name__ == "__main__":
    main()

