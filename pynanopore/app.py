import streamlit as st
import plotly.graph_objects as go
import pandas as pd
import os
import pyabf

from even_detection import ReadingData, CreatingChunks, EventDetection, Plotting 

# Function to load and process your data
def process_data(file_path):
    reader = ReadingData(file_path)
    data = reader.get_data()

    all_events = []

    chunker = CreatingChunks(data)
    detector = EventDetection(std_multiplier=0.5, threshold_multiplier=2.5)

    for sweepNumber in data.sweepList:
        data.setSweep(sweepNumber)
        sweep_data = data.sweepY * (-1)
        sweep_time = data.sweepX

        for chunk_start in range(0, len(sweep_data), chunker.points_per_interval):
            chunk_end = chunk_start + chunker.points_per_interval
            data_chunk = sweep_data[chunk_start:chunk_end]
            time_chunk = sweep_time[chunk_start:chunk_end]
            events_data = detector.detect_events(data_chunk, time_chunk)
            all_events.extend(events_data)
        
    events_df = pd.DataFrame(all_events)
    ind = ((all_events[100]['end_time']+(((all_events[100]['end_time'])/100))*2) * 50000)
    return sweep_time[:int(ind)], sweep_data[:int(ind)], all_events[:100], events_df

# Main function for Streamlit app
def main():
    st.title('Single Molecule Electrophysiology Analysis')

    # Folder selector
    with st.sidebar:
        folder_path = st.text_input('Enter folder path here:')
        if folder_path:
            files = os.listdir(folder_path)
            selected_file = st.selectbox("Choose a file from the folder:", files)
    if selected_file:
        data_file = os.path.join(folder_path, selected_file)
        
        with st.container():
            col1, col2 = st.columns(2)

            with col1:
                try:

                    st.markdown("Ion Current Trace:")
                    sweep_time, sweep_data, all_events, events_df = process_data(data_file)
                    #st.write(events_df)  # Display the events data as a table in Streamlit

                    # Use your plotter to display the plot
                    plotter = Plotting()
                    plot_figure = plotter.plot_data_series(sweep_time, sweep_data)
                    st.plotly_chart(plot_figure)

                except Exception as e:
                    st.write("Error: ", e)

            with col2:
                try:

                    st.markdown("Ion Current Trace With Detected Events:")

                    # Use your plotter to display the plot
                    plotter = Plotting()
                    plot_figure = plotter.plot_data(sweep_time, sweep_data, all_events)
                    st.plotly_chart(plot_figure)

                except Exception as e:
                    st.write("Error: ", e)

if __name__ == "__main__":
    # setting the page configuration
    base="dark"
    primaryColor="purple"
    # Setting the page config first
    st.set_page_config(
        page_title="Real-Time Forex Dashboard",
        page_icon="icon.jpg",
        layout="wide"
    )
    main()
