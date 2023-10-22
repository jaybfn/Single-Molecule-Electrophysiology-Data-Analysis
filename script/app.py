import streamlit as st
import plotly.graph_objects as go
import pandas as pd
import os
from even_detection import ReadingData, CreatingChunks, EventDetection, Plotting 

# Function to load and process your data
def process_data(file_path):
    reader = ReadingData(file_path)
    abf = reader.get_data()

    chunker = CreatingChunks(abf)
    detector = EventDetection(std_multiplier=0.5, threshold_multiplier=2.5)

    all_events = []

    for sweepNumber in abf.sweepList:
        abf.setSweep(sweepNumber)
        sweep_data = abf.sweepY * (-1)
        sweep_time = abf.sweepX

        for chunk_start in range(0, len(sweep_data), chunker.points_per_interval):
            chunk_end = chunk_start + chunker.points_per_interval
            data_chunk = sweep_data[chunk_start:chunk_end]
            time_chunk = sweep_time[chunk_start:chunk_end]
            events_data = detector.detect_events(data_chunk, time_chunk)
            all_events.extend(events_data)

    events_df = pd.DataFrame(all_events)
    return sweep_time, sweep_data, all_events, events_df

# Main function for Streamlit app
def main():
    st.title('ABF File Data Visualization')

    # Folder selector
    folder_path = st.text_input('Enter folder path here:')
    if folder_path:
        files = os.listdir(folder_path)
        selected_file = st.selectbox("Choose a file from the folder:", files)
        if selected_file:
            data_file = os.path.join(folder_path, selected_file)

            try:
                sweep_time, sweep_data, all_events, events_df = process_data(data_file)
                st.write(events_df)  # Display the events data as a table in Streamlit

                # Use your plotter to display the plot
                plotter = Plotting()
                plot_figure = plotter.plot_data(sweep_time, sweep_data, all_events)
                st.plotly_chart(plot_figure)

            except Exception as e:
                st.write("Error: ", e)

if __name__ == "__main__":
    main()
