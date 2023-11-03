import streamlit as st
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
import pyabf


# from event_detection import ReadingData, CreatingChunks, EventDetection, Plotting 
# from powerspectrum import PSDAnalyzer, LorentzianFitter
# from dwelltime import DwellTime_ExponentialFit

from pynanopore.event_detection import ReadingData, CreatingChunks, EventDetection, Plotting 
from pynanopore.powerspectrum import PSDAnalyzer, LorentzianFitter
from pynanopore.dwelltime import DwellTime_ExponentialFit

# Function to load and process your data
def process_data(file_path):
    reader = ReadingData(file_path)
    data = reader.get_data()

    all_events = []

    chunker = CreatingChunks(data)
    detector = EventDetection(std_multiplier=0.5, threshold_multiplier=2.5)

    for sweepNumber in data.sweepList:
        data.setSweep(sweepNumber)
        if data.sweepY.mean() > 0:
            sweep_data = data.sweepY 
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

        else:

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
        selected_file = None
        folder_path = st.text_input('Enter folder path here:')
        if folder_path and os.path.isdir(folder_path):  # Check if it's a valid directory
            files = os.listdir(folder_path)
            if files:  # Check if the directory has any files
                selected_file = st.selectbox("Choose a file from the folder:", files)
            else:
                st.warning("The provided directory is empty!")
    if selected_file:
        data_file = os.path.join(folder_path, selected_file)
        
        event_detection,statistical_analysis, power_spectrum = st.tabs(['Event Detection','Statistical Analysis','Power Spectrum Analysis'])

        with event_detection:
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
                    on = st.toggle("Detect Events")

                with col2:
                    try:
                        if on:
                            st.markdown("Ion Current Trace With Detected Events:")

                            # Use your plotter to display the plot
                            plotter = Plotting()
                            plot_figure = plotter.plot_data(sweep_time, sweep_data, all_events)
                            st.plotly_chart(plot_figure)

                    except Exception as e:
                        st.write("Error: ", e)

        with statistical_analysis:

            col1, col2,col3,col4,col5 = st.columns(5)
            with col1:
                options = st.selectbox("Exponential Fit Options (Single or Double)?", 
                                            ('single', 'double'),
                                            index = 0,
                                            placeholder = "Select Fit Options")
            with col2:
                bins = st.number_input("Select bins for Exponential Fit (Default value set to 20 bins)", value=20)
    
            with st.container():
                col1, col2 = st.columns(2)
                # Calculate the 99.9th percentile value of the 'difference' column
                percentile_99_9 = np.percentile(events_df['difference'], 99.9)
                # Filter the DataFrame to only include rows where 'difference' is less than the 99.9th percentile
                events_df = events_df[events_df['difference'] < percentile_99_9]

                with col1:
                    
                    fit = DwellTime_ExponentialFit(events_df, bins = bins)
                    hist= fit.plot_hist_data()
                    st.plotly_chart(hist)

                with col2:
                    
                    
                    if options == 'single':
                        fit = DwellTime_ExponentialFit(events_df, bins = bins)  # Create instance of the class
                        fit.fit_data('single')  # Fit the data
                        hist_fit = fit.plot_data('single')  # Plot the data
                        st.plotly_chart(hist_fit)
                        parms = fit.print_parameters('single')

                        with st.container():
                            fit1, fit2 = st.columns(2)
                            with fit1:
                                st.text('Fit Params:')
                                a = round(parms[0], 4)
                                tau = round(parms[1], 4)

                                # Create a dictionary with the data
                                data = {'Parameter': ['a', 'tau'],
                                        'Value': [a, tau]}
                                
                                # Convert the dictionary to a DataFrame
                                parameters_df = pd.DataFrame(data)
                                parameters_df = parameters_df.set_index('Parameter')
                                # display the datafrrame
                                st.dataframe(parameters_df.transpose())


                    elif options == 'double':
                        st.text('testing!')
                        fit = DwellTime_ExponentialFit(events_df, bins = bins)  # Create instance of the class
                        fit.fit_data('double')  # Fit the data
                        hist_fit = fit.plot_data('double')  # Plot the data
                        st.plotly_chart(hist_fit)
                        parms = fit.print_parameters('double')

                        with st.container():
                            

                            st.text('Fit Params:')
                            a1 = round(parms[0], 4)
                            tau1 = round(parms[1], 4)
                            a2 = round(parms[2], 4)
                            tau2 = round(parms[3], 4)

                            # Create a dictionary with the data
                            data = {'Parameter': ['a1', 'tau1','a2', 'tau2'],
                                    'Value': [a1, tau1, a2, tau2]}
                            
                            # Convert the dictionary to a DataFrame
                            parameters_df = pd.DataFrame(data)
                            parameters_df = parameters_df.set_index('Parameter')
                            # display the datafrrame
                            st.dataframe(parameters_df.transpose())

        with power_spectrum:

            with st.container():
                col1, col2 = st.columns(2)

                with col1:
                    # First, compute the power spectrum
                    analyzer = PSDAnalyzer(fs=50000)
                    frequencies, power_spectrum = analyzer.compute_psd_with_hamming(sweep_data)
                    psd_plot = analyzer._plot_psd(frequencies,power_spectrum)
                    st.plotly_chart(psd_plot)
                
                with col2:

                    lorentzian_fitter = LorentzianFitter(frequencies, power_spectrum)
                    lorentzian_fitter.fit_lorentzian()
                    psd_plot_fit = lorentzian_fitter.plot_fit(frequencies, power_spectrum)
                    st.plotly_chart(psd_plot_fit)
                    # fitting params
                    with st.container():
                    
                        S0 = round(lorentzian_fitter.S_0_opt,2)
                        fc = round(lorentzian_fitter.f_c_opt,2)
                        # Create a dictionary with the data
                        data = {'Parameter': ['S0 (pA^2/Hz)', 'fc (Hz)'],
                                'Value': [S0, fc]}
                        
                        # Convert the dictionary to a DataFrame
                        parameters_df = pd.DataFrame(data)
                        parameters_df = parameters_df.set_index('Parameter')
                        # display the datafrrame
                        st.text('Fit Params:')
                        st.dataframe(parameters_df.transpose())



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
