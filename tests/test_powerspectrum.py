# test_psd_analysis.py

import numpy as np
import pytest
from pynanopore.powerspectrum import PSDAnalyzer, LorentzianFitter

# Test for PSDAnalyzer
def test_compute_psd_with_hamming():
    # Generate a synthetic current data signal
    np.random.seed(0)  # Ensures reproducibility
    current_data = np.random.randn(1000)
    
    # Initialize PSDAnalyzer
    analyzer = PSDAnalyzer(fs=1000)
    
    # Compute PSD
    frequencies, power_spectrum = analyzer.compute_psd_with_hamming(current_data)
    
    # Assertions to check if compute_psd_with_hamming is working as expected
    assert len(frequencies) > 0, "No frequencies returned"
    assert len(power_spectrum) > 0, "No power spectrum returned"
    assert len(frequencies) == len(power_spectrum), "Frequencies and power spectrum lengths do not match"

# Test for LorentzianFitter
def test_fit_lorentzian():
    # Generate synthetic data
    np.random.seed(0)
    frequencies = np.linspace(1, 1000, 100)
    power_spectrum = 1 / (frequencies**2 + 1) + np.random.normal(0, 0.1, 100)
    
    # Ensure no zero or negative frequencies and power spectrum values
    frequencies = np.where(frequencies <= 0, np.finfo(float).eps, frequencies)
    power_spectrum = np.where(power_spectrum <= 0, np.finfo(float).eps, power_spectrum)

    # Initialize LorentzianFitter
    fitter = LorentzianFitter(frequencies, power_spectrum)
    
    # Try to fit the model and catch any ValueError
    try:
        fitter.fit_lorentzian()
    except ValueError as e:
        print(f"ValueError: {e}")
        print("Initial frequencies:", frequencies)
        print("Initial power spectrum:", power_spectrum)
        print("Initial guess for fitting might be causing the problem.")
        raise  # Re-raise the exception after logging
    
    # Check if the S_0_opt and f_c_opt have been set
    assert fitter.S_0_opt is not None, "S_0_opt is None"
    assert fitter.f_c_opt is not None, "f_c_opt is None"
    
    # Check if the optimized parameters are within the expected range
    assert 0 < fitter.S_0_opt < 10, "S_0_opt is out of expected range"
    assert 0 < fitter.f_c_opt < 2000, "f_c_opt is out of expected range"