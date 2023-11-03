import pytest
import numpy as np
import pandas as pd
from pynanopore.dwelltime import DwellTime_ExponentialFit 

class TestDwellTimeExponentialFit:
    """
    Test suite for the DwellTime_ExponentialFit class.
    """
    
    @pytest.fixture
    def sample_events_df(self) -> pd.DataFrame:
        """
        Provides a sample DataFrame for testing.
        """
        data = {'difference': np.random.exponential(scale=1.0, size=1000)}
        return pd.DataFrame(data)
    
    @pytest.fixture
    def exp_fit(self, sample_events_df) -> DwellTime_ExponentialFit:
        """
        Provides a DwellTime_ExponentialFit instance with a sample dataframe for testing.
        """
        return DwellTime_ExponentialFit(sample_events_df)
    
    def test_init(self, exp_fit: DwellTime_ExponentialFit):
        """
        Test the initialization of the DwellTime_ExponentialFit class.
        """
        assert isinstance(exp_fit.events_df, pd.DataFrame), "Initialization of events_df failed."
        assert exp_fit.bins == 250, "Default bins value should be 250."
    
    def test_prepare_histogram(self, exp_fit: DwellTime_ExponentialFit):
        """
        Test the histogram preparation.
        """
        hist, bin_centers = exp_fit._prepare_histogram()
        assert len(hist) == exp_fit.bins, "Histogram bins count mismatch."
        assert len(bin_centers) == exp_fit.bins, "Bin centers count mismatch."
    
    def test_fit_data_single(self, exp_fit: DwellTime_ExponentialFit):
        """
        Test fitting data with a single exponential function.
        """
        exp_fit.fit_data('single')
        assert exp_fit.params_single is not None, "Single exponential parameters should not be None."
    
    # def test_fit_data_double(self, exp_fit: DwellTime_ExponentialFit):
    #     """
    #     Test fitting data with a double exponential function.
    #     """
    #     exp_fit.fit_data('double')
    #     assert exp_fit.params_double is not None, "Double exponential parameters should not be None."
    
    def test_fit_data_invalid(self, exp_fit: DwellTime_ExponentialFit):
        """
        Test fitting data with an invalid fit type.
        """
        with pytest.raises(ValueError):
            exp_fit.fit_data('invalid')
    
    def test_plot_hist_data(self, exp_fit: DwellTime_ExponentialFit):
        """
        Test the histogram plotting function.
        """
        fig = exp_fit.plot_hist_data()
        assert fig is not None, "Histogram plot should not be None."
    
    def test_plot_data_single(self, exp_fit: DwellTime_ExponentialFit):
        """
        Test plotting with single exponential fit data.
        """
        exp_fit.fit_data('single')
        fig = exp_fit.plot_data('single')
        assert fig is not None, "Single fit plot should not be None."
    
    def test_plot_data_double(self, exp_fit: DwellTime_ExponentialFit):
        """
        Test plotting with double exponential fit data.
        """
        exp_fit.fit_data('double')
        fig = exp_fit.plot_data('double')
        assert fig is not None, "Double fit plot should not be None."
    
    def test_print_parameters_single(self, exp_fit: DwellTime_ExponentialFit):
        """
        Test printing parameters for single exponential fit.
        """
        exp_fit.fit_data('single')
        a, b = exp_fit.print_parameters('single')
        assert a is not None and b is not None, "Parameters for single exponential should not be None."
    
    # def test_print_parameters_double(self, exp_fit: DwellTime_ExponentialFit):
    #     """
    #     Test printing parameters for double exponential fit.
    #     """
    #     exp_fit.fit_data('double')
    #     a, b, c, d = exp_fit.print_parameters('double')
    #     assert None not in (a, b, c, d), "Parameters for double exponential should not be None."
    
    def test_print_parameters_invalid(self, exp_fit: DwellTime_ExponentialFit):
        """
        Test printing parameters with an invalid fit type.
        """
        with pytest.raises(ValueError):
            exp_fit.print_parameters('invalid')


