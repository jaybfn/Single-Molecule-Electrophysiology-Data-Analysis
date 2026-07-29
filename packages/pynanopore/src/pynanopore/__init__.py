"""Pynanopore: single-molecule nanopore electrophysiology analysis."""

from pynanopore.detection.chunking import ChunkGenerator
from pynanopore.detection.events import Event, EventDetector
from pynanopore.dwelltime.fit import DwellTimeExponentialFit
from pynanopore.io.readers import load_trace
from pynanopore.io.trace import Trace
from pynanopore.psd.analyzer import PSDAnalyzer
from pynanopore.psd.lorentzian import LorentzianFitter

__version__ = "2.0.0"

__all__ = [
    "Trace",
    "load_trace",
    "ChunkGenerator",
    "Event",
    "EventDetector",
    "DwellTimeExponentialFit",
    "PSDAnalyzer",
    "LorentzianFitter",
    "__version__",
]
