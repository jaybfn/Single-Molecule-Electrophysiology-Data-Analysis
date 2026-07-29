"""Pynanopore: single-molecule nanopore electrophysiology analysis."""

from pynanopore._version import __version__
from pynanopore.batch import BatchDetectConfig, batch_detect
from pynanopore.detection.baseline import ConstantBaseline, MedianBaseline, NoneBaseline
from pynanopore.detection.chunking import ChunkGenerator
from pynanopore.detection.events import Event, EventDetector
from pynanopore.detection.pulse_shape import PulseShapeIdealizer, PulseShapeResult
from pynanopore.dwelltime.fit import DwellTimeExponentialFit, DwellTimeFitResult
from pynanopore.io.readers import load_trace
from pynanopore.io.trace import Trace
from pynanopore.psd.analyzer import PSDAnalyzer
from pynanopore.psd.lorentzian import CompositePSDFitter, LorentzianFitter, PSDFitDiagnostics

__all__ = [
    "Trace",
    "load_trace",
    "ChunkGenerator",
    "Event",
    "EventDetector",
    "NoneBaseline",
    "ConstantBaseline",
    "MedianBaseline",
    "PulseShapeIdealizer",
    "PulseShapeResult",
    "DwellTimeExponentialFit",
    "DwellTimeFitResult",
    "PSDAnalyzer",
    "LorentzianFitter",
    "CompositePSDFitter",
    "PSDFitDiagnostics",
    "BatchDetectConfig",
    "batch_detect",
    "__version__",
]
