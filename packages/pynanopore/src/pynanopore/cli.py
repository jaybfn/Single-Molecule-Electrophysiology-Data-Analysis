"""Command-line interface for batch analysis."""

from __future__ import annotations

import argparse
import json
import sys

import pandas as pd

from pynanopore import (
    BatchDetectConfig,
    DwellTimeExponentialFit,
    EventDetector,
    LorentzianFitter,
    PSDAnalyzer,
    __version__,
    batch_detect,
    load_trace,
)
from pynanopore.detection.baseline import (
    ConstantBaseline,
    MedianBaseline,
    NoneBaseline,
    PercentileBaseline,
)
from pynanopore.psd.lorentzian import (
    CompositePSDFitter,
    LorentzianWhiteFitter,
    MultiLorentzianFitter,
)


def _make_baseline(name: str, window_s: float, percentile: float = 90.0):
    if name == "median":
        return MedianBaseline(window_s=window_s)
    if name == "constant":
        return ConstantBaseline()
    if name == "percentile":
        return PercentileBaseline(percentile=percentile, window_s=max(window_s, 0.5))
    return NoneBaseline()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="pynanopore",
        description="Single-molecule nanopore electrophysiology analysis CLI",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    detect = sub.add_parser("detect", help="Detect events in a recording")
    detect.add_argument("file", help="Path to .abf or .csv file")
    detect.add_argument("--std-multiplier", type=float, default=0.25)
    detect.add_argument("--threshold-multiplier", type=float, default=1.5)
    detect.add_argument("--interval", type=float, default=5.0)
    detect.add_argument("--overlap", type=float, default=0.0, help="Chunk overlap in seconds")
    detect.add_argument(
        "--direction",
        choices=["down", "up"],
        default="down",
        help="Event polarity relative to baseline",
    )
    detect.add_argument(
        "--baseline",
        choices=["none", "median", "constant", "percentile"],
        default="none",
        help="Baseline estimator",
    )
    detect.add_argument("--baseline-window", type=float, default=0.05, help="Baseline window (s)")
    detect.add_argument(
        "--baseline-percentile",
        type=float,
        default=90.0,
        help="Percentile for percentile baseline (use ~10 for upward events)",
    )
    detect.add_argument("--no-levels", action="store_true", help="Skip multi-level analysis")
    detect.add_argument("--output", "-o", help="Write events CSV to this path")

    batch = sub.add_parser("batch-detect", help="Detect events for all ABF/CSV files in a folder")
    batch.add_argument("input_dir", help="Folder containing .abf / .csv recordings")
    batch.add_argument("-o", "--output-dir", required=True, help="Output directory for results")
    batch.add_argument("--std-multiplier", type=float, default=0.25)
    batch.add_argument("--threshold-multiplier", type=float, default=1.5)
    batch.add_argument("--interval", type=float, default=5.0)
    batch.add_argument("--overlap", type=float, default=0.0)
    batch.add_argument("--direction", choices=["down", "up"], default="down")
    batch.add_argument(
        "--baseline", choices=["none", "median", "constant", "percentile"], default="none"
    )
    batch.add_argument("--baseline-window", type=float, default=0.05)
    batch.add_argument("--baseline-percentile", type=float, default=90.0)
    batch.add_argument("--no-dwell-fit", action="store_true", help="Skip per-file dwell MLE")
    batch.add_argument("--dwell-fit", choices=["single", "double", "auto"], default="single")
    batch.add_argument("--no-levels", action="store_true")
    batch.add_argument(
        "--n-jobs",
        type=int,
        default=1,
        help="Parallel workers (1=serial, -1=all CPUs)",
    )

    dwell = sub.add_parser("dwelltime", help="Fit dwell-time histogram from events CSV")
    dwell.add_argument("events_csv", help="CSV with a 'difference' column")
    dwell.add_argument("--fit", choices=["single", "double", "auto"], default="single")
    dwell.add_argument("--method", choices=["mle", "histogram"], default="mle")
    dwell.add_argument("--binning", choices=["linear", "log"], default="linear")
    dwell.add_argument("--bins", type=int, default=50)

    psd = sub.add_parser("psd", help="Compute PSD (+ optional model fit)")
    psd.add_argument("file", help="Path to .abf or .csv file")
    psd.add_argument("--fs", type=float, default=None, help="Override sample rate")
    psd.add_argument("--fit", action="store_true", help="Fit a spectral model")
    psd.add_argument(
        "--fit-model",
        choices=["lorentzian", "composite", "lorentzian_white", "double_lorentzian"],
        default="lorentzian",
    )
    psd.add_argument("--nperseg", type=int, default=None)
    psd.add_argument("--noverlap", type=int, default=None)
    psd.add_argument("--window", default="hamming")
    psd.add_argument("--scaling", choices=["spectrum", "density"], default="spectrum")
    psd.add_argument("--max-frequency", type=float, default=10000.0)

    args = parser.parse_args(argv)

    if args.command == "detect":
        trace = load_trace(args.file)
        detector = EventDetector(
            std_multiplier=args.std_multiplier,
            threshold_multiplier=args.threshold_multiplier,
            direction=args.direction,
            baseline=_make_baseline(args.baseline, args.baseline_window, args.baseline_percentile),
            analyze_levels=not args.no_levels,
        )
        events = detector.detect_trace(trace, interval_length=args.interval, overlap=args.overlap)
        df = pd.DataFrame([e.to_dict() for e in events])
        if args.output:
            df.to_csv(args.output, index=False)
            print(f"Wrote {len(df)} events to {args.output}")
        else:
            print(df.to_string(index=False) if not df.empty else "No events detected")
        return 0

    if args.command == "batch-detect":
        cfg = BatchDetectConfig(
            std_multiplier=args.std_multiplier,
            threshold_multiplier=args.threshold_multiplier,
            direction=args.direction,
            baseline=args.baseline,
            baseline_window=args.baseline_window,
            baseline_percentile=args.baseline_percentile,
            interval_length=args.interval,
            overlap=args.overlap,
            fit_dwelltime=not args.no_dwell_fit,
            dwell_fit_type=args.dwell_fit,
            analyze_levels=not args.no_levels,
            n_jobs=args.n_jobs,
        )
        summary = batch_detect(args.input_dir, args.output_dir, cfg)
        ok = int((summary["status"] == "ok").sum()) if "status" in summary.columns else 0
        print(f"Processed {len(summary)} files ({ok} ok). Summary: {args.output_dir}/summary.csv")
        return 0

    if args.command == "dwelltime":
        events_df = pd.read_csv(args.events_csv)
        fit = DwellTimeExponentialFit(events_df, bins=args.bins, binning=args.binning)
        result = fit.fit(args.fit, method=args.method)
        print(json.dumps(result.to_dict(), indent=2))
        return 0

    if args.command == "psd":
        trace = load_trace(args.file)
        fs = args.fs if args.fs is not None else trace.sample_rate
        analyzer = PSDAnalyzer(fs=fs)
        frequencies, power_spectrum = analyzer.compute_psd(
            trace.current,
            nperseg=args.nperseg,
            noverlap=args.noverlap,
            window=args.window,
            scaling=args.scaling,
        )
        psd_result: dict = {
            "n_frequencies": len(frequencies),
            "fs": fs,
            "window": args.window,
            "scaling": args.scaling,
        }
        if args.fit:
            fitter: (
                CompositePSDFitter
                | LorentzianWhiteFitter
                | MultiLorentzianFitter
                | LorentzianFitter
            )
            if args.fit_model == "composite":
                fitter = CompositePSDFitter(
                    frequencies, power_spectrum, max_frequency=args.max_frequency
                )
                psd_result.update(fitter.fit())
            elif args.fit_model == "lorentzian_white":
                fitter = LorentzianWhiteFitter(
                    frequencies, power_spectrum, max_frequency=args.max_frequency
                )
                psd_result.update(fitter.fit())
            elif args.fit_model == "double_lorentzian":
                fitter = MultiLorentzianFitter(
                    frequencies,
                    power_spectrum,
                    n_components=2,
                    include_white=True,
                    max_frequency=args.max_frequency,
                )
                psd_result.update(fitter.fit())
            else:
                fitter = LorentzianFitter(
                    frequencies, power_spectrum, max_frequency=args.max_frequency
                )
                s0, fc = fitter.fit_lorentzian()
                psd_result["S0"] = s0
                psd_result["fc"] = fc
            diagnostics = fitter.diagnostics
            if diagnostics is not None:
                psd_result["diagnostics"] = diagnostics.to_dict()
        print(json.dumps(psd_result, indent=2))
        return 0

    return 1


if __name__ == "__main__":
    sys.exit(main())
