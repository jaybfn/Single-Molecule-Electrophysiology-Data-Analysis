"""Command-line interface for batch analysis."""

from __future__ import annotations

import argparse
import json
import sys

import pandas as pd

from pynanopore import (
    DwellTimeExponentialFit,
    EventDetector,
    LorentzianFitter,
    PSDAnalyzer,
    __version__,
    load_trace,
)


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
        choices=["none", "median", "constant"],
        default="none",
        help="Baseline estimator",
    )
    detect.add_argument("--baseline-window", type=float, default=0.05, help="Median window (s)")
    detect.add_argument("--output", "-o", help="Write events CSV to this path")

    dwell = sub.add_parser("dwelltime", help="Fit dwell-time histogram from events CSV")
    dwell.add_argument("events_csv", help="CSV with a 'difference' column")
    dwell.add_argument("--fit", choices=["single", "double"], default="single")
    dwell.add_argument("--bins", type=int, default=50)

    psd = sub.add_parser("psd", help="Compute PSD (+ optional Lorentzian fit)")
    psd.add_argument("file", help="Path to .abf or .csv file")
    psd.add_argument("--fs", type=float, default=None, help="Override sample rate")
    psd.add_argument("--fit", action="store_true", help="Fit Lorentzian model")

    args = parser.parse_args(argv)

    if args.command == "detect":
        from pynanopore.detection.baseline import ConstantBaseline, MedianBaseline, NoneBaseline

        trace = load_trace(args.file)
        if args.baseline == "median":
            baseline: MedianBaseline | ConstantBaseline | NoneBaseline = MedianBaseline(
                window_s=args.baseline_window
            )
        elif args.baseline == "constant":
            baseline = ConstantBaseline()
        else:
            baseline = NoneBaseline()
        detector = EventDetector(
            std_multiplier=args.std_multiplier,
            threshold_multiplier=args.threshold_multiplier,
            direction=args.direction,
            baseline=baseline,
        )
        events = detector.detect_trace(
            trace, interval_length=args.interval, overlap=args.overlap
        )
        df = pd.DataFrame([e.to_dict() for e in events])
        if args.output:
            df.to_csv(args.output, index=False)
            print(f"Wrote {len(df)} events to {args.output}")
        else:
            print(df.to_string(index=False) if not df.empty else "No events detected")
        return 0

    if args.command == "dwelltime":
        events_df = pd.read_csv(args.events_csv)
        fit = DwellTimeExponentialFit(events_df, bins=args.bins)
        fit.fit_data(args.fit)
        params = fit.get_parameters(args.fit)
        print(json.dumps({"fit": args.fit, "parameters": params}, indent=2))
        return 0

    if args.command == "psd":
        trace = load_trace(args.file)
        fs = args.fs if args.fs is not None else trace.sample_rate
        analyzer = PSDAnalyzer(fs=fs)
        frequencies, power_spectrum = analyzer.compute_psd_with_hamming(trace.current)
        result: dict = {
            "n_frequencies": len(frequencies),
            "fs": fs,
        }
        if args.fit:
            fitter = LorentzianFitter(frequencies, power_spectrum)
            s0, fc = fitter.fit_lorentzian()
            result["S0"] = s0
            result["fc"] = fc
        print(json.dumps(result, indent=2))
        return 0

    return 1


if __name__ == "__main__":
    sys.exit(main())
