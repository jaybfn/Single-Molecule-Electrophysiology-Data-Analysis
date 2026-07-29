# Event detection mathematics

This document describes the algorithms implemented in
`packages/pynanopore/src/pynanopore/detection/`.

Related code:

| Module | Role |
|--------|------|
| `baseline.py` | Open-pore baseline estimation |
| `events.py` | Dual-threshold detection + event features |
| `pulse_shape.py` | Rectangular pulse idealization |
| `viz/plotting.py` → `plot_pulse_shape` | Raw + ideal overlay (screenshot-style) |

---

## 1. Signal model

A recording is a sampled ion current \(I[n]\) at rate \(f_s\) with times \(t[n]\).

Many ABF files have a **negative** open-pore current. Events may appear as:

- **down** — excursions *below* the open pore (classic positive-current convention after polarity flip), or  
- **up** — excursions *above* a negative baseline (as in the example pulse-shape figure).

Pynanopore always detects on a **canonical residual** where events are downward (see §3).

---

## 2. Baseline estimation

Let \(B[n]\) be an estimate of the slowly varying open-pore level.

| Estimator | Definition |
|-----------|------------|
| `NoneBaseline` | \(B[n] = \mathrm{mean}(I)\) over the analysis window (constant) |
| `ConstantBaseline` | \(B[n] = c\) (user value or median) |
| `MedianBaseline(window_s)` | \(B[n] = \mathrm{median}\{I[k] : \|k-n\| \le W/2\}\) with \(W \approx f_s\cdot\mathrm{window\_s}\) |

Residual:

\[
R[n] = I[n] - B[n]
\]

Median baseline removes slow drift so local thresholds are not biased by a tilting open pore.

---

## 3. Dual-threshold detection

### 3.1 Canonical work signal

\[
W[n] =
\begin{cases}
R[n] & \text{if direction = down} \\
-R[n] & \text{if direction = up}
\end{cases}
\]

Events are always sought as **negative** excursions of \(W\).

### 3.2 Thresholds

Over a chunk of \(N\) samples:

\[
\mu_W = \frac{1}{N}\sum_n W[n],
\qquad
\sigma_W = \sqrt{\frac{1}{N}\sum_n (W[n]-\mu_W)^2}
\]

With multipliers \(k_{\mathrm{std}}\) (`std_multiplier`) and \(k_{\mathrm{thr}}\) (`threshold_multiplier`):

\[
T_{\mathrm{entry}} = \mu_W - k_{\mathrm{std}}\,\sigma_W
\qquad
T_{\mathrm{deep}} = \mu_W - k_{\mathrm{thr}}\,\sigma_W
\]

Defaults: \(k_{\mathrm{std}}=0.25\), \(k_{\mathrm{thr}}=1.5\).  
Require \(k_{\mathrm{thr}} \ge k_{\mathrm{std}}\) so \(T_{\mathrm{deep}} \le T_{\mathrm{entry}}\).

```text
W (canonical)
  ^
  | ——— μ_W
  | · · · T_entry   ← start / end crossings
  |   · · T_deep    ← must be visited inside event
  |     ╲___╱
  +------------→ n
```

### 3.3 State machine

For \(n = 1 \ldots N-1\):

1. **Start** when \(W\) crosses below entry:

\[
W[n-1] \ge T_{\mathrm{entry}} \;\wedge\; W[n] < T_{\mathrm{entry}}
\]

2. **Confirm** if any sample while open satisfies \(W[n] < T_{\mathrm{deep}}\).

3. **End** when \(W\) crosses back above entry:

\[
W[n-1] < T_{\mathrm{entry}} \;\wedge\; W[n] \ge T_{\mathrm{entry}}
\]

4. **Accept** if confirmed and dwell \(\tau = t_{\mathrm{end}} - t_{\mathrm{start}} \ge \tau_{\min}\)
   (default \(\tau_{\min} = 10^{-4}\,\mathrm{s}\)).

This is a hysteresis-style rule: a sensitive edge threshold plus a deeper confirmation level to reject shallow noise.

### 3.4 Chunking and overlap

Long traces are processed in windows of length \(\Delta t\) (default 5 s):

\[
N_{\mathrm{chunk}} = \lfloor f_s \Delta t \rfloor
\]

Optional overlap \(\delta\) (seconds) advances the window by \(\Delta t - \delta\).
Duplicate events are removed by unique absolute `start_idx`.

---

## 4. Event features

For an accepted event on samples \([n_s, n_e]\) in the **original** current \(I\):

| Symbol | Field | Definition |
|--------|-------|------------|
| \(t_s, t_e\) | `start_time`, `end_time` | Event bounds (s) |
| \(\tau\) | `difference`, `dwell_time` | \(t_e - t_s\) |
| \(I_0\) | `i0` | Local open pore \(B[n_s]\) |
| \(I_b\) | `blockade_mean` | \(\mathrm{mean}(I[n_s:n_e])\) |
| | `blockade_min` / `blockade_max` | Extremes inside the event |
| \(A_{\mathrm{ext}}\) | `amplitude` | \(\min I\) (down) or \(\max I\) (up) |
| \(\Delta I\) | `delta_i` | \(\lvert I_0 - I_b\rvert\) |
| \(\Delta I/I_0\) | `delta_i_over_i0` | \(\Delta I / \lvert I_0\rvert\) |
| Area | `area` | \(\sum_n \lvert I_0 - I[n]\rvert / f_s\) |
| | `rise_time`, `fall_time` | 10–90% transition estimates (§4.1) |

### 4.1 Rise / fall times

Define the fractional depth inside the event:

\[
f[n] = \frac{I[n] - I_0}{I_b - I_0}
\]

- **Rise time:** time between first samples with \(f \ge 0.1\) and \(f \ge 0.9\).  
- **Fall time:** analogous from the end of the event (return toward \(I_0\)).

---

## 5. Pulse-shape idealization

Given events \(\{E_k\}\), the idealized trace is piecewise constant:

\[
I_{\mathrm{ideal}}[n] =
\begin{cases}
I_b^{(k)} & n \in [n_s^{(k)}, n_e^{(k)}] \text{ for some event } k \\
I_{\mathrm{open}} & \text{otherwise}
\end{cases}
\]

where \(I_b^{(k)}\) is `blockade_mean` and \(I_{\mathrm{open}}\) is the median of event `i0` values (or a user global open level).

Plotting (`plot_pulse_shape`):

- raw \(I[n]\) (gray)
- \(I_{\mathrm{ideal}}[n]\) as a step trace (`shape='hv'`)
- **red** markers at rising edges \((t_s, I_b)\)
- **blue** markers at falling edges \((t_e, I_0)\)
- horizontal guides at open and blocked levels  

This matches the visual style of idealized single-channel / nanopore pulse overlays.

---

## 6. Usage example

```python
from pynanopore import (
    load_trace,
    EventDetector,
    MedianBaseline,
    PulseShapeIdealizer,
)
from pynanopore.viz import plot_pulse_shape

trace = load_trace("recording.abf")  # or CSV

# Upward pulses from a negative baseline (screenshot-like)
detector = EventDetector(
    std_multiplier=0.25,
    threshold_multiplier=1.5,
    direction="up",
    baseline=MedianBaseline(window_s=0.05),
)
events = detector.detect_trace(trace, interval_length=5.0, overlap=0.5)

pulse = PulseShapeIdealizer.from_events(trace, events)
fig = plot_pulse_shape(trace.time, trace.current, pulse)
fig.show()
```

For downward blockades after polarity correction, use `direction="down"` (default).

---

## 7. Parameter guidance

| Parameter | Effect if increased |
|-----------|---------------------|
| `std_multiplier` | Harder to start/end events (fewer, shorter edges) |
| `threshold_multiplier` | Requires deeper blockade (fewer false positives) |
| `min_duration` | Rejects short spikes |
| `MedianBaseline.window_s` | Smooths faster baseline wiggles; too large follows events |
| `overlap` | Fewer missed events at chunk boundaries |

Always validate with `plot_pulse_shape` on a short zoom before batch processing.
