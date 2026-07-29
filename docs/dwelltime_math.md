# Dwell-time lifetime fitting mathematics

This document describes the algorithms in
`packages/pynanopore/src/pynanopore/dwelltime/fit.py`.

---

## 1. Data

Dwell times \(\{t_i\}_{i=1}^{N}\) are taken from event column `difference` (or `dwell_time`),
keeping only positive finite values.

Optional percentile clipping (in the stats service) removes extreme outliers before fitting.

---

## 2. Histogram (display)

Density histogram with either:

- **linear** bins over \([\min t, \max t]\), or  
- **log** bins via \(\mathrm{geomspace}(\min t, \max t)\).

The histogram is for visualization and legacy curve fitting; MLE uses the unbinned times.

---

## 3. Single-exponential lifetime (MLE)

Model PDF:

\[
p(t \mid \tau) = \frac{1}{\tau}\, e^{-t/\tau}, \qquad t > 0,\; \tau > 0
\]

Log-likelihood:

\[
\ell(\tau) = -N\log\tau - \frac{1}{\tau}\sum_i t_i
\]

MLE:

\[
\hat\tau = \bar t = \frac{1}{N}\sum_i t_i
\]

---

## 4. Double-exponential mixture (MLE)

\[
p(t \mid w,\tau_1,\tau_2) =
w\,\frac{1}{\tau_1}e^{-t/\tau_1}
+ (1-w)\,\frac{1}{\tau_2}e^{-t/\tau_2}
\]

with \(0 < w < 1\), \(\tau_1,\tau_2 > 0\).

Parameters are optimized by maximizing \(\sum_i \log p(t_i)\) (Nelder–Mead on
transformed variables \(\mathrm{logit}(w)\), \(\log\tau_1\), \(\log\tau_2\)).
Components are ordered so \(\tau_1 \le \tau_2\).

---

## 5. Model selection

\[
\mathrm{AIC} = 2k - 2\ell,\qquad
\mathrm{BIC} = k\ln N - 2\ell
\]

| Model | \(k\) |
|-------|------|
| single | 1 (\(\tau\)) |
| double | 3 (\(w,\tau_1,\tau_2\)) |

`fit_type='auto'` chooses the model with **lower AIC**.

---

## 6. Legacy histogram fit

`method='histogram'` still supports unconstrained

\[
a\,e^{b x}\quad\text{or}\quad a\,e^{b x}+c\,e^{d x}
\]

via `scipy.optimize.curve_fit` on the density histogram. When \(b<0\), an approximate
lifetime \(\tau \approx -1/b\) is reported for convenience. Prefer **MLE** for physical \(\tau\).

---

## 7. Usage

```python
from pynanopore import DwellTimeExponentialFit
import pandas as pd

fit = DwellTimeExponentialFit(events_df, bins=50, binning="log")
result = fit.fit("auto", method="mle")
print(result.parameters, result.aic, result.bic)
```

CLI:

```bash
pynanopore dwelltime events.csv --fit auto --method mle --binning log
```
