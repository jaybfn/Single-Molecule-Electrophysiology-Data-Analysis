# PSD analysis mathematics

Algorithms live in `packages/pynanopore/src/pynanopore/psd/`.

## 1. Welch PSD

For a current trace $I[n]$ sampled at $f_s$, Welch’s method estimates the one-sided spectrum
$S(f)$ by averaging modified periodograms of overlapping segments.

Configurable knobs (via `PSDAnalyzer.compute_psd`):

| Parameter | Meaning |
|-----------|---------|
| `nperseg` | Samples per segment |
| `noverlap` | Overlap between segments |
| `window` | e.g. `hamming`, `hann`, `blackman` |
| `scaling` | `spectrum` or `density` (SciPy `welch`) |
| `skip_bins` | Drop lowest frequency bins (DC leakage) |

## 2. Lorentzian (power-1)

$$
S(f) = \frac{S_0}{1 + (f / f_c)^2}
$$

Fit in log–log space with Trust Region Reflective least squares.
Diagnostics:

$$
R^2_{\log} = 1 - \frac{\sum (y_i - \hat y_i)^2}{\sum (y_i - \bar y)^2},
\quad y = \log_{10} S
$$

plus RMSE in log10 space.

## 3. Composite Lorentzian + $1/f^\alpha$

$$
S(f) = \frac{S_0}{1 + (f / f_c)^2} + \frac{A}{f^{\alpha}}
$$

Same log-space residual fitting; returns $S_0, f_c, A, \alpha$.

## 4. Usage

```python
from pynanopore import PSDAnalyzer, LorentzianFitter, CompositePSDFitter

f, p = PSDAnalyzer(fs).compute_psd(current, nperseg=4096, window="hann")
params = CompositePSDFitter(f, p).fit()
```

```bash
pynanopore psd file.abf --fit --fit-model composite --window hann --nperseg 4096
```
