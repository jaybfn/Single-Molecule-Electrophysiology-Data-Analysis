# Science Phase E mathematics

Algorithms added in **v2.6** for analysis quality:

| Topic | Module |
|-------|--------|
| Percentile open-pore baseline | `detection/baseline.py` → `PercentileBaseline` |
| Multi-level conductance in events | `detection/levels.py` |
| Lorentzian + white floor | `psd/lorentzian.py` → `LorentzianWhiteFitter` |
| Double Lorentzian (+ white) | `psd/lorentzian.py` → `MultiLorentzianFitter` |
| Parallel batch | `batch.py` (`n_jobs`) |

---

## 1. Percentile open-pore baseline

When event occupancy is high, mean/median baselines are pulled toward the blocked
level. A sliding **percentile** tracks the open pore more robustly:

- Downward events → high percentile (e.g. $p=90$)
- Upward events → low percentile (e.g. $p=10$)

For each sample $n$,

$$
I_0[n] = \mathrm{percentile}_{p}\big(I[n-W/2:n+W/2]\big)
$$

with window $W = f_s \cdot w$ seconds (`window_s`, typically $\ge 0.5\,\mathrm{s}$).

Residual: $r[n] = I[n] - I_0[n]$.

---

## 2. Multi-level conductance inside an event

After dual-threshold detection, the event segment $\{I[k]\}_{k=s}^{e}$ is clustered
with **1-D k-means** for $k\in\{1,2\}$. Model order uses a Gaussian **BIC**:

$$
\mathrm{BIC}(k) = -2\,\ell(k) + (k+1)\ln N
$$

Levels are ordered by distance from the local open pore $I_0$. Exports:

- `n_levels`, `level1_current`, `level2_current`
- `level1_fraction`, `level2_fraction`
- `level_sep` $= |L_2-L_1|/|I_0|$, `level_rms`

This is a fast post-hoc feature, not a full QuB/HMM idealization.

---

## 3. PSD models

### Lorentzian + white floor

$$
S(f)=\frac{S_0}{1+(f/f_c)^2}+N
$$

### Double Lorentzian + white

$$
S(f)=\sum_{i=1}^{2}\frac{S_{0,i}}{1+(f/f_{c,i})^2}+N
$$

Fits remain in log-space (`least_squares` / TRF) with the same diagnostics
(`r2_log`, `rmse_log`) as the single Lorentzian / composite models.

---

## 4. Parallel batch

`BatchDetectConfig.n_jobs`:

- `1` — serial (default)
- `N>1` — `ProcessPoolExecutor` with $N$ workers
- `-1` — all CPUs

Schema version for batch metadata: **1.1.0** (adds `n_jobs`, multilevel summary columns).
