"""
================================================================
  ORIFICE METER ANALYSIS  —  Empirical vs ISO 5167 Baseline
================================================================
  Author  : Fluid Mechanics Test Rig Analysis Script
  Output  : orifice_analysis_high_res.pdf  (one plot per page)

  PHYSICAL SETUP
  ──────────────
  Pipe Inner Diameter    D  = 24.5 mm
  Orifice Bore           d  = 13.3 mm
  Plate Design : 6 mm thick, 45–60° downstream bevel (draft)
                 for sharp-edge flow separation at the upstream face.
  Tapping Type : Custom — 2D upstream, 1D downstream
                 (NOT standard corner / D-D/2 / flange taps)
                 → Empirical Cd will intentionally deviate from ISO 5167 RHG.

  EQUATIONS
  ─────────
  EMPIRICAL  Cd (measured from rig):
      Cd_emp = Q_actual / ( A_d * sqrt( 2*delta_P / (rho*(1-beta^4)) ) )

  THEORETICAL baseline — ISO 5167-2:2003 Reader-Harris/Gallagher (RHG),
  corner-tap configuration, with small-pipe correction (D < 71.12 mm):
      Cd_RHG = f(Re_D, beta)   [see function Cd_RHG() below]

  NOTE: Deviations between empirical and RHG are EXPECTED because:
    - Tapping positions differ from corner-tap standard (2D/1D vs corner)
    - Custom bevel geometry alters vena-contracta location
    - Physical rig losses not captured in ideal ISO model

  INPUTS
  ──────
  fluid_test_data.csv           — Flow_Rate(L/min), Diff_Pressure(Pa), Temp(C)
  water_properties_IAPWS95.csv  — Temperature (degC), Density rho (kg/m3),
                                   Dynamic Viscosity mu (mPa.s),
                                   Kinematic Viscosity nu (mm2/s)

  USAGE
  ─────
  python orifice_analysis.py
  python orifice_analysis.py --csv fluid_test_data.csv
                              --iapws water_properties_IAPWS95.csv
                              --bin 0.5

  DEPENDENCIES
  ────────────
  pip install pandas numpy matplotlib scipy
================================================================
"""

import argparse
import datetime
import os
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.ticker import AutoMinorLocator
from scipy.interpolate import interp1d
from scipy.special import erfc

warnings.filterwarnings("ignore")

# =============================================================================
#  SECTION 1 — GEOMETRY  (SI units throughout)
# =============================================================================
D_m   = 24.5e-3           # pipe inner diameter  [m]
d_m   = 13.3e-3           # orifice bore         [m]
# Plate: 6 mm thick, 45-60 degree downstream bevel — ensures sharp-edge
#        flow separation at the upstream face of the orifice plate.
# Tapping: Custom — 2D upstream, 1D downstream (non-standard positions).
#          ISO 5167 RHG corner-tap equation used only as a theoretical baseline.

beta  = d_m / D_m
A_d   = np.pi / 4.0 * d_m ** 2   # orifice throat area  [m^2]
A_D   = np.pi / 4.0 * D_m ** 2   # pipe cross-section   [m^2]
g     = 9.81                      # gravitational acceleration [m/s^2]

print("\n" + "="*62)
print("  ORIFICE METER ANALYSIS — Empirical vs ISO 5167 RHG")
print("="*62)
print(f"  D  = {D_m*1000:.1f} mm      d  = {d_m*1000:.1f} mm")
print(f"  beta = {beta:.5f}    A_d = {A_d*1e6:.4f} mm^2")
print(f"  Tapping: Custom (2D upstream, 1D downstream)")
print(f"  Plate  : 6 mm, 45-60 deg downstream bevel")


# =============================================================================
#  SECTION 2 — FILE RESOLUTION
# =============================================================================
def locate_file(filename: str) -> str:
    candidates = [
        Path.cwd() / filename,
        Path(__file__).resolve().parent / filename,
        Path.home() / "Desktop" / filename,
    ]
    for p in candidates:
        if p.exists():
            return str(p)
    sys.exit(
        f"\n[ERROR] Cannot locate '{filename}'.\n"
        f"  Place it in the same folder as this script, the working directory,\n"
        f"  or on the Desktop.\n"
    )


# =============================================================================
#  SECTION 3 — IAPWS-95 PROPERTY TABLE
#  Columns: Temperature (degC) | Density rho (kg/m3) |
#           Dynamic Viscosity mu (mPa.s) | Kinematic Viscosity nu (mm2/s)
# =============================================================================
def load_iapws(path: str):
    df = pd.read_csv(path)
    t_col = next((c for c in df.columns if "temp" in c.lower()), None)
    r_col = next((c for c in df.columns
                  if ("dens" in c.lower() or "rho" in c.lower() or "\u03c1" in c)
                  and "kinematic" not in c.lower()), None)
    m_col = next((c for c in df.columns
                  if "dynamic" in c.lower()
                  or ("\u03bc" in c and "kinematic" not in c.lower())), None)

    if not all([t_col, r_col, m_col]):
        sys.exit(
            f"[ERROR] Cannot identify required columns in IAPWS-95 file.\n"
            f"  Detected: {list(df.columns)}\n"
            f"  Need: Temperature, Density, Dynamic Viscosity columns.\n"
        )

    T   = df[t_col].values.astype(float)
    rho = df[r_col].values.astype(float)
    mu  = df[m_col].values.astype(float) * 1e-3   # mPa.s -> Pa.s

    rho_fn = interp1d(T, rho, kind="cubic", fill_value="extrapolate")
    mu_fn  = interp1d(T, mu,  kind="cubic", fill_value="extrapolate")

    print(f"\n  [IAPWS-95] Loaded {len(T)} rows — T = {T.min():.0f}–{T.max():.0f} degC")
    return rho_fn, mu_fn


# =============================================================================
#  SECTION 4 — SENSOR DATA INGESTION
# =============================================================================
def load_sensor_data(path: str) -> pd.DataFrame:
    print(f"\n  [DATA] Loading: {path}")
    df = pd.read_csv(path)

    col_map = {}
    for col in df.columns:
        lc = col.lower()
        if "flow" in lc:
            col_map[col] = "flow_lpm"
        elif "press" in lc or "diff" in lc:
            col_map[col] = "dp_Pa"
        elif "temp" in lc:
            col_map[col] = "temp_C"
    df.rename(columns=col_map, inplace=True)

    missing = {"flow_lpm", "dp_Pa", "temp_C"} - set(df.columns)
    if missing:
        sys.exit(f"[ERROR] Unmatched columns: {missing}\n  Found: {list(df.columns)}\n")

    df = (df[["flow_lpm", "dp_Pa", "temp_C"]]
          .apply(pd.to_numeric, errors="coerce")
          .dropna()
          .reset_index(drop=True))

    n_raw = len(df)

    # ── Physical range filters ──────────────────────────────────────────────
    # 1. Zero or negative flow rate — no flow condition, unusable for analysis
    mask_flow = df["flow_lpm"] > 0
    n_zero_flow = (~mask_flow).sum()

    # 2. Pressure spikes above 10 kPa — sensor artefacts / transient events
    mask_pres = (df["dp_Pa"] > 0) & (df["dp_Pa"] <= 10_000)
    n_pres_spike = (~mask_pres & mask_flow).sum()

    # 3. Temperature out of physical range for liquid water (0–100 °C)
    #    Negative values → sensor fault; above 100 °C → boiling / bad reading
    mask_temp = (df["temp_C"] >= 0) & (df["temp_C"] <= 100)
    n_bad_temp = (~mask_temp).sum()

    df = df[mask_flow & mask_pres & mask_temp].reset_index(drop=True)

    print(f"  [DATA] Raw rows              : {n_raw}")
    print(f"  [DATA] Removed — zero flow   : {n_zero_flow}")
    print(f"  [DATA] Removed — dP > 10 kPa : {n_pres_spike}")
    print(f"  [DATA] Removed — bad temp    : {n_bad_temp}")
    print(f"  [DATA] Clean rows remaining  : {len(df)}")

    if len(df) == 0:
        sys.exit("[ERROR] No rows remain after physical range filtering. "
                 "Check sensor data and filter thresholds.\n")
    return df


# =============================================================================
#  SECTION 5 — SLIDING-WINDOW TEMPERATURE STABILISATION
# =============================================================================
def find_stable_temperature(temp_series: pd.Series,
                             window_sizes=None):
    if window_sizes is None:
        window_sizes = [10, 20, 30, 50]

    arr  = temp_series.values
    n    = len(arr)
    best = dict(std=np.inf, start=0, end=n, win=window_sizes[0])

    for win in window_sizes:
        if win > n:
            continue
        stds = np.array([arr[i:i + win].std() for i in range(n - win + 1)])
        idx  = int(np.argmin(stds))
        if stds[idx] < best["std"]:
            best = dict(std=stds[idx], start=idx, end=idx + win, win=win)

    T_star  = arr[best["start"]:best["end"]].mean()
    T_sigma = best["std"]

    print(f"\n  [TEMP] Stable window -> rows {best['start']}-{best['end']}"
          f"  (n = {best['win']})")
    print(f"  [TEMP] T* = {T_star:.3f} degC    sigma(T) = {T_sigma:.4f} degC")
    return T_star, T_sigma, best["start"], best["end"]


# =============================================================================
#  SECTION 6 — FLOW-RATE BINNING
# =============================================================================
def _zscore_filter_bin(group: pd.DataFrame, col: str = "dp_Pa",
                       z_thresh: float = 2.0) -> pd.DataFrame:
    """
    Within a single flow-rate bin, remove rows where the z-score of `col`
    exceeds z_thresh (default ±2σ).

    Rationale: pressure fluctuations within a nominally steady flow setpoint
    can produce a cluster of valid readings plus occasional spikes that pull
    the median away from the true operating value. Removing intra-bin outliers
    before aggregation gives a cleaner representative ΔP per flow rate.

    IQR-based rejection was considered but z-score is more appropriate here
    because the within-bin distributions are approximately Gaussian (sensor
    noise) and the sample sizes are typically 5–50 points — enough for a
    reliable mean/std estimate.

    If a bin has fewer than 3 points, or zero std (all identical), filtering
    is skipped and the full group is returned as-is.
    """
    if len(group) < 3:
        return group
    mu  = group[col].mean()
    sig = group[col].std()
    if sig == 0:
        return group
    z = (group[col] - mu).abs() / sig
    kept = group[z <= z_thresh]
    # Safety: always keep at least 1 point
    return kept if len(kept) >= 1 else group


def bin_by_flow(df: pd.DataFrame, bin_w: float = 0.5,
                z_thresh: float = 2.0) -> pd.DataFrame:
    """
    1. Round flow readings to nearest bin_w (L/min).
    2. Within each bin, apply z-score filter on ΔP (removes intra-bin spikes).
    3. Aggregate: mean flow, median ΔP, std ΔP, count (post-filter).
    """
    df = df.copy()
    df["flow_bin"] = (df["flow_lpm"] / bin_w).round() * bin_w

    # Apply intra-bin z-score filtering per group
    n_before = len(df)
    df_filtered = (df.groupby("flow_bin", group_keys=False)
                     .apply(lambda g: _zscore_filter_bin(g, "dp_Pa", z_thresh)))
    n_removed = n_before - len(df_filtered)

    if n_removed > 0:
        print(f"\n  [BIN] Intra-bin z-score filter (|z| > {z_thresh}): "
              f"removed {n_removed} rows from within-bin clusters")

    grp = (df_filtered
             .groupby("flow_bin")
             .agg(
                 flow_lpm=("flow_lpm", "mean"),
                 dp_Pa   =("dp_Pa",    "median"),
                 dp_std  =("dp_Pa",    "std"),
                 count   =("dp_Pa",    "count"),
             )
             .reset_index(drop=True)
             .sort_values("flow_lpm")
             .reset_index(drop=True))

    grp["dp_std"] = grp["dp_std"].fillna(0.0)

    # ── POST-BINNING TREND RESIDUAL FILTER ───────────────────────────────────
    # Problem the intra-bin z-score cannot catch:
    #   If ALL readings within a bin are consistently elevated (e.g. due to a
    #   sustained transient or valve perturbation during that flow setpoint),
    #   the intra-bin z-score sees a tight cluster and passes every point.
    #   The bin median then becomes a cross-bin outlier — visible as the spike
    #   at Q≈20 L/min in the ΔP vs Q plot.
    #
    # Fix — fit a power-law ΔP = a·Q^b to the binned medians in log-space,
    # compute each bin's relative residual:
    #   r_i = (ΔP_i − ΔP_fit_i) / ΔP_fit_i
    # then reject bins where |r_i| > residual_z_thresh × σ(residuals).
    # This is equivalent to ±2σ on the normalised residuals of the global
    # trend — independent of absolute pressure magnitude, so it works across
    # the full flow range.
    grp = _trend_residual_filter(grp, residual_z_thresh=2.0)

    print(f"\n  [BIN] Bin width = {bin_w} L/min  ->  {len(grp)} operating points after all filters")
    print(f"  {'Q (L/min)':>10}  {'dP med (Pa)':>12}  {'sigma(dP)':>10}  {'n_used':>6}")
    print(f"  {'-'*10}  {'-'*12}  {'-'*10}  {'-'*6}")
    for _, r in grp.iterrows():
        print(f"  {r.flow_lpm:>10.2f}  {r.dp_Pa:>12.2f}  {r.dp_std:>10.2f}  {int(r['count']):>6}")

    return grp


def _trend_residual_filter(grp: pd.DataFrame,
                            residual_z_thresh: float = 2.0) -> pd.DataFrame:
    """
    Post-binning cross-bin outlier rejection based on power-law trend residuals.

    Steps:
      1. Fit  log(ΔP) = log(a) + b·log(Q)  by OLS in log-log space.
         (For an ideal orifice b ≈ 2, but we fit empirically to allow for
          rig-specific deviations.)
      2. Compute the normalised residual for each bin:
             r_i = (ΔP_i − ΔP_fit_i) / ΔP_fit_i
      3. Compute  μ_r, σ_r  of all residuals.
      4. Reject bins where  |r_i − μ_r| / σ_r > residual_z_thresh.

    Why relative (not absolute) residuals?
      ΔP spans orders of magnitude from low to high flow, so a fixed
      absolute threshold would be too loose at high Q and too tight at low Q.
      Normalising by the fitted value makes the criterion scale-invariant.

    Requires at least 4 bins to fit reliably; if fewer, returns grp unchanged.
    """
    if len(grp) < 4:
        return grp

    try:
        log_q  = np.log(grp["flow_lpm"].values)
        log_dp = np.log(grp["dp_Pa"].values)
        b_fit, log_a = np.polyfit(log_q, log_dp, 1)
        dp_fit = np.exp(log_a) * grp["flow_lpm"].values ** b_fit

        rel_resid = (grp["dp_Pa"].values - dp_fit) / dp_fit
        mu_r  = rel_resid.mean()
        sig_r = rel_resid.std()

        if sig_r == 0:
            return grp

        z_resid = np.abs(rel_resid - mu_r) / sig_r
        keep    = z_resid <= residual_z_thresh
        n_drop  = (~keep).sum()

        if n_drop > 0:
            dropped_q  = grp.loc[~keep, "flow_lpm"].values
            dropped_dp = grp.loc[~keep, "dp_Pa"].values
            print(f"\n  [TREND FILTER] Power-law residual filter "
                  f"(|z_resid| > {residual_z_thresh}): "
                  f"removed {n_drop} bin(s)")
            for q_v, dp_v, z_v, r_v in zip(
                    dropped_q, dropped_dp,
                    z_resid[~keep], rel_resid[~keep]):
                print(f"    -> Q = {q_v:.2f} L/min, "
                      f"ΔP = {dp_v:.1f} Pa, "
                      f"residual = {r_v*100:+.1f}%, "
                      f"|z| = {z_v:.2f}")
            print(f"     Fit exponent b = {b_fit:.4f}  "
                  f"(ideal orifice = 2.000)")

        return grp[keep].reset_index(drop=True)

    except Exception as e:
        print(f"  [TREND FILTER] Skipped — {e}")
        return grp


# =============================================================================
#  SECTION 7 — THEORETICAL BASELINE: ISO 5167-2 RHG (corner taps)
# =============================================================================
def Cd_RHG(Re_D: float, b: float = beta) -> float:
    """
    ISO 5167-2:2003 Reader-Harris/Gallagher equation — corner tap configuration.
    Validity range: 5e3 <= Re_D <= 1e7,  0.1 <= beta <= 0.75.
    Includes small-pipe correction for D < 71.12 mm.

    Used as THEORETICAL BASELINE only. Actual rig uses custom 2D/1D tapping,
    so empirical Cd will deviate. This deviation is intentional and expected.
    """
    if Re_D <= 0:
        return np.nan

    b4 = b ** 4
    A  = (19000.0 * b / Re_D) ** 0.8

    Cd = (0.5961
          + 0.0261 * b**2
          - 0.216  * b**8
          + 0.000521 * (1.0e6 * b / Re_D) ** 0.7
          + (0.0188 + 0.0063 * A) * b**3.5 * (1.0e6 / Re_D) ** 0.3
          + 0.043 * (1.0 - 0.11 * A) * b4 / (1.0 - b4))

    # Small-pipe correction: D < 71.12 mm (= 2.8 inches)
    if D_m < 0.07112:
        Cd += 0.011 * (0.75 - b) * (2.8 - D_m / 0.0254)

    return Cd


# =============================================================================
#  SECTION 8 — CORE CALCULATIONS
# =============================================================================
def compute(binned: pd.DataFrame, rho: float, mu: float) -> pd.DataFrame:
    """
    Derived quantities for each binned operating point.

    EMPIRICAL Cd (from rig measurements):
        Cd_emp = Q_actual / ( A_d * sqrt( 2*delta_P / (rho*(1-beta^4)) ) )

    THEORETICAL Cd (ISO 5167-2 RHG, corner taps — baseline reference):
        Cd_RHG = f(Re_D, beta)
    """
    df = binned.copy()

    # Volumetric flow rate  Q = flow_lpm / 60000  [m^3/s]
    df["Q_m3s"]   = df["flow_lpm"] / 60_000.0

    # Mean pipe velocity  V = Q / A_D  [m/s]
    df["V_pipe"]  = df["Q_m3s"] / A_D

    # Pipe Reynolds number  Re_D = rho * V * D / mu
    df["Re"]      = rho * df["V_pipe"] * D_m / mu

    # Differential head  delta_h = delta_P / (rho * g)  [m water column]
    df["delta_h"] = df["dp_Pa"]  / (rho * g)
    df["h_std"]   = df["dp_std"] / (rho * g)   # propagated sigma [m]

    # Theoretical flow rate (lossless orifice equation)
    #   Q_th = A_d * sqrt( 2*delta_P / (rho*(1-beta^4)) )
    df["Q_th"]    = A_d * np.sqrt(2.0 * df["dp_Pa"] / (rho * (1.0 - beta**4)))

    # EMPIRICAL discharge coefficient
    #   Cd_emp = Q_actual / Q_th
    df["Cd_emp"]  = df["Q_m3s"] / df["Q_th"]

    # THEORETICAL Cd — ISO 5167-2 RHG corner-tap baseline
    df["Cd_RHG"]  = df["Re"].apply(lambda r: Cd_RHG(r))

    # Permanent head loss using empirical Cd (ISO 5167 pressure-recovery formula)
    Cd_bar = df["Cd_emp"].median()
    denom  = np.sqrt(1.0 - beta**4 * Cd_bar**2)
    num    = denom - Cd_bar * beta**2
    den_d  = denom + Cd_bar * beta**2
    df["h_loss"]  = df["delta_h"] * (num / den_d)

    return df


# =============================================================================
#  SECTION 9 — CHAUVENET OUTLIER FILTER
# =============================================================================
def chauvenet_mask(series: pd.Series) -> pd.Series:
    """
    Returns boolean mask: True = keep, False = statistical outlier.
    Reject point i if:  n * erfc( |x_i - x_bar| / (sigma*sqrt(2)) ) < 0.5
    """
    n    = len(series)
    mean = series.mean()
    std  = series.std()
    if std == 0 or n < 3:
        return pd.Series([True] * n, index=series.index)
    d    = np.abs(series - mean) / std
    prob = n * erfc(d / np.sqrt(2))
    return prob > 0.5


# =============================================================================
#  SECTION 10 — CONSOLE SUMMARY
# =============================================================================
def print_summary(df: pd.DataFrame, rho: float, mu: float,
                  T_star: float, T_sigma: float) -> None:
    mask   = chauvenet_mask(df["Cd_emp"])
    valid  = df[mask]
    n_out  = (~mask).sum()

    Cd_m   = valid["Cd_emp"].mean()
    Cd_s   = valid["Cd_emp"].std()
    Re_m   = valid["Re"].mean()
    Cd_iso = Cd_RHG(Re_m)
    dev    = abs(Cd_m - Cd_iso) / Cd_iso * 100 if Cd_iso else float("nan")

    print("\n" + "="*64)
    print("  RESULTS SUMMARY")
    print("="*64)
    print(f"  Stable temperature T*       : {T_star:.3f} degC  (sigma = {T_sigma:.4f} degC)")
    print(f"  rho  (IAPWS-95 at T*)       : {rho:.4f} kg/m^3")
    print(f"  mu   (IAPWS-95 at T*)       : {mu:.4e} Pa.s")
    print("-"*64)
    print(f"  Flow-rate bins              : {len(df)}")
    print(f"  Outliers removed (Chauv.)   : {n_out}")
    print(f"  Valid bins                  : {len(valid)}")
    print("-"*64)
    print(f"  Empirical Cd (mean)         : {Cd_m:.4f}")
    print(f"  sigma(Cd_emp)               : {Cd_s:.4f}")
    print(f"  Coefficient of Variation    : {Cd_s/Cd_m*100:.2f} %")
    print(f"  95% CI                      : {Cd_m:.4f} +/- {1.96*Cd_s:.4f}")
    print(f"  ISO 5167 RHG Cd (at Re_bar) : {Cd_iso:.4f}")
    print(f"  Empirical deviation vs ISO  : {dev:.2f} %  <- expected (custom taps)")
    print("-"*64)
    print(f"  Mean Re_D                   : {Re_m:.0f}")
    print(f"  Flow range                  : "
          f"{df['flow_lpm'].min():.2f} - {df['flow_lpm'].max():.2f} L/min")
    print(f"  dP range                    : "
          f"{df['dp_Pa'].min():.1f} - {df['dp_Pa'].max():.1f} Pa")
    print(f"  Mean differential head      : {df['delta_h'].mean():.4f} m")
    print(f"  Mean permanent head loss    : {df['h_loss'].mean():.4f} m")
    print("="*64 + "\n")


# =============================================================================
#  SECTION 11 — PLOT STYLE
# =============================================================================
C_EMP   = "#1565C0"   # empirical data     — strong blue
C_THEO  = "#C62828"   # theoretical (ISO)  — deep red
C_FIT   = "#E65100"   # power/quad fit     — amber
C_LOSS  = "#2E7D32"   # head loss          — forest green
C_OUT   = "#6A1B9A"   # outliers           — purple
C_SIGMA = "#9E9E9E"   # error bars/bands   — mid grey

plt.rcParams.update({
    "figure.facecolor":  "white",
    "axes.facecolor":    "#f8f9fa",
    "axes.edgecolor":    "#bbbbbb",
    "axes.grid":         True,
    "grid.color":        "#e0e0e0",
    "grid.linewidth":    0.65,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "font.family":       "DejaVu Sans",
    "font.size":         11,
    "axes.titlesize":    13,
    "axes.titleweight":  "bold",
    "axes.titlepad":     10,
    "axes.labelsize":    11,
    "legend.framealpha": 0.92,
    "legend.fontsize":   10,
    "lines.linewidth":   1.8,
})

def _minor_ticks(ax):
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator())

def _subtitle(T_star, T_sigma, rho):
    return (f"D = {D_m*1000:.1f} mm  |  d = {d_m*1000:.1f} mm  |  "
            f"beta = {beta:.4f}  |  "
            f"T* = {T_star:.2f} degC (sigma = {T_sigma:.4f} degC)  |  "
            f"rho = {rho:.2f} kg/m^3\n"
            f"Tapping: Custom 2D/1D  ·  Plate: 6 mm, 45-60 deg downstream bevel")


# =============================================================================
#  SECTION 12 — INDIVIDUAL PLOT FUNCTIONS
#  Each returns a Figure. Caller writes it to PdfPages (one page each).
# =============================================================================

def fig_q_actual_vs_q_theoretical(d_in, d_out, sub):
    """
    Graph 1 — Q_actual vs Q_theoretical  (the Cd slope)

    Physics:
        Q_actual = Cd · Q_theoretical  →  y = m·x  (line through origin).
        Forced-origin OLS slope m = Cd.
        Each data point is labelled with its individual Cd value so the
        variation across the flow range is immediately visible.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    fig.suptitle(sub, fontsize=8.5, color="#555555", y=0.98)

    # Convert to L/min for readable axis labels
    Q_act = d_in["Q_m3s"] * 60_000.0   # L/min actual
    Q_th  = d_in["Q_th"]  * 60_000.0   # L/min theoretical

    # Forced-origin OLS:  Cd = Σ(Q_act · Q_th) / Σ(Q_th²)
    Cd_slope = float(np.sum(Q_act * Q_th) / np.sum(Q_th ** 2))

    # Per-point Cd  (= Q_actual / Q_theoretical, already computed as Cd_emp)
    Cd_per_point = d_in["Cd_emp"].values

    # ── Scatter ──────────────────────────────────────────────────────────────
    ax.scatter(Q_th, Q_act, s=55, color=C_EMP, alpha=0.88, zorder=4,
               label=f"Measured data  (n = {len(d_in)})")

    # Label each point with its individual Cd value (below the marker)
    for q_t, q_a, cd_v in zip(Q_th, Q_act, Cd_per_point):
        ax.annotate(f"{cd_v:.3f}",
                    xy=(q_t, q_a),
                    xytext=(0, -14), textcoords="offset points",
                    ha="center", va="top",
                    fontsize=3.5, color="#1a4a8a",
                    arrowprops=None)

    if len(d_out):
        Q_act_out = d_out["Q_m3s"] * 60_000.0
        Q_th_out  = d_out["Q_th"]  * 60_000.0
        ax.scatter(Q_th_out, Q_act_out, s=60, color=C_OUT,
                   marker="x", linewidths=2.0, zorder=5,
                   label=f"Outlier — Chauvenet (n={len(d_out)})")

    # Regression line  Q_act = Cd_mean · Q_th

    ax.set_xlabel("Theoretical Flow Rate  $Q_{th}$  (L/min)", fontsize=11)
    ax.set_ylabel("Actual Flow Rate  $Q_{actual}$  (L/min)", fontsize=11)
    ax.set_title(f"$Q_{{actual}}$ vs $Q_{{theoretical}}$  —  Calibration Slope\n"
                 f"(individual $C_d$ values labelled on each point)",
                 fontsize=12)

    ax.legend(fontsize=10)
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
    _minor_ticks(ax)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    return fig


def fig_loglog_q_vs_head(d_in, d_out, sub):
    """
    Graph 2 — Log-Log plot of Q_actual vs Differential Head H

    Physics:
        Q = k · H^n   →   log(Q) = n·log(H) + log(k)
        On log-log axes: straight line, slope = n.
        Bernoulli theory predicts n = 0.5 exactly.
    """
    from scipy.stats import linregress

    fig, ax = plt.subplots(figsize=(10, 6))
    fig.suptitle(sub, fontsize=8.5, color="#555555", y=0.98)

    H     = d_in["delta_h"].values
    Q_act = d_in["Q_m3s"].values

    # Log-log regression:  log(Q) = n·log(H) + log(k)
    log_H = np.log10(H)
    log_Q = np.log10(Q_act)
    n_slope, log_k, r_val, _, _ = linregress(log_H, log_Q)
    k_intercept = 10 ** log_k

    # Scatter
    ax.scatter(H, Q_act * 60_000, s=55, color=C_EMP, alpha=0.88,
               zorder=4, label=f"Measured data  (n = {len(d_in)})")

    if len(d_out):
        ax.scatter(d_out["delta_h"], d_out["Q_m3s"] * 60_000,
                   s=60, color=C_OUT, marker="x", linewidths=2.0,
                   zorder=5, label=f"Outlier — Chauvenet (n={len(d_out)})")

    # Empirical power-law fit line
    H_fit = np.logspace(np.log10(H.min() * 0.8),
                        np.log10(H.max() * 1.2), 400)
    ax.plot(H_fit, k_intercept * H_fit ** n_slope * 60_000,
            color=C_THEO, lw=2.2, zorder=3,
            label=f"Empirical fit:  $Q = k \\cdot H^n$\n"
                  f"Slope  $n$ = {n_slope:.4f}   "
                  f"(Bernoulli ideal = 0.5000)\n"
                  f"$k$ = {k_intercept:.6f}  m$^{{(3-3n)}}$/s")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Differential Head  $H = \\Delta P / (\\rho g)$  (m water column)",
                  fontsize=11)
    ax.set_ylabel("Actual Flow Rate  $Q_{actual}$  (L/min)", fontsize=11)
    ax.set_title(f"Log-Log: $Q_{{actual}}$ vs Differential Head  $H$\n"
                 f"Empirical exponent  $n$ = {n_slope:.4f}   "
                 f"(Bernoulli square-root law: $n$ = 0.5000)",
                 fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, which="both", color="#e0e0e0", linewidth=0.5)
    ax.grid(True, which="major", color="#cccccc", linewidth=0.8)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    return fig


def fig_dp_vs_q(d_in, d_out, sub):
    """Plot 3 — Differential Pressure vs Flow Rate"""
    fig, ax = plt.subplots(figsize=(10, 6))
    fig.suptitle(sub, fontsize=8.5, color="#555555", y=0.98)

    ax.errorbar(d_in["flow_lpm"], d_in["dp_Pa"] / 1000,
                yerr=d_in["dp_std"] / 1000,
                fmt="o", color=C_EMP, ms=6, ecolor=C_SIGMA,
                elinewidth=1.2, capsize=4, alpha=0.88,
                label="Empirical: median ΔP ± σ", zorder=3)

    if len(d_out):
        ax.errorbar(d_out["flow_lpm"], d_out["dp_Pa"] / 1000,
                    yerr=d_out["dp_std"] / 1000,
                    fmt="x", color=C_OUT, ms=8, ecolor=C_OUT,
                    elinewidth=1.2, capsize=4,
                    label=f"Outlier (Chauvenet, n={len(d_out)})", zorder=4)

    try:
        lx, ly  = np.log(d_in["flow_lpm"]), np.log(d_in["dp_Pa"])
        b_f, la = np.polyfit(lx, ly, 1)
        q_fit   = np.linspace(d_in["flow_lpm"].min(), d_in["flow_lpm"].max(), 300)
        ax.plot(q_fit, np.exp(la) * q_fit**b_f / 1000,
                color=C_FIT, lw=2, ls="--",
                label=f"Power-law fit: ΔP ∝ Q^{b_f:.3f}  (ideal exponent = 2.000)")
    except Exception:
        pass

    ax.set_xlabel("Volumetric Flow Rate  Q  (L/min)")
    ax.set_ylabel("Differential Pressure  ΔP  (kPa)")
    ax.set_title("Differential Pressure vs Flow Rate")
    ax.legend()
    _minor_ticks(ax)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    return fig


def fig_cd_vs_re(d_in, d_out, sub):
    """Plot 2 — Empirical Cd vs Re (scatter only, no theoretical overlay)"""
    fig, ax = plt.subplots(figsize=(10, 6))
    fig.suptitle(sub, fontsize=8.5, color="#555555", y=0.98)

    # Empirical scatter — one point per flow-rate bin
    sc = ax.scatter(d_in["Re"], d_in["Cd_emp"],
                    s=50, color=C_EMP, alpha=0.85, zorder=4,
                    label="Empirical  Cd_emp  (per bin)")

    if len(d_out):
        ax.scatter(d_out["Re"], d_out["Cd_emp"],
                   s=55, color=C_OUT, marker="x", linewidths=2.0, zorder=5,
                   label=f"Outlier — Chauvenet (n={len(d_out)})")

    # Annotate each point with its flow rate for traceability
    for _, row in d_in.iterrows():
        ax.annotate(f"{row['flow_lpm']:.1f}",
                    xy=(row["Re"], row["Cd_emp"]),
                    xytext=(4, 4), textcoords="offset points",
                    fontsize=7.5, color="#444444", alpha=0.75)

    ax.set_xlabel("Pipe Reynolds Number  Re$_D$")
    ax.set_ylabel("Empirical Discharge Coefficient  C$_d$")
    ax.set_title("Empirical C$_d$ vs Reynolds Number\n"
                 "(Each point = one binned flow setpoint; "
                 "labels show Q in L/min)")
    ax.legend()
    _minor_ticks(ax)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    return fig


def fig_linearity(d_in, d_out, rho, sub):
    """Plot 4 — dP vs Q^2 linearity check"""
    fig, ax = plt.subplots(figsize=(10, 6))
    fig.suptitle(sub, fontsize=8.5, color="#555555", y=0.98)

    Q2   = d_in["Q_m3s"] ** 2
    Q2_x = Q2 * 1e6      # [x10^-6 m^6/s^2]
    DP   = d_in["dp_Pa"]

    ax.errorbar(Q2_x, DP / 1000, yerr=d_in["dp_std"] / 1000,
                fmt="o", color=C_EMP, ms=6, ecolor=C_SIGMA,
                elinewidth=1.2, capsize=4, alpha=0.88,
                label="Empirical: median ΔP ± σ", zorder=3)

    if len(d_out):
        ax.scatter((d_out["Q_m3s"]**2) * 1e6, d_out["dp_Pa"] / 1000,
                   s=45, color=C_OUT, marker="x", linewidths=1.8,
                   label="Outlier", zorder=4)

    # Empirical fit slope only — no ISO baseline
    Cd_emp_mean = d_in["Cd_emp"].mean()
    slope_emp   = rho * (1.0 - beta**4) / (2.0 * Cd_emp_mean**2 * A_d**2)
    Q2_fit      = np.linspace(0, Q2_x.max() * 1.1, 300)
    ax.plot(Q2_fit, slope_emp * Q2_fit * 1e-6 / 1000,
            color=C_EMP, lw=2, ls="--",
            label=f"Empirical fit  ($\\bar{{C}}_d$ = {Cd_emp_mean:.4f})")

    ax.set_xlabel("Q²  (×10⁻⁶ m⁶/s²)")
    ax.set_ylabel("ΔP  (kPa)")
    ax.set_title("Linearity Check: ΔP vs Q²\n"
                 "(Ideal orifice: straight line through origin; slope encodes C$_d$)")
    ax.legend()
    _minor_ticks(ax)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    return fig


def fig_velocity_vs_head(d_in, d_out, sub):
    """Plot 5 — Pipe Velocity vs Differential Head"""
    fig, ax = plt.subplots(figsize=(10, 6))
    fig.suptitle(sub, fontsize=8.5, color="#555555", y=0.98)

    Cd_emp_mean = d_in["Cd_emp"].mean()

    ax.errorbar(d_in["V_pipe"], d_in["delta_h"], yerr=d_in["h_std"],
                fmt="o", color=C_EMP, ms=6, ecolor=C_SIGMA,
                elinewidth=1.2, capsize=4, alpha=0.88,
                label="Empirical  V vs Δh  (measured)", zorder=3)

    if len(d_out):
        ax.scatter(d_out["V_pipe"], d_out["delta_h"], s=45, color=C_OUT,
                   marker="x", linewidths=1.8, label="Outlier", zorder=4)

    V_fit = np.linspace(0, d_in["V_pipe"].max() * 1.15, 300)

    # Empirical curve only: Δh = V²(1−β⁴) / (2g·C̄d²)
    dh_emp = V_fit**2 * (1.0 - beta**4) / (2.0 * g * Cd_emp_mean**2)
    ax.plot(V_fit, dh_emp, color=C_EMP, lw=2, ls="--",
            label=f"Empirical: $\\Delta h = V^2(1-\\beta^4)/(2g\\bar{{C}}_d^2)$   "
                  f"($\\bar{{C}}_d$ = {Cd_emp_mean:.4f})")

    ax.set_xlabel("Mean Pipe Velocity  V  (m/s)")
    ax.set_ylabel("Differential Head  Δh = ΔP/(ρg)  (m water column)")
    ax.set_title("Pipe Velocity vs Differential Head")
    ax.legend()
    _minor_ticks(ax)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    return fig


def fig_head_loss_vs_q(d_in, d_out, sub):
    """Plot 6 — Permanent Head Loss vs Flow Rate"""
    fig, ax = plt.subplots(figsize=(10, 6))
    fig.suptitle(sub, fontsize=8.5, color="#555555", y=0.98)

    # Differential head (lighter)
    ax.errorbar(d_in["flow_lpm"], d_in["delta_h"], yerr=d_in["h_std"],
                fmt="o", color=C_EMP, ms=5, ecolor=C_SIGMA,
                elinewidth=1.0, capsize=3, alpha=0.50,
                label="Differential head  Δh  (empirical)", zorder=2)

    # Permanent head loss (from empirical Cd)
    h_loss_err = d_in["h_std"] * d_in["h_loss"] / d_in["delta_h"]
    ax.errorbar(d_in["flow_lpm"], d_in["h_loss"], yerr=h_loss_err,
                fmt="s", color=C_LOSS, ms=6, ecolor=C_SIGMA,
                elinewidth=1.2, capsize=4, alpha=0.90,
                label="Permanent head loss  h$_L$  (empirical)", zorder=3)

    if len(d_out):
        ax.scatter(d_out["flow_lpm"], d_out["h_loss"], s=45, color=C_OUT,
                   marker="x", linewidths=1.8, label="Outlier", zorder=5)

    # Quadratic fit on h_loss only — no ISO baseline
    try:
        q_arr = d_in["Q_m3s"].values
        h_arr = d_in["h_loss"].values
        c     = np.polyfit(q_arr, h_arr, 2)
        q_fit = np.linspace(0, q_arr.max() * 1.08, 300)
        ax.plot(q_fit * 60_000, np.polyval(c, q_fit),
                color=C_FIT, lw=2, ls="--",
                label="Quadratic fit  $h_L \\propto Q^2$  (empirical)")
    except Exception:
        pass

    ax.set_xlabel("Volumetric Flow Rate  Q  (L/min)")
    ax.set_ylabel("Head  (m water column)")
    ax.set_title("Permanent Head Loss vs Flow Rate\n"
                 "(Δh = total differential head; h$_L$ = irrecoverable pressure loss)")
    ax.legend()
    _minor_ticks(ax)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    return fig


# =============================================================================
#  SECTION 13 — PDF EXPORT  (one plot per page, 300 DPI)
# =============================================================================
def export_pdf(figures, titles, output_path="orifice_analysis_high_res.pdf"):
    print(f"\n  [PDF] Writing {len(figures)} pages -> {output_path}")

    with PdfPages(output_path) as pdf:
        d = pdf.infodict()
        d["Title"]        = "Orifice Meter Analysis — Empirical vs ISO 5167"
        d["Author"]       = "Fluid Mechanics Test Rig Analysis Script"
        d["Subject"]      = (f"D={D_m*1000:.1f}mm, d={d_m*1000:.1f}mm, "
                             f"beta={beta:.4f}  |  Custom 2D/1D tapping, "
                             f"6mm bevel plate")
        d["CreationDate"] = datetime.datetime.today()

        for i, (fig, title) in enumerate(zip(figures, titles), start=1):
            pdf.savefig(fig, dpi=300, bbox_inches="tight")
            plt.close(fig)
            print(f"    Page {i:02d}: {title}")

    print(f"  [PDF] Saved -> {output_path}")


# =============================================================================
#  MAIN
# =============================================================================
def main():
    parser = argparse.ArgumentParser(
        description="Orifice Meter Analysis — Empirical vs ISO 5167 RHG"
    )
    parser.add_argument("--csv",    default="fluid_test_data.csv")
    parser.add_argument("--iapws",  default="water_properties_IAPWS95.csv")
    parser.add_argument("--bin",    default=0.5, type=float,
                        help="Flow-rate bin width in L/min (default: 0.5)")
    parser.add_argument("--output", default="orifice_analysis_high_res.pdf")
    args = parser.parse_args()

    # 1. Locate input files
    data_path  = locate_file(args.csv)
    iapws_path = locate_file(args.iapws)

    # 2. Load IAPWS-95 interpolators
    rho_fn, mu_fn = load_iapws(iapws_path)

    # 3. Load raw sensor data
    df_raw = load_sensor_data(data_path)

    # 4. Find most stable temperature window -> single T* for all fluid props
    T_star, T_sigma, _, _ = find_stable_temperature(df_raw["temp_C"])
    rho = float(rho_fn(T_star))
    mu  = float(mu_fn(T_star))
    print(f"  [FLUID]  rho = {rho:.4f} kg/m^3    mu = {mu:.4e} Pa.s")

    # 5. Bin by flow rate -> intra-bin z-score filter -> representative median dP
    df_binned = bin_by_flow(df_raw, bin_w=args.bin, z_thresh=2.0)

    # 6. Compute all derived quantities (empirical + theoretical)
    df = compute(df_binned, rho, mu)

    # 7. Chauvenet filter on empirical Cd
    mask  = chauvenet_mask(df["Cd_emp"])
    d_in  = df[mask].reset_index(drop=True)
    d_out = df[~mask].reset_index(drop=True)

    # 8. Console summary
    print_summary(df, rho, mu, T_star, T_sigma)

    # 9. Shared subtitle for all figures
    sub = _subtitle(T_star, T_sigma, rho)

    # 10. Build seven figures (one per PDF page) — new graphs 1 & 2 first
    plot_specs = [
        ("Q_actual vs Q_theoretical — Cd Slope",
         fig_q_actual_vs_q_theoretical(d_in, d_out, sub)),
        ("Log-Log Q_actual vs Head — Bernoulli Exponent n",
         fig_loglog_q_vs_head(d_in, d_out, sub)),
        ("dP vs Flow Rate",
         fig_dp_vs_q(d_in, d_out, sub)),
        ("Cd vs Reynolds Number",
         fig_cd_vs_re(d_in, d_out, sub)),
        ("Linearity Check: dP vs Q^2",
         fig_linearity(d_in, d_out, rho, sub)),
        ("Velocity vs Differential Head",
         fig_velocity_vs_head(d_in, d_out, sub)),
        ("Head Loss vs Flow Rate",
         fig_head_loss_vs_q(d_in, d_out, sub)),
    ]

    titles  = [t for t, _ in plot_specs]
    figures = [f for _, f in plot_specs]

    # 11. Export PDF — one plot per page at 300 DPI
    export_pdf(figures, titles, output_path=args.output)


if __name__ == "__main__":
    main()
