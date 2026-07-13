"""ROOT file access: uproot patch, key introspection, and statistics (Clopper-Pearson, inverse-variance)."""

from __future__ import annotations

import struct
from typing import NamedTuple

import numpy as np
import uproot
import uproot.containers
from scipy.stats import beta as beta_dist

PLOTTABLE_CLASSNAMES = ("TEfficiency", "TProfile")

_patched = False


def patch_uproot_asvector() -> None:
    """Patch uproot to manually parse `vector<pair<double,double>>` members.

    uproot's generic AsVector reader raises NotImplementedError for the
    streamerless memberwise serialization ROOT uses for this specific
    container type; we intercept only that exact failure and hand-decode
    the two parallel big-endian float64 arrays it actually contains.
    """
    global _patched
    if _patched:
        return

    orig_read = uproot.containers.AsVector.read
    stl_u32 = struct.Struct(">I")

    def _patched_read(self, chunk, cursor, context, file, selffile, parent, header=True):
        try:
            return orig_read(self, chunk, cursor, context, file, selffile, parent, header)
        except NotImplementedError as exc:
            if "streamerless memberwise serialization of class AsVector(pair<double,double>)" not in str(exc):
                raise
            cursor.skip(6)
            n = cursor.field(chunk, stl_u32, context)
            if n == 0:
                return np.empty(0, dtype=object)
            first = cursor.array(chunk, n, np.dtype(">f8"), context)
            second = cursor.array(chunk, n, np.dtype(">f8"), context)
            return list(zip(first.tolist(), second.tolist()))

    uproot.containers.AsVector.read = _patched_read
    _patched = True


patch_uproot_asvector()


def _is_one_dimensional(obj, classname: str) -> bool:
    if classname == "TEfficiency":
        return len(obj.member("fPassedHistogram").axes) == 1
    return len(obj.axes) == 1


def list_plottable_keys(path: str) -> dict[str, str]:
    """Return {base_key_name: classname} for 1D TEfficiency/TProfile keys, deduped by highest cycle.

    Multi-dimensional TEfficiency/TProfile objects (e.g. trackeff_vs_eta_phi) share the same
    ROOT classname as their 1D counterparts but aren't representable by our single-x-axis panels,
    so dimensionality is checked explicitly rather than relying on the classname alone.
    """
    f = uproot.open(path)
    classnames = f.classnames()
    best_cycle: dict[str, int] = {}
    result: dict[str, str] = {}
    for full_key, classname in classnames.items():
        if classname not in PLOTTABLE_CLASSNAMES:
            continue
        if ";" in full_key:
            base_key, cycle_str = full_key.rsplit(";", 1)
            cycle = int(cycle_str)
        else:
            base_key, cycle = full_key, 0
        if base_key in best_cycle and cycle <= best_cycle[base_key]:
            continue
        if not _is_one_dimensional(f[full_key], classname):
            continue
        best_cycle[base_key] = cycle
        result[base_key] = classname
    return result


def read_efficiency_raw(path: str, key: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (x_centers, n_pass, n_tot) for a TEfficiency key, with no statistics applied."""
    teff = uproot.open(path)[key]
    passed = teff.member("fPassedHistogram")
    total = teff.member("fTotalHistogram")
    x = passed.axes[0].centers()
    return x, passed.values(), total.values()


def clopper_pearson(n_pass, n_tot) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute (efficiency, err_lo, err_hi) as a 68% Clopper-Pearson CI. Works on scalars or arrays."""
    n_pass = np.asarray(n_pass, dtype=float)
    n_tot = np.asarray(n_tot, dtype=float)

    eff = np.where(n_tot > 0, n_pass / n_tot, np.nan)
    lo = np.where(n_tot > 0, beta_dist.ppf(0.1587, n_pass, n_tot - n_pass + 1), np.nan)
    hi = np.where(n_tot > 0, beta_dist.ppf(0.8413, n_pass + 1, n_tot - n_pass), np.nan)

    err_lo = eff - lo
    err_hi = np.where(np.isnan(hi), 0.0, hi - eff)
    return eff, err_lo, err_hi


def fold_absolute(x, n_pass, n_tot) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Pool raw pass/total counts of bins at +v and -v into a single |v| bin."""
    x = np.asarray(x, dtype=float)
    n_pass = np.asarray(n_pass, dtype=float)
    n_tot = np.asarray(n_tot, dtype=float)

    abs_x = np.abs(x)
    unique_x = np.unique(abs_x)
    folded_pass = np.array([n_pass[abs_x == ux].sum() for ux in unique_x])
    folded_tot = np.array([n_tot[abs_x == ux].sum() for ux in unique_x])
    return unique_x, folded_pass, folded_tot


class EfficiencyData(NamedTuple):
    x: np.ndarray
    eff: np.ndarray
    err_lo: np.ndarray
    err_hi: np.ndarray


def read_efficiency(path: str, key: str, *, fold_abs: bool = False) -> EfficiencyData:
    x, n_pass, n_tot = read_efficiency_raw(path, key)
    if fold_abs:
        x, n_pass, n_tot = fold_absolute(x, n_pass, n_tot)
    eff, err_lo, err_hi = clopper_pearson(n_pass, n_tot)
    return EfficiencyData(x, eff, err_lo, err_hi)


class PooledEfficiency(NamedTuple):
    eff: float
    err_lo: float
    err_hi: float
    n_pass_total: float
    n_tot_total: float


def pooled_efficiency(path: str, key: str) -> PooledEfficiency:
    """Clopper-Pearson CI on pass/total counts pooled across all bins."""
    _, n_pass, n_tot = read_efficiency_raw(path, key)
    n_pass_total = float(n_pass.sum())
    n_tot_total = float(n_tot.sum())
    eff, err_lo, err_hi = clopper_pearson(n_pass_total, n_tot_total)
    return PooledEfficiency(float(eff), float(err_lo), float(err_hi), n_pass_total, n_tot_total)


class ProfileData(NamedTuple):
    x: np.ndarray
    values: np.ndarray
    errors: np.ndarray


def read_profile(path: str, key: str) -> ProfileData:
    obj = uproot.open(path)[key]
    return ProfileData(obj.axes[0].centers(), obj.values(), obj.errors())


class ProfileCombined(NamedTuple):
    mean: float
    err: float


def inverse_variance_combine(values, errors) -> ProfileCombined:
    """Inverse-variance-weighted combination of TProfile bins, excluding non-finite/zero-error bins."""
    values = np.asarray(values, dtype=float)
    errors = np.asarray(errors, dtype=float)

    valid = np.isfinite(values) & np.isfinite(errors) & (errors > 0)
    if not np.any(valid):
        return ProfileCombined(float("nan"), float("nan"))

    weights = 1.0 / errors[valid] ** 2
    mean = float(np.sum(weights * values[valid]) / np.sum(weights))
    err = float(1.0 / np.sqrt(np.sum(weights)))
    return ProfileCombined(mean, err)


def combine_profile_inverse_variance(path: str, key: str) -> ProfileCombined:
    profile = read_profile(path, key)
    return inverse_variance_combine(profile.values, profile.errors)
