from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ResampleResult:
    t_ms: np.ndarray
    y: np.ndarray
    valid: np.ndarray


def make_regular_grid(t_ms: np.ndarray, dt_ms: float) -> np.ndarray:
    t_ms = np.asarray(t_ms, dtype=np.float64)
    if t_ms.size == 0:
        return np.zeros((0,), dtype=np.float64)
    t0 = float(np.nanmin(t_ms))
    t1 = float(np.nanmax(t_ms))
    if not np.isfinite(t0) or not np.isfinite(t1) or t1 <= t0:
        return np.array([t0], dtype=np.float64)
    n = int(np.floor((t1 - t0) / float(dt_ms))) + 1
    return (t0 + float(dt_ms) * np.arange(n, dtype=np.float64))


def _interp_1d(t_src: np.ndarray, y_src: np.ndarray, t_dst: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Linear interpolation with NaN-aware validity.
    Returns (y_dst, valid_dst).
    """
    t_src = np.asarray(t_src, dtype=np.float64)
    y_src = np.asarray(y_src, dtype=np.float64)
    t_dst = np.asarray(t_dst, dtype=np.float64)

    ok = np.isfinite(t_src) & np.isfinite(y_src)
    if int(np.count_nonzero(ok)) < 2:
        return np.full_like(t_dst, np.nan, dtype=np.float64), np.zeros_like(t_dst, dtype=bool)

    ts = t_src[ok]
    ys = y_src[ok]
    order = np.argsort(ts)
    ts = ts[order]
    ys = ys[order]

    y = np.interp(t_dst, ts, ys, left=np.nan, right=np.nan)
    valid = np.isfinite(y)
    return y, valid


def resample_linear(
    t_ms: np.ndarray,
    y: np.ndarray,
    dt_ms: float,
    *,
    axis_time: int = 0,
) -> ResampleResult:
    """
    Resample y(t) onto a regular time grid using linear interpolation.

    Supports y with shape (T, ...) where axis_time indicates the time axis.
    Missing values should be NaN; validity is returned.
    """
    t_ms = np.asarray(t_ms, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    t_dst = make_regular_grid(t_ms, dt_ms=float(dt_ms))
    if t_dst.size == 0:
        return ResampleResult(t_ms=np.zeros((0,), dtype=np.float64), y=y[:0], valid=np.zeros((0,), dtype=bool))

    y_moved = np.moveaxis(y, axis_time, 0)  # (T, ...)
    t_src = t_ms
    t_src = t_src.reshape(-1)
    if y_moved.shape[0] != t_src.shape[0]:
        raise ValueError(f"Time axis mismatch: t has {t_src.shape[0]} but y has {y_moved.shape[0]}")

    out = np.empty((t_dst.shape[0],) + y_moved.shape[1:], dtype=np.float64)
    valid_all = np.ones((t_dst.shape[0],), dtype=bool)

    # Flatten remaining dims and interp each component.
    flat = y_moved.reshape(y_moved.shape[0], -1)  # (T,K)
    flat_out = np.empty((t_dst.shape[0], flat.shape[1]), dtype=np.float64)

    for k in range(flat.shape[1]):
        yk, valid_k = _interp_1d(t_src, flat[:, k], t_dst)
        flat_out[:, k] = yk
        valid_all &= valid_k

    out = flat_out.reshape((t_dst.shape[0],) + y_moved.shape[1:])
    out = np.moveaxis(out, 0, axis_time)
    return ResampleResult(t_ms=t_dst, y=out, valid=valid_all)


def finite_diff(
    t_ms: np.ndarray,
    x: np.ndarray,
    *,
    axis_time: int = 0,
) -> np.ndarray:
    """
    Central-difference derivative dx/dt with variable dt; NaNs propagate.
    Returns same shape as x.
    """
    t = np.asarray(t_ms, dtype=np.float64) / 1000.0  # seconds
    x = np.asarray(x, dtype=np.float64)
    x_m = np.moveaxis(x, axis_time, 0)  # (T,...)
    if x_m.shape[0] != t.shape[0]:
        raise ValueError("Time axis mismatch for finite_diff")

    if t.shape[0] < 2:
        return np.full_like(x, np.nan, dtype=np.float64)

    dx = np.full_like(x_m, np.nan, dtype=np.float64)
    dt = np.diff(t)
    if np.any(dt <= 0):
        # non-monotonic time, fall back to index-based dt=1
        dt = np.ones_like(dt)

    # forward/backward for edges, central for middle
    dx[0] = (x_m[1] - x_m[0]) / dt[0]
    dx[-1] = (x_m[-1] - x_m[-2]) / dt[-1]
    if t.shape[0] > 2:
        dt_mid = (t[2:] - t[:-2]).reshape(-1, *([1] * (x_m.ndim - 1)))
        dx[1:-1] = (x_m[2:] - x_m[:-2]) / dt_mid

    return np.moveaxis(dx, 0, axis_time)

