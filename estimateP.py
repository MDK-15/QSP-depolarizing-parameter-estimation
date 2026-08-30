import numpy as np
import torch
from noise_signal_design import f_E, initial_state
 
 
def build_f_E_numpy(theta_opt, Phi_opt, rho0=None):
    """Wrap the torch f_E as a plain float -> float function of p."""
    theta_t = torch.tensor(theta_opt)
    Phi_t = torch.tensor(Phi_opt)
    if rho0 is None:
        rho0 = initial_state()
 
    def f(p):
        return f_E(theta_t, Phi_t, torch.tensor(float(p)), rho0).item()
 
    return f

def estimate_p_bisect(theta_opt, Phi_opt, X, p_lo, p_hi, tol=1e-8, max_iter=60):
    """
    Root-find f_E(p) = s via bisection, clipping to the interval endpoints
    if s falls outside the achievable range (matches Theorem 9's structure:
    clip at boundary rather than extrapolate). Assumes f_E is monotone on
    [p_lo, p_hi], which is exactly what optimize_noise_sensitivity was
    trying to arrange -- check this if you see poor convergence.
    """
    f = build_f_E_numpy(theta_opt, Phi_opt)
    s = np.mean(X)
 
    f_lo, f_hi = f(p_lo), f(p_hi)
    increasing = f_hi >= f_lo
 
    # clip to range
    lo_val, hi_val = (f_lo, f_hi) if increasing else (f_hi, f_lo)
    if s <= lo_val:
        return p_lo if increasing else p_hi
    if s >= hi_val:
        return p_hi if increasing else p_lo
 
    a, b = p_lo, p_hi
    for _ in range(max_iter):
        mid = 0.5 * (a + b)
        f_mid = f(mid)
        if abs(f_mid - s) < tol:
            return mid
        if (f_mid < s) == increasing:
            a = mid
        else:
            b = mid
    return 0.5 * (a + b)

def estimate_p_bisect_baseline(theta_opt, Phi_opt, X, p_lo, p_hi, rho0, tol=1e-8, max_iter=60):
    """
    Root-find f_E(p) = s via bisection, clipping to the interval endpoints
    if s falls outside the achievable range (matches Theorem 9's structure:
    clip at boundary rather than extrapolate). Assumes f_E is monotone on
    [p_lo, p_hi], which is exactly what optimize_noise_sensitivity was
    trying to arrange -- check this if you see poor convergence.
    """
    f = build_f_E_numpy(theta_opt, Phi_opt, rho0)
    s = np.mean(X)
 
    f_lo, f_hi = f(p_lo), f(p_hi)
    increasing = f_hi >= f_lo
 
    # clip to range
    lo_val, hi_val = (f_lo, f_hi) if increasing else (f_hi, f_lo)
    if s <= lo_val:
        return p_lo if increasing else p_hi
    if s >= hi_val:
        return p_hi if increasing else p_lo
 
    a, b = p_lo, p_hi
    for _ in range(max_iter):
        mid = 0.5 * (a + b)
        f_mid = f(mid)
        if abs(f_mid - s) < tol:
            return mid
        if (f_mid < s) == increasing:
            a = mid
        else:
            b = mid
    return 0.5 * (a + b)