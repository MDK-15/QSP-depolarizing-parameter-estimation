import numpy as np
from numpy.polynomial.chebyshev import Chebyshev, chebval
from numpy.polynomial import Polynomial
from noise_signal_design import optimize_noise_sensitivity
from qsppack.solver import solve
from estimateP import estimate_p_bisect

def rot_Z(theta):
    return np.diag([np.exp(1j*theta), np.exp(-1j*theta)])

def rot_X(phi):
    c, s = np.cos(phi), 1j*np.sin(phi)
    return np.array([[c, s], [s, c]])

def depolarizing_channel(rho, p):
    I = np.eye(2, dtype=complex)
    X = np.array([[0,1], [1,0]], dtype=complex)
    Y = np.array([[0,-1j], [1j,0]], dtype=complex)
    Z = np.array([[1,0],[0,-1]], dtype=complex)

    K0 = np.sqrt(1-p) * I
    K1 = np.sqrt(p/3) * X
    K2 = np.sqrt(p/3) * Y
    K3 = np.sqrt(p/3) * Z

    return (K0 @ rho @ K0.conj().T + K1 @ rho @ K1.conj().T + K2 @ rho @ K2.conj().T + K3 @ rho @ K3.conj().T)

def build_rho(theta, Phi, p, rho):
    rho = rot_X(Phi[0]) @ rho @ rot_X(Phi[0]).conj().T
    for phi_j in Phi[1:]:
        rho = rot_Z(theta) @ rho @ rot_Z(theta).conj().T
        rho = depolarizing_channel(rho, p)
        rho = rot_X(phi_j) @ rho @ rot_X(phi_j).conj().T
    return rho

def hadamard_gate():
    return (1/np.sqrt(2)) * np.array([[1, 1], [1, -1]], dtype=complex)

def QSP_single_iteration(mk, C, dk, rk_prev, q, p_real, p_prev, rho):
    theta, Phi, L_E, kappa = optimize_noise_sensitivity(d=dk, p_lo=p_prev-rk_prev, p_hi=p_prev+rk_prev, n_grid=20)

    rho = build_rho(theta, Phi, p_real, rho)
    prob0 = float(np.clip(rho[0,0].real, 0.0, 1.0))
    X = np.random.choice([0, 1], size=mk, p=[1-prob0, prob0])
    
    p_hat = estimate_p_bisect(theta, Phi, X, p_prev-rk_prev, p_prev+rk_prev)

    rk = 1/(4*C*dk*q)
    
    return (p_hat, rk, L_E, kappa)

