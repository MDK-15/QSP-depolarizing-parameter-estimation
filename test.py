from QSP_PE import QSP_PE
from single_iteration import QSP_single_iteration
import matplotlib.pyplot as plt
import numpy as np

zero = np.array([1, 0], dtype=complex)
rho = np.outer(zero, zero.conj().T)

p_real = 0.01
p_prev = 0.015
dk = 8
C = 5
q = 2
rk_prev = 0.01
mk = 1000

results = np.zeros(30)

for k in range(30):
    estimate = QSP_single_iteration(mk=mk, C=C, dk=dk, rk_prev=rk_prev, q=q, p_real=p_real, p_prev=p_prev, rho=rho)
    results[k] = estimate[0]

# print(f"k: {k}")
# print(f"Real p: {p_real}")
# print(f"Predicted p: {estimate[0]}")

print(f"Real p: {p_real}\nInitial estimate: {p_prev}\nAverage estimate: {np.round(np.mean(results), 4)}\nMin estimate: {np.round(np.min(results), 4)}\nMax estimate: {np.round(np.max(results), 4)}")