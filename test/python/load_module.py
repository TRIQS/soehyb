
import numpy as np

from triqs_soehyb.impurity import Fastdiagram

beta = 1.0
lamb = 100.0
eps = 1e-10

n = 4
nops = 1
F = np.zeros((nops, n, n), dtype=complex)
F[:] = np.eye(n)[None, ...]
F_dag = F

diagramsolver = Fastdiagram(beta, lamb, eps, F, F_dag)
