
import numpy as np

from triqs_soehyb.pycppdlr import build_dlr_rf
from triqs_soehyb.pycppdlr import ImTimeOps
from triqs_soehyb.impurity import Fastdiagram

beta = 1.0
lamb = 100.0
eps = 1e-10

n = 4
nops = 1
F = np.zeros((nops, n, n), dtype=complex)
F[:] = np.eye(n)[None, ...]
F_dag = F

dlr_rf = build_dlr_rf(lamb, eps)
ito = ImTimeOps(lamb, dlr_rf)
diagramsolver = Fastdiagram(beta, lamb, ito, F, F_dag)
