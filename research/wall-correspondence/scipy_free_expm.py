import numpy as np


def expm_sym(A):
    w, U = np.linalg.eigh(A)
    return U @ np.diag(np.exp(w)) @ U.T
