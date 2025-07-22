import numpy as np

def get_full_h_atomic(ad):

    H_mat = np.zeros([ad.full_hilbert_space_dim]*2, dtype=float)

    for sidx in range(ad.n_subspaces):
        U = ad.unitary_matrices[sidx]
        E = ad.energies[sidx]
        H_block_diag = np.diag(E + ad.gs_energy)
        H_block = U @ H_block_diag @ U.T.conj()
        fidx = ad.fock_states[sidx]
        bidx = np.ix_(fidx, fidx)
        H_mat[bidx] = H_block

    return H_mat

def get_full_x_matrix(ad, oidx, x_connection, x_matrix):

    op_mat = np.zeros([ad.full_hilbert_space_dim]*2, dtype=float)

    for s1 in range(ad.n_subspaces):
        
        s2 = x_connection(oidx, s1)
        if s2 < 0: continue
        
        block_mat = x_matrix(oidx, s1)
        U1 = ad.unitary_matrices[s1]
        U2 = ad.unitary_matrices[s2]
        block_mat_fock = U2 @ block_mat @ U1.T.conj()

        f1 = ad.fock_states[s1]
        f2 = ad.fock_states[s2]
        bidx = np.ix_(f2, f1)
        op_mat[bidx] = block_mat_fock

    return op_mat


def get_full_c_matrix(ad, oidx):
    return get_full_x_matrix(ad, oidx, ad.c_connection, ad.c_matrix)


def get_full_cdag_matrix(ad, oidx):
    return get_full_x_matrix(ad, oidx, ad.cdag_connection, ad.cdag_matrix)