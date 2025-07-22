import numpy as np
import matplotlib as plt
from triqs.operators import c, c_dag, n
from triqs.atom_diag import AtomDiag
from h5 import HDFArchive
import atom_diag_utils as utils

if __name__ == "__main__":
    # parameters to tune
    norb = 2
    h5_fname = "spin_flip_fermion.h5"
    
    N = 0
    for kap in range(norb):
        for sig in ['up', 'do']:
            N += n(sig, kap)

    sym_ops = [N]

    # fixed parameters
    beta = 2.0
    mu = 0.25
    U = 1.0
    V = 0.1

    # construct Hamiltonion, fundamental operator set, AtomDiag object
    H = 0
    fops = []
    for i in range(0, norb):
        H += U * n('up', i) * n('do', i) + mu * ( n('up', i) + n('do', i) ) + \
            V * ( c_dag('up', i) * c('do', i) + c_dag('do', i) * c('up', i) )
        fops += [ ('do', i) ]
    for i in range(0, norb):
        fops += [ ('up', i)]
    ad = AtomDiag(H, fops, sym_ops)

    # get Hamiltonian as a matrix
    H_mat = utils.get_full_h_atomic(ad)
    
    # permute this matrix so that it matches the blocks found by atom_diag
    H_perm = [] # permutation to apply to the rows and columns of H
    H_mat_blocks = [] # array for the blocks of H 
    H_mat_block_inds = np.zeros(ad.n_subspaces, np.int64) # array for block-column indices of H
    for s, state in enumerate(ad.fock_states): # loop over Fock state ordering that induces block structure
        H_perm = np.concatenate((H_perm,state)) 
        H_block = H_mat[state][:,state].reshape(len(state), len(state))
        H_mat_blocks = H_mat_blocks + [H_block]
        if np.all(np.abs(H_block) < 1e-16):
            H_mat_block_inds[s] = -1 # set -1 if block-column has no nonzero block
    H_perm = H_perm.astype(np.int64)
    H_mat_perm = H_mat[H_perm][:,H_perm]

    # c_blocks = [[], [], [], []]
    # cdag_blocks = [[], [], [], []]
    c_blocks = [[] for _ in range(ad.n_subspaces)]
    cdag_blocks = [[] for _ in range(ad.n_subspaces)]

    # orbital indices: do 0, do 1, up 0, up 1
    # save blocks of creation and annihilation operators
    for oidx in range(2 * norb):
        for sidx in range(ad.n_subspaces):
            cidx = ad.c_connection(oidx, sidx)
            fock_idx = np.ix_(ad.fock_states[cidx], ad.fock_states[sidx])
            c_blocks[sidx] = c_blocks[sidx] + \
                [utils.get_full_c_matrix(ad, oidx)[fock_idx]]
            didx = ad.cdag_connection(oidx, sidx)
            fock_idx = np.ix_(ad.fock_states[didx], ad.fock_states[sidx])
            cdag_blocks[sidx] = cdag_blocks[sidx] + \
                [utils.get_full_cdag_matrix(ad, oidx)[fock_idx]]
    for sidx in range(ad.n_subspaces):        
        c_blocks[sidx] = np.array(c_blocks[sidx])
        cdag_blocks[sidx] = np.array(cdag_blocks[sidx])
    
    # save dense creation and annihilation operators
    c_dense = np.zeros((2 * norb, 4 ** norb, 4 ** norb))
    cdag_dense = np.zeros((2 * norb, 4 ** norb, 4 ** norb))
    fop_perm = [0, 1, 2, 3]
    for oidx in range(2 * norb):
        c_dense[oidx, :, :] = utils.get_full_c_matrix(ad, fop_perm[oidx])[H_perm][:, H_perm]
        cdag_dense[oidx, :, :] = utils.get_full_cdag_matrix(ad, fop_perm[oidx])[H_perm][:, H_perm]

    with HDFArchive("/home/paco/feynman/soehyb/test/c++/h5/" + h5_fname) as ar:
        ar['norb'] = norb
        ar['num_blocks'] = ad.n_subspaces
        ar['ad'] = ad
        ar['H_mat_blocks'] = H_mat_blocks
        ar['H_mat_block_inds'] = H_mat_block_inds
        ar['c_blocks'] = c_blocks
        ar['cdag_blocks'] = cdag_blocks
        ar['H_mat_dense'] = H_mat_perm
        ar['c_dense'] = c_dense
        ar['cdag_dense'] = cdag_dense
