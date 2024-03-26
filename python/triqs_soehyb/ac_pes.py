# This is a python implementation for analytic continuation of Fermionic Green's functions/self energy
# using PES (ES) method
# Reference: PhysRevB.107.075151
import numpy as np 
import scipy
import scipy.optimize
# import cvxpy as cp
# import mosek
from triqs_soehyb.aaa import *
def kernel(tau, omega):
    kernel = np.empty((len(tau), len(omega)))

    p, = np.where(omega > 0.)
    m, = np.where(omega <= 0.)
    w_p, w_m = omega[p].T, omega[m].T

    tau = tau[:, None]

    kernel[:, p] = np.exp(-tau*w_p) / (1 + np.exp(-w_p))
    kernel[:, m] = np.exp((1. - tau)*w_m) / (1 + np.exp(w_m))

    return kernel
def eval_with_pole(pol, Z, weight):
    pol_t = np.reshape(pol,[pol.size,1])
    M = 1/(Z-pol_t)
    M = M.transpose()
    if len(weight.shape)==1:
        return M@weight
    else:
        G = M@np.reshape(weight, (weight.shape[0], weight.shape[1]*weight.shape[2]))
        return np.reshape(G, (G.shape[0],  weight.shape[1], weight.shape[2]))
def get_weight_t(pol, tgrid, Deltat, beta, maxiter=100000):
    M = -kernel(tgrid/beta, pol*beta)
    shape_iaa = Deltat.shape
    shape_iA = (shape_iaa[0], shape_iaa[1]*shape_iaa[2])
    shape_xaa = (len(pol), shape_iaa[1], shape_iaa[2])
    weight = np.linalg.lstsq(M, Deltat.reshape(shape_iA), rcond=None)[0]
    residue = (Deltat.reshape(shape_iA) - M@weight).reshape(shape_iaa)
    
    weight = weight.reshape(shape_xaa)
    return weight, M, residue

def erroreval_t(pol,  tgrid, Deltat, beta, maxiter=100000):
    R, M, residue = get_weight_t(pol, tgrid, Deltat, beta, maxiter=100000)
    if len(Deltat.shape)==1:
        y = np.linalg.norm(residue)
        grad = np.real(np.dot(np.conj(residue) ,(R*(M**2))))
    else:
        y = np.linalg.norm(residue.flatten())

        Np = len(pol)
        grad = np.zeros(Np)
        Nw = len(tgrid)
        for k in range(Np):
            for w in range(Nw):
                grad[k] = grad[k] + np.real(np.sum((M[w,k]**2)*(np.conj(residue[w,:,:]) * R[k])))

    grad = -grad/y
    return y, grad

def get_weight(pol, Z, G, cleanflag=True, maxiter=100000, Hermitian = True):
    pol_t = np.reshape(pol,[pol.size,1])
    M = 1/(Z-pol_t)
    M = M.transpose()
    MM = np.concatenate([M.real,M.imag])
    if len(G.shape)==1: 
        GG = np.concatenate([G.real,G.imag])
        if cleanflag == True:
            R = np.linalg.lstsq(MM, GG,rcond=0)[0]
        else:
            [R,rnorm] = scipy.optimize.nnls(MM, GG,maxiter=maxiter)
        residue = G - M@R
    else:
        Np = len(pol)
        Norb = G.shape[1]
        R = np.zeros((Np, Norb, Norb), dtype = np.complex128)
        if cleanflag == True:
            if Hermitian == False:
                for i in range(Norb):
                    for j in range(Norb):
                        
                        R[:,i,j] = np.linalg.lstsq(M, G[:,i,j],rcond=0)[0]
                #not implemented
            else:
                for i in range(Norb):
                    GG = np.concatenate([G[:,i,i].real,G[:,i,i].imag])
                    R[:,i,i] = np.linalg.lstsq(MM, GG,rcond=0)[0]
                    for j in range(i+1,Norb):
                        g1 = (G[:,j,i] + G[:,i,j])/2.0
                        g2 = (G[:,i,j] - G[:,j,i])/2.0 
                        GG1 = np.concatenate([g1.real,g1.imag])
                        GG2 = np.concatenate([g2.imag, -g2.real])
                        R1 = np.linalg.lstsq(MM, GG1,rcond=0)[0]
                        R2 = np.linalg.lstsq(MM, GG2,rcond=0)[0]
                        R[:,i,j] = R1 + 1j*R2
                        R[:,j,i] = R1 - 1j*R2
        else:
            X = [cp.Variable((Norb, Norb), hermitian=True) for i in range(Np) ]
            Nw = len(Z)
            constraints = [X[i] >> 0 for i in range(Np)]
            Gfit = []
            for w in range(Nw):
                Gfit.append(cp.sum_squares(sum([ M[w,i]*X[i] for i in range(Np)]) - G[w,:,:]))

            objective = cp.Minimize(sum(Gfit))
            prob = cp.Problem(objective, constraints)
            result = prob.solve(solver = "SCS",verbose = False, eps = 1.e-8)
            # mosek_params_dict = {"MSK_DPAR_INTPNT_CO_TOL_PFEAS": 1.e-8,\
            #                     "MSK_DPAR_INTPNT_CO_TOL_DFEAS": 1.e-8,
            #                     "MSK_DPAR_INTPNT_CO_TOL_REL_GAP": 1.e-8, 
            #                     "MSK_DPAR_INTPNT_CO_TOL_NEAR_REL": 1000}
            # result = prob.solve(solver = "MOSEK", verbose=False,\
            #                 mosek_params = mosek_params_dict)
           
            for i in range(Np):
                R[i] = X[i].value
                
       
        residue = 1.0*G
        for i in range(Norb):
            for j in range(Norb):
                residue[:,i,j] = residue[:,i,j] -  M@ R[:,i,j]
    return R, M, residue
    
    
def aaa_reduce(pol, R, eps=1e-6):
    Np = R.shape[0]
    Rnorm = np.zeros(Np)
    for i in range(Np):
        Rnorm[i] = np.linalg.norm(R[i])
    nonz_index = Rnorm>eps
    return pol[nonz_index], R[nonz_index]

def erroreval(pol,  Z, G, cleanflag=True, maxiter=100000,Hermitian=True):
    R, M, residue = get_weight(pol,  Z, G, cleanflag=cleanflag, maxiter=maxiter,Hermitian=Hermitian)
    if len(G.shape)==1:
        y = np.linalg.norm(residue)
        grad = np.real(np.dot(np.conj(residue) ,(R*(M**2))))
    else:
        y = np.linalg.norm(residue.flatten())

        Np = len(pol)
        grad = np.zeros(Np)
        Nw = len(Z)
        for k in range(Np):
            for w in range(Nw):
                grad[k] = grad[k] + np.real(np.sum((M[w,k]**2)*(np.conj(residue[w,:,:]) * R[k])))

    grad = -grad/y
    return y, grad

def polefitting(Deltaiw, Z, Deltat,tgrid,beta, Np_max=50,eps = 1e-5,Hermitian=True):
    for mmax in range(4,Np_max,2):
        r = aaa_matrix_real(Deltaiw, Z, mmax=mmax)
        pol = r.pol()
        # pol = pol[np.abs(np.imag(pol))<0.1]
        pol = np.real(pol)
        weight, _, residue = get_weight_t(pol, tgrid, Deltat,beta)
        pol, weight = aaa_reduce(pol, weight,eps)
        fhere = lambda pole: erroreval_t(pol, tgrid, Deltat,beta)
        res = scipy.optimize.minimize(fhere,pol, method='L-BFGS-B', jac=True,options= {"disp" :False,"gtol":1e-10,"ftol":1e-10})
        weight, _, residue = get_weight_t(pol, tgrid, Deltat,beta)
        error = np.linalg.norm(residue) / np.sqrt(len(tgrid))
        if error<eps:
            return weight, res.x, error
        
    return weight, res.x, np.linalg.norm(residue)
# def polefitting(Deltaiw, Z, Np_max=50,eps = 1e-5,Hermitian=True):
#     for mmax in range(4,Np_max,2):
#         r = aaa_matrix_real(Deltaiw, Z, mmax=mmax)
#         pol = r.pol()
#         # pol = pol[np.abs(np.imag(pol))<0.1]
#         pol = np.real(pol)
#         weight, _, residue = get_weight(pol, Z, Deltaiw,cleanflag=True,Hermitian=Hermitian)
#         pol, weight = aaa_reduce(pol, weight,eps)
#         fhere = lambda pole: erroreval(pole,Z, Deltaiw,cleanflag=True,Hermitian=Hermitian)
#         res = scipy.optimize.minimize(fhere,pol, method='L-BFGS-B', jac=True,options= {"disp" :False,"gtol":1e-10,"ftol":1e-10})
#         weight, _, residue = get_weight(res.x, Z, Deltaiw,cleanflag=True, Hermitian=Hermitian)
#         # print(np.linalg.norm(residue))
#         if np.linalg.norm(residue)<eps:
#             return weight, res.x, np.linalg.norm(residue)
        
#     return weight, res.x, np.linalg.norm(residue)
