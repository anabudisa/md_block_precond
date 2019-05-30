import numpy as np
import scipy as sp
import scipy.sparse as sps

import porepy as pp

# -------------------------------------
# import ctypes for HAZMATH -- Xiaozhe
import ctypes
from ctypes.util import find_library


class SolverHazmath(object):

    def __init__(self, gb, solver):
        # Grid bucket
        self.gb = gb
        # Discretization
        self.solver = solver
        # stiffness matrix in 2x2 block form: 0 -> (u, lambda), 1 -> p
        self.M = None
        # right hand side in 2x1 block form: : 0 -> (u, lambda), 1 -> p
        self.f = None
        # permutation matrix from original porepy order to new 2x2 block order
        self.P = None
        # list of block degrees of freedom: 0 -> (u, lambda), 1 -> p
        self.block_dof_list = None
        # extension operator
        self.R = None

    # ------------------------------------------------------------------------ #

    def solve(self, tol=1e-5, maxit=100):
        # solve the system using HAZMATH
        #          M x = f

        # right hand side
        print("--- Make right hand side")
        ff = np.concatenate(tuple(self.f))
        print("--- Done")

        # Mass matrix of pressure
        print("--- Get pressure mass matrix")
        Mp = self.P0_mass_matrix()
        Mp_diag = Mp.diagonal()
        print("--- Done")
        
        # ------------------------------------
        # prepare HAZMATH solver
        # ------------------------------------
        # call HAZMATH solver library
        libHAZMATHsolver = ctypes.cdll.LoadLibrary(
            '/home/abudis01/hazmath/lib/libhazmath.so')

        # parameters for HAZMATH solver
        prtlvl = ctypes.c_int(3)
        tol = ctypes.c_double(tol)
        maxit = ctypes.c_int(maxit)

        # ------------------------------------
        # convert
        # ------------------------------------
        print("--- Store information about the matrix in ctype variables")
        # information about the matrix
        Muu_size = self.M[0, 0].shape
        nrowp1_uu = Muu_size[0] + 1
        nrow_uu = ctypes.c_int(Muu_size[0])
        ncol_uu = ctypes.c_int(Muu_size[1])
        nnz_uu = ctypes.c_int(self.M[0, 0].nnz)

        Mup_size = self.M[0, 1].shape
        nrowp1_up = Mup_size[0] + 1
        nrow_up = ctypes.c_int(Mup_size[0])
        ncol_up = ctypes.c_int(Mup_size[1])
        nnz_up = ctypes.c_int(self.M[0, 1].nnz)

        Mpu_size = self.M[1, 0].shape
        nrowp1_pu = Mpu_size[0] + 1
        nrow_pu = ctypes.c_int(Mpu_size[0])
        ncol_pu = ctypes.c_int(Mpu_size[1])
        nnz_pu = ctypes.c_int(self.M[1, 0].nnz)

        Mpp_size = self.M[1, 1].shape
        nrowp1_pp = Mpp_size[0] + 1
        nrow_pp = ctypes.c_int(Mpp_size[0])
        ncol_pp = ctypes.c_int(Mpp_size[1])
        nnz_pp = ctypes.c_int(self.M[1, 1].nnz)

        # allocate solution
        nrow = Muu_size[0] + Mpp_size[0]
        nrow_double = ctypes.c_double * nrow
        hazmath_sol = nrow_double()
        numiters = ctypes.c_int(-1)
        print("--- Done")

        # ------------------------------------
        # solve using HAZMATH
        # ------------------------------------
        print("--- Call hazmath solver")
        libHAZMATHsolver.python_wrapper_krylov_mixed_darcy(
            ctypes.byref(nrow_uu), ctypes.byref(ncol_uu),
            ctypes.byref(nnz_uu),
            (ctypes.c_int * nrowp1_uu)(*self.M[0, 0].indptr),
            (ctypes.c_int * self.M[0, 0].nnz)(*self.M[0, 0].indices),
            (ctypes.c_double * self.M[0, 0].nnz)(*self.M[0, 0].data),
            ctypes.byref(nrow_up),
            ctypes.byref(ncol_up), ctypes.byref(nnz_up),
            (ctypes.c_int * nrowp1_up)(*self.M[0, 1].indptr),
            (ctypes.c_int * self.M[0, 1].nnz)(*self.M[0, 1].indices),
            (ctypes.c_double * self.M[0, 1].nnz)(*self.M[0, 1].data),
            ctypes.byref(nrow_pu), ctypes.byref(ncol_pu),
            ctypes.byref(nnz_pu),
            (ctypes.c_int * nrowp1_pu)(*self.M[1, 0].indptr),
            (ctypes.c_int * self.M[1, 0].nnz)(*self.M[1, 0].indices),
            (ctypes.c_double * self.M[1, 0].nnz)(*self.M[1, 0].data),
            ctypes.byref(nrow_pp),
            ctypes.byref(ncol_pp), ctypes.byref(nnz_pp),
            (ctypes.c_int * nrowp1_pp)(*self.M[1, 1].indptr),
            (ctypes.c_int * self.M[1, 1].nnz)(*self.M[1, 1].indices),
            (ctypes.c_double * self.M[1, 1].nnz)(*self.M[1, 1].data),
            (ctypes.c_double * Mpp_size[0])(*Mp_diag),
            (ctypes.c_double * nrow)(*ff),
            ctypes.byref(hazmath_sol), ctypes.byref(tol),
            ctypes.byref(maxit),
            ctypes.byref(prtlvl), ctypes.byref(numiters))
        print("--- Done")

        # ------------------------------------
        # convert solution
        # ------------------------------------
        print("--- Convert solution")
        x = sp.array(hazmath_sol)
        numiters = sp.array(numiters)
        print("--- Done")
        
        return x, numiters

    # ------------------------------------------------------------------------ #

    def P0_mass_matrix(self):
        matrices = []

        # assemble matrix for each grid in GridBucket
        for g, d in self.gb:
            M_g = sps.diags(g.cell_volumes)
            matrices.append(M_g)

        return sps.block_diag(tuple(matrices), format="csr")

    # ------------------------------------------------------------------------ #

    def permutation_matrix(self, perm):
        self.P = sps.identity(len(perm)).tocoo()

        self.P.row = np.argsort(perm)

    # ------------------------------------------------------------------------ #

    def permute_dofs(self):
        dof_u = np.array([], dtype=int)
        dof_p = np.array([], dtype=int)
        dof_l = np.array([], dtype=int)

        dof_list = self.solver.solver._dof_start_of_grids(self.gb)
        for g, d in self.gb:
            nn = d['node_number']
            local_dof_list = np.arange(dof_list[nn], dof_list[nn + 1])
            dim_u = d['dof_u']
            # dim_p = d['dof_p']
            dof_u = np.append(dof_u, local_dof_list[:dim_u])
            dof_p = np.append(dof_p, local_dof_list[dim_u:])

        for e, d in self.gb.edges():
            nn = d['edge_number'] + self.gb.num_graph_nodes()
            dof_l = np.append(dof_l, np.arange(dof_list[nn], dof_list[nn + 1]))

        perm = np.hstack((dof_u, dof_l, dof_p))
        self.block_dof_list = [np.concatenate((dof_u, dof_l)), dof_p]

        return perm

    # ------------------------------------------------------------------------ #

    def extension_operator(self):
        R_list = []
        # assemble extension operator over all mortar grids
        for e, d in self.gb.edges():
            mg = d['mortar_grid']
            if mg.dim == self.gb.dim_max() - 1:
                Rg = mg.high_to_mortar_int
                R_list.append(Rg)

        self.R = sps.hstack(R_list, format="csc")

    # ------------------------------------------------------------------------ #

    def setup_system(self, A, b):
        # get the permutation
        perm = self.permute_dofs()
        self.permutation_matrix(perm)

        AA = A.copy()
        bb = np.copy(b)

        # setup block structure
        blocks_no = len(self.block_dof_list)
        self.M = np.empty(shape=(blocks_no, blocks_no), dtype=np.object)
        self.f = np.empty(shape=(blocks_no,), dtype=np.object)

        for row in np.arange(blocks_no):
            for col in np.arange(blocks_no):
                self.M[row, col] = AA[self.block_dof_list[row], :].tocsc()[:,
                              self.block_dof_list[col]].tocsr()
            self.f[row] = bb[self.block_dof_list[row]]

    # ------------------------------------------------------------------------ #

