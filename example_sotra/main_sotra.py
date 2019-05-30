import numpy as np
import scipy.sparse as sps
import time
from tabulate import tabulate
import random

import porepy as pp

from data_sotra import Data
from solver_hazmath import SolverHazmath


# export solution to .vtk
def visualize(gb, solver_flow, x, file_name="solution", folder="solution_rt0"):
    # extract solution
    solver_flow.split(gb, "up", x)
    solver_flow.extract_p(gb, "up", "pressure")

    # save solution as vtk
    save = pp.Exporter(gb, file_name, folder=folder)
    save.write_vtk(["pressure"])

    return

# ---------------------------------------------------------------------------- #


# solve with direct python solver
def solve_direct(M, f, bmat=False):
    # direct solve of system
    #       Mx = f

    # if M in bmat form
    if bmat:
        M = sps.bmat(M, format="csc")
        f = np.concatenate(tuple(f))

    start_time = time.time()

    upl = sps.linalg.spsolve(M, f)

    print("Elapsed time direct solver: ", time.time() - start_time)

    return upl

# ---------------------------------------------------------------------------- #


# set up and solve the system with the HAZMATH solver
def main_HAZMATH(name_="one_fracture", mesh_size=1./8, aperture=1.,kf=1.,kn=1.):
    file_name = name_ + ".csv"

    # create grids
    data = Data({"mesh_size": mesh_size, "aperture": aperture, "km": 1.,
                 "kf": kf, "kn": kn}, file_name)

    # choose discretization
    discr = pp.RT0MixedDim("flow")

    # assemble
    # get matrix and rhs
    A, b = discr.matrix_rhs(data.gb)

    # choose linear solver
    solver_hazmath = SolverHazmath(data.gb, discr)

    # get the block structure of M, f
    solver_hazmath.setup_system(A, b)

    # solve with HAZMATH
    x_hazmath, iters = solver_hazmath.solve()

    # solve (directly),u
    x_direct = solve_direct(solver_hazmath.M, solver_hazmath.f, bmat=True)

    # permute solutions
    # y_hazmath = solver_hazmath.P.T * x_hazmath
    # y_direct = solver_hazmath.P.T * x_direct

    # write to vtk
    # visualize(data.gb, discr, y_hazmath, file_name="sol_hazmath")
    # visualize(data.gb, discr, y_direct, file_name="sol_direct")

    error = np.linalg.norm(x_direct - x_hazmath) / np.linalg.norm(x_direct)
    print("Error: ", error)

    return iters

# ---------------------------------------------------------------------------- #


# run the solver for different mesh sizes
def test_mesh_size(name_="one_fracture"):
    aperture = 1. / (10. ** 2)
    kn = 10.**0
    kf = 10.**0

    table_h = []
    for k in np.arange(2, 7):
        mesh_size = 600./(2. ** k)

        it = main_HAZMATH(name_, mesh_size, aperture, kf, kn)

        table_h.append([mesh_size, it])

    # np.savetxt("tables/"+name_+"_mesh_size_iter.csv", table_h)

    return tabulate(table_h, headers=["h", "iter"])

# ---------------------------------------------------------------------------- #

def test_perm_random(name_="network_sotra", k=10):
    # pick k fractures that will be 
    frac_random = random.sample(range(1,63), k)
    

# ---------------------------------------------------------------------------- #


# run the solver for different tangential and normal fracture permeabilities
def test_permeabilities(name_="one_fracture"):
    mesh_size = 1./(2. ** 4)
    aperture = 1. / (10. ** 2)

    table_kf = []
    for k in np.arange(-4, 5, 4):
        kf = 10. ** k
        table_kn = []
        for l in np.arange(-4, 5, 4):
            kn = 10. ** l

            it = main_HAZMATH(name_, mesh_size, aperture, kf, kn)

            table_kn.append(it)

        # header
        table_kn.insert(0, kf)
        table_kf.append(table_kn)

    np.savetxt("tables/"+name_+"_perm_iter.csv", table_kf)

    return tabulate(table_kf, headers=["Kf / Kn", "1e-4", "1e0", "1e4"])

# ---------------------------------------------------------------------------- #


# call functions
if __name__ == "__main__":
    # grid config file (without extension!)
    name = "network_sotra"

    # table1 = test_mesh_size(name)
    # table2 = test_aperture(name)
    # table3 = test_permeabilities(name)

    # print(table1)
    # print(table2)
    # print(table3)
    main_HAZMATH(name_=name, mesh_size=10, aperture=1e-2, kf=1.,
                 kn=1.)
