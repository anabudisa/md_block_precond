import numpy as np
import porepy as pp
from tabulate import tabulate


class Data(object):

    def __init__(self, data, file_name):

        self.gb = None
        self.domain = None
        self.data = data

        self.tol = 1e-8

        self.create_gb(file_name)
        self.add_data()

    # ------------------------------------------------------------------------ #

    def create_gb(self, file_name):

        mesh_kwargs = {}
        mesh_kwargs = {"mesh_size": self.data["mesh_size"],
                       "mesh_size_frac": self.data["mesh_size"],
                       "mesh_size_bound": self.data["mesh_size"],
                       "mesh_size_min": self.data["mesh_size"]}

        self.domain = {"xmin": 0, "xmax": 700, "ymin": 0, "ymax": 600}

        self.gb = pp.importer.dfm_2d_from_csv(file_name, mesh_kwargs,
                                             self.domain)

        self.gb.compute_geometry()
        # pp.coarsening.coarsen(self.gb, "by_volume")
        self.gb.assign_node_ordering()

        self.up_dof()

    # ------------------------------------------------------------------------ #

    def add_data(self):

        self.gb.add_node_props(["param", "is_tangential"])

        for g, d in self.gb:
            param = pp.Parameters(g)

            ones = np.ones(g.num_cells)
            zeros = np.zeros(g.num_cells)
            empty = np.empty(0)
            
            # Tangential permeability
            if g.dim == 2:
                kxx = ones * self.data["km"]
                perm = pp.SecondOrderTensor(g.dim, kxx=kxx, kyy=kxx, kzz=1)
            else:
                kxx = ones * self.data["kf"]
                perm = pp.SecondOrderTensor(g.dim, kxx=kxx, kyy=1, kzz=1)
            param.set_tensor("flow", perm)

            # Aperture
            aperture = np.power(self.data["aperture"], self.gb.dim_max() -
                                g.dim)
            param.set_aperture(ones  * aperture)

            # Source term
            param.set_source("flow", zeros)

            # Boundaries
            bound_faces = g.get_boundary_faces()
            if bound_faces.size == 0:
                bc = pp.BoundaryCondition(g, empty, empty)
                param.set_bc("flow", bc)
            else:
                bound_face_centers = g.face_centers[:, bound_faces]

                right = bound_face_centers[0, :] > self.domain["xmax"] - \
                                                   self.tol
                left = bound_face_centers[0, :] < self.domain['xmin'] + self.tol

                labels = np.array(["neu"] * bound_faces.size)
                labels[np.logical_or(right, left)] = "dir"

                bc_val = np.zeros(g.num_faces)
                bc_val[bound_faces[left]] = 1013250

                param.set_bc("flow", pp.BoundaryCondition(g, bound_faces,
                                                          labels))
                param.set_bc_val("flow", bc_val)

            d["is_tangential"] = True
            d["param"] = param

        # Normal permeability
        self.gb.add_edge_props("kn")
        for e, d in self.gb.edges():
            g_l = self.gb.nodes_of_edge(e)[0]
            mg = d['mortar_grid']
            check_P = mg.low_to_mortar_avg()

            gamma = np.power(
                check_P * self.gb.node_props(g_l, "param").get_aperture(),
                1. / (self.gb.dim_max() - g_l.dim))
            d["kn"] = self.data["kn"] * np.ones(mg.num_cells) / gamma

    # ------------------------------------------------------------------------ #

    def up_dof(self):
        # add attribute to all node and edge grids - number of dof for each
        # variable
        self.gb.add_node_props(['dof_u', 'dof_p'])
        for g, d in self.gb:
            d['dof_u'] = g.num_faces
            d['dof_p'] = g.num_cells

        self.gb.add_edge_props('dof_lmbd')
        for _, d in self.gb.edges():
            d['dof_lmbd'] = d['mortar_grid'].num_cells

    # ------------------------------------------------------------------------ #

    def print_setup(self):
        print(" ------------------------------------------------------------- ")
        print(" -------------------- PROBLEM SETUP -------------------------- ")

        table = [["Mesh size", self.data["mesh_size"]],
                 ["Aperture", self.data["aperture"]],
                 ["Km", self.data["km"]],
                 ["Kf", self.data["kf"]],
                 ["Kn", self.data["kn"]]]

        print(tabulate(table, headers=["Parameter", "Value"]))

        print(" ------------------------------------------------------------- ")

    # ------------------------------------------------------------------------ #
