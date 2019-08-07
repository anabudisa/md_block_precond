# Research

Block Preconditioners for Mixed-dimensional Discretization of Flow in Fractured Porous Media, by Ana Budiša and Xiaozhe Hu.  [arXiv:1905.13513](https://arxiv.org/abs/1905.13513).

# Abstract

We are interested in an efficient numerical method for the mixed-dimensional approach to modeling single-phase flow in fractured porous media. The model introduces fractures and their intersections as lower-dimensional structures, and the mortar variable is used for flow coupling between the matrix and fractures. We consider a stable mixed finite element discretization of the problem, which results in a parameter-dependent linear system. For this, we develop block preconditioners based on the well-posedness of the discretization choice. The preconditioned iterative method demonstrates robustness with regards to discretization and physical parameters. The analytical results are verified on several examples of fracture network configurations, and notable results in reduction of number of iterations and computational time are obtained.

# Code

- This code works with [PorePy](https://github.com/pmgbergen/porepy) at commit [0f5d28aced2259ec762bdb3fdff100445ccb8e92](https://github.com/pmgbergen/porepy/tree/0f5d28aced2259ec762bdb3fdff100445ccb8e92). It is not guaranteed that the code works with later versions.
- This code requires [HAZmath](https://bitbucket.org/hazmath/hazmath/) library. (private)

# Cite

We kindly ask you to cite the following publication [arXiv:1905.13513](https://arxiv.org/abs/1905.13513) when using the work contained in this repository.

# Licence

See [LICENCE](https://github.com/anabudisa/md_block_precond/blob/master/LICENSE).
