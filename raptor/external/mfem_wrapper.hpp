// Copyright (c) 2015-2017, RAPtor Developer Team
// License: Simplified BSD, http://opensource.org/licenses/BSD-2-Clause

#ifndef RAPTOR_MFEM_WRAPPER_H
#define RAPTOR_MFEM_WRAPPER_H

#include "raptor/external/hypre_wrapper.hpp"
#include "raptor/core/types.hpp"
#include "raptor/core/par_matrix.hpp"
#include "raptor/core/par_vector.hpp"
#include "mfem.hpp"
#include <fstream>
#include <iostream>
#include <vector>

raptor::ParCSRMatrix* mfem_linear_elasticity(raptor::ParVector& x_raptor,
        raptor::ParVector& b_raptor, int* num_variables,
        const char* mesh_file,
        int order = 3, int seq_n_refines = 2, int par_n_refines = 2,
        RAPtor_MPI_Comm comm_mat = RAPtor_MPI_COMM_WORLD,
        std::vector<double>* rigid_body_candidates = nullptr,
        int* num_rigid_body_candidates = nullptr,
        bool static_cond = true, 
        double material_contrast = 1.0);

raptor::ParCSRMatrix* mfem_dg_elasticity(raptor::ParVector& x_raptor,
        raptor::ParVector& b_raptor, int* num_variables,
        const char* mesh_file,
        int order = 3, int seq_n_refines = 2, int par_n_refines = 2,
        RAPtor_MPI_Comm comm_mat = RAPtor_MPI_COMM_WORLD);

raptor::ParCSRMatrix* mfem_grad_div(raptor::ParVector& x_raptor,
        raptor::ParVector& b_raptor,
        const char* mesh_file,
        int order = 3, int seq_n_refines = 2, int par_n_refines = 2,
        RAPtor_MPI_Comm comm_mat = RAPtor_MPI_COMM_WORLD);

raptor::ParCSRMatrix* mfem_adaptive_laplacian(raptor::ParVector& x_raptor,
        raptor::ParVector& b_raptor,
        const char* mesh_file,
        int order = 3, int max_dofs = 1000000,
        RAPtor_MPI_Comm comm_mat = RAPtor_MPI_COMM_WORLD);

raptor::ParCSRMatrix* mfem_dg_diffusion(raptor::ParVector& x_raptor,
        raptor::ParVector& b_raptor,
        const char* mesh_file,
        int order = 3, int seq_n_refines = 2, int par_n_refines = 2,
        RAPtor_MPI_Comm comm_mat = RAPtor_MPI_COMM_WORLD);

raptor::ParCSRMatrix* mfem_laplacian(raptor::ParVector& x_raptor,
        raptor::ParVector& b_raptor,
        const char* mesh_file,
        int order = 3, int seq_n_refines = 2, int par_n_refines = 2,
        RAPtor_MPI_Comm comm_mat = RAPtor_MPI_COMM_WORLD);

#endif
