// Copyright (c) 2015-2017, RAPtor Developer Team
// License: Simplified BSD, http://opensource.org/licenses/BSD-2-Clause

#ifndef RAPTOR_MFEM_WRAPPER_H
#define RAPTOR_MFEM_WRAPPER_H

#include "gallery/external/hypre_wrapper.hpp"
#include "core/types.hpp"
#include "core/par_matrix.hpp"
#include "core/par_vector.hpp"
#include "mfem.hpp"
#include <fstream>
#include <iostream>

using namespace std;

raptor::ParCSRMatrix* mfem_linear_elasticity(raptor::ParVector& x_raptor, 
        raptor::ParVector& b_raptor, int* num_variables,
        const char* mesh_file, 
        int order = 3, int seq_n_refines = 2, int par_n_refines = 2,
        MPI_Comm comm_mat = MPI_COMM_WORLD);

raptor::ParCSRMatrix* mfem_dg_elasticity(raptor::ParVector& x_raptor, 
        raptor::ParVector& b_raptor, int* num_variables,
        const char* mesh_file, 
        int order = 3, int seq_n_refines = 2, int par_n_refines = 2,
        MPI_Comm comm_mat = MPI_COMM_WORLD);

raptor::ParCSRMatrix* mfem_laplacian(raptor::ParVector& x_raptor, 
        raptor::ParVector& b_raptor,
        const char* mesh_file, 
        int order = 3, int seq_n_refines = 2, int par_n_refines = 2,
        MPI_Comm comm_mat = MPI_COMM_WORLD);

raptor::ParCSRMatrix* mfem_grad_div(raptor::ParVector& x_raptor, 
        raptor::ParVector& b_raptor,
        const char* mesh_file, 
        int order = 3, int seq_n_refines = 2, int par_n_refines = 2,
        MPI_Comm comm_mat = MPI_COMM_WORLD);

raptor::ParCSRMatrix* mfem_adaptive_laplacian(raptor::ParVector& x_raptor, 
        raptor::ParVector& b_raptor,
        const char* mesh_file, 
        int order = 3, MPI_Comm comm_mat = MPI_COMM_WORLD);

raptor::ParCSRMatrix* mfem_dg_diffusion(raptor::ParVector& x_raptor, 
        raptor::ParVector& b_raptor,
        const char* mesh_file, 
        int order = 3, int seq_n_refines = 2, int par_n_refines = 2,
        MPI_Comm comm_mat = MPI_COMM_WORLD);

#endif
