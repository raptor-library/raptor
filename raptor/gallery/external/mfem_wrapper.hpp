#ifndef RAPTOR_MFEM_WRAPPER_H
#define RAPTOR_MFEM_WRAPPER_H

#include "gallery/external/hypre_wrapper.hpp"
#include "core/types.hpp"
#include "core/par_matrix.hpp"
#include "core/par_vector.hpp"
#include <fstream>
#include <iostream>

using namespace std;

//void mfem_laplace(raptor::ParMatrix** A_ptr, raptor::ParVector** x_ptr, raptor::ParVector** b_ptr, const char* mesh_file, int num_elements, int order = 3, MPI_Comm comm_mat = MPI_COMM_WORLD);

raptor::ParCSRMatrix* mfem_linear_elasticity(const char* mesh_file, int num_elements, 
        int order = 3, MPI_Comm comm_mat = MPI_COMM_WORLD);

//void mfem_darcy(raptor::ParMatrix** A_ptr, raptor::ParVector** x_ptr, raptor::ParVector** b_ptr, const char* mesh_file, int num_elements, int order = 3, MPI_Comm comm_mat = MPI_COMM_WORLD);


#endif
