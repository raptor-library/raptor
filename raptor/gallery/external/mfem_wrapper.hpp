#ifndef RAPTOR_MFEM_WRAPPER_H
#define RAPTOR_MFEM_WRAPPER_H

#include "core/types.hpp"
#include "core/par_matrix.hpp"
#include "core/par_vector.hpp"

void mfem_laplace(raptor::ParMatrix** A_ptr, raptor::ParVector** x_ptr, raptor::ParVector** b_ptr, const char* mesh_file, int num_elements, int order = 3, bool visualization = false);

void mfem_linear_elasticity(raptor::ParMatrix** A_ptr, raptor::ParVector** x_ptr, raptor::ParVector** b_ptr, const char* mesh_file, int num_elements, int order = 3, bool visualization = false);

void mfem_electromagnetic_diffusion(raptor::ParMatrix** A_ptr, raptor::ParVector** x_ptr, raptor::ParVector** b_ptr, const char* mesh_file, int num_elements, int order = 3, bool visualization = false);

void mfem_hdiv_diffusion(raptor::ParMatrix** A_ptr, raptor::ParVector** x_ptr, raptor::ParVector** b_ptr, const char* mesh_file, int num_elements, int order = 3, bool set_bc = true, bool visualization = false);

//void mfem_mixed_Darcy(raptor::ParMatrix** A_ptr, raptor::ParVector** x_ptr, raptor::ParVector** b_ptr, const char* mesh_file, int num_elements, int order = 3, bool visualization = false);

//void mfem_isoparametric_laplace(raptor::ParMatrix** A_ptr, raptor::ParVector** x_ptr, raptor::ParVector** b_ptr, int elem_type, int ref_levels, int order, bool always_snap = false, bool visualization = false);

#endif
