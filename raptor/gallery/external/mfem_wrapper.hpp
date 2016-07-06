// Copyright (c) 2015, Raptor Developer Team, University of Illinois at Urbana-Champaign
// License: Simplified BSD, http://opensource.org/licenses/BSD-2-Clause

#ifndef RAPTOR_MFEM_WRAPPER_H
#define RAPTOR_MFEM_WRAPPER_H

#include "core/types.hpp"
#include "core/par_matrix.hpp"
#include "core/par_vector.hpp"

/**************************************************************
 *****   MFEM Laplace Matrix
 **************************************************************
 ***** Creates a Laplacian matrix from MFEM
 *****
 ***** Parameters
 ***** -------------
 ***** A : raptor::ParMatrix**
 *****    Pointer to uninitialized parallel matrix for laplacian to be stored in
 ***** x : ParVector**
 *****    Pointer to uninitialized parallel vector for solution vector
 ***** y : ParVector**
 *****    Pointer to uninitialized parallel vector for rhs vector
 ***** mesh_file : char (const)
 *****    Location of file containing mesh for MFEM to use
 ***** num_elements: int
 *****    Maximum size for refined serial mesh
 ***** order : int (optional)
 *****    Use continuous Lagrange finite elements of this order.
 *****    If order < 1, uses isoparametric/isogeometric space.
 *****    Default = 3
 ***** comm_mat : MPI_Comm (optional)
 *****    MPI_Communicator for A.  Default = MPI_COMM_WORLD.
 **************************************************************/
void mfem_laplace(raptor::ParMatrix** A_ptr, raptor::ParVector** x_ptr, raptor::ParVector** b_ptr, const char* mesh_file, int num_elements, int order = 3, MPI_Comm comm_mat = MPI_COMM_WORLD);

/**************************************************************
 *****   MFEM Linear Elasticity Matrix
 **************************************************************
 ***** Creates a linear elasticity matrix from MFEM
 *****
 ***** Parameters
 ***** -------------
 ***** A : raptor::ParMatrix**
 *****    Pointer to uninitialized parallel matrix for laplacian to be stored in
 ***** x : ParVector**
 *****    Pointer to uninitialized parallel vector for solution vector
 ***** y : ParVector**
 *****    Pointer to uninitialized parallel vector for rhs vector
 ***** mesh_file : char (const)
 *****    Location of file containing mesh for MFEM to use
 ***** num_elements: int
 *****    Maximum size for refined serial mesh
 ***** order : int (optional)
 *****    Use continuous Lagrange finite elements of this order.
 *****    If order < 1, uses isoparametric/isogeometric space.
 *****    Default = 3
 ***** comm_mat : MPI_Comm (optional)
 *****    MPI_Communicator for A.  Default = MPI_COMM_WORLD.
 **************************************************************/
void mfem_linear_elasticity(raptor::ParMatrix** A_ptr, raptor::ParVector** x_ptr, raptor::ParVector** b_ptr, const char* mesh_file, int num_elements, int order = 3, MPI_Comm comm_mat = MPI_COMM_WORLD);

/**************************************************************
 *****   MFEM Electomagnetic Diffusion
 **************************************************************
 ***** Creates a linear elasticity matrix from MFEM
 *****
 ***** Parameters
 ***** -------------
 ***** A : raptor::ParMatrix**
 *****    Pointer to uninitialized parallel matrix for laplacian to be stored in
 ***** x : ParVector**
 *****    Pointer to uninitialized parallel vector for solution vector
 ***** y : ParVector**
 *****    Pointer to uninitialized parallel vector for rhs vector
 ***** mesh_file : char (const)
 *****    Location of file containing mesh for MFEM to use
 ***** num_elements: int
 *****    Maximum size for refined serial mesh
 ***** order : int (optional)
 *****    Use continuous Lagrange finite elements of this order.
 *****    If order < 1, uses isoparametric/isogeometric space.
 *****    Default = 3
 ***** comm_mat : MPI_Comm (optional)
 *****    MPI_Communicator for A.  Default = MPI_COMM_WORLD.
 **************************************************************/
void mfem_electromagnetic_diffusion(raptor::ParMatrix** A_ptr, raptor::ParVector** x_ptr, raptor::ParVector** b_ptr, const char* mesh_file, int num_elements, int order = 3, MPI_Comm comm_mat = MPI_COMM_WORLD);

/**************************************************************
 *****   MFEM Hdiv Diffusion
 **************************************************************
 ***** Creates a linear elasticity matrix from MFEM
 *****
 ***** Parameters
 ***** -------------
 ***** A : raptor::ParMatrix**
 *****    Pointer to uninitialized parallel matrix for laplacian to be stored in
 ***** x : ParVector**
 *****    Pointer to uninitialized parallel vector for solution vector
 ***** y : ParVector**
 *****    Pointer to uninitialized parallel vector for rhs vector
 ***** mesh_file : char (const)
 *****    Location of file containing mesh for MFEM to use
 ***** num_elements: int
 *****    Maximum size for refined serial mesh
 ***** order : int (optional)
 *****    Use continuous Lagrange finite elements of this order.
 *****    If order < 1, uses isoparametric/isogeometric space.
 *****    Default = 3
 ***** set_bc : Bool (Optional)
 *****    Whether to set boundary conditions.  Default = True
 ***** comm_mat : MPI_Comm (optional)
 *****    MPI_Communicator for A.  Default = MPI_COMM_WORLD.
 **************************************************************/
void mfem_hdiv_diffusion(raptor::ParMatrix** A_ptr, raptor::ParVector** x_ptr, raptor::ParVector** b_ptr, const char* mesh_file, int num_elements, int order = 3, bool set_bc = true, MPI_Comm comm_mat = MPI_COMM_WORLD);

#endif
