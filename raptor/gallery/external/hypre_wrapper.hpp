// Copyright (c) 2015, Raptor Developer Team, University of Illinois at Urbana-Champaign
// License: Simplified BSD, http://opensource.org/licenses/BSD-2-Clause

#ifndef RAPTOR_HYPRE_WRAPPER_H
#define RAPTOR_HYPRE_WRAPPER_H

#include "core/types.hpp"
#include "core/par_matrix.hpp"
#include "core/par_comm.hpp"
#include "core/hierarchy.hpp"
#include "_hypre_utilities.h"
#include "HYPRE.h"
#include "_hypre_parcsr_mv.h"
#include "_hypre_parcsr_ls.h"

/**************************************************************
 *****   Convert Raptor Vector to HYPRE_IJVector
 **************************************************************
 ***** Converts a raptor::ParVector* to a HYPRE_IJVector
 ***** (Deep Copy)
 ***** TODO -- Change this to shallow copy?
 *****
 ***** Parameters
 ***** -------------
 ***** x_rap : raptor::ParVector*
 *****    Parllel vector to be converted
 ***** comm_mat : MPI_Comm (optional)
 *****    Communicator of processes that contain
 *****    portions of the vector.  Default = MPI_COMM_WORLD
 *****
 ***** Returns
 ***** -------------
 ***** HYPRE_IJVector
 *****    A HYPRE Vector containing the same size and data 
 *****    as the raptor vector
 **************************************************************/
HYPRE_IJVector convert(raptor::ParVector* x_rap,
                       MPI_Comm comm_mat = MPI_COMM_WORLD);

/**************************************************************
 *****   Convert Raptor Matrix to HYPRE_IJMatrix
 **************************************************************
 ***** Converts a raptor::ParMatrix* to a HYPRE_IJMatrix
 ***** (Deep Copy)
 ***** TODO -- change this to shallow copy?
 *****
 ***** Parameters
 ***** -------------
 ***** A_rap : raptor::ParMarix*
 *****    Parallel matrix to be converted
 ***** comm_mat : MPI_Comm (optional)
 *****    Communicator of matrix.  Default = MPI_COMM_WORLD
 *****
 ***** Returns
 ***** -------------
 ***** HYPRE_IJMatrix
 *****    A HYPRE Matrix containing the same size and data 
 *****    as the raptor parallel matrix
 **************************************************************/
HYPRE_IJMatrix convert(raptor::ParMatrix* A_rap,
                       MPI_Comm comm_mat = MPI_COMM_WORLD);

/**************************************************************
 *****   Convert hypre_ParCSRMatrix* to raptor ParMatrix*
 **************************************************************
 ***** Converts a hypre_ParCSRMatrix* to a raptor::ParMatrix*
 ***** (Shallow Copy)
 *****
 ***** Parameters
 ***** -------------
 ***** A_hypre : hypre_ParCSRMatrix* 
 *****    Parallel matrix to be converted
 ***** comm_mat : MPI_Comm (optional)
 *****    Communicator of matrix.  Default = MPI_COMM_WORLD
 *****
 ***** Returns
 ***** -------------
 ***** raptor::ParMatrix*
 *****    A raptor parallel Matrix containing the same size and data 
 *****    as the original hypre_ParCSRMatrix* object
 **************************************************************/
raptor::ParMatrix* convert(hypre_ParCSRMatrix* A_hypre,
                           MPI_Comm comm_mat = MPI_COMM_WORLD);

/**************************************************************
 *****   Convert HYPRE Hierarchy to Raptor Hierarchy
 **************************************************************
 ***** Converts a HYPRE AMG hierarchy into Raptor type objects
 ***** (Shallow Copy)
 *****
 ***** Parameters
 ***** -------------
 ***** amg_data : hypre_ParAMGData*
 *****    Structure containing all objects in the AMG hierarchy
 ***** comm_mat : MPI_Comm (optional)
 *****    Communicator for original fine level matrix
 *****    Default = MPI_COMM_WORLD
 *****
 ***** Returns
 ***** -------------
 ***** raptor::Hierarchy*
 *****    AMG Hierarchy containing all coarse grid matrices (A) 
 *****    and prolongation operators (P) that were in the original
 *****    HYPRE hierarchy.
 **************************************************************/
raptor::Hierarchy* convert(hypre_ParAMGData* amg_data, 
                           MPI_Comm comm_mat = MPI_COMM_WORLD);

/**************************************************************
 *****   Remove shared pointers from HYPRE matrix
 **************************************************************
 ***** After conversion, many pointers are shared between
 ***** the original hypre matrix and the shallow copy into
 ***** the raptor matrix.  To delete both without errors, 
 ***** shared pointers need to be removed from the HYPRE matrix.
 *****
 ***** Parameters
 ***** -------------
 ***** A_hypre : hypre_ParCSRMatrix*
 *****    HYPRE matrix that has previously been shallow copied
 *****    into a raptor ParMatrix*
 **************************************************************/
void remove_shared_ptrs(hypre_ParCSRMatrix* A_hypre);

/**************************************************************
 *****   Remove shared pointers from HYPRE AMG Hierarchy
 **************************************************************
 ***** After conversion, many pointers are shared between
 ***** the original hypre hierarhcy and the shallow copy into
 ***** the raptor hierarchy.  To delete both without errors, 
 ***** shared pointers need to be removed from the HYPRE hierarchy.
 *****
 ***** Parameters
 ***** -------------
 ***** amg_data : hypre_ParAMGData*
 *****    Structure containing all objects in the AMG hierarchy
 **************************************************************/
void remove_shared_ptrs(hypre_ParAMGData* amg_data);

/**************************************************************
 *****   Create HYPRE AMG Hierarchy
 **************************************************************
 ***** Creates an AMG hierachy using the HYPRE setup phase
 *****
 ***** Parameters
 ***** -------------
 ***** A : hypre_ParCSRMatrix*
 *****    Fine-level matrix in HYPRE format
 ***** b : hypre_ParVector* 
 *****    Fine-level right hand size (HYPRE format)
 ***** x : hypre_ParVector*
 *****    Fine-level solution vector (HYPRE format)
 ***** coarsen_type : HYPRE_Int (optional)
 *****    Type of coarsening strategy.
 *****    Default = Falgout (6)
 ***** interp_type : HYPRE_Int (optional)
 *****    Type of interpolation strategy
 *****    Default = Classical modified interpolation (0)
 ***** p_max_elmts : HYPRE_Int (optional)
 *****    Max number of elements per row in interpolation
 *****    Default = 0
 ***** agg_num_levels : HYPRE_Int (optional)
 *****    Number of levels to be aggressively coarsened
 *****    Default = 0
 ***** strong_threshold : HYPRE_Real (optional)
 *****    Strength threshold for strongly connected entries
 *****    Default = 0.25
 ***** Returns
 ***** -------------
 ***** raptor::Hierarchy*
 *****    AMG Hierarchy containing all coarse grid matrices (A) 
 *****    and prolongation operators (P) that were in the original
 *****    HYPRE hierarchy.
 **************************************************************/
HYPRE_Solver hypre_create_hierarchy(hypre_ParCSRMatrix* A,
                                hypre_ParVector* x,
                                hypre_ParVector* b,
                                int coarsen_type = 6,
                                int interp_type = 0,
                                int p_max_elmts = 0,
                                int agg_num_levels = 0,
                                double strong_threshold = 0.25);

/**************************************************************
 *****   Create Raptor AMG Hierarchy (VIA HYPRE)
 **************************************************************
 ***** Converts raptor system to HYPRE formats, and creates
 ***** and AMG hierarchy using the HYPRE setup phase.
 ***** Shallow copies the hierarchy to Raptor objects, 
 ***** removes shared pointers, and deletes all HYPRE objects.
 *****
 ***** Parameters
 ***** -------------
 ***** A_rap : raptor::ParMatrix*
 *****    Fine-level matrix in HYPRE format
 ***** x_rap : raptor::ParVector*
 *****    Fine-level solution vector (HYPRE format)
 ***** b_rap : raptor::ParVector* 
 *****    Fine-level right hand size (HYPRE format)
 ***** coarsen_type : HYPRE_Int (optional)
 *****    Type of coarsening strategy.
 *****    Default = Falgout (6)
 ***** interp_type : HYPRE_Int (optional)
 *****    Type of interpolation strategy
 *****    Default = Classical modified interpolation (0)
 ***** p_max_elmts : HYPRE_Int (optional)
 *****    Max number of elements per row in interpolation
 *****    Default = 0
 ***** agg_num_levels : HYPRE_Int (optional)
 *****    Number of levels to be aggressively coarsened
 *****    Default = 0
 ***** strong_threshold : HYPRE_Real (optional)
 *****    Strength threshold for strongly connected entries
 *****    Default = 0.25
 ***** comm_mat : MPI_Comm (optional)
 *****    Communicator for original raptor matrix
 *****    Default = MPI_COMM_WORLD
 *****
 ***** Returns
 ***** -------------
 ***** raptor::Hierarchy*
 *****    AMG Hierarchy containing all coarse grid matrices (A) 
 *****    and prolongation operators (P) for the system and
 *****    options passed to HYPRE
 **************************************************************/
raptor::Hierarchy* create_wrapped_hierarchy(raptor::ParMatrix* A_rap,
                                raptor::ParVector* x_rap,
                                raptor::ParVector* b_rap,
                                int coarsen_type = 6,
                                int interp_type = 0,
                                int p_max_elmts = 0,
                                int agg_num_levels = 0,
                                double strong_threshold = 0.25,
                                MPI_Comm comm_mat = MPI_COMM_WORLD);

#endif
