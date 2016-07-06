// Copyright (c) 2015, Raptor Developer Team, University of Illinois at Urbana-Champaign
// License: Simplified BSD, http://opensource.org/licenses/BSD-2-Clause

#include "hypre_wrapper.hpp"

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
HYPRE_IJVector convert(raptor::ParVector* x_rap, MPI_Comm comm_mat)
{
    HYPRE_IJVector x;

    HYPRE_Int first_local = x_rap->first_local;
    HYPRE_Int local_n = x_rap->local_n;

    HYPRE_IJVectorCreate(comm_mat, first_local, first_local + local_n, &x);
    HYPRE_IJVectorSetObjectType(x, HYPRE_PARCSR);
    HYPRE_IJVectorInitialize(x);

    HYPRE_Int* rows = new HYPRE_Int[local_n];
    for (HYPRE_Int i = 0; i < local_n; i++)
    {
        rows[i] = i+first_local;
    }
    HYPRE_Real* x_data = x_rap->local->data();
    HYPRE_IJVectorSetValues(x, local_n, rows, x_data);

    HYPRE_IJVectorAssemble(x);

    delete[] rows;

    return x;
}

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
HYPRE_IJMatrix convert(raptor::ParMatrix* A_rap, MPI_Comm comm_mat)
{
    // Declare variables
    HYPRE_IJMatrix A;
    HYPRE_Int n_rows = 0;
    HYPRE_Int n_cols = 0;
    HYPRE_Int local_row_start = 0;
    HYPRE_Int local_col_start = 0;
    HYPRE_Int rank, num_procs;
    HYPRE_Int one = 1;

    MPI_Comm_rank(comm_mat, &rank);
    MPI_Comm_size(comm_mat, &num_procs);

    // Determine dimensions
    n_rows = A_rap->local_rows;
    if (n_rows)
    {
        n_cols = A_rap->local_cols;
        local_row_start = A_rap->first_row;
        local_col_start = A_rap->first_col_diag;
    }

    // Create HYPRE_Matrix
    HYPRE_IJMatrixCreate(comm_mat, local_row_start, local_row_start + n_rows - 1, local_col_start, local_col_start + n_cols - 1, &A);
    HYPRE_IJMatrixSetObjectType(A, HYPRE_PARCSR);
    HYPRE_IJMatrixInitialize(A);

    // Add diagonal block values
    for (int i = 0; i < A_rap->diag->n_rows; i++)
    {
        HYPRE_Int row_start = A_rap->diag->indptr[i];
        HYPRE_Int row_end = A_rap->diag->indptr[i+1];
        HYPRE_Int global_row = i + A_rap->first_row;
        for (int j = row_start; j < row_end; j++)
        {
            HYPRE_Int global_col = A_rap->diag->indices[j] + A_rap->first_col_diag;
            HYPRE_Real value = A_rap->diag->data[j];
            HYPRE_IJMatrixSetValues(A, 1, &one, &global_row, &global_col, &value);
        }
    }

    // Add off-diagonal block values
    for (int i = 0; i < A_rap->offd->n_cols; i++)
    {
        HYPRE_Int col_start = A_rap->offd->indptr[i];
        HYPRE_Int col_end = A_rap->offd->indptr[i+1];
        HYPRE_Int global_col = A_rap->local_to_global[i];
        for (int j = col_start; j < col_end; j++)
        {
            HYPRE_Int global_row = A_rap->offd->indices[j] + A_rap->first_row;
            HYPRE_Real value = A_rap->offd->data[j];
            HYPRE_IJMatrixSetValues(A, 1, &one, &global_row, &global_col, &value);
        }
    }

    // Assemble hypre matrix
    HYPRE_IJMatrixAssemble(A);

    return A;
}

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
raptor::ParMatrix* convert(hypre_ParCSRMatrix* A_hypre, MPI_Comm comm_mat)
{
    HYPRE_Int num_procs;
    MPI_Comm_size(comm_mat, &num_procs);

    // Get HYPRE diagonal block
    hypre_CSRMatrix* A_hypre_diag = hypre_ParCSRMatrixDiag(A_hypre);
    HYPRE_Real* diag_data = hypre_CSRMatrixData(A_hypre_diag);
    HYPRE_Int* diag_i = hypre_CSRMatrixI(A_hypre_diag);
    HYPRE_Int* diag_j = hypre_CSRMatrixJ(A_hypre_diag);
    HYPRE_Int  diag_nnz = hypre_CSRMatrixNumNonzeros(A_hypre_diag);
    HYPRE_Int diag_rows = hypre_CSRMatrixNumRows(A_hypre_diag);
    HYPRE_Int diag_cols = hypre_CSRMatrixNumCols(A_hypre_diag);

    // Get HYPRE off-diagonal block
    hypre_CSRMatrix* A_hypre_offd = hypre_ParCSRMatrixOffd(A_hypre);
    HYPRE_Real* offd_data = hypre_CSRMatrixData(A_hypre_offd);
    HYPRE_Int* offd_i = hypre_CSRMatrixI(A_hypre_offd);
    HYPRE_Int* offd_j = hypre_CSRMatrixJ(A_hypre_offd);
    HYPRE_Int  offd_nnz = hypre_CSRMatrixNumNonzeros(A_hypre_offd);
    HYPRE_Int offd_rows = hypre_CSRMatrixNumRows(A_hypre_offd);
    HYPRE_Int offd_cols = hypre_CSRMatrixNumCols(A_hypre_offd);

    // Get HYPRE local information
    HYPRE_Int first_row = hypre_ParCSRMatrixFirstRowIndex(A_hypre);
    HYPRE_Int first_col_diag = hypre_ParCSRMatrixFirstColDiag(A_hypre);
    HYPRE_Int* col_map_offd = hypre_ParCSRMatrixColMapOffd(A_hypre);
    HYPRE_Int global_rows = hypre_ParCSRMatrixGlobalNumRows(A_hypre);
    HYPRE_Int global_cols = hypre_ParCSRMatrixGlobalNumCols(A_hypre);
    HYPRE_Int* global_col_starts = hypre_ParCSRMatrixColStarts(A_hypre);

    // Get HYPRE communication information
    hypre_ParCSRCommPkg* comm_pkg = hypre_ParCSRMatrixCommPkg(A_hypre);
    HYPRE_Int num_sends = hypre_ParCSRCommPkgNumSends(comm_pkg);
    HYPRE_Int num_recvs = hypre_ParCSRCommPkgNumRecvs(comm_pkg);
    HYPRE_Int* send_procs = hypre_ParCSRCommPkgSendProcs(comm_pkg);
    HYPRE_Int* recv_procs = hypre_ParCSRCommPkgRecvProcs(comm_pkg);
    HYPRE_Int* send_map_starts = hypre_ParCSRCommPkgSendMapStarts(comm_pkg);
    HYPRE_Int* send_map_elmts = hypre_ParCSRCommPkgSendMapElmts(comm_pkg);
    HYPRE_Int* recv_vec_starts = hypre_ParCSRCommPkgRecvVecStarts(comm_pkg);
  
    // Declare other variables
    HYPRE_Int row_start, row_end;
    HYPRE_Int global_row, global_col;
    HYPRE_Real value;

    // Create empty raptor Matrix
    raptor::ParMatrix* A = new ParMatrix();
    A->global_rows = global_rows;
    A->global_cols = global_cols;
    A->local_rows = diag_rows;
    A->local_cols = diag_cols;
    A->first_row = first_row;
    A->first_col_diag = first_col_diag;
    A->comm_mat = comm_mat;

    // Copy local to global index data
    A->offd_num_cols = offd_cols;
    A->local_to_global.set_data(offd_cols, col_map_offd);

    // Copy diagonal matrix
    A->diag = new Matrix(diag_rows, diag_cols, CSR);
    A->diag->n_rows = diag_rows;
    A->diag->n_cols = diag_cols;
    A->diag->n_outer = diag_rows;
    A->diag->n_inner = diag_cols;
    A->diag->nnz = diag_nnz;
    A->diag->format = CSR;
    if (diag_rows)
    {
        A->diag->indptr.set_data(diag_rows + 1, diag_i);
    }
    if (diag_nnz)
    {
        A->diag->indices.set_data(diag_nnz, diag_j);
        A->diag->data.set_data(diag_nnz, diag_data);
    }

    // Copy off-diagonal matrix
    if (offd_cols)
    {
        A->offd = new Matrix(offd_rows, offd_cols, CSR);
        A->offd->n_rows = offd_rows;
        A->offd->n_cols = offd_cols;
        A->offd->n_outer = offd_rows;
        A->offd->n_inner = offd_cols;
        A->offd->nnz = offd_nnz;
        A->offd->format = CSR;
        if (offd_rows)
        {
            A->offd->indptr.set_data(offd_rows + 1, offd_i);
        }
        if (offd_nnz)
        {
            A->offd->indices.set_data(offd_nnz, offd_j);
            A->offd->data.set_data(offd_nnz, offd_data);
        }

        // Convert offd matrix to CSC
        A->offd->convert(CSC);
    }


    // Create empty communicator
    A->comm = new ParComm();
    A->comm->num_sends = num_sends;
    A->comm->num_recvs = num_recvs;
    if (num_sends)
    {
        A->comm->size_sends = send_map_starts[num_sends];
    }
    else
    {
        A->comm->size_sends = 0;
    }
    A->comm->size_recvs = offd_cols;

    // Add send information to communicator
    if (num_sends)
    {
        A->comm->send_procs.set_data(A->comm->num_sends, send_procs);
        A->comm->send_row_starts.set_data(A->comm->num_sends + 1, send_map_starts);
        A->comm->send_requests = new MPI_Request [num_sends];
        A->comm->send_buffer = new data_t[A->comm->size_sends];
    }
    if (A->comm->size_sends)
    {
        A->comm->send_row_indices.set_data(A->comm->size_sends, send_map_elmts);
    }

    // Add recv information to communicator
    if (num_recvs)
    {
        A->comm->recv_procs.set_data(A->comm->num_recvs, recv_procs);
        A->comm->recv_col_starts.set_data(A->comm->num_recvs, recv_vec_starts);
        A->comm->recv_col_starts.resize(A->comm->num_recvs + 1);
        A->comm->recv_col_starts[A->comm->num_recvs] = offd_cols;
        hypre_ParCSRCommPkgRecvVecStarts(comm_pkg) = A->comm->recv_col_starts.data();

        A->comm->recv_requests = new MPI_Request [A->comm->num_recvs];
        A->comm->recv_buffer = new data_t [A->comm->size_recvs];
    }
    
    //HYPRE matrices always keep diagonal entry first
    //A->diag->move_diag_first();

    return A;
}

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
raptor::Hierarchy* convert(hypre_ParAMGData* amg_data, MPI_Comm comm_mat)
{
    HYPRE_Int num_levels = hypre_ParAMGDataNumLevels(amg_data);
    hypre_ParCSRMatrix** A_array = hypre_ParAMGDataAArray(amg_data);
    hypre_ParCSRMatrix** P_array = hypre_ParAMGDataPArray(amg_data);

    raptor::Hierarchy* ml = new raptor::Hierarchy();

    for (int i = 0; i < num_levels - 1; i++)
    {
        ParMatrix* A = convert(A_array[i], comm_mat);
        ParMatrix* P = convert(P_array[i], comm_mat);

        ml->add_level(A, P);
    }

    ParMatrix* A = convert(A_array[num_levels-1], comm_mat);
    ml->add_level(A);

    return ml;
}

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
void remove_shared_ptrs(hypre_ParCSRMatrix* A_hypre)
{
    hypre_CSRMatrix* A_hypre_diag = hypre_ParCSRMatrixDiag(A_hypre);
    hypre_CSRMatrix* A_hypre_offd = hypre_ParCSRMatrixOffd(A_hypre);
    hypre_ParCSRCommPkg* comm_pkg = hypre_ParCSRMatrixCommPkg(A_hypre);

    if (hypre_CSRMatrixNumRows(A_hypre_diag))
    {
        hypre_CSRMatrixData(A_hypre_diag) = NULL;
        hypre_CSRMatrixI(A_hypre_diag) = NULL;
        hypre_CSRMatrixJ(A_hypre_diag) = NULL;
    }

    if (hypre_CSRMatrixNumCols(A_hypre_offd))
    {
        hypre_CSRMatrixData(A_hypre_offd) = NULL;
        hypre_CSRMatrixI(A_hypre_offd) = NULL;
        hypre_CSRMatrixJ(A_hypre_offd) = NULL;
    }

    if (hypre_CSRMatrixNumCols(A_hypre_offd))
    {
        hypre_ParCSRMatrixColMapOffd(A_hypre) = NULL;
    }

    if (hypre_ParCSRCommPkgNumSends(comm_pkg))
    {
        hypre_ParCSRCommPkgSendProcs(comm_pkg) = NULL;
        hypre_ParCSRCommPkgSendMapStarts(comm_pkg) = NULL;
        hypre_ParCSRCommPkgSendMapElmts(comm_pkg) = NULL;
    }

    if (hypre_ParCSRCommPkgNumRecvs(comm_pkg))
    {
        hypre_ParCSRCommPkgRecvProcs(comm_pkg) = NULL;
        hypre_ParCSRCommPkgRecvVecStarts(comm_pkg) = NULL; 
    }
}

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
void remove_shared_ptrs(hypre_ParAMGData* amg_data)
{
    HYPRE_Int num_levels = hypre_ParAMGDataNumLevels(amg_data);
    hypre_ParCSRMatrix** A_array = hypre_ParAMGDataAArray(amg_data);
    hypre_ParCSRMatrix** P_array = hypre_ParAMGDataPArray(amg_data);

    for (int i = 0; i < num_levels - 1; i++)
    {
        remove_shared_ptrs(A_array[i]);
        remove_shared_ptrs(P_array[i]);
    }
    remove_shared_ptrs(A_array[num_levels-1]);
}

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
                                hypre_ParVector* b,
                                hypre_ParVector* x, 
                                HYPRE_Int coarsen_type,
                                HYPRE_Int interp_type,
                                HYPRE_Int p_max_elmts,
                                HYPRE_Int agg_num_levels,
                                HYPRE_Real strong_threshold)
{
    // Create AMG solver struct
    HYPRE_Solver amg_data;
    HYPRE_BoomerAMGCreate(&amg_data);
      
    // Set Boomer AMG Parameters
    HYPRE_BoomerAMGSetPrintLevel(amg_data, 1);
    HYPRE_BoomerAMGSetCoarsenType(amg_data, coarsen_type);
    HYPRE_BoomerAMGSetInterpType(amg_data, interp_type);
    HYPRE_BoomerAMGSetPMaxElmts(amg_data, p_max_elmts);
    HYPRE_BoomerAMGSetAggNumLevels(amg_data, agg_num_levels);
    HYPRE_BoomerAMGSetStrongThreshold(amg_data, strong_threshold);
    HYPRE_BoomerAMGSetMaxCoarseSize(amg_data, 25);
    HYPRE_BoomerAMGSetMinCoarseSize(amg_data, 10);

    // Setup AMG
    HYPRE_BoomerAMGSetup(amg_data, A, b, x);

    return amg_data;
}

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
                                int coarsen_type,
                                int interp_type,
                                int p_max_elmts,
                                int agg_num_levels,
                                double strong_threshold,
                                MPI_Comm comm_mat)
{
    HYPRE_IJMatrix A = convert(A_rap, comm_mat);
    HYPRE_IJVector x = convert(x_rap, comm_mat);
    HYPRE_IJVector b = convert(b_rap, comm_mat);

    hypre_ParCSRMatrix* parcsr_A;
    HYPRE_IJMatrixGetObject(A, (void**) &parcsr_A);
    hypre_ParVector* par_x;
    HYPRE_IJVectorGetObject(x, (void **) &par_x);
    hypre_ParVector* par_b;
    HYPRE_IJVectorGetObject(b, (void **) &par_b);

    HYPRE_Solver amg_data = hypre_create_hierarchy(parcsr_A, par_x, par_b, 
                            coarsen_type, interp_type, p_max_elmts, agg_num_levels);

    raptor::Hierarchy* ml;
    ml = convert((hypre_ParAMGData*)amg_data, comm_mat);

    //Clean up TODO -- can we set arrays to NULL and still delete these?
    remove_shared_ptrs((hypre_ParAMGData*) amg_data);
    hypre_BoomerAMGDestroy(amg_data); 
    HYPRE_IJMatrixDestroy(A);
    HYPRE_IJVectorDestroy(x);
    HYPRE_IJVectorDestroy(b);

    return ml;
}

