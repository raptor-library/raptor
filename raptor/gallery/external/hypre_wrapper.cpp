// Copyright (c) 2015-2017, RAPtor Developer Team
// License: Simplified BSD, http://opensource.org/licenses/BSD-2-Clause


#include "hypre_wrapper.hpp"

HYPRE_IJVector convert(raptor::ParVector& x_rap, MPI_Comm comm_mat)
{
    HYPRE_IJVector x;

    HYPRE_Int first_local = x_rap.first_local;
    HYPRE_Int local_n = x_rap.local_n;
    HYPRE_Int last_local = first_local + local_n - 1;

    HYPRE_IJVectorCreate(comm_mat, first_local, last_local, &x);
    HYPRE_IJVectorSetObjectType(x, HYPRE_PARCSR);
    HYPRE_IJVectorInitialize(x);

    HYPRE_Int* rows = new HYPRE_Int[local_n];
    for (HYPRE_Int i = 0; i < local_n; i++)
    {
        rows[i] = i+first_local;
    }
    HYPRE_Real* x_data = x_rap.local.data();
    HYPRE_IJVectorSetValues(x, local_n, rows, x_data);

    HYPRE_IJVectorAssemble(x);

    delete[] rows;

    return x;
}

HYPRE_IJMatrix convert(raptor::ParCSRMatrix* A_rap, MPI_Comm comm_mat)
{
    HYPRE_IJMatrix A;

    HYPRE_Int n_rows = 0;
    HYPRE_Int n_cols = 0;
    HYPRE_Int local_row_start = 0;
    HYPRE_Int local_col_start = 0;
    HYPRE_Int rank, num_procs;
    HYPRE_Int one = 1;

    MPI_Comm_rank(comm_mat, &rank);
    MPI_Comm_size(comm_mat, &num_procs);

    aligned_vector<int> row_sizes(num_procs);
    aligned_vector<int> col_sizes(num_procs);
    MPI_Allgather(&(A_rap->local_num_rows), 1, MPI_INT, row_sizes.data(), 1, MPI_INT,
            MPI_COMM_WORLD);
    MPI_Allgather(&(A_rap->on_proc_num_cols), 1, MPI_INT, col_sizes.data(), 1, MPI_INT,
            MPI_COMM_WORLD);
    for (int i = 0; i < rank; i++)
    {
        local_row_start += row_sizes[i];
        local_col_start += col_sizes[i];
    }

    n_rows = A_rap->local_num_rows;
    if (n_rows)
    {
        n_cols = A_rap->on_proc_num_cols;
    }

    /**********************************
     ****** CREATE HYPRE MATRIX
     ************************************/
    HYPRE_IJMatrixCreate(comm_mat, local_row_start, local_row_start + n_rows - 1, local_col_start, local_col_start + n_cols - 1, &A);
    HYPRE_IJMatrixSetObjectType(A, HYPRE_PARCSR);
    HYPRE_IJMatrixInitialize(A);

    HYPRE_Int row_start, row_end;
    HYPRE_Int global_row, global_col;
    HYPRE_Real value;

    for (int i = 0; i < A_rap->on_proc->n_rows; i++)
    {
        row_start = A_rap->on_proc->idx1[i];
        row_end = A_rap->on_proc->idx1[i+1];
        global_row = A_rap->local_row_map[i];
        for (int j = row_start; j < row_end; j++)
        {
            global_col = A_rap->on_proc_column_map[A_rap->on_proc->idx2[j]];
            value = A_rap->on_proc->vals[j];
            HYPRE_IJMatrixSetValues(A, 1, &one, &global_row, &global_col, &value);
        }
    }
    for (int i = 0; i < A_rap->off_proc->n_rows; i++)
    {
        row_start = A_rap->off_proc->idx1[i];
        row_end = A_rap->off_proc->idx1[i+1];
        global_row = A_rap->local_row_map[i];
        for (int j = row_start; j < row_end; j++)
        {
            global_col = A_rap->off_proc_column_map[A_rap->off_proc->idx2[j]];
            value = A_rap->off_proc->vals[j];
            HYPRE_IJMatrixSetValues(A, 1, &one, &global_row, &global_col, &value);
        }
    }
    HYPRE_IJMatrixAssemble(A);

    return A;
}

raptor::ParCSRMatrix* convert(hypre_ParCSRMatrix* A_hypre, MPI_Comm comm_mat)
{
    int num_procs;
    int rank;
    MPI_Comm_size(comm_mat, &num_procs);
    MPI_Comm_rank(comm_mat, &rank);

    hypre_CSRMatrix* A_hypre_diag = hypre_ParCSRMatrixDiag(A_hypre);
    HYPRE_Real* diag_data = hypre_CSRMatrixData(A_hypre_diag);
    HYPRE_Int* diag_i = hypre_CSRMatrixI(A_hypre_diag);
    HYPRE_Int* diag_j = hypre_CSRMatrixJ(A_hypre_diag);
    HYPRE_Int  diag_nnz = hypre_CSRMatrixNumNonzeros(A_hypre_diag);
    HYPRE_Int diag_rows = hypre_CSRMatrixNumRows(A_hypre_diag);
    HYPRE_Int diag_cols = hypre_CSRMatrixNumCols(A_hypre_diag);

    hypre_CSRMatrix* A_hypre_offd = hypre_ParCSRMatrixOffd(A_hypre);
    HYPRE_Int num_cols_offd = hypre_CSRMatrixNumCols(A_hypre_offd);
    HYPRE_Int* offd_i = hypre_CSRMatrixI(A_hypre_offd);
    HYPRE_Int* offd_j = hypre_CSRMatrixJ(A_hypre_offd);
    HYPRE_Real* offd_data;
    if (num_cols_offd)
    {
        offd_data = hypre_CSRMatrixData(A_hypre_offd);
    }

    HYPRE_Int first_local_row = hypre_ParCSRMatrixFirstRowIndex(A_hypre);
    HYPRE_Int first_local_col = hypre_ParCSRMatrixFirstColDiag(A_hypre);
    HYPRE_Int* col_map_offd = hypre_ParCSRMatrixColMapOffd(A_hypre);
    HYPRE_Int global_rows = hypre_ParCSRMatrixGlobalNumRows(A_hypre);
    HYPRE_Int global_cols = hypre_ParCSRMatrixGlobalNumCols(A_hypre);

    ParCSRMatrix* A = new ParCSRMatrix(global_rows, global_cols, 
            diag_rows, diag_cols, 
            first_local_row, first_local_col);

    A->on_proc->idx1[0] = 0;
    for (int i = 0; i < diag_rows; i++)
    {
        int row_start = diag_i[i];
        int row_end = diag_i[i+1];
        for (int j = row_start; j < row_end; j++)
        {
            A->on_proc->idx2.push_back(diag_j[j]);
            A->on_proc->vals.push_back(diag_data[j]);
        }
        A->on_proc->idx1[i+1] = A->on_proc->idx2.size();
    }
    A->on_proc->nnz = A->on_proc->idx2.size();
    
    A->off_proc->idx1[0] = 0;
    for (int i = 0; i < diag_rows; i++)
    {
        if (num_cols_offd)
        {
            int row_start = offd_i[i];
            int row_end = offd_i[i+1];
            for (int j = row_start; j < row_end; j++)
            {
                int global_col = col_map_offd[offd_j[j]];
                A->off_proc->idx2.push_back(global_col);
                A->off_proc->vals.push_back(offd_data[j]);
            }
        }
        A->off_proc->idx1[i+1] = A->off_proc->idx2.size();
    }
    A->off_proc->nnz = A->off_proc->idx2.size();

    A->finalize();

    return A;

}

HYPRE_Solver hypre_create_hierarchy(hypre_ParCSRMatrix* A,
                                hypre_ParVector* b,
                                hypre_ParVector* x, 
                                HYPRE_Int coarsen_type,
                                HYPRE_Int interp_type,
                                HYPRE_Int p_max_elmts,
                                HYPRE_Int agg_num_levels,
                                HYPRE_Real strong_threshold,
                                HYPRE_Int num_functions)
{
    // Create AMG solver struct
    HYPRE_Solver amg_data;
    HYPRE_BoomerAMGCreate(&amg_data);
      
    // Set Boomer AMG Parameters
    HYPRE_BoomerAMGSetMaxRowSum(amg_data, 1);
    HYPRE_BoomerAMGSetCoarsenType(amg_data, coarsen_type);
    HYPRE_BoomerAMGSetInterpType(amg_data, interp_type);
    HYPRE_BoomerAMGSetPMaxElmts(amg_data, p_max_elmts);
    HYPRE_BoomerAMGSetAggNumLevels(amg_data, agg_num_levels);
    HYPRE_BoomerAMGSetStrongThreshold(amg_data, strong_threshold);
    HYPRE_BoomerAMGSetMaxCoarseSize(amg_data, 50);
    HYPRE_BoomerAMGSetRelaxType(amg_data, 3);
    HYPRE_BoomerAMGSetNumFunctions(amg_data, num_functions);

    HYPRE_BoomerAMGSetPrintLevel(amg_data, 3);
    HYPRE_BoomerAMGSetMaxIter(amg_data, 100);

    // Setup AMG
    HYPRE_BoomerAMGSetup(amg_data, A, b, x);

    return amg_data;
}

HYPRE_Solver hypre_create_GMRES(hypre_ParCSRMatrix* A,
                                hypre_ParVector* b,
                                hypre_ParVector* x, HYPRE_Solver* precond_ptr,
                                HYPRE_Int coarsen_type,
                                HYPRE_Int interp_type,
                                HYPRE_Int p_max_elmts,
                                HYPRE_Int agg_num_levels,
                                HYPRE_Real strong_threshold,
                                HYPRE_Int num_functions)
{
    // Create AMG solver struct
    HYPRE_Solver gmres_data;
    HYPRE_ParCSRGMRESCreate(MPI_COMM_WORLD, &gmres_data);
    HYPRE_Solver amg_data;
    HYPRE_BoomerAMGCreate(&amg_data);

    // Set Boomer AMG Parameters
    HYPRE_BoomerAMGSetMaxRowSum(amg_data, 1);
    HYPRE_BoomerAMGSetCoarsenType(amg_data, coarsen_type);
    HYPRE_BoomerAMGSetInterpType(amg_data, interp_type);
    HYPRE_BoomerAMGSetPMaxElmts(amg_data, p_max_elmts);
    HYPRE_BoomerAMGSetAggNumLevels(amg_data, agg_num_levels);
    HYPRE_BoomerAMGSetStrongThreshold(amg_data, strong_threshold);
    HYPRE_BoomerAMGSetMaxCoarseSize(amg_data, 50);
    HYPRE_BoomerAMGSetRelaxType(amg_data, 3);
    HYPRE_BoomerAMGSetNumFunctions(amg_data, num_functions);

    HYPRE_BoomerAMGSetPrintLevel(amg_data, 0);
    HYPRE_BoomerAMGSetMaxIter(amg_data, 1);

    // Set GMRES Parameters
    HYPRE_GMRESSetMaxIter(gmres_data, 100);
    HYPRE_GMRESSetTol(gmres_data, 1e-6);
    HYPRE_GMRESSetPrintLevel(gmres_data, 2);

    // Setup AMG
    HYPRE_ParCSRGMRESSetPrecond(gmres_data, HYPRE_BoomerAMGSolve,
            HYPRE_BoomerAMGSetup, amg_data);

    HYPRE_ParCSRGMRESSetup(gmres_data, A, b, x);

    *precond_ptr = amg_data;
    return gmres_data;
}

HYPRE_Solver hypre_create_BiCGSTAB(hypre_ParCSRMatrix* A,
                                hypre_ParVector* b,
                                hypre_ParVector* x, HYPRE_Solver* precond_ptr,
                                HYPRE_Int coarsen_type,
                                HYPRE_Int interp_type,
                                HYPRE_Int p_max_elmts,
                                HYPRE_Int agg_num_levels,
                                HYPRE_Real strong_threshold,
                                HYPRE_Int num_functions)
{
    // Create AMG solver struct
    HYPRE_Solver bicgstab_data;
    HYPRE_ParCSRBiCGSTABCreate(MPI_COMM_WORLD, &bicgstab_data);
    HYPRE_Solver amg_data;
    HYPRE_BoomerAMGCreate(&amg_data);

    // Set Boomer AMG Parameters
    HYPRE_BoomerAMGSetMaxRowSum(amg_data, 1);
    HYPRE_BoomerAMGSetCoarsenType(amg_data, coarsen_type);
    HYPRE_BoomerAMGSetInterpType(amg_data, interp_type);
    HYPRE_BoomerAMGSetPMaxElmts(amg_data, p_max_elmts);
    HYPRE_BoomerAMGSetAggNumLevels(amg_data, agg_num_levels);
    HYPRE_BoomerAMGSetStrongThreshold(amg_data, strong_threshold);
    HYPRE_BoomerAMGSetMaxCoarseSize(amg_data, 50);
    HYPRE_BoomerAMGSetRelaxType(amg_data, 3);
    HYPRE_BoomerAMGSetNumFunctions(amg_data, num_functions);

    HYPRE_BoomerAMGSetPrintLevel(amg_data, 0);
    HYPRE_BoomerAMGSetMaxIter(amg_data, 1);

    // Set GMRES Parameters
    HYPRE_BiCGSTABSetMaxIter(bicgstab_data, 100);
    HYPRE_BiCGSTABSetTol(bicgstab_data, 1e-6);
    HYPRE_BiCGSTABSetPrintLevel(bicgstab_data, 2);

    // Setup AMG
    HYPRE_ParCSRBiCGSTABSetPrecond(bicgstab_data, HYPRE_BoomerAMGSolve,
            HYPRE_BoomerAMGSetup, amg_data);

    HYPRE_ParCSRBiCGSTABSetup(bicgstab_data, A, b, x);

    *precond_ptr = amg_data;
    return bicgstab_data;
}
