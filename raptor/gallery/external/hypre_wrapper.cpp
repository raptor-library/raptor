#include "hypre_wrapper.hpp"

HYPRE_IJVector convert(raptor::ParVector* x_rap)
{
    HYPRE_IJVector x;

    HYPRE_Int first_local = x_rap->first_local;
    HYPRE_Int local_n = x_rap->local_n;

    HYPRE_IJVectorCreate(MPI_COMM_WORLD, first_local, first_local + local_n, &x);
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

HYPRE_IJMatrix convert(raptor::ParMatrix* A_rap)
{
    HYPRE_IJMatrix A;

    HYPRE_Int n_rows, n_cols;
    HYPRE_Int local_row_start, local_col_start;
    HYPRE_Int rank, num_procs;
    HYPRE_Int one = 1;

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    n_rows = A_rap->local_rows;
    n_cols = A_rap->local_cols;
    local_row_start = A_rap->first_row;
    local_col_start = A_rap->first_col_diag;

    /**********************************
     ****** CREATE HYPRE MATRIX
     ************************************/
    HYPRE_IJMatrixCreate(MPI_COMM_WORLD, local_row_start, local_row_start + n_rows - 1, local_col_start, local_col_start + n_cols - 1, &A);
    HYPRE_IJMatrixSetObjectType(A, HYPRE_PARCSR);
    HYPRE_IJMatrixInitialize(A);

    for (int i = 0; i < A_rap->diag->n_rows; i++)
    {
        HYPRE_Int row_start = A_rap->diag->indptr[i];
        HYPRE_Int row_end = A_rap->diag->indptr[i+1];
        HYPRE_Int global_row = i + A_rap->first_row;
        for (int j = row_start; j < row_end; j++)
        {
            HYPRE_Int global_col = A_rap->diag->indices[j] + A_rap->first_col_diag;
            HYPRE_Real value = A_rap->diag->data[j];
            HYPRE_IJMatrixAddToValues(A, 1, &one, &global_row, &global_col, &value);
        }
    }

    for (int i = 0; i < A_rap->offd->n_cols; i++)
    {
        HYPRE_Int col_start = A_rap->offd->indptr[i];
        HYPRE_Int col_end = A_rap->offd->indptr[i+1];
        HYPRE_Int global_col = A_rap->local_to_global[i];
        for (int j = col_start; j < col_end; j++)
        {
            HYPRE_Int global_row = A_rap->offd->indices[j] + A_rap->first_row;
            HYPRE_Real value = A_rap->offd->data[j];
            HYPRE_IJMatrixAddToValues(A, 1, &one, &global_row, &global_col, &value);
        }
    }
    HYPRE_IJMatrixAssemble(A);

    return A;
}

raptor::ParMatrix* convert(hypre_ParCSRMatrix* A_hypre)
{
    hypre_CSRMatrix* A_hypre_diag = hypre_ParCSRMatrixDiag(A_hypre);
    HYPRE_Real* diag_data = hypre_CSRMatrixData(A_hypre_diag);
    HYPRE_Int* diag_i = hypre_CSRMatrixI(A_hypre_diag);
    HYPRE_Int* diag_j = hypre_CSRMatrixJ(A_hypre_diag);

    hypre_CSRMatrix* A_hypre_offd = hypre_ParCSRMatrixOffd(A_hypre);
    HYPRE_Real* offd_data = hypre_CSRMatrixData(A_hypre_offd);
    HYPRE_Int* offd_i = hypre_CSRMatrixI(A_hypre_offd);
    HYPRE_Int* offd_j = hypre_CSRMatrixJ(A_hypre_offd);

    HYPRE_Int local_rows = hypre_CSRMatrixNumRows(A_hypre_diag);
    HYPRE_Int local_cols = hypre_CSRMatrixNumCols(A_hypre_diag);
    HYPRE_Int first_row = hypre_ParCSRMatrixFirstRowIndex(A_hypre);
    HYPRE_Int first_col_diag = hypre_ParCSRMatrixFirstColDiag(A_hypre);
    HYPRE_Int* col_map_offd = hypre_ParCSRMatrixColMapOffd(A_hypre);
    HYPRE_Int global_rows = hypre_ParCSRMatrixGlobalNumRows(A_hypre);
    HYPRE_Int global_cols = hypre_ParCSRMatrixGlobalNumCols(A_hypre);

    HYPRE_Int row_start, row_end;
    HYPRE_Int global_row, global_col;
    HYPRE_Real value;

    raptor::ParMatrix* A = new ParMatrix(global_rows, global_cols, local_rows, local_cols, first_row, first_col_diag);

    for (int i = 0; i < local_rows; i++)
    {
        row_start = diag_i[i];
        row_end = diag_i[i+1];

        for (int j = row_start; j < row_end; j++)
        {
            global_col = diag_j[j] + first_col_diag;
            value = diag_data[j];
            if (fabs(value) < zero_tol) continue;
            A->add_value(i, global_col, value);
        }

        row_start = offd_i[i];
        row_end = offd_i[i+1];

        for (int j = row_start; j < row_end; j++)
        {
            global_col = col_map_offd[offd_j[j]];
            value = offd_data[j];
            if (fabs(value) < zero_tol) continue;
            A->add_value(i, global_col, value);
        }
    }

    A->finalize(0);

    return A;
}

raptor::Hierarchy* convert(hypre_ParAMGData* amg_data)
{
    HYPRE_Int num_levels = hypre_ParAMGDataNumLevels(amg_data);
    hypre_ParCSRMatrix** A_array = hypre_ParAMGDataAArray(amg_data);
    hypre_ParCSRMatrix** P_array = hypre_ParAMGDataPArray(amg_data);

    raptor::Hierarchy* ml = new raptor::Hierarchy();

    for (int i = 0; i < num_levels - 1; i++)
    {
        ParMatrix* A = convert(A_array[i]);
        ParMatrix* P = convert(P_array[i]);

        ml->add_level(A, P);
    }

    ParMatrix* A = convert(A_array[num_levels-1]);
    ml->add_level(A);

    return ml;
}

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

raptor::Hierarchy* create_wrapped_hierarchy(raptor::ParMatrix* A_rap,
                                raptor::ParVector* x_rap,
                                raptor::ParVector* b_rap,
                                int coarsen_type,
                                int interp_type,
                                int p_max_elmts,
                                int agg_num_levels,
                                double strong_threshold)
{
    HYPRE_IJMatrix A = convert(A_rap);
    HYPRE_IJVector x = convert(x_rap);
    HYPRE_IJVector b = convert(b_rap);

    hypre_ParCSRMatrix* parcsr_A;
    HYPRE_IJMatrixGetObject(A, (void**) &parcsr_A);
    hypre_ParVector* par_x;
    HYPRE_IJVectorGetObject(x, (void **) &par_x);
    hypre_ParVector* par_b;
    HYPRE_IJVectorGetObject(b, (void **) &par_b);

    //A_rap = convert(parcsr_A);
    //x_rap = convert(par_x);
    //b_rap = convert(par_b);

    HYPRE_Solver amg_data = hypre_create_hierarchy(parcsr_A, par_x, par_b, coarsen_type, interp_type, p_max_elmts, agg_num_levels);

    raptor::Hierarchy* ml = convert((hypre_ParAMGData*)amg_data);

    //Clean up
    hypre_BoomerAMGDestroy(amg_data); 
    HYPRE_IJMatrixDestroy(A);
    HYPRE_IJVectorDestroy(x);
    HYPRE_IJVectorDestroy(b);

    return ml;
}

