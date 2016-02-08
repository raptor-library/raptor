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

// TODO - Create a shallow copy for conversion
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

// TODO -- Create A Shallow Copy for Conversion
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
    HYPRE_Int num_cols_offd = hypre_CSRMatrixNumCols(A_hypre_offd);
    HYPRE_Int global_rows = hypre_ParCSRMatrixGlobalNumRows(A_hypre);
    HYPRE_Int global_cols = hypre_ParCSRMatrixGlobalNumCols(A_hypre);
    HYPRE_Int* global_col_starts = hypre_ParCSRMatrixColStarts(A_hypre);

    hypre_ParCSRCommPkg* comm_pkg = hypre_ParCSRMatrixCommPkg(A_hypre);
    HYPRE_Int num_sends = hypre_ParCSRCommPkgNumSends(comm_pkg);
    HYPRE_Int num_recvs = hypre_ParCSRCommPkgNumRecvs(comm_pkg);
    HYPRE_Int* send_procs = hypre_ParCSRCommPkgSendProcs(comm_pkg);
    HYPRE_Int* recv_procs = hypre_ParCSRCommPkgRecvProcs(comm_pkg);
    HYPRE_Int* send_map_starts = hypre_ParCSRCommPkgSendMapStarts(comm_pkg);
    HYPRE_Int* send_map_elmts = hypre_ParCSRCommPkgSendMapElmts(comm_pkg);
    HYPRE_Int* recv_vec_starts = hypre_ParCSRCommPkgRecvVecStarts(comm_pkg);
  
    HYPRE_Int row_start, row_end;
    HYPRE_Int global_row, global_col;
    HYPRE_Real value;

    // Create empty raptor Matrix
    raptor::ParMatrix* A = new ParMatrix(global_rows, global_cols, local_rows, local_cols, first_row, first_col_diag, global_col_starts);

    // Copy diagonal matrix
    raptor::index_t* rap_diag_indptr = A->diag->indptr.data();
    raptor::index_t* rap_diag_indices = A->diag->indices.data();
    raptor::data_t* rap_diag_data = A->diag->data.data();
    rap_diag_indptr = diag_i;
    rap_diag_indices = diag_j;
    rap_diag_data = diag_data;
    A->diag->n_rows = local_rows;
    A->diag->n_cols = local_cols;
    A->diag->n_outer = local_rows;
    A->diag->n_inner = local_cols;
    A->diag->format = CSR;

    // Copy off-diagonal matrix
    raptor::index_t* rap_offd_indptr = A->offd->indptr.data();
    raptor::index_t* rap_offd_indices = A->offd->indices.data();
    raptor::data_t* rap_offd_data = A->offd->data.data();
    rap_offd_indptr = offd_i;
    rap_offd_indices = offd_j;
    rap_offd_data = offd_data;
    A->offd->n_rows = local_rows;
    A->offd->n_cols = num_cols_offd;
    A->offd->n_outer = local_rows;
    A->offd->n_inner = local_cols;
    A->offd->format = CSR;

    // Convert offd matrix to CSC
    A->offd->convert(CSC);

    // Copy local to global index data
    A->offd_num_cols = num_cols_offd;
    raptor::index_t* rap_local_to_global = A->local_to_global.data();
    rap_local_to_global = col_map_offd;

    // Create empty communicator
    A->comm = new ParComm();

    index_t* rap_send_procs = A->comm->send_procs.data();
    index_t* rap_send_row_starts = A->comm->send_row_starts.data();
    index_t* rap_send_row_indices = A->comm->send_row_indices.data();
    rap_send_procs = send_procs;
    rap_send_row_starts = send_map_starts;
    rap_send_row_indices = send_map_elmts;

    index_t* rap_recv_procs = A->comm->recv_procs.data();
    index_t* rap_recv_col_starts = A->comm->recv_col_starts.data();
    rap_recv_procs = recv_procs;
    rap_recv_col_starts = recv_vec_starts;

    A->comm->send_procs.resize(num_sends);
    A->comm->recv_procs.resize(num_recvs);
 
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

    //Clean up TODO -- can we set arrays to NULL and still delete these?
    //hypre_BoomerAMGDestroy(amg_data); 
    //HYPRE_IJMatrixDestroy(A);
    //HYPRE_IJVectorDestroy(x);
    //HYPRE_IJVectorDestroy(b);

    return ml;
}

