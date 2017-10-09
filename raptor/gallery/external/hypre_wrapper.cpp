#include "hypre_wrapper.hpp"

HYPRE_IJVector convert(raptor::ParVector* x_rap, MPI_Comm comm_mat)
{
    HYPRE_IJVector x;

    HYPRE_Int first_local = x_rap->first_local;
    HYPRE_Int local_n = x_rap->local_n;
    HYPRE_Int last_local = first_local + local_n - 1;

    HYPRE_IJVectorCreate(comm_mat, first_local, last_local, &x);
    HYPRE_IJVectorSetObjectType(x, HYPRE_PARCSR);
    HYPRE_IJVectorInitialize(x);

    HYPRE_Int* rows = new HYPRE_Int[local_n];
    for (HYPRE_Int i = 0; i < local_n; i++)
    {
        rows[i] = i+first_local;
    }
    HYPRE_Real* x_data = x_rap->local.data();
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

    std::vector<int> row_sizes(num_procs);
    std::vector<int> col_sizes(num_procs);
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

// TODO -- Create A Shallow Copy for Conversion
/*ParMatrix* convert_shallow(hypre_ParCSRMatrix* A_hypre, MPI_Comm comm_mat)
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
//    HYPRE_Real* offd_data = hypre_CSRMatrixData(A_hypre_offd);
    HYPRE_Int* offd_i = hypre_CSRMatrixI(A_hypre_offd);
    HYPRE_Int* offd_j = hypre_CSRMatrixJ(A_hypre_offd);
    HYPRE_Int  offd_nnz = hypre_CSRMatrixNumNonzeros(A_hypre_offd);
    HYPRE_Int offd_rows = hypre_CSRMatrixNumRows(A_hypre_offd);
    HYPRE_Int offd_cols = hypre_CSRMatrixNumCols(A_hypre_offd);
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

    hypre_ParCSRCommPkg* comm_pkg = hypre_ParCSRMatrixCommPkg(A_hypre);
    if (!comm_pkg)
    {
        hypre_MatvecCommPkgCreate(A_hypre);
        comm_pkg = hypre_ParCSRMatrixCommPkg(A_hypre);
    }
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
    raptor::ParMatrix* A = new ParMatrix();
    A->global_rows = global_rows;
    A->global_cols = global_cols;
    A->global_col_starts.resize(num_procs);
    MPI_Allgather(&first_local_col, 1, MPI_INT, A->global_col_starts.data(), 1, MPI_INT, MPI_COMM_WORLD);
    A->local_num_rows = diag_rows;
    A->local_num_cols = diag_cols;
    A->first_local_row = first_local_row;
    A->first_local_col = first_local_col;
    A->comm_mat = comm_mat;

    // Copy local to global index data
    A->offd_num_cols = offd_cols;
    A->offd_column_map.set_data(offd_cols, col_map_offd);

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

    return A;
}

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
}*/

/*raptor::Hierarchy* convert(hypre_ParAMGData* amg_data, MPI_Comm comm_mat)
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
}*/

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
    HYPRE_BoomerAMGSetCoarsenType(amg_data, coarsen_type);
    HYPRE_BoomerAMGSetInterpType(amg_data, interp_type);
    HYPRE_BoomerAMGSetPMaxElmts(amg_data, p_max_elmts);
    HYPRE_BoomerAMGSetAggNumLevels(amg_data, agg_num_levels);
    HYPRE_BoomerAMGSetStrongThreshold(amg_data, strong_threshold);
    HYPRE_BoomerAMGSetMaxCoarseSize(amg_data, 50);
    HYPRE_BoomerAMGSetRelaxType(amg_data, 3);

    HYPRE_BoomerAMGSetPrintLevel(amg_data, 3);
    HYPRE_BoomerAMGSetMaxIter(amg_data, 100);

    // Setup AMG
    HYPRE_BoomerAMGSetup(amg_data, A, b, x);

    return amg_data;
}

/*raptor::Hierarchy* create_wrapped_hierarchy(raptor::ParMatrix* A_rap,
                                raptor::ParVector* x_rap,
                                raptor::ParVector* b_rap,
                                int coarsen_type,
                                int interp_type,
                                int p_max_elmts,
                                int agg_num_levels,
                                double strong_threshold,
                                MPI_Comm comm_mat)
{
    raptor::Hierarchy* ml = NULL;
    HYPRE_IJMatrix A = convert(A_rap, comm_mat);
    HYPRE_IJVector x = convert(x_rap, comm_mat);
    HYPRE_IJVector b = convert(b_rap, comm_mat);

    hypre_ParCSRMatrix* parcsr_A;
    HYPRE_IJMatrixGetObject(A, (void**) &parcsr_A);
    hypre_ParVector* par_x;
    HYPRE_IJVectorGetObject(x, (void **) &par_x);
    hypre_ParVector* par_b;
    HYPRE_IJVectorGetObject(b, (void **) &par_b);

    hypre_CSRMatrix* A_offd = hypre_ParCSRMatrixOffd(parcsr_A);
    HYPRE_Int num_cols_offd = hypre_CSRMatrixNumCols(A_offd);

    HYPRE_Solver amg_data = hypre_create_hierarchy(parcsr_A, par_x, par_b, 
                            coarsen_type, interp_type, p_max_elmts, agg_num_levels);

    ml = convert((hypre_ParAMGData*)amg_data, comm_mat);

    //Clean up TODO -- can we set arrays to NULL and still delete these?
    //remove_shared_ptrs((hypre_ParAMGData*) amg_data);
    hypre_BoomerAMGDestroy(amg_data); 
    HYPRE_IJMatrixDestroy(A);
    HYPRE_IJVectorDestroy(x);
    HYPRE_IJVectorDestroy(b);

    return ml;
}*/

