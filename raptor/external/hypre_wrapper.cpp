// Copyright (c) 2015-2017, RAPtor Developer Team
// License: Simplified BSD, http://opensource.org/licenses/BSD-2-Clause

#include <iostream>

#include "hypre_wrapper.hpp"

namespace raptor {

HYPRE_IJVector convert(raptor::ParVector& x_rap, RAPtor_MPI_Comm comm_mat)
{
    int num_procs, rank;
    RAPtor_MPI_Comm_rank(comm_mat, &rank);
    RAPtor_MPI_Comm_size(comm_mat, &num_procs);

    HYPRE_IJVector x;

    HYPRE_Int local_n = x_rap.local_n;
    std::vector<int> first_n(num_procs);
    RAPtor_MPI_Allgather(&local_n, 1, RAPtor_MPI_INT, first_n.data(), 1, RAPtor_MPI_INT, comm_mat);
    HYPRE_Int first_local = 0;
    for (int i = 0; i < rank; i++)
    {
        first_local += first_n[i];
    }
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

HYPRE_IJMatrix convert(raptor::ParCSRMatrix* A_rap, RAPtor_MPI_Comm comm_mat)
{
    HYPRE_IJMatrix A;

    HYPRE_Int n_rows = 0;
    HYPRE_Int n_cols = 0;
    HYPRE_Int local_row_start = 0;
    HYPRE_Int local_col_start = 0;
    HYPRE_Int rank, num_procs;
    HYPRE_Int one = 1;

    RAPtor_MPI_Comm_rank(comm_mat, &rank);
    RAPtor_MPI_Comm_size(comm_mat, &num_procs);

    std::vector<int> row_sizes(num_procs);
    std::vector<int> col_sizes(num_procs);
    RAPtor_MPI_Allgather(&(A_rap->local_num_rows), 1, RAPtor_MPI_INT, row_sizes.data(), 1, RAPtor_MPI_INT,
            RAPtor_MPI_COMM_WORLD);
    RAPtor_MPI_Allgather(&(A_rap->on_proc_num_cols), 1, RAPtor_MPI_INT, col_sizes.data(), 1, RAPtor_MPI_INT,
            RAPtor_MPI_COMM_WORLD);
    for (int i = 0; i < rank; i++)
    {
        local_row_start += row_sizes[i];
        local_col_start += col_sizes[i];
    }

    // Send new global cols (to update off_proc_col_map)
    std::vector<int> new_cols(A_rap->on_proc_num_cols);
    for (int i = 0; i < A_rap->on_proc_num_cols; i++)
    {
        new_cols[i] = i + local_col_start;
    }
    if (!A_rap->comm) A_rap->comm = new ParComm(A_rap->partition,
            A_rap->off_proc_column_map, A_rap->on_proc_column_map);
    std::vector<int>& recvbuf = A_rap->comm->communicate(new_cols);
    std::vector<int> off_proc_cols(A_rap->off_proc_num_cols);
    std::copy(recvbuf.begin(), recvbuf.begin() + A_rap->off_proc_num_cols,
            off_proc_cols.begin());


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
        global_row = i + local_row_start;
        for (int j = row_start; j < row_end; j++)
        {
            global_col = A_rap->on_proc->idx2[j] + local_col_start;
            value = A_rap->on_proc->vals[j];
            HYPRE_IJMatrixSetValues(A, 1, &one, &global_row, &global_col, &value);
        }
    }
    for (int i = 0; i < A_rap->off_proc->n_rows; i++)
    {
        row_start = A_rap->off_proc->idx1[i];
        row_end = A_rap->off_proc->idx1[i+1];
        global_row = i + local_row_start;
        for (int j = row_start; j < row_end; j++)
        {
            global_col = off_proc_cols[A_rap->off_proc->idx2[j]];
            value = A_rap->off_proc->vals[j];
            HYPRE_IJMatrixSetValues(A, 1, &one, &global_row, &global_col, &value);
        }
    }
    HYPRE_IJMatrixAssemble(A);

    return A;
}

raptor::ParCSRMatrix* convert(hypre_ParCSRMatrix* A_hypre, RAPtor_MPI_Comm comm_mat)
{
    int num_procs;
    int rank;
    RAPtor_MPI_Comm_size(comm_mat, &num_procs);
    RAPtor_MPI_Comm_rank(comm_mat, &rank);

    hypre_CSRMatrix* A_hypre_diag = hypre_ParCSRMatrixDiag(A_hypre);
    HYPRE_Real* diag_data = hypre_CSRMatrixData(A_hypre_diag);
    HYPRE_Int* diag_i = hypre_CSRMatrixI(A_hypre_diag);
    HYPRE_Int* diag_j = hypre_CSRMatrixJ(A_hypre_diag);
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
            A->on_proc->idx2.emplace_back(diag_j[j]);
            A->on_proc->vals.emplace_back(diag_data[j]);
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
                A->off_proc->idx2.emplace_back(global_col);
                A->off_proc->vals.emplace_back(offd_data[j]);
            }
        }
        A->off_proc->idx1[i+1] = A->off_proc->idx2.size();
    }
    A->off_proc->nnz = A->off_proc->idx2.size();

    A->finalize();

    return A;

}

raptor::ParBSRMatrix* convert(hypre_ParCSRMatrix* A_hypre,
                              int b_rows, int b_cols,
                              RAPtor_MPI_Comm comm_mat)
{
    int num_procs;
    int rank;
    RAPtor_MPI_Comm_size(comm_mat, &num_procs);
    RAPtor_MPI_Comm_rank(comm_mat, &rank);

    hypre_CSRMatrix* A_hypre_diag = hypre_ParCSRMatrixDiag(A_hypre);
    HYPRE_Real* diag_data = hypre_CSRMatrixData(A_hypre_diag);
    HYPRE_Int* diag_i = hypre_CSRMatrixI(A_hypre_diag);
    HYPRE_Int* diag_j = hypre_CSRMatrixJ(A_hypre_diag);
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

    { // check inputs
	    // TODO: reduce on local check and check row_starts
	    if ((diag_rows % b_rows) ||
	        (diag_cols % b_cols) ||
	        (global_rows % b_rows) ||
	        (global_cols % b_cols)) {
		    if (rank == 0)
			    std::cout << "raptor: Global and local dimensions must evenly block dimensions for ParBSR conversion" << std::endl;
		    return nullptr;
	    }
    }

    auto *A = new ParBSRMatrix(global_rows / b_rows,
                               global_cols / b_cols,
                               diag_rows / b_rows,
                               diag_cols / b_cols,
                               first_local_row / b_rows,
                               first_local_col / b_cols,
                               b_rows, b_cols);

    using block_map = std::map<int, double*>;
    { // process on_proc
	    auto & diag = dynamic_cast<raptor::BSRMatrix&>(*A->on_proc);
	    auto & browptr = diag.idx1;
	    auto & bcolind = diag.idx2;
	    auto & bvalues = diag.block_vals;
	    for (std::size_t block_index = 0; block_index < browptr.size() - 1; ++block_index) {
		    block_map bmap;
		    // local sort likely faster than map insertion
		    for (int i = 0; i < b_rows; ++i) {
			    int row = block_index * b_rows + i;
			    for (int off = diag_i[row]; off < diag_i[row+1]; ++off) {
				    int pos = diag_j[off] / b_cols;
				    auto blkit = bmap.find(pos);
				    double * blk;
				    if (blkit == bmap.end()) {
					    blk = new double[b_rows * b_cols]();
					    bmap[pos] = blk;
				    } else
					    blk = blkit->second;
				    blk[i * b_cols + (diag_j[off] % b_cols)] = diag_data[off];
			    }
		    }
		    // add blocks to A
		    for (auto & blk : bmap) {
			    bcolind.emplace_back(blk.first);
			    bvalues.emplace_back(blk.second);
		    }
		    browptr[block_index + 1] = browptr[block_index] + bmap.size();
	    }
    }

    { // process off_proc
	    auto & offd = dynamic_cast<raptor::BSRMatrix&>(*A->off_proc);
	    auto & browptr = offd.idx1;
	    auto & bcolind = offd.idx2;
	    auto & bvalues = offd.block_vals;
	    for (std::size_t block_index = 0; block_index < browptr.size() - 1; ++block_index) {
		    if (num_cols_offd) {
			    block_map bmap;
			    // local sort likely faster than map insertion
			    for (int i = 0; i < b_rows; ++i) {
				    int row = block_index * b_rows + i;
				    for (int off = offd_i[row]; off < offd_i[row+1]; ++off) {
					    int global_col = col_map_offd[offd_j[off]];
					    int pos = global_col / b_cols;
					    auto blkit = bmap.find(pos);
					    double * blk;
					    if (blkit == bmap.end()) {
						    blk = new double[b_rows * b_cols]();
						    bmap[pos] = blk;
					    } else
						    blk = blkit->second;
					    blk[i * b_cols + (global_col % b_cols)] = offd_data[off];
				    }
			    }
			    // add blocks to A
			    for (auto [colind, val] : bmap) {
				    bcolind.emplace_back(colind);
				    bvalues.emplace_back(val);
			    }
			    browptr[block_index + 1] = browptr[block_index] + bmap.size();
		    }
	    }
    }

    A->off_proc->nnz = A->off_proc->idx2.size();
    A->on_proc->nnz = A->on_proc->idx2.size();

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
                                HYPRE_Real filter_threshold,
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
    HYPRE_BoomerAMGSetRelaxOrder(amg_data, 0);
//    HYPRE_BoomerAMGSetRelaxWt(amg_data, 3.0/4); // set omega for SOR
    HYPRE_BoomerAMGSetNumFunctions(amg_data, num_functions);
    //HYPRE_BoomerAMGSetRAP2(amg_data, 1);

    HYPRE_BoomerAMGSetTruncFactor(amg_data, filter_threshold);

    HYPRE_BoomerAMGSetPrintLevel(amg_data, 0);
    HYPRE_BoomerAMGSetMaxIter(amg_data, 1000);

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
    HYPRE_ParCSRGMRESCreate(RAPtor_MPI_COMM_WORLD, &gmres_data);
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

    HYPRE_BoomerAMGSetRelaxOrder(amg_data, 1);
    HYPRE_BoomerAMGSetTruncFactor(amg_data, 0.3);

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
    HYPRE_ParCSRBiCGSTABCreate(RAPtor_MPI_COMM_WORLD, &bicgstab_data);
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

}
