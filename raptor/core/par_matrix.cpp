// Copyright (c) 2015-2017, RAPtor Developer Team
// License: Simplified BSD, http://opensource.org/licenses/BSD-2-Clause
#include "par_matrix.hpp"

using namespace raptor;

// Declare private methods
void bsr_to_csr_copy_helper(ParBSRMatrix* A, ParCSRMatrix* B);

/**************************************************************
*****   ParMatrix Add Value
**************************************************************
***** Adds a value to the local portion of the parallel matrix,
***** determining whether it should be added to diagonal or
***** off-diagonal block.
*****
***** Parameters
***** -------------
***** row : index_t
*****    Local row of value
***** global_col : index_t
*****    Global column of value
***** value : data_t
*****    Value to be added to parallel matrix
**************************************************************/
void ParMatrix::add_value(
        int row,
        index_t global_col,
        data_t value)
{
    if (global_col >= partition->first_local_col
            && global_col <= partition->last_local_col)
    {
        on_proc->add_value(row, global_col - partition->first_local_col, value);
    }
    else
    {
        off_proc->add_value(row, global_col, value);
    }
}

/**************************************************************
*****   ParMatrix Add Global Value
**************************************************************
***** Adds a value to the local portion of the parallel matrix,
***** determining whether it should be added to diagonal or
***** off-diagonal block.
*****
***** Parameters
***** -------------
***** global_row : index_t
*****    Global row of value
***** global_col : index_t
*****    Global column of value
***** value : data_t
*****    Value to be added to parallel matrix
**************************************************************/
void ParMatrix::add_global_value(
        index_t global_row,
        index_t global_col,
        data_t value)
{
    add_value(global_row - partition->first_local_row, global_col, value);
}

/**************************************************************
*****   ParMatrix Finalize
**************************************************************
***** Finalizes the diagonal and off-diagonal matrices.  Sorts
***** the local_to_global indices, and creates the parallel
***** communicator
*****
***** Parameters
***** -------------
***** create_comm : bool (optional)
*****    Boolean for whether parallel communicator should be
*****    created (default is true)
**************************************************************/
void ParMatrix::condense_off_proc()
{
    if (off_proc->nnz == 0)
    {
        return;
    }

    int prev_col = -1;

    std::map<int, int> orig_to_new;

    std::copy(off_proc->idx2.begin(), off_proc->idx2.end(),
            std::back_inserter(off_proc_column_map));
    std::sort(off_proc_column_map.begin(), off_proc_column_map.end());

    off_proc_num_cols = 0;
    for (std::vector<int>::iterator it = off_proc_column_map.begin();
            it != off_proc_column_map.end(); ++it)
    {
        if (*it != prev_col)
        {
            orig_to_new[*it] = off_proc_num_cols;
            off_proc_column_map[off_proc_num_cols++] = *it;
            prev_col = *it;
        }
    }
    off_proc_column_map.resize(off_proc_num_cols);

    for (std::vector<int>::iterator it = off_proc->idx2.begin();
            it != off_proc->idx2.end(); ++it)
    {
        *it = orig_to_new[*it];
    }
}

void ParMatrix::finalize(bool create_comm)
{
    on_proc->sort();
    on_proc->remove_duplicates();
    off_proc->sort();
    off_proc->remove_duplicates();

    int rank, num_procs;
    RAPtor_MPI_Comm_size(RAPtor_MPI_COMM_WORLD, &num_procs);
    RAPtor_MPI_Comm_rank(RAPtor_MPI_COMM_WORLD, &rank);

    // Assume nonzeros in each on_proc column
    if (on_proc_num_cols > (int)on_proc_column_map.size())
    {
        on_proc_column_map.resize(on_proc_num_cols);
        for (int i = 0; i < on_proc_num_cols; i++)
        {
            on_proc_column_map[i] = i + partition->first_local_col;
        }
    }

    if (local_num_rows > (int)local_row_map.size())
    {
        local_row_map.resize(local_num_rows);
        for (int i = 0; i < local_num_rows; i++)
        {
            local_row_map[i] = i + partition->first_local_row;
        }
    }

    // Condense columns in off_proc, storing global
    // columns as 0-num_cols, and store mapping
    if (off_proc->nnz)
    {
        condense_off_proc();
    }
    else
    {
        off_proc_num_cols = 0;
    }
    off_proc->resize(local_num_rows, off_proc_num_cols);
    local_nnz = on_proc->nnz + off_proc->nnz;

    if (create_comm){
        comm = new ParComm(partition, off_proc_column_map);
    }
    else
        comm = new ParComm(partition);
}

std::vector<int> ParMatrix::map_partition_to_local() const
{
	std::vector<int> on_proc_partition_to_col(partition->local_num_cols+1, -1);
    for (int i = 0; i < on_proc_num_cols; i++)
    {
        on_proc_partition_to_col[on_proc_column_map[i] - partition->first_local_col] = i;
    }

    return on_proc_partition_to_col;
}


/**************************************************************
*****  ParBSRMatrix to ParCSRMatrix Convert
**************************************************************/
void bsr_to_csr_copy_helper(ParBSRMatrix* A, ParCSRMatrix* B)
{
    if (B->on_proc)
    {
        delete B->on_proc;
    }
    if (B->off_proc)
    {
        delete B->off_proc;
    }

    // Convert on and off proc to CSR
    B->on_proc = A->on_proc->to_CSR();
    B->off_proc = A->off_proc->to_CSR();

    B->local_nnz = B->on_proc->nnz + B->off_proc->nnz;
    B->global_num_rows = A->global_num_rows * A->on_proc->b_rows;
    B->global_num_cols = A->global_num_cols * A->on_proc->b_cols;

    B->on_proc_num_cols = B->on_proc->n_cols;
    B->off_proc_num_cols = B->off_proc->n_cols;

    // Updated partition
    B->partition = new Partition(B->global_num_rows, B->global_num_cols,
                        B->on_proc->n_rows, B->on_proc->n_cols,
                        A->partition->first_local_row * A->on_proc->b_rows,
                        A->partition->first_local_col * A->on_proc->b_cols);
    B->local_num_rows = B->partition->local_num_rows;

    // Updated column and row maps -
    B->finalize(false);

    // Determine which cols of blocks are non-zero
    bool* off_proc_nz_cols = new bool[A->off_proc_num_cols * A->off_proc->b_cols];
    A->off_proc->block_removal_col_check(off_proc_nz_cols);

    // Update off_proc_column_map
    int first_col;
    int off_proc_map_indx = 0;
    for (int i = 0; i < A->off_proc_num_cols; i++)
    {
        first_col = A->off_proc_column_map[i] * A->off_proc->b_cols;
        for (int j = 0; j < A->off_proc->b_cols; j++)
        {
            if (off_proc_nz_cols[i*A->off_proc->b_cols + j])
            {
                B->off_proc_column_map[off_proc_map_indx] = first_col + j;
                off_proc_map_indx++;
            }
        }
    }

    // Updated how communicators are created
    if (A->comm)
    {
        B->comm = new ParComm(B->partition, B->off_proc_column_map, B->on_proc_column_map);
    }
    else
    {
        B->comm = NULL;
    }

    if (A->tap_comm)
    {
        B->tap_comm = new TAPComm(B->partition, B->off_proc_column_map, B->on_proc_column_map);
    }
    else
    {
        B->tap_comm = NULL;
    }

    if (A->tap_mat_comm)
    {
        B->tap_mat_comm = new TAPComm(B->partition, B->off_proc_column_map, B->on_proc_column_map);
    }
    else
    {
        B->tap_mat_comm = NULL;
    }

    delete[] off_proc_nz_cols;
}



/**************************************************************
*****  ParMatrix Convert
**************************************************************
***** Convert from one type of parmatrix to another
***** No copies if parmatrix type remains the same
***** If blocked parmatrix, converts to block matrix
**************************************************************/
ParCOOMatrix* ParCOOMatrix::to_ParCOO()
{
    return this;
}
ParCOOMatrix* ParCOOMatrix::to_ParBCOO()
{
    return this->to_ParCOO();
}
ParCOOMatrix* ParBCOOMatrix::to_ParCOO()
{
    return this->to_ParBCOO();
}
ParCOOMatrix* ParBCOOMatrix::to_ParBCOO()
{
    return this;
}
ParCSRMatrix* ParCOOMatrix::to_ParCSR()
{
    ParCSRMatrix* A = new ParCSRMatrix();
    A->copy_helper(this);
    return A;
}
ParCSRMatrix* ParCOOMatrix::to_ParBSR()
{
    return this->to_ParCSR();
}
ParCSRMatrix* ParBCOOMatrix::to_ParCSR()
{
    return this->to_ParBSR();
}
ParCSRMatrix* ParBCOOMatrix::to_ParBSR()
{
    ParBSRMatrix* A = new ParBSRMatrix();
    A->copy_helper(this);
    return A;
}
ParCSCMatrix* ParCOOMatrix::to_ParCSC()
{
    ParCSCMatrix* A = new ParCSCMatrix();
    A->copy_helper(this);
    return A;
}
ParCSCMatrix* ParCOOMatrix::to_ParBSC()
{
    return this->to_ParCSC();
}
ParCSCMatrix* ParBCOOMatrix::to_ParCSC()
{
    return this->to_ParBSC();
}
ParCSCMatrix* ParBCOOMatrix::to_ParBSC()
{
    ParBSCMatrix* A = new ParBSCMatrix();
    A->copy_helper(this);
    return A;
}

ParCOOMatrix* ParCSRMatrix::to_ParCOO()
{
    ParCOOMatrix* A = new ParCOOMatrix();
    A->copy_helper(this);
    return A;
}
ParCOOMatrix* ParCSRMatrix::to_ParBCOO()
{
    return this->to_ParCOO();
}
ParCOOMatrix* ParBSRMatrix::to_ParCOO()
{
    return this->to_ParBCOO();
}
ParCOOMatrix* ParBSRMatrix::to_ParBCOO()
{
    ParBCOOMatrix* A = new ParBCOOMatrix();
    A->copy_helper(this);
    return A;
}
ParCSRMatrix* ParCSRMatrix::to_ParCSR()
{
    return this;
}
ParCSRMatrix* ParCSRMatrix::to_ParBSR()
{
    return this->to_ParCSR();
}
ParCSRMatrix* ParBSRMatrix::to_ParCSR()
{
    ParCSRMatrix* A = new ParCSRMatrix();
    bsr_to_csr_copy_helper(this, A);
    return A;
}
ParCSRMatrix* ParBSRMatrix::to_ParBSR()
{
    return this;
}
ParCSCMatrix* ParCSRMatrix::to_ParCSC()
{
    ParCSCMatrix* A = new ParCSCMatrix();
    A->copy_helper(this);
    return A;
}
ParCSCMatrix* ParCSRMatrix::to_ParBSC()
{
    return this->to_ParCSC();
}
ParCSCMatrix* ParBSRMatrix::to_ParCSC()
{
    return this->to_ParBSC();
}
ParCSCMatrix* ParBSRMatrix::to_ParBSC()
{
    ParBSCMatrix* A = new ParBSCMatrix();
    A->copy_helper(this);
    return A;
}

ParCOOMatrix* ParCSCMatrix::to_ParCOO()
{
    ParCOOMatrix* A = new ParCOOMatrix();
    A->copy_helper(this);
    return A;
}
ParCOOMatrix* ParCSCMatrix::to_ParBCOO()
{
    return this->to_ParCOO();
}
ParCOOMatrix* ParBSCMatrix::to_ParCOO()
{
    return this->to_ParBCOO();
}
ParCOOMatrix* ParBSCMatrix::to_ParBCOO()
{
    ParBCOOMatrix* A = new ParBCOOMatrix();
    A->copy_helper(this);
    return A;
}
ParCSRMatrix* ParCSCMatrix::to_ParCSR()
{
    ParCSRMatrix* A = new ParCSRMatrix();
    A->copy_helper(this);
    return A;
}
ParCSRMatrix* ParCSCMatrix::to_ParBSR()
{
    return this->to_ParCSR();
}
ParCSRMatrix* ParBSCMatrix::to_ParCSR()
{
    return this->to_ParBSR();
}
ParCSRMatrix* ParBSCMatrix::to_ParBSR()
{
    ParBSRMatrix* A = new ParBSRMatrix();
    A->copy_helper(this);
    return A;
}
ParCSCMatrix* ParCSCMatrix::to_ParCSC()
{
    return this;
}
ParCSCMatrix* ParCSCMatrix::to_ParBSC()
{
    return this->to_ParCSC();
}
ParCSCMatrix* ParBSCMatrix::to_ParCSC()
{
    return this->to_ParBSC();
}
ParCSCMatrix* ParBSCMatrix::to_ParBSC()
{
    return this;
}


void ParCSRMatrix::copy_structure(ParBSRMatrix* A)
{
    on_proc->idx1.clear();
    on_proc->idx2.clear();
    off_proc->idx1.clear();
    off_proc->idx2.clear();

    std::copy(A->on_proc->idx1.begin(), A->on_proc->idx1.end(),
		std::back_inserter(on_proc->idx1));
    std::copy(A->on_proc->idx2.begin(), A->on_proc->idx2.end(),
		std::back_inserter(on_proc->idx2));

    std::copy(A->off_proc->idx1.begin(), A->off_proc->idx1.end(),
		std::back_inserter(off_proc->idx1));
    std::copy(A->off_proc->idx2.begin(), A->off_proc->idx2.end(),
		std::back_inserter(off_proc->idx2));

    on_proc->n_rows = A->on_proc->n_rows;
    on_proc->n_cols = A->on_proc->n_cols;
    on_proc->nnz = A->on_proc->nnz;

    off_proc->n_rows = A->off_proc->n_rows;
    off_proc->n_cols = A->off_proc->n_cols;
    off_proc->nnz = A->off_proc->nnz;

    ParMatrix::copy_helper(A);
}


void ParMatrix::default_copy_helper(ParMatrix* A)
{
    partition = A->partition;
    partition->num_shared++;

    local_nnz = A->local_nnz;
    local_num_rows = A->local_num_rows;
    global_num_rows = A->global_num_rows;
    global_num_cols = A->global_num_cols;

    std::copy(A->off_proc_column_map.begin(), A->off_proc_column_map.end(),
            std::back_inserter(off_proc_column_map));
    std::copy(A->on_proc_column_map.begin(), A->on_proc_column_map.end(),
            std::back_inserter(on_proc_column_map));
    std::copy(A->local_row_map.begin(), A->local_row_map.end(),
            std::back_inserter(local_row_map));

    off_proc_num_cols = off_proc_column_map.size();
    on_proc_num_cols = on_proc_column_map.size();

    if (A->comm)
    {
        comm = A->comm;
        comm->num_shared++;
    }
    else
    {
        comm = NULL;
    }

    if (A->tap_comm)
    {
        tap_comm = A->tap_comm;
        tap_comm->num_shared++;
    }
    else
    {
        tap_comm = NULL;
    }

    if (A->tap_mat_comm)
    {
        tap_mat_comm = A->tap_mat_comm;
        tap_mat_comm->num_shared++;
    }
    else
    {
        tap_mat_comm = NULL;
    }
}

void ParMatrix::copy_helper(ParCOOMatrix* A)
{
    default_copy_helper(A);
}
void ParMatrix::copy_helper(ParCSRMatrix* A)
{
    default_copy_helper(A);
}
void ParMatrix::copy_helper(ParCSCMatrix* A)
{
    default_copy_helper(A);
}


void ParCOOMatrix::copy_helper(ParCOOMatrix* A)
{
    if (on_proc)
    {
        delete on_proc;
    }
    if (off_proc)
    {
        delete off_proc;
    }

    on_proc = A->on_proc->copy();
    off_proc = A->off_proc->copy();

    ParMatrix::copy_helper(A);
}

void ParCOOMatrix::copy_helper(ParCSRMatrix* A)
{
    if (on_proc)
    {
        delete on_proc;
    }
    if (off_proc)
    {
        delete off_proc;
    }

    on_proc = A->on_proc->to_COO();
    off_proc = A->off_proc->to_COO();

    ParMatrix::copy_helper(A);
}

void ParCOOMatrix::copy_helper(ParCSCMatrix* A)
{
    if (on_proc)
    {
        delete on_proc;
    }
    if (off_proc)
    {
        delete off_proc;
    }

    on_proc = A->on_proc->to_COO();
    off_proc = A->off_proc->to_COO();

    ParMatrix::copy_helper(A);
}

void ParCSRMatrix::copy_helper(ParCSRMatrix* A)
{
    if (on_proc)
    {
        delete on_proc;
    }
    if (off_proc)
    {
        delete off_proc;
    }

    on_proc = A->on_proc->copy();
    off_proc = A->off_proc->copy();

    ParMatrix::copy_helper(A);
}

void ParCSRMatrix::copy_helper(ParCSCMatrix* A)
{
    if (on_proc)
    {
        delete on_proc;
    }
    if (off_proc)
    {
        delete off_proc;
    }

    on_proc = A->on_proc->to_CSR();
    off_proc = A->off_proc->to_CSR();

    ParMatrix::copy_helper(A);
}

void ParCSRMatrix::copy_helper(ParCOOMatrix* A)
{
    if (on_proc)
    {
        delete on_proc;
    }
    if (off_proc)
    {
        delete off_proc;
    }

    on_proc = A->on_proc->to_CSR();
    off_proc = A->off_proc->to_CSR();

    ParMatrix::copy_helper(A);
}

void ParCSCMatrix::copy_helper(ParCSRMatrix* A)
{
    if (on_proc)
    {
        delete on_proc;
    }
    if (off_proc)
    {
        delete off_proc;
    }

    on_proc = A->on_proc->to_CSC();
    off_proc = A->off_proc->to_CSC();

    ParMatrix::copy_helper(A);
}

void ParCSCMatrix::copy_helper(ParCSCMatrix* A)
{
    if (on_proc)
    {
        delete on_proc;
    }
    if (off_proc)
    {
        delete off_proc;
    }

    on_proc = A->on_proc->copy();
    off_proc = A->off_proc->copy();

    ParMatrix::copy_helper(A);
}

void ParCSCMatrix::copy_helper(ParCOOMatrix* A)
{
    if (on_proc)
    {
        delete on_proc;
    }
    if (off_proc)
    {
        delete off_proc;
    }

    on_proc = A->on_proc->to_CSC();
    off_proc = A->off_proc->to_CSC();

    ParMatrix::copy_helper(A);
}

// Main transpose
ParCSRMatrix* ParCSRMatrix::transpose()
{
    int start, end;
    int proc;
    int col, col_start, col_end;
    int ctr, prev_ctr, size, bytes;
    int col_count, count;
    int col_size;
    int idx, row;
    RAPtor_MPI_Status recv_status;

    Partition* part_T;
    Matrix* on_proc_T;
    Matrix* off_proc_T;
    CSCMatrix* send_mat;
    CSCMatrix* recv_mat;
    ParCSRMatrix* T = NULL;

    std::vector<char> send_buffer;
    std::vector<char> recv_buffer;

    // Transpose partition
    part_T = partition->transpose();

    // Transpose local (on_proc) matrix
    on_proc_T = on_proc->transpose();

    // Allocate vectors for sending off_proc matrix
    send_mat = off_proc->to_CSC();
    recv_mat = new CSCMatrix(local_num_rows, comm->send_data->size_msgs);

    int int_size, dbl_size;
    MPI_Pack_size(1, RAPtor_MPI_INT, comm->mpi_comm, &int_size);
    MPI_Pack_size(1, RAPtor_MPI_DOUBLE, comm->mpi_comm, &dbl_size);

    bytes = 0;
    for (int i = 0; i < comm->recv_data->num_msgs; i++)
    {
        start = comm->recv_data->indptr[i];
        end = comm->recv_data->indptr[i+1];
        for (col = start; col < end; col++)
        {
            col_start = send_mat->idx1[col];
            col_end = send_mat->idx1[col+1];
            col_size = col_end - col_start;
            bytes += col_size * (int_size + dbl_size) + int_size;
        }
    }
    send_buffer.resize(bytes);
    std::vector<int> send_ptr(comm->recv_data->num_msgs+1);

    // Add off_proc cols of matrix to send buffer
    ctr = 0;
    prev_ctr = 0;
    for (int i = 0; i < comm->recv_data->num_msgs; i++)
    {
        proc = comm->recv_data->procs[i];
        start = comm->recv_data->indptr[i];
        end = comm->recv_data->indptr[i+1];
        for (col = start; col < end; col++)
        {
            col_start = send_mat->idx1[col];
            col_end = send_mat->idx1[col+1];
            col_size = col_end - col_start;
            RAPtor_MPI_Pack(&col_size, 1, RAPtor_MPI_INT, send_buffer.data(), bytes, &ctr, comm->mpi_comm);
            for (int k = col_start; k < col_end; k++)
            {
                RAPtor_MPI_Pack(&(local_row_map[send_mat->idx2[k]]), 1, RAPtor_MPI_INT,
                        send_buffer.data(), bytes, &ctr, comm->mpi_comm);
            }
            RAPtor_MPI_Pack(&(send_mat->vals[col_start]), col_end - col_start, RAPtor_MPI_DOUBLE,
                    send_buffer.data(), bytes, &ctr, comm->mpi_comm);
        }

        RAPtor_MPI_Isend(&(send_buffer[prev_ctr]), ctr - prev_ctr, RAPtor_MPI_PACKED, proc,
                comm->key, comm->mpi_comm, &(comm->recv_data->requests[i]));
        prev_ctr = ctr;
    }

    col_count = 0;
    recv_mat->idx1[0] = 0;
    recv_mat->nnz = 0;
    for (int i = 0; i < comm->send_data->num_msgs; i++)
    {
        proc = comm->send_data->procs[i];
        start = comm->send_data->indptr[i];
        end = comm->send_data->indptr[i+1];
        size = end - start;
        RAPtor_MPI_Probe(proc, comm->key, comm->mpi_comm, &recv_status);
        RAPtor_MPI_Get_count(&recv_status, RAPtor_MPI_PACKED, &count);
        if (count > (int)recv_buffer.size())
        {
            recv_buffer.resize(count);
        }
        RAPtor_MPI_Recv(&(recv_buffer[0]), count, RAPtor_MPI_PACKED, proc,
                comm->key, comm->mpi_comm, &recv_status);
        ctr = 0;
        for (int j = 0; j < size; j++)
        {
            RAPtor_MPI_Unpack(recv_buffer.data(), count, &ctr, &col_size, 1, RAPtor_MPI_INT, comm->mpi_comm);
            recv_mat->idx2.resize(recv_mat->nnz + col_size);
            recv_mat->vals.resize(recv_mat->nnz + col_size);
            RAPtor_MPI_Unpack(recv_buffer.data(), count, &ctr, &(recv_mat->idx2[recv_mat->nnz]), col_size,
                    RAPtor_MPI_INT, comm->mpi_comm);
            RAPtor_MPI_Unpack(recv_buffer.data(), count, &ctr, &(recv_mat->vals[recv_mat->nnz]), col_size,
                    RAPtor_MPI_DOUBLE, comm->mpi_comm);
            recv_mat->nnz += col_size;
            recv_mat->idx1[col_count+1] = recv_mat->nnz;
            col_count++;
        }
    }
    recv_mat->nnz = recv_mat->idx2.size();
    RAPtor_MPI_Waitall(comm->recv_data->num_msgs, comm->recv_data->requests.data(), RAPtor_MPI_STATUSES_IGNORE);

    off_proc_T = new CSRMatrix(on_proc_num_cols, -1);
    std::vector<int> off_T_sizes(on_proc_num_cols, 0);
    for (int i = 0; i < comm->send_data->size_msgs; i++)
    {
        row = comm->send_data->indices[i];
        start = recv_mat->idx1[i];
        end = recv_mat->idx1[i+1];
        off_T_sizes[row] += (end - start);
    }
    off_proc_T->idx1[0] = 0;
    for (int i = 0; i < off_proc_T->n_rows; i++)
    {
        off_proc_T->idx1[i+1] = off_proc_T->idx1[i] + off_T_sizes[i];
        off_T_sizes[i] = 0;
    }
    off_proc_T->nnz = off_proc_T->idx1[off_proc_T->n_rows];
    off_proc_T->idx2.resize(off_proc_T->nnz);
    off_proc_T->vals.resize(off_proc_T->nnz);
    for (int i = 0; i < comm->send_data->size_msgs; i++)
    {
        row = comm->send_data->indices[i];
        start = recv_mat->idx1[i];
        end = recv_mat->idx1[i+1];
        for (int j = start; j < end; j++)
        {
            idx = off_proc_T->idx1[row] + off_T_sizes[row]++;
            off_proc_T->idx2[idx] = recv_mat->idx2[j];
            off_proc_T->vals[idx] = recv_mat->vals[j];
        }
    }

    T = new ParCSRMatrix(part_T, on_proc_T, off_proc_T);

    delete send_mat;
    delete recv_mat;

    return T;
}

ParCOOMatrix* ParCOOMatrix::transpose()
{
    ParCSRMatrix* A_csr = to_ParCSR();
    ParCSRMatrix* AT_csr = A_csr->transpose();
    delete A_csr;

    ParCOOMatrix* AT = AT_csr->to_ParCOO();
    delete AT_csr;

    return AT;
}
ParCSCMatrix* ParCSCMatrix::transpose()
{
    // TODO -- Shouldn't have to convert first
    ParCSRMatrix* A_csr = to_ParCSR();
    ParCSRMatrix* AT_csr = A_csr->transpose();
    delete A_csr;

    ParCSCMatrix* AT = AT_csr->to_ParCSC();
    delete AT_csr;

    return AT;
}

// Assumes block_row_size and block_col_size evenly divide local row/col sizes
ParBSRMatrix* ParCSRMatrix::to_ParBSR(const int block_row_size, const int block_col_size)
{
    int start, end, col;
    int prev_row, prev_col;
    int block_row, block_col;
    int block_pos, col_pos;
    int global_col, pos;
    double val;

    int global_block_rows = global_num_rows / block_row_size;
    int global_block_cols = global_num_cols / block_col_size;
    ParBSRMatrix* A = new ParBSRMatrix(global_block_rows, global_block_cols,
            block_row_size, block_col_size);

    // Get local to global mappings for block matrix
    prev_row = -1;
    for (std::vector<int>::iterator it = local_row_map.begin();
            it != local_row_map.end(); ++it)
    {
        block_row = *it / block_row_size;
        if (block_row != prev_row)
        {
            A->local_row_map.emplace_back(block_row);
            prev_row = block_row;
        }
    }
    if (global_num_rows == global_num_cols)
    {
        A->on_proc_column_map = A->get_local_row_map();
    }
    else
    {
        prev_col = -1;
        for (std::vector<int>::iterator it = on_proc_column_map.begin();
                it != on_proc_column_map.end(); ++it)
        {
            block_col = *it / block_col_size;
            if (block_col != prev_col)
            {
                A->on_proc_column_map.emplace_back(block_row);
                prev_col = block_row;
            }
        }
    }

    prev_col = -1;
    std::map<int, int> global_to_block_local;
    for (std::vector<int>::iterator it = off_proc_column_map.begin();
            it != off_proc_column_map.end(); ++it)
    {
        block_col = *it / block_col_size;
        if (block_col != prev_col)
        {
            global_to_block_local[block_col] = A->off_proc_column_map.size();
            A->off_proc_column_map.emplace_back(block_col);
            prev_col = block_col;
        }
    }
    A->local_num_rows = A->local_row_map.size();
    A->on_proc_num_cols = A->local_num_rows;
    A->off_proc_num_cols = A->off_proc_column_map.size();
    A->off_proc->n_cols = A->off_proc_num_cols;

    BSRMatrix* A_on_proc = (BSRMatrix*) A->on_proc;
    BSRMatrix* A_off_proc = (BSRMatrix*) A->off_proc;

    A_on_proc->idx1[0] = 0;
    A_off_proc->idx1[0] = 0;
    for (int i = 0; i < local_num_rows; i += block_row_size)
    {
        std::vector<int> on_proc_pos(A->on_proc_num_cols, -1);
        std::vector<int> off_proc_pos(A->off_proc_num_cols, -1);
        for (int row_pos = 0; row_pos < block_row_size; row_pos++)
        {
            start = on_proc->idx1[i+row_pos];
            end = on_proc->idx1[i+row_pos+1];
            for (int k = start; k < end; k++)
            {
                col = on_proc->idx2[k];
                block_col = col / block_col_size;
                if (on_proc_pos[block_col] == -1)
                {
                    on_proc_pos[block_col] = A_on_proc->idx2.size();
                    A_on_proc->idx2.emplace_back(block_col);
                    A_on_proc->block_vals.emplace_back(
                            new double[A_on_proc->b_size]());
                }
                val = on_proc->vals[k];
                pos = on_proc_pos[block_col];
                col_pos = col % block_col_size;
                block_pos = row_pos * block_col_size + col_pos;
                A_on_proc->block_vals[pos][block_pos] = val;
            }

            start = off_proc->idx1[i+row_pos];
            end = off_proc->idx1[i+row_pos+1];
            for (int k = start; k < end; k++)
            {
                col = off_proc->idx2[k];
                global_col = off_proc_column_map[col];
                block_col = global_to_block_local[global_col / block_col_size];
                if (off_proc_pos[block_col] == -1)
                {
                    off_proc_pos[block_col] = A_off_proc->idx2.size();
                    A_off_proc->idx2.emplace_back(block_col);
                    A_off_proc->block_vals.emplace_back(
                            new double[A_off_proc->b_size]());
                }
                val = off_proc->vals[k];
                pos = off_proc_pos[block_col];
                col_pos = global_col % block_col_size;
                block_pos = row_pos * block_col_size + col_pos;
                A_off_proc->block_vals[pos][block_pos] = val;
            }
        }
        A_on_proc->idx1[i/block_row_size + 1] = A_on_proc->idx2.size();
        A_off_proc->idx1[i/block_row_size + 1] = A_off_proc->idx2.size();
    }
    A_on_proc->nnz = A_on_proc->idx2.size();
    A_off_proc->nnz = A_off_proc->idx2.size();

    A->comm = new ParComm(A->partition, A->off_proc_column_map);

    return A;
}

void ParMatrix::init_tap_communicators(RAPtor_MPI_Comm mpi_comm)
{
    /*********************************
     * Initialize
     * *******************************/
    // Get RAPtor_MPI Information
    int rank, num_procs;
    RAPtor_MPI_Comm_rank(mpi_comm, &rank);
    RAPtor_MPI_Comm_size(mpi_comm, &num_procs);

    // Initialize standard tap_comm
    tap_comm = new TAPComm(partition, true);

    // Initialize Variables
    std::vector<int> off_proc_col_to_proc;
    std::vector<int> on_node_column_map;
    std::vector<int> on_node_col_to_proc;
    std::vector<int> off_node_column_map;
    std::vector<int> off_node_col_to_proc;
    std::vector<int> on_node_to_off_proc;
    std::vector<int> off_node_to_off_proc;
    std::vector<int> recv_nodes;
    std::vector<int> orig_procs;
    std::vector<int> node_to_local_proc;
    std::vector<int> on_proc_to_new;
    int on_proc_nc = on_proc_column_map.size();
    if (partition->local_num_cols)
    {
        on_proc_to_new.resize(partition->local_num_cols);
        for (int i = 0; i < on_proc_nc; i++)
        {
            on_proc_to_new[on_proc_column_map[i] - partition->first_local_col] = i;
        }
    }

    /*********************************
     * Split columns by processes,
     * on-node, and off-node
     * *******************************/
    // Find process on which vector value associated with each column is
    // stored
    partition->form_col_to_proc(off_proc_column_map, off_proc_col_to_proc);

    // Partition off_proc cols into on_node and off_node
    tap_comm->split_off_proc_cols(off_proc_column_map, off_proc_col_to_proc,
           on_node_column_map, on_node_col_to_proc, on_node_to_off_proc,
           off_node_column_map, off_node_col_to_proc, off_node_to_off_proc);

    // Form local_L_par_comm: fully local communication (origin and
    // destination processes both local to node)
    tap_comm->form_local_L_par_comm(on_node_column_map, on_node_col_to_proc,
            partition->first_local_col);
    for (std::vector<int>::iterator it = tap_comm->local_L_par_comm->send_data->indices.begin();
            it != tap_comm->local_L_par_comm->send_data->indices.end(); ++it)
    {
        *it = on_proc_to_new[*it];
    }


    /*********************************
     * Form standard 3-step
     * node-aware communicator
     * *******************************/
    // Gather all nodes with which any local process must communication
    tap_comm->form_local_R_par_comm(off_node_column_map, off_node_col_to_proc,
            orig_procs);

    // Find global processes with which rank communications
    tap_comm->form_global_par_comm(orig_procs);

    // Form local_S_par_comm: initial distribution of values among local
    // processes, before inter-node communication
    tap_comm->form_local_S_par_comm(orig_procs);

    // Adjust send indices (currently global vector indices) to be index
    // of global vector value from previous recv
    tap_comm->adjust_send_indices(partition->first_local_col);


    tap_comm->update_recv(on_node_to_off_proc, off_node_to_off_proc);
    for (std::vector<int>::iterator it = tap_comm->local_S_par_comm->send_data->indices.begin();
            it != tap_comm->local_S_par_comm->send_data->indices.end(); ++it)
    {
        *it = on_proc_to_new[*it];
    }


    /*********************************
     * Form simple 2-step
     * node-aware communicator
     * *******************************/
    // Create simple (2-step) TAPComm for matrix communication
    // Copy local_L_par_comm from 3-step tap_comm
    tap_mat_comm = new TAPComm(partition, false, tap_comm->local_L_par_comm);

    // Form local recv communicator.  Will recv from local rank
    // corresponding to global rank on which data originates.  E.g. if
    // data is on rank r = (p, n), and my rank is s = (q, m), I will
    // recv data from (p, m).
    tap_mat_comm->form_simple_R_par_comm(off_node_column_map, off_node_col_to_proc);

    // Form global par comm.. Will recv from proc on which data
    // originates
    tap_mat_comm->form_simple_global_comm(off_node_col_to_proc);

    // Adjust send indices (currently global vector indices) to be
    // index of global vector value from previous recv (only updating
    // local_R to match position in global)
    tap_mat_comm->adjust_send_indices(partition->first_local_col);

    tap_mat_comm->update_recv(on_node_to_off_proc, off_node_to_off_proc, false);

    for (std::vector<int>::iterator it =
            tap_mat_comm->global_par_comm->send_data->indices.begin();
            it != tap_mat_comm->global_par_comm->send_data->indices.end(); ++it)
    {
        *it = on_proc_to_new[*it];
    }
}
