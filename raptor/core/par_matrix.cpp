// Copyright (c) 2015, Raptor Developer Team, University of Illinois at Urbana-Champaign
// License: Simplified BSD, http://opensource.org/licenses/BSD-2-Clause
#include "par_matrix.hpp"

using namespace raptor;

/**************************************************************
*****   ParMatrix Initialize Partition
**************************************************************
***** Initializes information about the local partition of
***** the parallel matrix.  Determines values for:
*****     local_num_rows
*****     local_num_cols
*****     first_local_row
*****     first_local_col
***** NOTE: The values for global_num_rows and global_num_cols
***** must be set before calling this method.
**************************************************************/
void ParMatrix::initialize_partition()
{
    // Find MPI Information
    int rank, num_procs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    // Determine the number of local rows per process
    int avg_local_num_rows = global_num_rows / num_procs;
    int extra_rows = global_num_rows % num_procs;

    // Initialize local matrix rows
    first_local_row = avg_local_num_rows * rank;
    local_num_rows = avg_local_num_rows;
    if (extra_rows > rank)
    {
        first_local_row += rank;
        local_num_rows++;
    }
    else
    {
        first_local_row += extra_rows;
    }

    // Determine the number of local columns per process
    if (global_num_rows < num_procs)
    {
        num_procs = global_num_rows;
    }
    int avg_local_num_cols = global_num_cols / num_procs;
    int extra_cols = global_num_cols % num_procs;

    // Initialize local matrix columns
    if (local_num_rows)
    {
        first_local_col = avg_local_num_cols * rank;
        local_num_cols = avg_local_num_cols;
        if (extra_cols > rank)
        {
            first_local_col += rank;
            local_num_cols++;
        }
        else
        {
            first_local_col += extra_cols;
        }
    }
}

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
    if (global_col >= first_local_col && global_col < first_local_col + local_num_cols)
    {
        on_proc->add_value(row, global_col - first_local_col, value);
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
    if (global_col >= first_local_col && global_col < first_local_col + local_num_cols)
    {
        on_proc->add_value(global_row-first_local_row, global_col - first_local_col, value);
    }
    else
    {
        off_proc->add_value(global_row-first_local_row, global_col, value);
    }
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
void ParMatrix::finalize(bool create_comm)
{
    // Condense columns in off_proc, storing global
    // columns as 0-num_cols, and store mapping
    off_proc->condense_cols();
    off_proc->sort();
    on_proc->sort();
    off_proc_column_map = off_proc->get_col_list();
    off_proc_num_cols = off_proc_column_map.size();   
        
    local_nnz = on_proc->nnz + off_proc->nnz;

    if (create_comm)
        comm = new ParComm(off_proc_column_map, first_local_row, first_local_col);
    else
        comm = new ParComm();
}

void ParMatrix::copy(ParCSRMatrix* A)
{
    global_num_rows = A->global_num_rows;
    global_num_cols = A->global_num_cols;
    first_local_row = A->first_local_row;
    first_local_col = A->first_local_col;
    local_nnz = A->local_nnz;
    local_num_rows = A->local_num_rows;
    local_num_cols = A->local_num_cols;

    off_proc_num_cols = A->off_proc_num_cols;
    off_proc_column_map.resize(off_proc_num_cols);
    for (int i = 0; i < off_proc_num_cols; i++)
    {
        off_proc_column_map[i] = A->off_proc_column_map[i];
    }

    if (A->comm)
    {
        comm = new ParComm(A->comm);
    }
}

void ParMatrix::copy(ParCSCMatrix* A)
{
    global_num_rows = A->global_num_rows;
    global_num_cols = A->global_num_cols;
    first_local_row = A->first_local_row;
    first_local_col = A->first_local_col;
    local_nnz = A->local_nnz;
    local_num_rows = A->local_num_rows;
    local_num_cols = A->local_num_cols;

    if (A->comm)
    {
        comm = new ParComm(A->comm);
    }
}

void ParMatrix::copy(ParCOOMatrix* A)
{
    global_num_rows = A->global_num_rows;
    global_num_cols = A->global_num_cols;
    first_local_row = A->first_local_row;
    first_local_col = A->first_local_col;
    local_nnz = A->local_nnz;
    local_num_rows = A->local_num_rows;
    local_num_cols = A->local_num_cols;

    if (A->comm)
    {
        comm = new ParComm(A->comm);
    }
}



// Communication helper -- on_proc and off_proc must be CSRMatrix*
CSRMatrix* comm_csr(ParComm* comm, 
        Matrix* on_proc, 
        Matrix* off_proc,
        int global_num_cols, 
        std::vector<int>& off_proc_column_map,
        int first_local_col)
{
    // Number of rows in recv_mat == size_recvs
    // Number of columns is unknown (and does not matter)
    CSRMatrix* recv_mat = new CSRMatrix(comm->recv_data->size_msgs,
            global_num_cols);

    // Calculate nnz/row, for each row to be communicated
    std::vector<int> send_row_sizes(comm->send_data->size_msgs, 0);
    std::vector<int> row_sizes;
    if (off_proc->n_cols)
    {
        row_sizes.resize(off_proc->n_cols, 0);
    }

    int send_mat_size = 0;
    for (int i = 0; i < comm->send_data->num_msgs; i++)
    {
        int start = comm->send_data->indptr[i];
        int end = comm->send_data->indptr[i+1];
        int proc = comm->send_data->procs[i];

        for (int j = start; j < end; j++)
        {
            int row = comm->send_data->indices[j];
            int row_size = (on_proc->idx1[row+1] - on_proc->idx1[row])
                + (off_proc->idx1[row+1] - off_proc->idx1[row]);
            send_row_sizes[j] = row_size;
            send_mat_size += row_size;
        }

        // Send nnz/row for each row to be communicated
        MPI_Isend(&(send_row_sizes[start]), end - start, MPI_INT, proc, comm->key,
                MPI_COMM_WORLD, &(comm->send_data->requests[i]));
    }

    // Recv row sizes corresponding to each off_proc column
    for (int i = 0; i < comm->recv_data->num_msgs; i++)
    {
        int start = comm->recv_data->indptr[i];
        int end = comm->recv_data->indptr[i+1];
        int proc = comm->recv_data->procs[i];

        MPI_Irecv(&(row_sizes[start]), end - start, MPI_INT, proc, comm->key,
               MPI_COMM_WORLD, &(comm->recv_data->requests[i])); 
    }

    // Wait for row communication to complete
    MPI_Waitall(comm->recv_data->num_msgs,
            comm->recv_data->requests,
            MPI_STATUS_IGNORE);

    MPI_Waitall(comm->send_data->num_msgs,
            comm->send_data->requests,
            MPI_STATUS_IGNORE);
        
    // Allocate Matrix Space
    recv_mat->idx1[0] = 0;
    for (int i = 0; i < off_proc->n_cols; i++)
    {
        recv_mat->idx1[i+1] = recv_mat->idx1[i] + row_sizes[i];
    }
    recv_mat->nnz = recv_mat->idx1[off_proc->n_cols];
    recv_mat->idx2.resize(recv_mat->nnz);
    recv_mat->vals.resize(recv_mat->nnz);

    struct PairData{ 
        double val; 
        int   index; 
    }; 
    PairData* send_data = new PairData[send_mat_size];
    PairData* recv_data = new PairData[recv_mat->nnz];

    // Send pair_data (column indices + values) for each row
    // Can use MPI_DOUBLE_INT
    int ctr = 0;
    int prev_ctr = 0;
    for (int i = 0; i < comm->send_data->num_msgs; i++)
    {
        int start = comm->send_data->indptr[i];
        int end = comm->send_data->indptr[i+1];
        int proc = comm->send_data->procs[i];

        for (int j = start; j < end; j++)
        {
            int row = comm->send_data->indices[j];

            // Send value / column indices for all nonzeros in row
            int row_start = on_proc->idx1[row];
            int row_end = on_proc->idx1[row+1];
            for (int k = row_start; k < row_end; k++)
            {
                send_data[ctr].val = on_proc->vals[k];
                send_data[ctr++].index = on_proc->idx2[k] + first_local_col;
            }

            row_start = off_proc->idx1[row];
            row_end = off_proc->idx1[row+1];
            for (int k = row_start; k < row_end; k++)
            {
                send_data[ctr].val = off_proc->vals[k];
                send_data[ctr++].index = off_proc_column_map[off_proc->idx2[k]];
            }
        }

        // Send nnz/row for each row to be communicated
        MPI_Isend(&(send_data[prev_ctr]), ctr - prev_ctr, MPI_DOUBLE_INT, 
                proc, comm->key, MPI_COMM_WORLD, &(comm->send_data->requests[i]));
        prev_ctr = ctr;
    }

    // Recv pair_data corresponding to each off_proc column
    // and add it to correct location in matrix
    for (int i = 0; i < comm->recv_data->num_msgs; i++)
    {
        int start = comm->recv_data->indptr[i];
        int end = comm->recv_data->indptr[i+1];
        int proc = comm->recv_data->procs[i];

        int start_idx = recv_mat->idx1[start];
        int end_idx = recv_mat->idx1[end];

        MPI_Irecv(&(recv_data[start_idx]), end_idx - start_idx, MPI_DOUBLE_INT,
                proc, comm->key, MPI_COMM_WORLD, &(comm->recv_data->requests[i]));
    }

    MPI_Waitall(comm->recv_data->num_msgs,
            comm->recv_data->requests,
            MPI_STATUS_IGNORE);

    MPI_Waitall(comm->send_data->num_msgs,
            comm->send_data->requests,
            MPI_STATUS_IGNORE);

    for (int i = 0; i < recv_mat->nnz; i++)
    {
        recv_mat->idx2[i] = recv_data[i].index;
        recv_mat->vals[i] = recv_data[i].val;
    }

    delete[] send_data;
    delete[] recv_data;

    return recv_mat;
}



void ParCOOMatrix::copy(ParCSRMatrix* A)
{
    ParMatrix::copy(A);

    delete on_proc;
    delete off_proc;

    on_proc = new COOMatrix((CSRMatrix*) A->on_proc);
    off_proc = new COOMatrix((CSRMatrix*) A->off_proc);
}

void ParCOOMatrix::copy(ParCSCMatrix* A)
{
    ParMatrix::copy(A);

    delete on_proc;
    delete off_proc;

    on_proc = new COOMatrix((CSCMatrix*) A->on_proc);
    off_proc = new COOMatrix((CSCMatrix*) A->off_proc);
}

void ParCOOMatrix::copy(ParCOOMatrix* A)
{
    ParMatrix::copy(A);

    delete on_proc;
    delete off_proc;

    on_proc = new COOMatrix((COOMatrix*) A->on_proc);
    off_proc = new COOMatrix((COOMatrix*) A->off_proc);
}

Matrix* ParCOOMatrix::communicate(ParComm* comm)
{
    printf("Not implemented for COO Matrices.\n");
    return NULL;
}

void ParCSRMatrix::copy(ParCSRMatrix* A)
{
    ParMatrix::copy(A);

    delete on_proc;
    delete off_proc;

    on_proc = new CSRMatrix((CSRMatrix*) A->on_proc);
    off_proc = new CSRMatrix((CSRMatrix*) A->off_proc);
}

void ParCSRMatrix::copy(ParCSCMatrix* A)
{
    ParMatrix::copy(A);

    delete on_proc;
    delete off_proc;

    on_proc = new CSRMatrix((CSCMatrix*) A->on_proc);
    off_proc = new CSRMatrix((CSCMatrix*) A->off_proc);
}

void ParCSRMatrix::copy(ParCOOMatrix* A)
{
    ParMatrix::copy(A);

    on_proc = new CSRMatrix((COOMatrix*) A->on_proc);
    off_proc = new CSRMatrix((COOMatrix*) A->off_proc);
}

Matrix* ParCSRMatrix::communicate(ParComm* comm)
{
    return comm_csr(comm, on_proc, off_proc, global_num_cols,
            off_proc_column_map, first_local_col);
}

void ParCSCMatrix::copy(ParCSRMatrix* A)
{
    ParMatrix::copy(A);
    
    delete on_proc;
    delete off_proc;

    on_proc = new CSCMatrix((CSRMatrix*) A->on_proc);
    off_proc = new CSCMatrix((CSRMatrix*) A->off_proc);
}

void ParCSCMatrix::copy(ParCSCMatrix* A)
{
    ParMatrix::copy(A);

    delete on_proc;
    delete off_proc;

    on_proc = new CSCMatrix((CSCMatrix*) A->on_proc);
    off_proc = new CSCMatrix((CSCMatrix*) A->off_proc);
}

void ParCSCMatrix::copy(ParCOOMatrix* A)
{
    ParMatrix::copy(A);

    delete on_proc;
    delete off_proc;

    on_proc = new CSCMatrix((COOMatrix*) A->on_proc);
    off_proc = new CSCMatrix((COOMatrix*) A->off_proc);
}

Matrix* ParCSCMatrix::communicate(ParComm* comm)
{
    Matrix* on_proc_csr = new CSRMatrix((CSCMatrix*) on_proc);
    Matrix* off_proc_csr = new CSRMatrix((CSCMatrix*) off_proc);

    CSRMatrix* recv_mat_csr = comm_csr(comm, on_proc_csr, off_proc_csr, 
            global_num_cols, off_proc_column_map, first_local_col);
    recv_mat_csr->condense_cols();
    delete on_proc_csr;
    delete off_proc_csr;


    CSCMatrix* recv_mat = new CSCMatrix(recv_mat_csr);
    delete recv_mat_csr;

    return recv_mat;
}
