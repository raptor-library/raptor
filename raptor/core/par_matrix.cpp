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
    off_proc->resize(local_num_rows, off_proc_num_cols);
        
    local_nnz = on_proc->nnz + off_proc->nnz;

    if (create_comm)
        comm = new ParComm(off_proc_column_map, first_local_row, first_local_col,
                global_num_cols, local_num_cols);
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
    if (off_proc_num_cols)
    {
        off_proc_column_map.resize(off_proc_num_cols);
        for (int i = 0; i < off_proc_num_cols; i++)
        {
            off_proc_column_map[i] = A->off_proc_column_map[i];
        }
    }

    if (A->comm)
    {
        comm = new ParComm((ParComm*) A->comm);
    }
    else
    {   
        comm = NULL;
    }
    
    if (A->tap_comm)
    {
        tap_comm = new TAPComm((TAPComm*) A->tap_comm);
    }
    else
    {
        tap_comm = NULL;
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

    off_proc_num_cols = A->off_proc_num_cols;
    if (off_proc_num_cols)
    {
        off_proc_column_map.resize(off_proc_num_cols);
        for (int i = 0; i < off_proc_num_cols; i++)
        {
            off_proc_column_map[i] = A->off_proc_column_map[i];
        }
    }

    if (A->comm)
    {
        comm = new ParComm((ParComm*) A->comm);
    }
    else
    {   
        comm = NULL;
    }
    
    if (A->tap_comm)
    {
        tap_comm = new TAPComm((TAPComm*) A->tap_comm);
    }
    else
    {
        tap_comm = NULL;
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

    off_proc_num_cols = A->off_proc_column_map.size();
    if (off_proc_num_cols)
    {
        off_proc_column_map.resize(off_proc_num_cols);
        for (int i = 0; i < off_proc_num_cols; i++)
        {
            off_proc_column_map[i] = A->off_proc_column_map[i];
        }
    }

    if (A->comm)
    {
        comm = new ParComm((ParComm*) A->comm);
    }
    else
    {   
        comm = NULL;
    }
    
    if (A->tap_comm)
    {
        tap_comm = new TAPComm((TAPComm*) A->tap_comm);
    }
    else
    {
        tap_comm = NULL;
    }
}



// Communication helper -- on_proc and off_proc must be CSRMatrix*
// TODO -- fix this so it works for ParComm or TAPComm
CSRMatrix* comm_csr(CommPkg* comm, 
        Matrix* on_proc, 
        Matrix* off_proc,
        int global_num_cols, 
        std::vector<int>& off_proc_column_map,
        int first_local_col)
{
    Vector& recvbuf = comm->get_recv_buffer();

    int start, end;
    int ctr;
    int global_col;

    int nnz = on_proc->nnz + off_proc->nnz;
    std::vector<int> rowptr(on_proc->n_rows + 1);
    std::vector<int> col_indices;
    std::vector<double> values;
    if (nnz)
    {
        col_indices.resize(nnz);
        values.resize(nnz);
    }

    ctr = 0;
    rowptr[0] = ctr;
    for (int i = 0; i < on_proc->n_rows; i++)
    {
        start = on_proc->idx1[i];
        end = on_proc->idx1[i+1];
        for (int j = start; j < end; j++)
        {
            global_col = on_proc->idx2[j] + first_local_col;
            col_indices[ctr] = global_col;
            values[ctr++] = on_proc->vals[j];
        }

        start = off_proc->idx1[i];
        end = off_proc->idx1[i+1];
        for (int j = start; j < end; j++)
        {
            global_col = off_proc_column_map[off_proc->idx2[j]];
            col_indices[ctr] = global_col;
            values[ctr++] = off_proc->vals[j];
        }
        rowptr[i+1] = ctr;
    }

    return comm->communicate(rowptr, col_indices, values, MPI_COMM_WORLD);
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

Matrix* ParCOOMatrix::communicate(CommPkg* comm)
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

Matrix* ParCSRMatrix::communicate(CommPkg* comm)
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

Matrix* ParCSCMatrix::communicate(CommPkg* comm)
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
