// Copyright (c) 2015, Raptor Developer Team, University of Illinois at Urbana-Champaign
// License: Simplified BSD, http://opensource.org/licenses/BSD-2-Clause
#include "par_matrix.hpp"

using namespace raptor;

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
            && global_col < partition->last_local_col)
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
void ParMatrix::finalize(bool create_comm)
{
    // Condense columns in off_proc, storing global
    // columns as 0-num_cols, and store mapping
    if (off_proc->nnz)
    {
        off_proc->condense_cols();
        off_proc->sort();
        off_proc_column_map = off_proc->get_col_list();
        off_proc_num_cols = off_proc_column_map.size();   
        off_proc->resize(local_num_rows, off_proc_num_cols);
    }
    else
    {
        off_proc_num_cols = 0;
    }

    if (on_proc->nnz)
    {
        on_proc->sort();
    }
        
    local_nnz = on_proc->nnz + off_proc->nnz;

    if (create_comm)
        comm = new ParComm(partition, off_proc_column_map);
    else
        comm = new ParComm();
}

void ParMatrix::copy(ParCSRMatrix* A)
{
    partition = A->partition;
    partition->num_shared++;

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
    partition = A->partition;
    partition->num_shared++;

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
    partition = A->partition;
    partition->num_shared++;

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

void ParCOOMatrix::copy(ParCSRMatrix* A)
{
    ParMatrix::copy(A);

    if (on_proc)
    {   
        delete on_proc;
    }
    if (off_proc)
    {
        delete off_proc;
    }

    on_proc = new COOMatrix((CSRMatrix*) A->on_proc);
    off_proc = new COOMatrix((CSRMatrix*) A->off_proc);
}

void ParCOOMatrix::copy(ParCSCMatrix* A)
{
    ParMatrix::copy(A);

    if (on_proc)
    {   
        delete on_proc;
    }
    if (off_proc)
    {
        delete off_proc;
    }

    on_proc = new COOMatrix((CSCMatrix*) A->on_proc);
    off_proc = new COOMatrix((CSCMatrix*) A->off_proc);
}

void ParCOOMatrix::copy(ParCOOMatrix* A)
{
    ParMatrix::copy(A);

    if (on_proc)
    {   
        delete on_proc;
    }
    if (off_proc)
    {
        delete off_proc;
    }

    on_proc = new COOMatrix((COOMatrix*) A->on_proc);
    off_proc = new COOMatrix((COOMatrix*) A->off_proc);
}

COOMatrix* ParCOOMatrix::communicate(CommPkg* comm)
{
    printf("Not implemented for COO Matrices.\n");
    return NULL;
}

void ParCSRMatrix::copy(ParCSRMatrix* A)
{
    ParMatrix::copy(A);

    if (on_proc)
    {   
        delete on_proc;
    }
    if (off_proc)
    {
        delete off_proc;
    }
    on_proc = new CSRMatrix((CSRMatrix*) A->on_proc);
    off_proc = new CSRMatrix((CSRMatrix*) A->off_proc);
}

void ParCSRMatrix::copy(ParCSCMatrix* A)
{
    ParMatrix::copy(A);

    if (on_proc)
    {   
        delete on_proc;
    }
    if (off_proc)
    {
        delete off_proc;
    }
    on_proc = new CSRMatrix((CSCMatrix*) A->on_proc);
    off_proc = new CSRMatrix((CSCMatrix*) A->off_proc);
}

void ParCSRMatrix::copy(ParCOOMatrix* A)
{
    ParMatrix::copy(A);

    if (on_proc)
    {   
        delete on_proc;
    }
    if (off_proc)
    {
        delete off_proc;
    }

    on_proc = new CSRMatrix((COOMatrix*) A->on_proc);
    off_proc = new CSRMatrix((COOMatrix*) A->off_proc);
}

CSRMatrix* ParCSRMatrix::communicate(CommPkg* comm)
{
    int start, end;
    int ctr;
    int global_col;

    int nnz = on_proc->nnz + off_proc->nnz;
    std::vector<int> rowptr(local_num_rows + 1);
    std::vector<int> col_indices;
    std::vector<double> values;
    if (nnz)
    {
        col_indices.resize(nnz);
        values.resize(nnz);
    }

    ctr = 0;
    rowptr[0] = ctr;
    for (int i = 0; i < local_num_rows; i++)
    {
        start = on_proc->idx1[i];
        end = on_proc->idx1[i+1];
        for (int j = start; j < end; j++)
        {
            global_col = on_proc->idx2[j] + partition->first_local_col;
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

void ParCSCMatrix::copy(ParCSRMatrix* A)
{
    ParMatrix::copy(A);
 
    if (on_proc)
    {   
        delete on_proc;
    }
    if (off_proc)
    {
        delete off_proc;
    }

    on_proc = new CSCMatrix((CSRMatrix*) A->on_proc);
    off_proc = new CSCMatrix((CSRMatrix*) A->off_proc);
}

void ParCSCMatrix::copy(ParCSCMatrix* A)
{
    ParMatrix::copy(A);

    if (on_proc)
    {   
        delete on_proc;
    }
    if (off_proc)
    {
        delete off_proc;
    }

    on_proc = new CSCMatrix((CSCMatrix*) A->on_proc);
    off_proc = new CSCMatrix((CSCMatrix*) A->off_proc);
}

void ParCSCMatrix::copy(ParCOOMatrix* A)
{
    ParMatrix::copy(A);

    if (on_proc)
    {   
        delete on_proc;
    }
    if (off_proc)
    {
        delete off_proc;
    }

    on_proc = new CSCMatrix((COOMatrix*) A->on_proc);
    off_proc = new CSCMatrix((COOMatrix*) A->off_proc);
}

CSCMatrix* ParCSCMatrix::communicate(CommPkg* comm)
{
    int start, end;
    int ctr, row, idx;
    int global_col;

    int nnz = on_proc->nnz + off_proc->nnz;
    std::vector<int> rowptr(local_num_rows + 1);
    std::vector<int> col_indices;
    std::vector<double> values;
    if (nnz)
    {
        col_indices.resize(nnz);
        values.resize(nnz);
    }
    std::vector<int> row_ctr;
    if (local_num_rows)
    {
        row_ctr.resize(local_num_rows, 0);
    }

    // Determine nnz per row
    for (int i = 0; i < local_num_cols; i++)
    {
        start = on_proc->idx1[i];
        end = on_proc->idx1[i+1];
        for (int j = start; j < end; j++)
        {
            row_ctr[on_proc->idx2[j]]++;
        }
    }
    for (int i = 0; i < off_proc_num_cols; i++)
    {
        start = off_proc->idx1[i];
        end = off_proc->idx1[i+1];
        for (int j = start; j < end; j++)
        {
            row_ctr[off_proc->idx2[j]]++;
        }
    }

    // Set rowptr values
    rowptr[0] = 0;
    for (int i = 0; i < local_num_rows; i++)
    {
        rowptr[i+1] = rowptr[i] + row_ctr[i];
        row_ctr[i] = 0;
    }

    // Set col_indices / values
    for (int i = 0; i < local_num_cols; i++)
    {
        global_col = i + partition->first_local_col;
        start = on_proc->idx1[i];
        end = on_proc->idx1[i+1];
        for (int j = start; j < end; j++)
        {
            row = on_proc->idx2[j];
            idx = rowptr[row] + row_ctr[row]++;
            col_indices[idx] = global_col;
            values[idx] = on_proc->vals[j];
        }
    }
    for (int i = 0; i < off_proc_num_cols; i++)
    {
        global_col = off_proc_column_map[i];
        start = off_proc->idx1[i];
        end = off_proc->idx1[i+1];
        for (int j = start; j < end; j++)
        {
            row = off_proc->idx2[j];
            idx = rowptr[row] + row_ctr[row]++;
            col_indices[idx] = global_col;
            values[idx] = off_proc->vals[j];
        }
    }

    CSRMatrix* recv_mat_csr = comm->communicate(rowptr, 
            col_indices, values, MPI_COMM_WORLD);
    recv_mat_csr->condense_cols();
    CSCMatrix* recv_mat = new CSCMatrix(recv_mat_csr);
    delete recv_mat_csr;

    return recv_mat;
}


