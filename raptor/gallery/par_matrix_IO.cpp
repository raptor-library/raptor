// Copyright (c) 2015-2017, RAPtor Developer Team
// License: Simplified BSD, http://opensource.org/licenses/BSD-2-Clause

#include "par_matrix_IO.hpp"
#include "matrix_IO.hpp"
#include <stdio.h>
#include "limits.h"

bool little_endian()
{
    int num = 1;
    return (*(char *)&num == 1);
}

template <class T>
void endian_swap(T *objp)
{
  unsigned char *memp = reinterpret_cast<unsigned char*>(objp);
  std::reverse(memp, memp + sizeof(T));
}

ParCSRMatrix* readParMatrix(const char* filename, 
        int local_num_rows, int local_num_cols,
        int first_local_row, int first_local_col, 
        RAPtor_MPI_Comm comm)
{
    int rank, num_procs;
    RAPtor_MPI_Comm_rank(comm, &rank);
    RAPtor_MPI_Comm_size(comm, &num_procs);

    ParCSRMatrix* A = NULL;

    int64_t pos;
    int32_t code;
    int32_t global_num_rows;
    int32_t global_num_cols;
    int32_t global_nnz;
    int32_t idx;
    int n_items_read;
    double val;

    bool is_little_endian = false;

    int ctr, size;

    int sizeof_dbl = sizeof(val);
    int sizeof_int32 = sizeof(code);

    FILE* ifile = fopen(filename, "rb");
    if (fseek(ifile, 0, SEEK_SET)) printf("Error seeking beginning of file\n"); 
    
    // Read code, and determine if little endian, or if long int
    int32_t header[4];
    n_items_read = fread(header, sizeof_int32, 4, ifile);
    code = header[0];
    global_num_rows = header[1];
    global_num_cols = header[2];
    global_nnz = header[3];
    if (code != PETSC_MAT_CODE)
    {
        endian_swap(&code);
        endian_swap(&global_num_rows);
        endian_swap(&global_num_cols);
        endian_swap(&global_nnz);
        is_little_endian = true;
    }

    if (first_local_col >= 0)
    {
        A = new ParCSRMatrix(global_num_rows, global_num_cols,
                local_num_rows, local_num_cols,
                first_local_row, first_local_col);
    }
    else
    {
        A = new ParCSRMatrix(global_num_rows, global_num_cols);
    }

    aligned_vector<int32_t> row_sizes;
    aligned_vector<int32_t> col_indices;
    aligned_vector<double> vals;
    aligned_vector<int> proc_nnz(num_procs);
    if (A->local_num_rows)
        row_sizes.resize(A->local_num_rows);
    int nnz = 0;

    // Find row sizes
    pos = (4 + A->partition->first_local_row) * sizeof_int32;
    if (fseek(ifile, pos, SEEK_SET)) printf("Error seeking pos\n"); 
    if (A->local_num_rows)
    {
        n_items_read = fread(row_sizes.data(), sizeof_int32, A->local_num_rows, ifile);
        if (n_items_read == EOF) printf("EOF reading code\n");
        if (ferror(ifile)) printf("Error reading row_size\n");
        if (is_little_endian)
        {
            for (int i = 0; i < A->local_num_rows; i++)
            {
                endian_swap(&(row_sizes[i]));
                nnz += row_sizes[i];
            }
        }
        else
        {
            for (int i = 0; i < A->local_num_rows; i++)
            {
                nnz += idx;
            }
        }
    }

    // Find nnz per proc (to find first_nnz)
    RAPtor_MPI_Allgather(&nnz, 1, RAPtor_MPI_INT, proc_nnz.data(), 1, RAPtor_MPI_INT, comm);
    long first_nnz = 0;
    for (int i = 0; i < rank; i++)
        first_nnz += proc_nnz[i];
    long total_nnz = first_nnz;
    for (int i = rank; i < num_procs; i++)
        total_nnz += proc_nnz[i];

    // Resize variables
    if (nnz)
    {
        col_indices.resize(nnz);
        vals.resize(nnz);
    }

    // Read in col_indices
    pos = (4 + A->global_num_rows + first_nnz) * sizeof_int32;
    if (fseek(ifile, pos, SEEK_SET)) printf("Error seeking pos\n"); 
    n_items_read = fread(col_indices.data(), sizeof_int32, nnz, ifile);
    if (n_items_read == EOF) printf("EOF reading code\n");
    if (ferror(ifile)) printf("Error reading col idx\n");
    
    pos = (4 + A->global_num_rows + total_nnz) * sizeof_int32 + (first_nnz * sizeof_dbl);
    if (fseek(ifile, pos, SEEK_SET)) printf("Error seeking pos\n"); 
    n_items_read = fread(vals.data(), sizeof_dbl, nnz, ifile);
    if (n_items_read == EOF) printf("EOF reading code\n");
    if (ferror(ifile)) printf("Error reading value\n");
   
    if (is_little_endian)
    {
        for (int i = 0; i < nnz; i++)
        {
            endian_swap(&(col_indices[i]));
            endian_swap(&(vals[i]));
        }
    }

    fclose(ifile);

    A->on_proc->idx1[0] = 0;
    A->off_proc->idx1[0] = 0;
    ctr = 0;
    for (int i = 0; i < A->local_num_rows; i++)
    {
        size = row_sizes[i];
        for (int j = 0; j < size; j++)
        {
            idx = col_indices[ctr];
            val = vals[ctr++];
            if ((int) idx >= A->partition->first_local_col &&
                    (int) idx <= A->partition->last_local_col)
            {
                A->on_proc->idx2.emplace_back(idx - A->partition->first_local_col);
                A->on_proc->vals.emplace_back(val);
            }
            else
            {
                A->off_proc->idx2.emplace_back(idx);
                A->off_proc->vals.emplace_back(val);
            }
        } 
        A->on_proc->idx1[i+1] = A->on_proc->idx2.size();
        A->off_proc->idx1[i+1] = A->off_proc->idx2.size();
    }
    A->on_proc->nnz = A->on_proc->idx2.size();
    A->off_proc->nnz = A->off_proc->idx2.size();

    A->finalize();

    return A;
}
