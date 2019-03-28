// Copyright (c) 2015-2017, RAPtor Developer Team
// License: Simplified BSD, http://opensource.org/licenses/BSD-2-Clause

#include "par_matrix_IO.hpp"
#include "matrix_IO.hpp"
#include <float.h>
#include <stdio.h>

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

    ParCSRMatrix* A;

    int32_t code;
    int32_t global_num_rows;
    int32_t global_num_cols;
    int32_t global_nnz;
    int32_t idx;
    double val;
    bool is_little_endian = false;

    int ctr, size;

    int sizeof_dbl = sizeof(val);
    int sizeof_int32 = sizeof(code);

    std::ifstream ifs (filename, std::ifstream::binary);

    ifs.read(reinterpret_cast<char *>(&code), sizeof_int32);
    ifs.read(reinterpret_cast<char *>(&global_num_rows), sizeof_int32);
    ifs.read(reinterpret_cast<char *>(&global_num_cols), sizeof_int32);
    ifs.read(reinterpret_cast<char *>(&global_nnz), sizeof_int32);

    if (code != PETSC_MAT_CODE)
    {
        is_little_endian = true;
        endian_swap(&code);
        endian_swap(&global_num_rows);
        endian_swap(&global_num_cols);
        endian_swap(&global_nnz);
    }

    assert(code == PETSC_MAT_CODE);

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

    aligned_vector<int> proc_nnz(num_procs);
    aligned_vector<int> row_sizes;
    aligned_vector<int> col_indices;
    aligned_vector<double> vals;
    int nnz = 0;
    if (A->local_num_rows)
        row_sizes.resize(A->local_num_rows);

    if (is_little_endian)
    {
        // Find row sizes
        ifs.seekg(A->partition->first_local_row * sizeof_int32, ifs.cur);
        for (int i = 0; i < A->local_num_rows; i++)
        {
            ifs.read(reinterpret_cast<char *>(&idx), sizeof_int32);
            endian_swap(&idx);
            row_sizes[i] = idx;
            nnz += idx;
        }
        ifs.seekg((A->global_num_rows - A->partition->last_local_row - 1) * sizeof_int32, ifs.cur);

        // Find nnz per proc (to find first_nnz)
        RAPtor_MPI_Allgather(&nnz, 1, RAPtor_MPI_INT, proc_nnz.data(), 1, RAPtor_MPI_INT, comm);
        int first_nnz = 0;
        for (int i = 0; i < rank; i++)
        {
            first_nnz += proc_nnz[i];
        }
        int remaining_nnz = global_nnz - first_nnz - nnz;

        // Resize variables
        if (nnz)
        {
            col_indices.resize(nnz);
            vals.resize(nnz);
        }

        // Read in col_indices
        ifs.seekg(first_nnz * sizeof_int32, ifs.cur);
        for (int i = 0; i < nnz; i++)
        {
            ifs.read(reinterpret_cast<char *>(&idx), sizeof_int32);
            endian_swap(&idx);
            col_indices[i] = idx;
        }
        ifs.seekg(remaining_nnz * sizeof_int32, ifs.cur);
        ifs.seekg(first_nnz * sizeof_dbl, ifs.cur);
        for (int i = 0; i < nnz; i++)
        {
            ifs.read(reinterpret_cast<char *>(&val), sizeof_dbl);
            endian_swap(&val);
            vals[i] = val;
        }
        ifs.seekg(remaining_nnz * sizeof_dbl, ifs.cur);
    }
    else
    {
        // Find row sizes
        ifs.seekg(A->partition->first_local_row * sizeof_int32, ifs.cur);
        for (int i = 0; i < A->local_num_rows; i++)
        {
            ifs.read(reinterpret_cast<char *>(&idx), sizeof_int32);
            row_sizes[i] = idx;
            nnz += idx;
        }
        ifs.seekg((A->global_num_rows - A->partition->last_local_row - 1) * sizeof_int32, ifs.cur);

        // Find nnz per proc (to find first_nnz)
        RAPtor_MPI_Allgather(&nnz, 1, RAPtor_MPI_INT, proc_nnz.data(), 1, RAPtor_MPI_INT, comm);
        int first_nnz = 0;
        for (int i = 0; i < rank; i++)
        {
            first_nnz += proc_nnz[i];
        }
        int remaining_nnz = global_nnz - first_nnz - nnz;

        // Resize variables
        if (nnz)
        {
            col_indices.resize(nnz);
            vals.resize(nnz);
        }

        // Read in col_indices
        ifs.seekg(first_nnz * sizeof_int32, ifs.cur);
        for (int i = 0; i < nnz; i++)
        {
            ifs.read(reinterpret_cast<char *>(&idx), sizeof_int32);
            col_indices[i] = idx;
        }
        ifs.seekg(remaining_nnz * sizeof_int32, ifs.cur);
        ifs.seekg(first_nnz * sizeof_dbl, ifs.cur);
        for (int i = 0; i < nnz; i++)
        {
            ifs.read(reinterpret_cast<char *>(&val), sizeof_dbl);
            vals[i] = val;
        }
        ifs.seekg(remaining_nnz * sizeof_dbl, ifs.cur);
    }

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
