// Copyright (c) 2015, Raptor Developer Team, University of Illinois at Urbana-Champaign
// License: Simplified BSD, http://opensource.org/licenses/BSD-2-Clause
#ifndef PARMATRIX_HPP
#define PARMATRIX_HPP

#include <mpi.h>
#include <math.h>
#include <Eigen/Dense>

#include "core/matrix.hpp"
#include "core/par_comm.hpp"
#include "core/types.hpp"

class ParMatrix
{
public:

    void reserve(index_t offd_cols, index_t nnz_per_row, index_t nnz_per_col);
    void create_partition(index_t global_rows, index_t global_cols, MPI_Comm _comm_mat);
    void add_value(index_t row, index_t global_col, data_t value, index_t row_global = 0);
    void finalize(index_t symmetric, format_t diag_f = CSR, format_t offd_f = CSC);

    ParMatrix(index_t _glob_rows, index_t _glob_cols)
    {
        // Initialize matrix dimensions
        global_rows = _glob_rows;
        global_cols = _glob_cols;
        offd_num_cols = 0;

        // Create Partition
        create_partition(global_rows, global_cols, MPI_COMM_WORLD);

        diag = new Matrix(local_rows, local_cols);
        offd = new Matrix(local_rows, local_cols);
    }

    ParMatrix(index_t _glob_rows, index_t _glob_cols, MPI_Comm _comm_mat)
    {
        // Initialize matrix dimensions
        global_rows = _glob_rows;
        global_cols = _glob_cols;
        offd_num_cols = 0;

        // Create Partition
        create_partition(global_rows, global_cols, _comm_mat);

        diag = new Matrix(local_rows, local_cols);
        offd = new Matrix(local_rows, local_cols);
    }

    ParMatrix(index_t _glob_rows, index_t _glob_cols, data_t* values)
    {
        global_rows = _glob_rows; 
        global_cols = _glob_cols;
        offd_num_cols = 0;
       
        // Create Partition
        create_partition(global_rows, global_cols, MPI_COMM_WORLD);

        diag = new Matrix(local_rows, local_cols);
        offd = new Matrix(local_rows, local_cols);

        index_t val_start = first_row * global_cols;
        index_t val_end = (first_row + local_rows) * global_cols;

        for (index_t i = val_start; i < val_end; i++)
        {
            if (fabs(values[i]) > zero_tol)
            {
                index_t global_col = i % global_cols;
                index_t global_row = i / global_cols;
                add_value(global_row - first_row, global_col, values[i]);
            }
        }

        finalize(0);
    }

    ParMatrix(index_t _globalRows, index_t _globalCols, Matrix* _diag, Matrix* _offd)
    {
        this->global_rows = _globalRows;
        this->global_cols = _globalCols;
        this->diag = _diag;
        this->offd = _offd;
    }

    ParMatrix(ParMatrix* A)
    {
        this->global_rows = A->global_rows;
        this->global_cols = A->global_cols;
        this->diag = A->diag; // should we mark as not owning? (we should think about move semantics or if people really love pointers we could use smart pointers).
        this->offd = A->offd;
    }

    ParMatrix()
    {
        this->local_rows = 0;
        this->local_cols = 0;
        this->offd_num_cols = 0;
    }

    ~ParMatrix()
    {
        if (this->offd_num_cols)
        {
            delete this->offd;
        }
        if (this->local_rows)
        {
            delete this->diag;
            delete this->comm;
        }
        delete[] this-> global_col_starts;
        local_to_global.clear();
        global_to_local.clear();
    }

    index_t global_rows;
    index_t global_cols;
    index_t local_nnz;
    index_t local_rows;
    index_t local_cols;
    Matrix* diag;
    Matrix* offd;
    std::vector<index_t> local_to_global;
    std::map<index_t, index_t> global_to_local;
    index_t offd_num_cols;
    index_t first_col_diag;
    index_t first_row;
    ParComm* comm;
    index_t* global_col_starts;
    MPI_Comm comm_mat;
};
#endif
