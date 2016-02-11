// Copyright (c) 2015, Raptor Developer Team, University of Illinois at Urbana-Champaign
// License: Simplified BSD, http://opensource.org/licenses/BSD-2-Clause
#ifndef PARMATRIX_HPP
#define PARMATRIX_HPP

#include <mpi.h>
#include <math.h>

#include "array.hpp"
#include "matrix.hpp"
#include "par_comm.hpp"
#include "types.hpp"

namespace raptor
{
class ParMatrix
{
public:

    void create_partition(index_t global_rows, index_t global_cols, MPI_Comm _comm_mat = MPI_COMM_WORLD);
    void create_partition(index_t global_rows, index_t global_cols, index_t first_row, index_t local_rows, index_t first_col_diag);
    void create_comm_mat(MPI_Comm _comm_mat = MPI_COMM_WORLD);
    void add_value(index_t row, index_t global_col, data_t value, index_t row_global = 0);
    void finalize(int symmetric);

    ParMatrix(index_t _glob_rows, index_t _glob_cols, format_t diag_f = CSR, format_t offd_f = CSC)
    {
        // Initialize matrix dimensions
        global_rows = _glob_rows;
        global_cols = _glob_cols;
        offd_num_cols = 0;

        // Create Partition
        create_partition(global_rows, global_cols, MPI_COMM_WORLD);

        diag = new Matrix(local_rows, local_cols, diag_f);
        offd = new Matrix(local_rows, local_cols, offd_f);
        if (local_rows)
        {
            diag_elmts = new data_t[local_rows]();
        }
    }

    ParMatrix(index_t _glob_rows, index_t _glob_cols, index_t _local_rows, index_t _local_cols, index_t _first_row, index_t _first_col_diag, index_t* _global_col_starts = NULL, format_t diag_f = CSR, format_t offd_f = CSC)
    {
        global_rows = _glob_rows;
        global_cols = _glob_cols;
        local_rows = _local_rows;
        local_cols = _local_cols;
        first_row = _first_row;
        first_col_diag = _first_col_diag;
        offd_num_cols = 0;

	if (_global_col_starts == NULL)
	{
            create_partition(global_rows, global_cols, first_row, local_rows, first_col_diag);
	}
	else
	{
	    global_col_starts = _global_col_starts;
	    create_comm_mat(MPI_COMM_WORLD);
	}
        diag = new Matrix(local_rows, local_cols, diag_f);
        offd = new Matrix(local_rows, local_cols, offd_f);

        if (local_rows)
        {
            diag_elmts = new data_t[local_rows]();
        }
    }

    ParMatrix(index_t _glob_rows, index_t _glob_cols, MPI_Comm _comm_mat, format_t diag_f = CSR, format_t offd_f = CSC)
    {
        // Initialize matrix dimensions
        global_rows = _glob_rows;
        global_cols = _glob_cols;
        offd_num_cols = 0;

        // Create Partition
        create_partition(global_rows, global_cols, _comm_mat);

        diag = new Matrix(local_rows, local_cols, diag_f);
        offd = new Matrix(local_rows, local_cols, offd_f);
        if (local_rows)
        {
            diag_elmts = new data_t[local_rows]();
        }
    }

    ParMatrix(index_t _glob_rows, index_t _glob_cols, data_t* values, format_t diag_f = CSR, format_t offd_f = CSC)
    {
        global_rows = _glob_rows; 
        global_cols = _glob_cols;
        offd_num_cols = 0;
       
        // Create Partition
        create_partition(global_rows, global_cols, MPI_COMM_WORLD);

        diag = new Matrix(local_rows, local_cols, diag_f);
        offd = new Matrix(local_rows, local_cols, offd_f);
        if (local_rows)
        {
            diag_elmts = new data_t[local_rows]();
        }

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
            delete diag_elmts;
        }
        delete[] this-> global_col_starts;
    }

    index_t global_rows;
    index_t global_cols;
    index_t local_nnz;
    index_t local_rows;
    index_t local_cols;
    Matrix* diag;
    Matrix* offd;
    data_t* diag_elmts;
    Array<index_t> local_to_global;
    std::map<index_t, index_t> global_to_local;
    index_t offd_num_cols;
    index_t first_col_diag;
    index_t first_row;
    ParComm* comm;
    index_t* global_col_starts;
    MPI_Comm comm_mat;
};
}
#endif
