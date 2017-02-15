// Copyright (c) 2015, Raptor Developer Team, University of Illinois at Urbana-Champaign
// License: Simplified BSD, http://opensource.org/licenses/BSD-2-Clause
#ifndef PARMATRIX_HPP
#define PARMATRIX_HPP

#include <mpi.h>
#include <math.h>
#include <set>

#include "matrix.hpp"
#include "par_comm.hpp"
#include "types.hpp"

/**************************************************************
 *****   ParMatrix Class
 **************************************************************
 ***** This class constructs a parallel matrix object, holding
 ***** a local diagonal matrix, a local off-diagonal block matrix,
 ***** and communication information
 *****
 ***** Attributes
 ***** -------------
 ***** global_num_rows : index_t
 *****    Number of rows in the global parallel matrix
 ***** global_num_cols : index_t
 *****    Number of columns in the parallel matrix
 ***** local_nnz : int
 *****    Number of nonzeros stored locally
 ***** local_num_rows : int
 *****    Number of rows stored locally
 ***** local_num_cols : int 
 *****    Number of columns stored locally
 ***** first_local_row : index_t
 *****    Global index of first row local to process
 ***** first_local_col : index_t
 *****    Global index of first column to fall in local block
 ***** diag : Matrix*
 *****    Matrix storing local diagonal block
 ***** offd : Matrix*
 *****    Matrix storing local off-diagonal block
 ***** offd_num_cols : index_t
 *****    Number of columns in the off-diagonal matrix
 ***** offd_column_map : std::vector<index_t>
 *****    Maps local columns of offd Matrix to global
 ***** comm : ParComm*
 *****    Parallel communicator for matrix
 ***** 
 ***** Methods
 ***** -------
 ***** initalize_partition()
 *****    Determines which rows are local to process and which 
 *****    columns fall in local block 
 ***** add_value()
 *****    Adds a value to a given local row and global column.  
 *****    Determines if this value is in the diagonal or 
 *****    off-diagonal block.
 ***** add_global_value()
 *****    Adds a value to a given global row and global column.  
 *****    Determines if this value is in the diagonal or 
 *****    off-diagonal block.
 ***** finalize()
 *****    Finalizes a matrix after values have been added.  
 *****    Converts the matrices to the appropriate formats and 
 *****    creates the parallel communicator.
 **************************************************************/
namespace raptor
{
class ParMatrix
{
public:
    /**************************************************************
    *****   ParMatrix Class Constructor
    **************************************************************
    ***** Initializes an empty ParMatrix, setting the dimensions
    ***** to the passed values.  The matrix is evenly partitioned 
    ***** across all processes.  The off-diagonal matrix is set
    ***** to have num_cols equal to the global number of columns
    ***** in the off-diagonal block (this should later be adjusted
    ***** to the number of columns with non-zeros)
    *****
    ***** Parameters
    ***** -------------
    ***** glob_rows : index_t
    *****    Global number of rows in parallel matrix
    ***** glob_cols : index_t
    *****    Global number of columns in parallel matrix
    ***** nnz_per_row : int (optional)
    *****    Estimate for number of non-zeros per row, used to 
    *****    reserve space in Matrix classes.
    **************************************************************/
    ParMatrix(index_t glob_rows, 
            index_t glob_cols, 
            int nnz_per_row = 5)
    {
        // Initialize matrix dimensions
        global_num_rows = glob_rows;
        global_num_cols = glob_cols;

        // Initialize matrix partition information
        initialize_partition();

        // Initialize diag and offd matrices as COO for adding entries
        // This should later be changed to CSR or CSC
        // A guess of 5 nnz per row is used for reserving matrix space
        diag = new COOMatrix(local_num_rows, local_num_cols, nnz_per_row);
        offd = new COOMatrix(local_num_rows, global_num_cols, nnz_per_row);
    }

    /**************************************************************
    *****   ParMatrix Class Constructor
    **************************************************************
    ***** Initializes an empty ParMatrix, setting the dimensions
    ***** to the passed values.  The local dimensions of the matrix
    ***** are passed as an argument, and the matrix is partitioned
    ***** according to the combination of all process's local dimensions.
    ***** The off-diagonal matrix is set
    ***** to have num_cols equal to the global number of columns
    ***** in the off-diagonal block (this should later be adjusted
    ***** to the number of columns with non-zeros).
    *****
    ***** Parameters
    ***** -------------
    ***** glob_rows : index_t
    *****    Global number of rows in parallel matrix
    ***** glob_cols : index_t
    *****    Global number of columns in parallel matrix
    ***** local_num_rows : int
    *****    Number of rows stored locally
    ***** local_num_cols : int
    *****    Number of columns in the diagonal block
    ***** first_local_row : index_t
    *****    Position of local matrix in global parallel matrix
    ***** first_col : index_t
    *****    First column of parallel matrix in diagonal block
    ***** nnz_per_row : int (optional)
    *****    Estimate for number of non-zeros per row, used to 
    *****    reserve space in Matrix classes.
    **************************************************************/
    ParMatrix(index_t glob_rows, 
            index_t glob_cols, 
            int local_num_rows, 
            int local_num_cols, 
            index_t first_local_row, 
            index_t first_col, 
            int nnz_per_row = 5)
    {
        global_num_rows = glob_rows;
        global_num_cols = glob_cols;
        local_num_rows = local_num_rows;
        local_num_cols = local_num_cols;
        first_local_row = first_local_row;
        first_local_col = first_col;

        diag = new COOMatrix(local_num_rows, local_num_cols, nnz_per_row);
        diag = new COOMatrix(local_num_rows, global_num_cols, nnz_per_row);
    }

    /**************************************************************
    *****   ParMatrix Class Constructor
    **************************************************************
    ***** Initializes a ParMatrix from a dense array of values.
    *****
    ***** Parameters
    ***** -------------
    ***** glob_rows : index_t
    *****    Global number of rows in parallel matrix
    ***** glob_cols : index_t
    *****    Global number of columns in parallel matrix
    ***** values : data_t*
    *****    Pointer to dense list of values to be put into 
    *****    the parallel matrix (zeros will be ignored)
    ***** nnz_per_row : int (optional)
    *****    Estimate for number of non-zeros per row, used to 
    *****    reserve space in Matrix classes.
    **************************************************************/
    ParMatrix(index_t glob_rows, 
            index_t glob_cols, 
            data_t* values,
            int nnz_per_row = 5)
    {
        // Set parallel matrix and local partition dimensions
        global_num_rows = glob_rows;
        global_num_cols = glob_cols;
        initialize_partition();

        // Initialize empty diag/offd matrices
        diag = new COOMatrix(local_num_rows, local_num_cols, nnz_per_row);
        offd = new COOMatrix(local_num_rows, global_num_cols, nnz_per_row);

        // Add values to diag/offd matrices
        index_t val_start = first_local_row * global_num_cols;
        index_t val_end = (first_local_row + local_num_rows) * global_num_cols;
        for (index_t i = val_start; i < val_end; i++)
        {
            if (fabs(values[i]) > zero_tol)
            {
                index_t global_col = i % global_num_cols;
                index_t global_row = i / global_num_cols;
                add_value(global_row - first_local_row, global_col, values[i]);
            }
        }

        // Convert diag/offd to compressed formats and
        // create parallel communicator
        finalize();
    }  
       
    /**************************************************************
    *****   ParMatrix Class Constructor
    **************************************************************
    ***** Initializes an empty ParMatrix, setting dimensions
    ***** to zero
    **************************************************************/
    ParMatrix()
    {
        local_num_rows = 0;
        local_num_cols = 0;
        offd_num_cols = 0;
        diag = new CSRMatrix(0, 0, 0);
        comm = new ParComm();

    }

    /**************************************************************
    *****   ParMatrix Class Destructor
    **************************************************************
    ***** Deletes local Matrices, Parallel Communicator, and 
    ***** array of diagonal elements
    **************************************************************/
    ~ParMatrix()
    {
        if (offd_num_cols)
        {
            delete offd;
        }
        delete diag;
        delete comm;
    }

    /**************************************************************
    *****   ParMatrix Add Value
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
    void initialize_partition()
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
    ***** local_row : index_t
    *****    Local row of value 
    ***** global_col : index_t 
    *****    Global column of value
    ***** value : data_t
    *****    Value to be added to parallel matrix
    **************************************************************/
    void add_value(index_t row, index_t global_col, data_t value);

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
    void add_global_value(int row, int global_col, double value);

    /**************************************************************
    *****   ParMatrix Finalize
    **************************************************************
    ***** Finalizes the diagonal and off-diagonal matrices.  Sorts
    ***** the local_to_global indices, and creates the parallel
    ***** communicator
    **************************************************************/
    void finalize(bool create_comm = true);

    void mult(ParVector* x, ParVector* b);
    void residual(ParVector* x, ParVector* b, ParVector* r);

    index_t global_num_rows;
    index_t global_num_cols;
    int local_nnz;
    int local_num_rows;
    int local_num_cols;
    index_t first_local_row;
    index_t first_local_col;
    Matrix* diag;
    Matrix* offd;
    std::vector<index_t> offd_column_map;
    int offd_num_cols;
    ParComm* comm;

};
}
#endif
