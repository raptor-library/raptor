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
 ***** global_rows : index_t
 *****    Number of rows in the global parallel matrix
 ***** global_cols : index_t
 *****    Number of columns in the parallel matrix
 ***** local_nnz : index_t
 *****    Number of nonzeros stored locally
 ***** local_rows : index_t
 *****    Number of rows stored locally
 ***** local_cols : index_t 
 *****    Number of columns stored locally
 ***** diag : Matrix*
 *****    Matrix storing local diagonal block
 ***** offd : Matrix*
 *****    Matrix storing local off-diagonal block
 ***** local_to_global : std::vector<index_t>
 *****    std::vector that converts local columns in offd matrix to global
 ***** global_to_local : std::map<index_t, index_t>
 *****    Maps global columns to local columns in offd matrix
 ***** global_col_starts : std::vector<index_t>
 *****    std::vector of first column to be in the diagonal block of
 *****    each process
 ***** offd_num_cols : index_t
 *****    Number of columns in the off-diagonal matrix
 ***** first_col_diag : index_t
 *****    First column in the diagonal block 
 ***** first_row : index_t
 *****    First row of parallel matrix stored locally
 ***** comm : ParComm*
 *****    Parallel communicator for matrix
 ***** comm_mat : MPI_Comm
 *****    MPI_Communicator for containing active processors (all
 *****    processes that hold at least one row of the matrix)
 ***** 
 ***** Methods
 ***** -------
 ***** create_partition()
 *****    Partitions the global matrix across the processors
 ***** create_comm_mat()
 *****    Creates an MPI_Communicator for active processors
 *****    (those holding rows of the global matrix)
 ***** add_value()
 *****    Adds a value to a given row/columns.  Determines
 *****    if this value is in the diagonal or off-diagonal block
 ***** finalize()
 *****    Finalizes a matrix after values have been added.  
 *****    Converts the matrix to the appropriate format and 
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
    ***** to have num_cols equal to the local number of columns
    ***** in the diagonal block (this should later be adjusted)
    *****
    ***** Parameters
    ***** -------------
    ***** _glob_rows : index_t
    *****    Global number of rows in parallel matrix
    ***** _glob_cols : index_t
    *****    Global number of columns in parallel matrix
    ***** diag_f : format_t (optional)
    *****    Format of diagonal matrix (default CSR)
    ***** offd_f : format_t (optional)
    *****    Format of off-diagonal matrix (default CSC)
    ***** comm_mat : MPI_Comm (optional)
    *****    Communicator containing all processes that are
    *****    creating this matrix
    **************************************************************/
    ParMatrix(index_t _glob_rows, index_t _glob_cols, MPI_Comm _comm_mat = MPI_COMM_WORLD)
    {
        // Initialize matrix dimensions
        global_rows = _glob_rows;
        global_cols = _glob_cols;
        offd_num_cols = 0;

        // Create Partition
        create_partition(_comm_mat);

        diag = new COOMatrix(local_rows, local_cols, 5);
        offd = new COOMatrix(local_rows, local_cols, 5);
    }

    /**************************************************************
    *****   ParMatrix Class Constructor
    **************************************************************
    ***** Initializes an empty ParMatrix, setting the dimensions
    ***** to the passed values.  The local dimensions of the matrix
    ***** are passed as an argument, and the matrix is partitioned
    ***** according to the combination of all process's local dimensions
    *****
    ***** Parameters
    ***** -------------
    ***** _glob_rows : index_t
    *****    Global number of rows in parallel matrix
    ***** _glob_cols : index_t
    *****    Global number of columns in parallel matrix
    ***** _local_rows : index_t
    *****    Number of rows stored locally
    ***** _local_cols : index_t
    *****    Number of columns in the diagonal block
    ***** _first_row : index_t
    *****    Position of local matrix in global parallel matrix
    ***** _first_col_diag : index_t
    *****    First column of parallel matrix in diagonal block
    ***** diag_f : format_t (optional)
    *****    Format of diagonal matrix (default CSR)
    ***** offd_f : format_t (optional)
    *****    Format of off-diagonal matrix (default CSC)
    **************************************************************/
    ParMatrix(index_t _glob_rows, index_t _glob_cols, index_t _local_rows, 
            index_t _local_cols, index_t _first_row, index_t _first_col_diag, 
            MPI_Comm _comm_mat = MPI_COMM_WORLD, int nnz_per_row_diag = 5, 
            int nnz_per_row_offd = 5, format_t diag_f = CSR, format_t offd_f = CSC, 
            bool need_part = true)
    {
        global_rows = _glob_rows;
        global_cols = _glob_cols;
        local_rows = _local_rows;
        local_cols = _local_cols;
        first_row = _first_row;
        first_col_diag = _first_col_diag;
        offd_num_cols = 0;

        if (need_part)
            gather_partition(_comm_mat);

        diag = new COOMatrix(local_rows, local_cols, nnz_per_row_diag);
        diag = new COOMatrix(local_rows, local_cols, nnz_per_row_offd);
    }

    /**************************************************************
    *****   ParMatrix Class Constructor
    **************************************************************
    ***** Initializes a ParMatrix from a dense array of values
    *****
    ***** Parameters
    ***** -------------
    ***** _glob_rows : index_t
    *****    Global number of rows in parallel matrix
    ***** _glob_cols : index_t
    *****    Global number of columns in parallel matrix
    ***** values : data_t*
    *****    Pointer to dense list of values to be put into 
    *****    the parallel matrix (zeros will be ignored)
    ***** diag_f : format_t (optional)
    *****    Format of diagonal matrix (default CSR)
    ***** offd_f : format_t (optional)
    *****    Format of off-diagonal matrix (default CSC)
    **************************************************************/
    ParMatrix(index_t _glob_rows, index_t _glob_cols, data_t* values, MPI_Comm _comm_mat = MPI_COMM_WORLD)
    {
        global_rows = _glob_rows; 
        global_cols = _glob_cols;
        offd_num_cols = 0;
       
        // Create Partition
        create_partition(_comm_mat);

        // Initialize empty diag/offd matrices
        diag = new COOMatrix(local_rows, local_cols, 5);
        offd = new COOMatrix(local_rows, local_cols, 5);

        // Add values to diag/offd matrices
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

        // Convert diag/offd to compressed formats, create parallel communicator
        finalize();
    }
       
    /**************************************************************
    *****   ParMatrix Class Constructor
    **************************************************************
    ***** Copies an existing parallel matrix 
    ***** TODO -- how do we want to copy matrices?
    *****
    ***** Parameters
    ***** -------------
    ***** A : ParMatrix* 
    *****    Parallel matrix to be copied
    **************************************************************/
    ParMatrix(ParMatrix* A)
    {
        global_rows = A->global_rows;
        global_cols = A->global_cols;
        diag = A->diag; // should we mark as not owning? (we should think about move semantics or if people really love pointers we could use smart pointers).
        offd = A->offd;
    }

    /**************************************************************
    *****   ParMatrix Class Constructor
    **************************************************************
    ***** Initializes an empty ParMatrix, setting dimensions
    ***** to zero
    **************************************************************/
    ParMatrix()
    {
        local_rows = 0;
        local_cols = 0;
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
        if (this->offd_num_cols)
        {
            delete offd;
        }
        delete diag;
        delete comm;
    }

    /**************************************************************
    *****   ParMatrix Create Partition
    **************************************************************
    ***** Partitions the matrix evenly across all processes
    ***** in the MPI communicator
    *****
    ***** Parameters
    ***** -------------
    ***** _comm_mat : MPI_Comm (optional)
    *****    MPI Communicator containing all processes that will
    *****    call this method (default MPI_COMM_WORLD)
    **************************************************************/
    void create_partition(MPI_Comm _comm_mat = MPI_COMM_WORLD);

    /**************************************************************
    *****   ParMatrix Gather Partition
    **************************************************************
    ***** All processes gather the previously created partition
    *****
    ***** Parameters
    ***** -------------
    ***** _comm_mat : MPI_Comm (optional)
    *****    MPI Communicator containing all processes that will
    *****    call this method (default MPI_COMM_WORLD)
    **************************************************************/
    void gather_partition(MPI_Comm _comm_mat = MPI_COMM_WORLD);

    /**************************************************************
    *****   ParMatrix Create Comm Mat
    **************************************************************
    ***** Creates MPI communicator (on top of that passed as
    ***** an argument) containing all active procs, or those
    ***** that will hold at least one row of the matrix.
    *****
    ***** Parameters
    ***** -------------
    ***** _comm_mat : MPI_Comm (optional)
    *****    MPI Communicator containing all processes that will
    *****    call this method (default MPI_COMM_WORLD)
    **************************************************************/
    void create_comm_mat(MPI_Comm _comm_mat = MPI_COMM_WORLD);

    /**************************************************************
    *****   ParMatrix Add Value
    **************************************************************
    ***** Adds a value to the local portion of the parallel matrix,
    ***** determining whether it should be added to diagonal or 
    ***** off-diagonal block.  Local_to_global, global_to_local, and
    ***** offd_num_cols are appended as necessary
    *****
    ***** Parameters
    ***** -------------
    ***** row : index_t
    *****    Row of value (default as local row)
    ***** global_col : index_t 
    *****    Global column of value
    ***** value : data_t
    *****    Value to be added to parallel matrix
    ***** row_global : index_t (optional)
    *****    Determines if the row is a global value (default 0 -- local)
    **************************************************************/
    void add_value(index_t row, index_t global_col, data_t value);
    void add_global_value(int row, int global_col, double value);

    /**************************************************************
    *****   ParMatrix Finalize
    **************************************************************
    ***** Finalizes the diagonal and off-diagonal matrices.  Sorts
    ***** the local_to_global indices, and creates the parallel
    ***** communicator
    **************************************************************/
    void finalize(bool create_comm = true);


    index_t global_rows;
    index_t global_cols;
    index_t local_nnz;
    index_t local_rows;
    index_t local_cols;
    Matrix* diag;
    Matrix* offd;
    std::vector<index_t> local_to_global;
    std::vector<index_t> global_col_starts;
    std::map<index_t, index_t> global_to_local;
    std::set<int> global_column_set;
    index_t offd_num_cols;
    index_t first_col_diag;
    index_t first_row;
    ParComm* comm;
    MPI_Comm comm_mat;


};
}
#endif
