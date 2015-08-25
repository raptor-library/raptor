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

//using namespace raptor;
using Eigen::VectorXd;

class ParMatrix
{
public:
    ParMatrix(index_t _glob_rows, index_t _glob_cols, Matrix<1>* _diag, Matrix<0>* _offd);
    ParMatrix(index_t _glob_rows, index_t _glob_cols, index_t _nnz, index_t* row_idx, index_t* col_idx,
             data_t* data, index_t* _global_row_starts, format_t format = CSR, int global_row_idx = 0, int symmetric = 1)
    {
        // Get MPI Information
        index_t rank, num_procs;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

        global_row_starts = _global_row_starts;

        //Declare matrix variables
        std::vector<Triplet>      diag_triplet(_nnz);
        std::vector<Triplet>      offd_triplet(_nnz);
        index_t                   diag_ctr;
        index_t                   offd_ctr;
        index_t                   last_col_diag;
        index_t                   row_start;
        index_t                   row_end;
        index_t                   global_col;

        // Initialize matrix dimensions
        global_rows = _glob_rows;
        global_cols = _glob_cols;
        local_nnz = _nnz;
        first_col_diag = global_row_starts[rank];
        local_rows = global_row_starts[rank+1] - first_col_diag;
        last_col_diag = first_col_diag + local_rows - 1;
        offd_num_cols = 0;

        // Split ParMat into diag and offd matrices
        if (global_row_idx && format == COO)
        {
            for (index_t i = 0; i < local_nnz; i++)
            {
                row_idx[i] -= first_col_diag;
            }
        }
    
        offd_ctr = 0;
        diag_ctr = 0;
        if (format == CSR)
        {
            // Assumes CSR Matrix
            for (index_t i = 0; i < local_rows; i++)
            {
                row_start = row_idx[i];
                row_end = row_idx[i+1];
                for (index_t j = row_start; j < row_end; j++)
                {
                    global_col = col_idx[j];
                    //In offd block
                    if (global_col < first_col_diag || global_col > last_col_diag)
                    {
                        if (global_to_local.count(global_col) == 0)
                        {
                            global_to_local[global_col] = offd_num_cols++;
                            local_to_global.push_back(global_col);
                        }
                        offd_triplet[offd_ctr++] = (Triplet(i, global_to_local[global_col], data[j]));
                    }
                    else //in diag block
                    {
                        diag_triplet[diag_ctr++] = (Triplet(i, global_col - first_col_diag, data[j]));
                    }
                }
            }
        }
        else if (format == COO)
        {
            // Assumes COO Matrix
            for (index_t i = 0; i < local_nnz; i++)
            {
                global_col = col_idx[i];
                //In offd block
                if (global_col < first_col_diag || global_col > last_col_diag)
                {
                    if (global_to_local.count(global_col) == 0)
                    {
                        global_to_local[global_col] = offd_num_cols++;
                        local_to_global.push_back(global_col);
                    }
                    offd_triplet[offd_ctr++] = (Triplet(row_idx[i], global_to_local[global_col], data[i]));
                }
                else //in diag block
                {
                    diag_triplet[diag_ctr++] = (Triplet(row_idx[i], global_col - first_col_diag, data[i]));
                }
            }
        }

        offd_nnz = offd_ctr;
        //offd_num_cols = 0;
        if (offd_nnz)
        {
            //Initialize off-diagonal-block matrix
            offd_triplet.resize(offd_nnz);
            offd = new CSC_Matrix(offd_triplet, local_rows, offd_num_cols);
            (offd->m)->makeCompressed();
        }

        //Initialize diagonal-block matrix
        diag = new CSR_Matrix(diag_triplet, local_rows, local_rows);
        (diag->m)->makeCompressed();

        //Initialize communication package
        comm = new ParComm(offd, local_to_global, global_to_local, global_row_starts, symmetric);


    }
    ParMatrix(ParMatrix* A);
    ~ParMatrix();

    index_t global_rows;
    index_t global_cols;
    index_t local_nnz;
    index_t local_rows;
    Matrix<1>* diag;
    Matrix<0>* offd;
    std::vector<index_t> local_to_global;
    std::map<index_t, index_t> global_to_local;
    index_t offd_num_cols;
    index_t first_col_diag;
    index_t offd_nnz;
    ParComm* comm;
    index_t* global_row_starts;

};
#endif
