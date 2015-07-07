// Copyright (c) 2015, Raptor Developer Team, University of Illinois at Urbana-Champaign
// License: Simplified BSD, http://opensource.org/licenses/BSD-2-Clause
#ifndef PARMATRIX_HPP
#define PARMATRIX_HPP

#include <mpi.h>
#include <math.h>
#include <Eigen/Dense>

#include "Matrix.hpp"
#include "ParComm.hpp"
#include "Types.hpp"

//using namespace raptor;
using Eigen::VectorXd;

class ParMatrix
{
public:
    ParMatrix(index_t _glob_rows, index_t _glob_cols, Matrix* _diag, Matrix* _offd);
    ParMatrix(index_t _glob_rows, index_t _glob_cols, index_t* ptr, index_t* idx,
             data_t* data, index_t* global_row_starts)
    {
        // Get MPI Information
        index_t rank, num_procs;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

        this->global_rows = _glob_rows;
        this->global_cols = _glob_cols;

        first_col_diag = global_row_starts[rank];
        this->local_rows = global_row_starts[rank+1] - first_col_diag;
        index_t last_col_diag = first_col_diag + local_rows - 1;

        std::vector<index_t> diag_i;
        std::vector<index_t> diag_j;
        std::vector<data_t> diag_data;
        std::vector<index_t> offd_i;
        std::vector<index_t> offd_j;
        std::vector<data_t> offd_data;
        index_t diag_ctr = 0;
        index_t offd_ctr = 0;

        //Split ParMat into diag and offd matrices
        diag_i.push_back(0);
        offd_i.push_back(0);
        for (index_t i = 0; i < local_rows; i++)
        {
            index_t row_start = ptr[i];
            index_t row_end = ptr[i+1];
            for (index_t j = row_start; j < row_end; j++)
            {
                index_t global_col = idx[j];

                //In offd block
                if (global_col < first_col_diag || global_col > last_col_diag)
                {
                    offd_ctr++;
                    offd_j.push_back(global_col);
                    offd_data.push_back(data[j]);
                }
                else //in diag block
                {
                    diag_ctr++;
                    diag_j.push_back(global_col - first_col_diag);
                    diag_data.push_back(data[j]);
                }
            }
            diag_i.push_back(diag_ctr);
            offd_i.push_back(offd_ctr);
        }

        //Create localToGlobal map and convert offd
        // cols to local (currently global)
        index_t offd_nnz = offd_j.size();
        offd_num_cols = 0;
        if (offd_nnz)
        {
            //New vector containing offdCols
            std::vector<index_t> offd_cols;
            for (index_t i = 0; i < offd_i[local_rows]; i++)
            {
                offd_cols.push_back(offd_j[i]);
            }

            //Sort columns
            std::sort(offd_cols.begin(), offd_cols.end());

            //Count columns with nonzeros in offd
            offd_num_cols = 1;
            for (index_t i = 0; i < offd_nnz-1; i++)
            {
                if (offd_cols[i+1] > offd_cols[i])
                {
                    offd_cols[offd_num_cols++] = offd_cols[i+1];
                }
            }

            //Map global columns to indices: 0 - offdNumCols
            for (index_t i = 0; i < offd_num_cols; i++)
            {
                this->local_to_global.push_back(offd_cols[i]);
                this->global_to_local[offd_cols[i]] = i;
            }

            //Convert offd cols to local
            for (index_t i = 0; i < local_rows; i++)
            {
                index_t row_start = offd_i[i];
                index_t row_end = offd_i[i+1];
                for (index_t j = row_start; j < row_end; j++)
                {
                    offd_j[j] = global_to_local[offd_j[j]];
                }
            }

            //Initialize two matrices (offd and diag)
            offd = new CSR_Matrix(offd_i.data(), offd_j.data(), offd_data.data(),
                          local_rows, offd_num_cols, offd_data.size());
            (offd->m)->makeCompressed();
        }

        diag = new CSR_Matrix(diag_i.data(), diag_j.data(), diag_data.data(),
                          local_rows, local_rows, diag_data.size());
        (diag->m)->makeCompressed();

        //Initialize communication package
        comm = new ParComm(offd, local_to_global, global_row_starts);


    }
    ParMatrix(ParMatrix* A);
    ~ParMatrix();

    index_t global_rows;
    index_t global_cols;
    index_t local_rows;
    Matrix* diag;
    Matrix* offd;
    std::vector<index_t> local_to_global;
    std::map<index_t, index_t> global_to_local;
    index_t offd_num_cols;
    index_t first_col_diag;
    ParComm* comm;
};
#endif
