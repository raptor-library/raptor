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
    ParMatrix(index_t _GlobRows, index_t _GlobCols, Matrix* _diag, Matrix* _offd);
    ParMatrix(index_t _GlobRows, index_t _GlobCols, index_t* ptr, index_t* idx,
             data_t* data, index_t* globalRowStarts)
    {
        index_t rank, numProcs;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        MPI_Comm_size(MPI_COMM_WORLD, &numProcs);

        this->globalRows = _GlobRows;
        this->globalCols = _GlobCols;

        firstColDiag = globalRowStarts[rank];
        this->localRows = globalRowStarts[rank+1] - firstColDiag;
        index_t lastColDiag = firstColDiag + localRows - 1;

        std::vector<index_t> diagI;
        std::vector<index_t> diagJ;
        std::vector<data_t> diagData;
        std::vector<index_t> offdI;
        std::vector<index_t> offdJ;
        std::vector<data_t> offdData;
        index_t diagCtr = 0;
        index_t offdCtr = 0;

        //Split ParMat into diag and offd matrices
        diagI.push_back(0);
        offdI.push_back(0);
        for (index_t i = 0; i < localRows; i++)
        {
            index_t rowStart = ptr[i];
            index_t rowEnd = ptr[i+1];
            for (index_t j = rowStart; j < rowEnd; j++)
            {
                index_t globalCol = idx[j];

                //In offd block
                if (globalCol < firstColDiag || globalCol > lastColDiag)
                {
                    offdCtr++;
                    offdJ.push_back(globalCol);
                    offdData.push_back(data[j]);
                }
                else //in diag block
                {
                    diagCtr++;
                    diagJ.push_back(globalCol - firstColDiag);
                    diagData.push_back(data[j]);
                }
            }
            diagI.push_back(diagCtr);
            offdI.push_back(offdCtr);
        }

        //Create localToGlobal map and convert offd
        // cols to local (currently global)
        index_t offdNNZ = offdJ.size();
        offdNumCols = 0;
        if (offdNNZ)
        {
            //New vector containing offdCols
            std::vector<index_t> offdCols;
            for (index_t i = 0; i < offdI[localRows]; i++)
            {
                offdCols.push_back(offdJ[i]);
            }

            //Sort columns
            std::sort(offdCols.begin(), offdCols.end());

            //Count columns with nonzeros in offd
            offdNumCols = 1;
            for (index_t i = 0; i < offdNNZ-1; i++)
            {
                if (offdCols[i+1] > offdCols[i])
                {
                    offdCols[offdNumCols++] = offdCols[i+1];
                }
            }

            //Map global columns to indices: 0 - offdNumCols
            for (index_t i = 0; i < offdNumCols; i++)
            {
                this->localToGlobal.push_back(offdCols[i]);
                this->globalToLocal[offdCols[i]] = i;
            }

            //Convert offd cols to local
            for (index_t i = 0; i < localRows; i++)
            {
                index_t rowStart = offdI[i];
                index_t rowEnd = offdI[i+1];
                for (index_t j = rowStart; j < rowEnd; j++)
                {
                    offdJ[j] = globalToLocal[offdJ[j]];
                }
            }

            //Initialize two matrices (offd and diag)
            offd = new CSR_Matrix(offdI.data(), offdJ.data(), offdData.data(),
                          localRows, offdNumCols, offdData.size());
            (offd->m)->makeCompressed();
        }

        diag = new CSR_Matrix(diagI.data(), diagJ.data(), diagData.data(),
                          localRows, localRows, diagData.size());
        (diag->m)->makeCompressed();

        //Initialize communication package
        comm = new ParComm(offd, localToGlobal, globalRowStarts);


    }
    ParMatrix(ParMatrix* A);
    ~ParMatrix();

    index_t globalRows;
    index_t globalCols;
    index_t localRows;
    Matrix* diag;
    Matrix* offd;
    std::vector<index_t> localToGlobal;
    std::map<index_t, index_t> globalToLocal;
    index_t offdNumCols;
    index_t firstColDiag;

    ParComm* comm;
};
#endif
