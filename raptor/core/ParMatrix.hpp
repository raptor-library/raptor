// Copyright (c) 2015, Raptor Developer Team, University of Illinois at Urbana-Champaign
// License: Simplified BSD, http://opensource.org/licenses/BSD-2-Clause
#ifndef PARMATRIX_HPP
#define PARMATRIX_HPP

#include <mpi.h>
#include <math.h>
#include <Eigen/Dense>

#include "Matrix.hpp"

using Eigen::VectorXd;

class ParMatrix
{
public:
    ParMatrix(_GlobRows, _GlobCols, Matrix* _diag, Matrix* _offd);
    ParMatrix(_GlobRows, _GlobCols, std:vector<int> ptr, std:vector<int> idx,
              std:vector<double> data, int* globalRowStarts)
    {
        int myID = MPI::COMM_WORLD.Get_rank( );

        int firstColDiag = globalRowStarts[myID];
        int localNumRows = globalRowStarts[myID+1] - firstColDiag;
        int lastColDiag = firstColDiag + localNumRows - 1;

        std::vector<int> diagI;
        std::vector<int> diagJ;
        std::vector<double> diagData;
        std::vector<int> offdI;
        std::vector<int> offdJ;
        std::vector<double> offdData;
        int diagCtr = 0;
        int offdCtr = 0;

        //Split ParMat into diag and offd matrices
        diagI.push(0);
        offdI.push(0);
        for (int i = 0; i < localNumRows; i++)
        {
            int rowStart = ptr[i];
            int rowEnd = ptr[i+1];
            for (int j = rowStart; j < rowEnd; j++)
            {
                int globalCol = idx[j];

                //In offd block
                if (globalCol < firstColDiag || globalCol > lastColDiag)
                {
                    offdCtr++;
                    offdJ.push(globalCol);
                    offdData.push(data[j]);
                }
                else //in diag block
                {
                    diagCtr++;
                    diagJ.push(globalCol - firstColDiag);
                    diagData.push(data[j]);
                }
            }
            diagI.push(diagCtr);
            offdI.push(offdCtr);
        }

        //Create localToGlobal map and convert offd
        // cols to local (currently global)
        int offdNNZ = OffdJ.size();

        //New vector containing offdCols
        int* offdCols[offdNNZ];
        for (int i = 0; i < offdI[localNumRows+1]; i++)
        {
            offdCols[i] = offdJ[i]
        }

        //Sort columns
        qsort0(offdCols, 0, offdNNZ - 1);

        //Count columns with nonzeros in offd
        int offdNumCols = 1;
        for (int i = 0; i < offdNNZ-1; i++)
        {
            if (offdCols[i+1] > offdCols[i])
            {
                offdCols[offdNumCols++] = offdCols[i+1];
            }
        }

        //Map global columns to indices: 0 - offdNumCols
        for (int i = 0; i < offdNumCols; i++)
        {
            this.localToGlobal.push(offdCols[i]);
            this.globalToLocal[offdCols[i]] = i;
        }

        //Convert offd cols to local
        for (int i = 0; i < localNumRows; i++)
        {
            int rowStart = offdI[i];
            int rowEnd = offdI[i+1];
            for (int j = rowStart; j < rowEnd; j++)
            {
                offdJ[j] = globalToLocal[offdJ[j]];
            }
        }

        //Initialize two matrices (offd and diag)
        offd = CSR_MATRIX(offdI, offdJ, offdData,
                          localNumRows, offdNumCols);
        diag = CSR_MATRIX(diagI, diagJ, diagData,
                          localNumRows, localNumRows);

    }
    ParMatrix(ParMatrix* A);
    ~ParMatrix();

    int globalRows;
    int globalCols;
    Matrix* diag;
    Matrix* offd;
std:vector<int> localToGlobal;
    map<int, int> globalToLocal;
};
#endif
