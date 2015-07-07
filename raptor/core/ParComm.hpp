// Copyright (c) 2015, Raptor Developer Team, University of Illinois at Urbana-Champaign
// License: Simplified BSD, http://opensource.org/licenses/BSD-2-Clause
#ifndef RAPTOR_CORE_PARCOMM_HPP
#define RAPTOR_CORE_PARCOMM_HPP

#include <mpi.h>
#include <math.h>
#include <Eigen/Dense>
using Eigen::VectorXd;

#include "Matrix.hpp"
#include <map>

class ParComm
{
public:
    // TODO
    ParComm();

    //Assumes symmetry (SPD A)
    ParComm(Matrix* offd, std::vector<int> mapToGlobal, int* globalRowStarts)
    {

        int rank, numProcs;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        MPI_Comm_size(MPI_COMM_WORLD, &numProcs);

        if (mapToGlobal.size() == 0)
        {
            return;
        }

        //Create map from columns to processors they lie on
        int proc = 0;
        int globalCol = 0;
        for (int col = 0; col < mapToGlobal.size(); col++)
        {
            globalCol = mapToGlobal[col];
            while (globalCol >= globalRowStarts[proc+1])
            {
                proc++;
            }
            colToProc.push_back(proc);   
        }

        //Find inital proc (local col 0 lies on)
        globalCol = mapToGlobal[0];
        std::vector<int> procCols;
        proc = colToProc[0];
        procCols.push_back(0);

        //Initialize list of columns that must be sent/recvd
        int first = 0;
        int last = 0;

        //For each offd col, find proc it lies on.  Add proc and list
        // of columns it holds to map sendIndices / recvIndices (same here)
        int oldProc = colToProc[0];
        for (int col = 0; col < mapToGlobal.size(); col++)
        {
            proc = colToProc[col];
            if (proc != oldProc)
            {
                this->recvProcs.push_back(oldProc);
                this->sendProcs.push_back(oldProc);
                first = last;
                last = col;
                std::vector<int> newvec(procCols.begin() + first, procCols.begin() + last);
                this->recvIndices[oldProc] = newvec;
            }
            oldProc = proc;
        }
        this->recvProcs.push_back(oldProc);
        this->sendProcs.push_back(oldProc);
        first = last;
        std::vector<int> newvec(procCols.begin() + first, procCols.begin() + mapToGlobal.size());
        this->recvIndices[oldProc] = newvec;



        int* procList = (int*) calloc(numProcs, sizeof(int));

        int* ptr = (offd->m)->outerIndexPtr();
        int* idx = (offd->m)->innerIndexPtr();
        double* values = (offd->m)->valuePtr();
        int numRows = (offd->m)->outerSize();
        for (int i = 0; i < numRows; i++)
        {
            int rowStart = ptr[i];
            int rowEnd = ptr[i+1];
            if (rowStart == rowEnd) 
            {
                continue;
            }
            oldProc = colToProc[idx[rowStart]];
            for (int j = rowStart; j < rowEnd; j++)
            {
                int col = idx[j];
                proc = colToProc[col];
                if (proc != oldProc)
                {
                    if (sendIndices.count(oldProc))
                    {
                        sendIndices[oldProc].push_back(i);
                    }
                    else
                    {
                        std::vector<int> tmp;
                        tmp.push_back(i);
                        sendIndices[oldProc] = tmp;
                    }
                }
                oldProc = proc;
            }
            if (sendIndices.count(oldProc))
            {
                sendIndices[oldProc].push_back(numRows - 1);
            }
            else
            {
                std::vector<int> tmp;
                tmp.push_back(numRows - 1);
                sendIndices[oldProc] = tmp;
            }
        }

        //Store total number of values to be sent/received
        this->sumSizeSends = mapToGlobal.size();
        this->sumSizeRecvs = mapToGlobal.size();

    }

    // TODO -- Does not assume square (P)
    ParComm(Matrix* offd, std::vector<int> mapToGlobal, int* globalRowStarts,
               int* possibleSendProcs)
    {

    }
    ~ParComm();

    int* globalRowStarts;
    int sumSizeSends;
    int sumSizeRecvs;
    std::map<int, std::vector<int>> sendIndices;
    std::map<int, std::vector<int>> recvIndices;
    std::vector<int> sendProcs;
    std::vector<int> recvProcs;
    std::vector<int> colToProc;
};
#endif
