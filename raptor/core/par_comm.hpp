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
    ParComm(Matrix* offd, std::vector<index_t> mapToGlobal, index_t* globalRowStarts)
    {

        index_t rank, numProcs;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        MPI_Comm_size(MPI_COMM_WORLD, &numProcs);

        if (mapToGlobal.size() == 0)
        {
            return;
        }

        //Create map from columns to processors they lie on
        index_t proc = 0;
        index_t globalCol = 0;
        for (index_t col = 0; col < mapToGlobal.size(); col++)
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
        std::vector<index_t> procCols;
        proc = colToProc[0];
        procCols.push_back(0);

        //Initialize list of columns that must be sent/recvd
        index_t first = 0;
        index_t last = 0;

        //For each offd col, find proc it lies on.  Add proc and list
        // of columns it holds to map sendIndices / recvIndices (same here)
        index_t oldProc = colToProc[0];
        for (index_t col = 0; col < mapToGlobal.size(); col++)
        {
            proc = colToProc[col];
            if (proc != oldProc)
            {
                this->recvProcs.push_back(oldProc);
                this->sendProcs.push_back(oldProc);
                first = last;
                last = col;
                std::vector<index_t> newvec(procCols.begin() + first, procCols.begin() + last);
                this->recvIndices[oldProc] = newvec;
            }
            oldProc = proc;
        }
        this->recvProcs.push_back(oldProc);
        this->sendProcs.push_back(oldProc);
        first = last;
        std::vector<index_t> newvec(procCols.begin() + first, procCols.begin() + mapToGlobal.size());
        this->recvIndices[oldProc] = newvec;



        index_t* procList = (index_t*) calloc(numProcs, sizeof(index_t));

        index_t* ptr = (offd->m)->outerIndexPtr();
        index_t* idx = (offd->m)->innerIndexPtr();
        double* values = (offd->m)->valuePtr();
        index_t numRows = (offd->m)->outerSize();
        for (index_t i = 0; i < numRows; i++)
        {
            index_t rowStart = ptr[i];
            index_t rowEnd = ptr[i+1];
            if (rowStart == rowEnd) 
            {
                continue;
            }
            oldProc = colToProc[idx[rowStart]];
            for (index_t j = rowStart; j < rowEnd; j++)
            {
                index_t col = idx[j];
                proc = colToProc[col];
                if (proc != oldProc)
                {
                    if (sendIndices.count(oldProc))
                    {
                        sendIndices[oldProc].push_back(i);
                    }
                    else
                    {
                        std::vector<index_t> tmp;
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
                std::vector<index_t> tmp;
                tmp.push_back(numRows - 1);
                sendIndices[oldProc] = tmp;
            }
        }

        //Store total number of values to be sent/received
        this->sumSizeSends = mapToGlobal.size();
        this->sumSizeRecvs = mapToGlobal.size();

    }

    // TODO -- Does not assume square (P)
    ParComm(Matrix* offd, std::vector<index_t> mapToGlobal, index_t* globalRowStarts,
               index_t* possibleSendProcs)
    {

    }
    ~ParComm();

    index_t* globalRowStarts;
    index_t sumSizeSends;
    index_t sumSizeRecvs;
    std::map<index_t, std::vector<index_t>> sendIndices;
    std::map<index_t, std::vector<index_t>> recvIndices;
    std::vector<index_t> sendProcs;
    std::vector<index_t> recvProcs;
    std::vector<index_t> colToProc;
};
#endif
