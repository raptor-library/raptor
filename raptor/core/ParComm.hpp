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

        int rank;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);

        if (mapToGlobal.size() == 0)
        {
            return;
        }

        //Find inital proc (local col 0 lies on)
        int proc = 0;
        int globalCol = mapToGlobal[0];
        std::vector<int> procCols;

        while (globalCol >= globalRowStarts[proc+1])
        {
            proc++;
        }
        this->sendProcs.push_back(proc);
        this->recvProcs.push_back(proc);
        procCols.push_back(0);

        //Initialize list of columns that must be sent/recvd
        int first = 0;
        int last = 0;

        //For each offd col, find proc it lies on.  Add proc and list
        // of columns it holds to map sendIndices / recvIndices (same here)
        for (int localCol = 1; localCol < mapToGlobal.size(); localCol++)
        {
            globalCol = mapToGlobal[localCol];
            procCols.push_back(localCol);

            // if globalCol lies on different proc than last
            // add to map, find new proc
            if (globalCol >= globalRowStarts[proc+1])
            {
                first = last;
                last = localCol;
                std::vector<int> newvec(procCols.begin() + first, procCols.begin() + last);
                this->sendIndices[proc] = newvec;
                this->recvIndices[proc] = newvec;



                while (globalCol >= globalRowStarts[proc+1])
                {
                    proc++;
                }
                this->sendProcs.push_back(proc);
                this->recvProcs.push_back(proc);
            }
        }

        //add last proc to map
        first = last;
        std::vector<int> newvec(procCols.begin() + first, procCols.begin() + mapToGlobal.size());
        this->sendIndices[proc] = newvec;
        this->recvIndices[proc] = newvec;

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
};
#endif
