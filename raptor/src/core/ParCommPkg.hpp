#ifndef PARCOMMPKG_HPP
#define PARCOMMPKG_HPP


#include <mpi.h>
#include <math.h>
#include <Eigen/Dense>
using Eigen::VectorXd;

#include "Matrix.hpp"
#include <map>

class ParMatrix
{
    public:
        //Assumes symmetry (SPD A)
        ParCommPkg(Matrix* offd, std:vector<int> mapToGlobal, int* globalRowStarts)
        {
            if (mapToGlobal.size() == 0)
                return;

            //Find inital proc (local col 0 lies on)
            int proc = 0;
            int globalCol = mapToGlobal[0];
            while (globalCol >= globalRowStarts[proc+1]) proc++;
            this.sendProcs.push(proc);
            this.recvProcs.push(proc);

            //Initialize list of columns that must be sent/recvd
            std:vector<int> procCols;
            std:vector<int>::const_iterator first = 0;
            std:vector<int>::const_iterator last = procCols.begin();

            //For each offd col, find proc it lies on.  Add proc and list 
            // of columns it holds to map sendIndices / recvIndices (same here)
            for (int localCol = 1; localCol < mapToGlobal.size(); localCol++)
            {
                globalCol = mapToGlobal[localCol];
                procCols.push(localCol);

                // if globalCol lies on different proc than last
                // add to map, find new proc
                if (globalCol >= globalRowStarts[proc+1])
                {
                    first = last;
                    last = procCols.begin() + localCol + 1;
                    std:vector<int> newvec(first, last);
                    this.sendIndices[proc] = newvec;
                    this.recvIndices[proc] = newvec;
                    
                    while (globalCol >= globalRowStarts[proc+1]) proc++;
                    this.sendProcs.push(proc);
                    this.recvProcs.push(proc);
                }
            }

            //add last proc to map
            first = last;
            last = procCols.begin() + localCol + 1;
            std:vector<int> newvec(first, last);
            this.sendIndices[proc] = newvec;
            this.recvIndices[proc] = newvec;
        }

        //Does not assume square (P)
        ParCommPkg(Matrix* offd, std:vector<int> mapToGlobal, int* globalRowStarts, int* possibleSendProcs)
        {

        }
        ~ParCommPkg();

        std:vector<int> getSendIndicies(int proc)
        {
            return sendIndices[proc];
        }

        std:vector<int> getRecvIndicies(int proc)
        {
            return recvIndices[proc];
        }
    
        std:vector<int> getSendProcs()
        {
            return sendProcs;
        }

        std:vector<int> getRecvProcs()
        {
            return recvProcs;
        }

    private:
        int* globalRowStarts;
        map<int, std:vector<int>> sendIndices;
        map<int, std:vector<int>> recvIndices;
        std:vector<int> sendProcs;
        std:vector<int> recvProcs;
};
#endif
