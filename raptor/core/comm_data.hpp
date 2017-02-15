// Copyright (c) 2015, Raptor Developer Team, University of Illinois at Urbana-Champaign
// License: Simplified BSD, http://opensource.org/licenses/BSD-2-Clause
#ifndef RAPTOR_CORE_COMMDATA_HPP
#define RAPTOR_CORE_COMMDATA_HPP

#include <mpi.h>
#include <math.h>

#include "matrix.hpp"
#include "par_vector.hpp"
#include <map>

/**************************************************************
 *****   CommData Class
 **************************************************************
 **************************************************************/
namespace raptor
{
class CommData
{
public:
    /**************************************************************
    *****   CommData Class Constructor
    **************************************************************
    ***** Initializes an empty CommData, setting number and size of 
    ***** messages to zero.

    **************************************************************/
    CommData()
    {
        num_msgs = 0;
        size_msgs = 0;
        indptr.push_back(0);
    }

    /**************************************************************
    *****   ParComm Class Destructor
    **************************************************************
    ***** 
    **************************************************************/
    ~CommData()
    {
        if (num_msgs)
        {
            delete[] requests;
        }
        if (size_msgs)
        {
            delete[] buffer;
        }
    };

    void add_msg(int proc,
            int msg_size,
            int* msg_indices)
    {
        int last_ptr = indptr[num_msgs];
        procs.push_back(proc);
        indptr.push_back(last_ptr + msg_size);
        for (int i = 0; i < msg_size; i++)
        {
            indices.push_back(msg_indices[i]);
        }

        num_msgs++;
        size_msgs += msg_size;
    }

    void add_msg(int proc,
            int msg_size)
    {
        int last_ptr = indptr[num_msgs];
        int idx_start = indices[size_msgs-1];
        procs.push_back(proc);
        indptr.push_back(last_ptr + msg_size);

        for (int i = 0; i < msg_size; i++)
        {
            indices.push_back(idx_start + i + 1);
        }

        num_msgs++;
        size_msgs += msg_size;
    }

    void finalize()
    {
        if (num_msgs)
        {
            requests = new MPI_Request[num_msgs];
        }
        if (size_msgs)
        {
            buffer = new data_t[size_msgs];
        }
    }

    int num_msgs;
    int size_msgs;
    std::vector<int> procs;
    std::vector<int> indptr;
    std::vector<int> indices;
    MPI_Request* requests;
    data_t* buffer;
};
}
#endif

