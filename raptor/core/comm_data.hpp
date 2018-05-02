// Copyright (c) 2015-2017, RAPtor Developer Team
// License: Simplified BSD, http://opensource.org/licenses/BSD-2-Clause
#ifndef RAPTOR_CORE_COMMDATA_HPP
#define RAPTOR_CORE_COMMDATA_HPP

#define WITH_MPI 1

#include <mpi.h>
#include "types.hpp"
#include "vector.hpp"

/**************************************************************
 *****   CommData Class
 **************************************************************
 **************************************************************/
namespace raptor
{
    // Forward Declaration
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

        vector_data = (OpData) {0, 0};
        matrix_data = (OpData) {0, 0};
    }

    CommData(CommData* data)
    {
        num_msgs = data->num_msgs;
        size_msgs = data->size_msgs;
        int n_procs = data->procs.size();
        int n_indptr = data->indptr.size();
        int n_indices = data->indices.size();
        if (n_procs)
        {
            procs.resize(n_procs);
            for (int i = 0; i < n_procs; i++)
            {
                procs[i] = data->procs[i];
            }
        }

        if (n_indptr)
        {
            indptr.resize(n_indptr);
            for (int i = 0; i < n_indptr; i++)
            {
                indptr[i] = data->indptr[i];
            }
        }

        if (n_indices)
        {
            indices.resize(n_indices);
            for (int i = 0; i < n_indices; i++)
            {
                indices[i] = data->indices[i];
            }
        }

        if (num_msgs)
        {
            requests.resize(num_msgs);
        }

        if (size_msgs)
        {
            buffer.resize(size_msgs);
            int_buffer.resize(size_msgs);
        }

        vector_data = (OpData) {0, 0};
        matrix_data = (OpData) {0, 0};
    }

    /**************************************************************
    *****   ParComm Class Destructor
    **************************************************************
    ***** 
    **************************************************************/
    ~CommData()
    {
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
        procs.push_back(proc);
        indptr.push_back(last_ptr + msg_size);

        num_msgs++;
        size_msgs += msg_size;
    }

    void finalize()
    {
        if (num_msgs)
        {
            requests.resize(num_msgs);
            buffer.resize(size_msgs);
            int_buffer.resize(size_msgs);
        }
    }

    void reset_data()
    {
        vector_data.num_msgs = 0;
        vector_data.size_msgs = 0;
        matrix_data.num_msgs = 0;
        matrix_data.size_msgs = 0;
    }

    void print_data(bool vec)
    {
        int n, s;
        if (vec)
        {
            MPI_Reduce(&(vector_data.num_msgs), &n, 1, MPI_INT, MPI_MAX, 0, MPI_COMM_WORLD);
            MPI_Reduce(&(vector_data.size_msgs), &s, 1, MPI_INT, MPI_MAX, 0, MPI_COMM_WORLD);
        }
        else
        {
            MPI_Reduce(&(matrix_data.num_msgs), &n, 1, MPI_INT, MPI_MAX, 0, MPI_COMM_WORLD);
            MPI_Reduce(&(matrix_data.size_msgs), &s, 1, MPI_INT, MPI_MAX, 0, MPI_COMM_WORLD);
        }
        int rank;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        if (rank == 0)
            printf("Num Msgs: %d, Size Msgs: %d\n", n, s);
    }

    template<typename T>
    aligned_vector<T>& get_buffer();

    int num_msgs;
    int size_msgs;
    aligned_vector<int> procs;
    aligned_vector<int> indptr;
    aligned_vector<int> indices;
    aligned_vector<int> indptr_T;
    aligned_vector<MPI_Request> requests;
    aligned_vector<double> buffer;
    aligned_vector<int> int_buffer;

    struct OpData
    {
        int num_msgs;
        int size_msgs;
    };

    OpData vector_data;
    OpData matrix_data;

};
}
#endif

