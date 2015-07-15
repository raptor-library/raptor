// Copyright (c) 2015, Raptor Developer Team, University of Illinois at Urbana-Champaign
// License: Simplified BSD, http://opensource.org/licenses/BSD-2-Clause
#ifndef RAPTOR_CORE_PARCOMM_HPP
#define RAPTOR_CORE_PARCOMM_HPP

#include <mpi.h>
#include <math.h>
#include <Eigen/Dense>
using Eigen::VectorXd;

#include "matrix.hpp"
#include <map>

class ParComm
{
public:
    void init_col_to_proc(index_t num_procs, index_t num_cols, std::vector<index_t> map_to_global, index_t* global_row_starts)
    {
        index_t proc = 0;
        index_t global_col = 0;
        for (index_t col = 0; col < num_cols; col++)
        {
            global_col = map_to_global[col];
            while (global_col >= global_row_starts[proc+1])
            {
                proc++;
            }
            col_to_proc.push_back(proc);   
        }
    }

    void init_comm_recvs(index_t num_cols, std::vector<index_t> map_to_global)
    {
        std::vector<index_t>    proc_cols;
        index_t proc;
        index_t old_proc;
        index_t first;
        index_t last;

        //Find inital proc (local col 0 lies on)
        proc = col_to_proc[0];

        // For each offd col, find proc it lies on.  Add proc and list
        // of columns it holds to map recvIndices
        last = 0;
        old_proc = col_to_proc[0];
        for (index_t local_col = 0; local_col < num_cols; local_col++)
        {   
            proc_cols.push_back(local_col);
            proc = col_to_proc[local_col];
            // Column lies on new processor, so add last
            // processor to communicator
            if (proc != old_proc)
            {
                first = last;
                last = local_col;
                std::vector<index_t> newvec(proc_cols.begin() + first, proc_cols.begin() + last);
                recv_indices[old_proc] = newvec;
                recv_procs.push_back(old_proc);
            }
            old_proc = proc;
        }
        // Add last processor to communicator
        first = last;
        std::vector<index_t> newvec(proc_cols.begin() + first, proc_cols.begin() + num_cols);
        recv_indices[old_proc] = newvec;
        recv_procs.push_back(old_proc);

        //Store total number of values to be sent/received
        size_recvs = num_cols;
    }

    void init_comm_sends_sym(Matrix* offd)
    {
        index_t* ptr;
        index_t* idx;
        index_t num_rows;
        index_t row_start;
        index_t row_end;
        index_t proc;
        index_t old_proc;
        index_t local_col;

        // Get CSR Matrix variables
        ptr = (offd->m)->outerIndexPtr();
        idx = (offd->m)->innerIndexPtr();
        num_rows = (offd->m)->outerSize();
        size_sends = 0;
        for (index_t i = 0; i < num_rows; i++)
        {
            row_start = ptr[i];
            row_end = ptr[i+1];
            if (row_start == row_end) 
            {
                continue;
            }
            old_proc = col_to_proc[idx[row_start]];
            for (index_t j = row_start; j < row_end; j++)
            {
                local_col = idx[j];
                proc = col_to_proc[local_col];
                // Column lies on new processor, so add last
                // processor to communicator
                if (proc != old_proc)
                {
                    if (send_indices.count(old_proc))
                    {
                        send_indices[old_proc].push_back(i);   
                    }
                    else
                    {
                        std::vector<index_t> tmp;
                        tmp.push_back(i);
                        send_indices[old_proc] = tmp;
                        send_procs.push_back(old_proc);
                    }
                    size_sends++;
                }
                old_proc = proc;
            }
            // Add last processor to communicator
            if (send_indices.count(old_proc))
            {
                send_indices[old_proc].push_back(i);
            }
            else
            {
                std::vector<index_t> tmp;
                tmp.push_back(i);
                send_indices[old_proc] = tmp;
                send_procs.push_back(old_proc);
            }
            size_sends++;
        }
    }

    void init_comm_sends_unsym(Matrix* offd, index_t rank, std::vector<index_t> map_to_global, index_t* global_row_starts)
    {
        index_t recv_proc = 0;
        index_t orig_ctr = 0;
        index_t ctr = 0;
        index_t req_ctr = 0;
        index_t unsym_tag = 1212;
        index_t recv_size = recv_procs.size();
        size_sends = 0;

        index_t send_buffer[size_recvs];
        MPI_Request send_requests[recv_size];
        MPI_Status send_status[recv_size];

        //Send everything in recv_idx[recv_proc] to recv_proc;
        for (index_t i = 0; i < recv_size; i++)
        {
            orig_ctr = ctr;
            recv_proc = recv_procs[i];
            for (auto recv_idx : recv_indices[recv_proc])
            {
                send_buffer[ctr++] = map_to_global[recv_idx];
            }
            MPI_Isend(&send_buffer[orig_ctr], ctr - orig_ctr, MPI_INT, recv_proc, unsym_tag, MPI_COMM_WORLD, &send_requests[req_ctr++]);
        }

        //While a proc has send_requests unfinished, probe
        index_t finished_flag = 0;
        index_t avail_flag = 0;
        index_t count = 0;
        MPI_Status recv_status;
        MPI_Request finished_request;
        while (!finished_flag)
        {
            //Probe for messages, and recv any found
            MPI_Iprobe(MPI_ANY_SOURCE, unsym_tag, MPI_COMM_WORLD, &avail_flag, &recv_status);
            if (avail_flag)
            {
                MPI_Get_count(&recv_status, MPI_INT, &count);
                index_t recv_buffer[count];
                MPI_Recv(&recv_buffer, count, MPI_INT, MPI_ANY_SOURCE, unsym_tag, MPI_COMM_WORLD, &recv_status);
                for (int i = 0; i < count; i++) recv_buffer[i] = recv_buffer[i] - global_row_starts[rank];
                send_procs.push_back(recv_status.MPI_SOURCE);
                std::vector<index_t> send_idx(recv_buffer, recv_buffer + count);
                send_indices[recv_status.MPI_SOURCE] = send_idx;
                size_sends += send_idx.size();
            }

            //Check if sends have finished
            MPI_Testall(recv_size, send_requests, &finished_flag, send_status);  
        }
        MPI_Ibarrier(MPI_COMM_WORLD, &finished_request);
       
        finished_flag = 0;
        MPI_Test(&finished_request, &finished_flag, &recv_status);
        while(!finished_flag)
        {
            //Probe for messages, and recv any found
            MPI_Iprobe(MPI_ANY_SOURCE, unsym_tag, MPI_COMM_WORLD, &avail_flag, &recv_status);
            if (avail_flag)
            {
                MPI_Get_count(&recv_status, MPI_INT, &count);
                index_t recv_buffer[count];
                MPI_Recv(&recv_buffer, count, MPI_INT, MPI_ANY_SOURCE, unsym_tag, MPI_COMM_WORLD, &recv_status);
                for (int i = 0; i < count; i++) recv_buffer[i] = recv_buffer[i] - global_row_starts[rank];
                send_procs.push_back(recv_status.MPI_SOURCE);
                std::vector<index_t> send_idx(recv_buffer, recv_buffer + count);
                send_indices[recv_status.MPI_SOURCE] = send_idx;
                size_sends += send_idx.size();
            }
            MPI_Test(&finished_request, &finished_flag, &recv_status);
        }
    }

    // TODO
    ParComm();

    ParComm(Matrix* offd, std::vector<index_t> map_to_global, index_t* global_row_starts, index_t symmetric = 1)
    {
        // Get MPI Information
        index_t rank, num_procs;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

        // Declare communication variables
        index_t                 num_cols;

        num_cols = map_to_global.size();

        // If no off-diagonal block, return empty comm package
        if (num_cols == 0)
        {
            return;
        }

        // Create map from columns to processors they lie on
        init_col_to_proc(num_procs, num_cols, map_to_global, global_row_starts);

        // For each offd col, find proc it lies on.  Add proc and list
        // of columns it holds to map recvIndices
        init_comm_recvs(num_cols, map_to_global);

        // Add processors needing to send to, and what to send to each
        if (symmetric)
        {
            init_comm_sends_sym(offd);
        }
        else
        {
            init_comm_sends_unsym(offd, rank, map_to_global, global_row_starts);
        }
    }

    ~ParComm(){};

    index_t size_sends;
    index_t size_recvs;
    std::map<index_t, std::vector<index_t>> send_indices;
    std::map<index_t, std::vector<index_t>> recv_indices;
    std::vector<index_t> send_procs;
    std::vector<index_t> recv_procs;
    std::vector<index_t> col_to_proc;
};
#endif
