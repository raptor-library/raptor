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
    // TODO
    ParComm();

    // Assumes symmetry (SPD A)
    ParComm(Matrix* offd, std::vector<index_t> map_to_global, index_t* global_row_starts)
    {
        // Get MPI Information
        index_t rank, num_procs;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

        // Declare communication variables
        std::vector<index_t>    proc_cols;
        index_t                 proc;
        index_t                 old_proc;
        index_t                 global_col;
        index_t                 local_col;
        index_t                 first;
        index_t                 last;
        index_t                 num_rows;
        index_t                 num_cols;
        index_t                 row_start;
        index_t                 row_end;
        index_t*                ptr;
        index_t*                idx;

        num_cols = map_to_global.size();

        // If no off-diagonal block, return empty comm package
        if (num_cols == 0)
        {
            return;
        }

        // Create map from columns to processors they lie on
        proc = 0;
        for (index_t col = 0; col < num_cols; col++)
        {
            global_col = map_to_global[col];
            while (global_col >= global_row_starts[proc+1])
            {
                proc++;
            }
        }

        //Find inital proc (local col 0 lies on)
        global_col = map_to_global[0];
        proc = col_to_proc[0];
        proc_cols.push_back(0);

        // For each offd col, find proc it lies on.  Add proc and list
        // of columns it holds to map recvIndices
        last = 0;
        old_proc = col_to_proc[0];
        for (local_col = 0; local_col < num_cols; local_col++)
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
                send_procs.push_back(old_proc);
            }
            old_proc = proc;
        }
        // Add last processor to communicator
        first = last;
        std::vector<index_t> newvec(proc_cols.begin() + first, proc_cols.begin() + num_cols);
        recv_indices[old_proc] = newvec;
        recv_procs.push_back(old_proc);
        send_procs.push_back(old_proc);

        // Add processors needing to send to, and what to send to each
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
            }
            size_sends++;
        }

        //Store total number of values to be sent/received
        size_recvs = num_cols;

    }

    // TODO -- Does not assume square (P)
    ParComm(Matrix* offd, std::vector<index_t> map_to_global, index_t* globalRowStarts,
               index_t* possibleSendProcs)
    {

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
