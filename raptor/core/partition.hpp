// Copyright (c) 2015-2017, RAPtor Developer Team
// License: Simplified BSD, http://opensource.org/licenses/BSD-2-Clause
#ifndef PARTITION_HPP 
#define PARTITION_HPP

#include <mpi.h>
#include <math.h>
#include <set>

#include "types.hpp"
#include "topology.hpp"

#define STANDARD_PPN 4
#define STANDARD_PROC_LAYOUT 1

/**************************************************************
 *****   Partition Class
 **************************************************************
 ***** This class holds the partition of a number of vertices 
 ***** across a number of processes
 *****
 ***** Attributes
 ***** -------------
 ***** global_num_indices : index_t
 *****    Number of rows to be partitioned
 ***** first_local_idx : index_t
 *****    First global index of a row in partition local to rank
 ***** local_num_indices : index_t
 *****    Number of rows local to rank's partition
 *****
 ***** Methods
 ***** ---------
 **************************************************************/
namespace raptor
{
  class Partition
  {
  public:
    Partition(index_t _global_num_rows, index_t _global_num_cols,
            Topology* _topology = NULL)
    {
        int rank, num_procs;
        int avg_num;
        int extra;

        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

        global_num_rows = _global_num_rows;
        global_num_cols = _global_num_cols;

        // Partition rows across processes
        avg_num = global_num_rows / num_procs;
        extra = global_num_rows % num_procs;
        first_local_row = avg_num * rank;
        local_num_rows = avg_num;
        if (extra > rank)
        {
            first_local_row += rank;
            local_num_rows++;
        }
        else
        {
            first_local_row += extra;
        }

        // Partition cols across processes
        if (global_num_rows < num_procs)
        {
            num_procs = global_num_rows;
        }
        avg_num = global_num_cols / num_procs;
        extra = global_num_cols % num_procs;
        if (local_num_rows)
        {
            first_local_col = avg_num * rank;
            local_num_cols = avg_num;
            if (extra > rank)
            {
                first_local_col += rank;
                local_num_cols++;
            }
            else
            {
                first_local_col += extra;
            }
        }
        else
        {
            local_num_cols = 0;
        }

        last_local_row = first_local_row + local_num_rows - 1;
        last_local_col = first_local_col + local_num_cols - 1;

        num_shared = 0;

        create_assumed_partition();

        if (_topology == NULL)
        {
            topology = new Topology();
        }
        else
        {
            topology = _topology;
            topology->num_shared++;
        }
    }

    Partition(index_t _global_num_rows, index_t _global_num_cols,
            index_t _brows, index_t _bcols, Topology* _topology = NULL)
    {
        int rank, num_procs;
        int avg_num_blocks, global_num_row_blocks, global_num_col_blocks;
        int extra;

        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

        global_num_rows = _global_num_rows;
        global_num_cols = _global_num_cols;

        // Partition rows across processes
	global_num_row_blocks = global_num_rows / _brows;
        avg_num_blocks = global_num_row_blocks / num_procs;
        extra = global_num_row_blocks % num_procs;
        first_local_row = avg_num_blocks * rank * _brows;
        local_num_rows = avg_num_blocks * _brows;
        if (extra > rank)
        {
            first_local_row += rank * _brows;
            local_num_rows += _brows;
        }
        else
        {
            first_local_row += extra * _brows;
        }

        // Partition cols across processes
	// local_num_cols = number of cols in on_proc matrix
        if (global_num_row_blocks < num_procs)
        {
            num_procs = global_num_row_blocks;
        }

	global_num_col_blocks = global_num_cols / _bcols;
        avg_num_blocks = global_num_col_blocks / num_procs;
        extra = global_num_col_blocks % num_procs;
        if (local_num_rows)
        {
            first_local_col = avg_num_blocks * rank * _bcols;
            local_num_cols = avg_num_blocks * _bcols;
            if (extra > rank)
            {
                first_local_col += rank * _bcols;
                local_num_cols += _bcols;
            }
            else
            {
                first_local_col += extra * _bcols;
            }
        }
        else
        {
            local_num_cols = 0;
        }

        last_local_row = first_local_row + local_num_rows - 1;
        last_local_col = first_local_col + local_num_cols - 1;

        num_shared = 0;

        create_assumed_partition();

        if (_topology == NULL)
        {
            topology = new Topology();
        }
        else
        {
            topology = _topology;
            topology->num_shared++;
        }
    }

    Partition(index_t _global_num_rows, index_t _global_num_cols,
            int _local_num_rows, int _local_num_cols,
            index_t _first_local_row, index_t _first_local_col,
            Topology* _topology = NULL)
    {
        int rank, num_procs;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

        global_num_rows = _global_num_rows;
        global_num_cols = _global_num_cols;
        local_num_rows = _local_num_rows;
        local_num_cols = _local_num_cols;
        first_local_row = _first_local_row;
        first_local_col = _first_local_col;
        last_local_row = first_local_row + local_num_rows - 1;
        last_local_col = first_local_col + local_num_cols - 1;

        num_shared = 0;

        create_assumed_partition();

        if (_topology == NULL)
        {
            topology = new Topology();
        }
        else
        {
            topology = _topology;
            topology->num_shared++;
        }
    }

    Partition(Topology* _topology = NULL)
    {
        if (_topology == NULL)
        {
            topology = new Topology();
        }
        else
        {
            topology = _topology;
            topology->num_shared++;
        }
    }

    Partition(Partition* A, Partition* B)
    {
        global_num_rows = A->global_num_rows;
        global_num_cols = B->global_num_cols;
        local_num_rows = A->local_num_rows;
        local_num_cols = B->local_num_cols;
        first_local_row = A->first_local_row;
        first_local_col = B->first_local_col;
        last_local_row = A->last_local_row;
        last_local_col = B->last_local_col;

        num_shared = 0;

        create_assumed_partition();

        topology = A->topology;
        topology->num_shared++;
    }

    ~Partition()
    {
        num_shared = 0;
        global_num_rows = 0;
        global_num_cols = 0;
        local_num_rows = 0;
        local_num_cols = 0;
        first_local_row = 0;
        first_local_col = 0;
        last_local_row = 0;
        last_local_col = 0;
        assumed_first_col = 0;
        assumed_last_col = 0;
        assumed_num_cols = 0;

        if (topology->num_shared)
        {
            topology->num_shared--;
        }
        else
        {
            delete topology;
        }
    }

    Partition* transpose()
    {
        Partition* part = new Partition(global_num_cols, global_num_rows,
                local_num_cols, local_num_rows, first_local_col,
                first_local_row, topology);
        return part;
    }

    void create_assumed_partition()
    {
        // Get MPI Information
        int rank, num_procs;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
        
        int proc, num_procs_extra;
        int ctr, tmp;
        int recvbuf;
        MPI_Status status;
        aligned_vector<int> send_buffer;
        aligned_vector<MPI_Request> send_requests;

        assumed_num_cols = global_num_cols / num_procs;
        if (global_num_cols % num_procs) assumed_num_cols++;

        // Find assumed first and last col on rank
        assumed_first_col = rank * assumed_num_cols;
        assumed_last_col = (rank + 1) * assumed_num_cols;
            
        // If last_local_col != last, exchange data with neighbors
        // Send to proc that is assumed to hold my last local col
        // and recv from all procs of which i hold their assumed last local cols
        proc = last_local_col / assumed_num_cols; // first col of rank + 1
        num_procs_extra = proc - rank; 
        ctr = 0;
        if (num_procs_extra > 0)
        {
            send_buffer.resize(num_procs_extra);
            send_requests.resize(num_procs_extra);
            for (int i = rank + 1; i <= proc; i++)
            {
                send_buffer[ctr] = first_local_col;
                MPI_Isend(&(send_buffer[ctr]), 1, MPI_INT, i, 2345, 
                        MPI_COMM_WORLD, &(send_requests[ctr]));
                ctr++;
            }
        }
        tmp = first_local_col;
        proc = rank - 1;
        assumed_col_ptr.emplace_back(tmp);
        while (assumed_first_col < tmp)
        {
            MPI_Recv(&recvbuf, 1, MPI_INT, proc, 2345, MPI_COMM_WORLD, &status);
            tmp = recvbuf;
            assumed_col_ptr.emplace_back(tmp);
            proc--;
        }
        if (ctr)
        {
            MPI_Waitall(ctr, send_requests.data(), MPI_STATUSES_IGNORE);
        }

        // Reverse order of col_starts (lowest proc first)
        std::reverse(assumed_col_ptr.begin(), assumed_col_ptr.end());
        for (int i = assumed_col_ptr.size() - 1; i >= 0; i--)
        {
            assumed_col_procs.emplace_back(rank - i);
        }

        // If first_local_col != first, exchange data with neighbors
        // Send to proc that is assumed to hold my first local col
        // and recv from all procs of which i hold their assumed first local cols
        proc = first_local_col / assumed_num_cols;
        num_procs_extra = rank - proc;
        ctr = 0;
        if (num_procs_extra > 0)
        {
            send_buffer.resize(num_procs_extra);
            send_requests.resize(num_procs_extra);
            for (int i = proc; i < rank; i++)
            {
                send_buffer[ctr] = last_local_col + 1;
                MPI_Isend(&(send_buffer[ctr]), 1, MPI_INT, i, 2345,
                        MPI_COMM_WORLD, &(send_requests[ctr]));
                ctr++;
            }
        }
        tmp = last_local_col + 1;
        proc = rank + 1;
        assumed_col_ptr.emplace_back(last_local_col + 1);
        assumed_col_procs.emplace_back(proc);
        while (assumed_last_col > tmp && proc < num_procs)
        {
            MPI_Recv(&recvbuf, 1, MPI_INT, proc, 2345, MPI_COMM_WORLD, &status);
            tmp = recvbuf;
            assumed_col_ptr.emplace_back(recvbuf);
            assumed_col_procs.emplace_back(++proc);
        }
        if (ctr)
        {
            MPI_Waitall(ctr, send_requests.data(), MPI_STATUSES_IGNORE);
        }
    }

    void form_col_to_proc (const aligned_vector<int>& off_proc_column_map,
            aligned_vector<int>& off_proc_col_to_proc) const
    {
        int rank;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);

        int off_proc_num_cols = off_proc_column_map.size();
        int assumed_col_ptr_size = assumed_col_ptr.size();
        int prev_proc;
        int num_sends = 0;
        int proc, start, end;
        int col, count;
        int finished, msg_avail;

        aligned_vector<int> send_procs;
        aligned_vector<int> send_proc_starts;
        aligned_vector<int> sendbuf_procs;
        aligned_vector<int> sendbuf_starts;
        aligned_vector<int> sendbuf;
        aligned_vector<MPI_Request> send_requests;
        MPI_Request barrier_request;
        MPI_Status status;

        if (off_proc_num_cols)
        {
            off_proc_col_to_proc.resize(off_proc_num_cols);
            prev_proc = off_proc_column_map[0] / assumed_num_cols;
            send_procs.emplace_back(prev_proc);
            send_proc_starts.emplace_back(0);
            for (int i = 1; i < off_proc_num_cols; i++)
            {
                proc = off_proc_column_map[i] / assumed_num_cols;
                if (proc != prev_proc)
                {
                    send_procs.emplace_back(proc);
                    send_proc_starts.emplace_back(i);
                    prev_proc = proc;
                }
            }
            send_proc_starts.emplace_back(off_proc_num_cols);

            num_sends = send_procs.size();
            send_requests.resize(num_sends);
            for (int i = 0; i < num_sends; i++)
            {
                proc = send_procs[i];
                start = send_proc_starts[i];
                end = send_proc_starts[i+1];

                if (proc == rank)
                {
                    int k = 0;
                    for (int j = start; j < end; j++)
                    {
                        col = off_proc_column_map[j];
                        while (k + 1 < assumed_col_ptr_size && 
                                col >= assumed_col_ptr[k+1])
                        {
                            k++;
                        }
                        proc = assumed_col_procs[k];
                        off_proc_col_to_proc[j] = proc;
                    }
                    send_requests[i] = MPI_REQUEST_NULL;
                }
                else
                {
                    MPI_Issend(&(off_proc_column_map[start]), end - start, MPI_INT,
                            proc, 9753, MPI_COMM_WORLD, &(send_requests[i]));
                }
            }
        }

        if (num_sends)
        {
            MPI_Testall(num_sends, send_requests.data(), &finished, MPI_STATUSES_IGNORE);
            while (!finished)
            {
                MPI_Iprobe(MPI_ANY_SOURCE, 9753, MPI_COMM_WORLD, &msg_avail, &status);
                if (msg_avail)
                {
                    MPI_Get_count(&status, MPI_INT, &count);
                    proc = status.MPI_SOURCE;
                    int recvbuf[count];
                    MPI_Recv(recvbuf, count, MPI_INT, proc, 9753, 
                            MPI_COMM_WORLD, &status);
                    sendbuf_procs.emplace_back(proc);
                    sendbuf_starts.emplace_back(sendbuf.size());
                    int k = 0;
                    for (int i = 0; i < count; i++)
                    {
                        col = recvbuf[i];
                        while (k + 1 < assumed_col_ptr_size &&
                                col >= assumed_col_ptr[k + 1])
                        {
                            k++;
                        }
                        proc = assumed_col_procs[k];
                        sendbuf.emplace_back(proc);    
                    }
                }
                MPI_Testall(num_sends, send_requests.data(), &finished, MPI_STATUSES_IGNORE);
            }
        }
        MPI_Ibarrier(MPI_COMM_WORLD, &barrier_request);
        MPI_Test(&barrier_request, &finished, MPI_STATUS_IGNORE);
        while (!finished)
        {
            MPI_Iprobe(MPI_ANY_SOURCE, 9753, MPI_COMM_WORLD, &msg_avail, &status);
            if (msg_avail)
            {
                MPI_Get_count(&status, MPI_INT, &count);
                proc = status.MPI_SOURCE;
                int recvbuf[count];
                MPI_Recv(recvbuf, count, MPI_INT, proc, 9753, 
                        MPI_COMM_WORLD, &status);
                sendbuf_procs.emplace_back(proc);
                sendbuf_starts.emplace_back(sendbuf.size());
                int k = 0;
                for (int i = 0; i < count; i++)
                {
                    col = recvbuf[i];
                    while (k + 1 < assumed_col_ptr_size &&
                            col >= assumed_col_ptr[k + 1])
                    {
                        k++;
                    }
                    proc = assumed_col_procs[k];
                    sendbuf.emplace_back(proc);
                }
            }
            MPI_Test(&barrier_request, &finished, MPI_STATUS_IGNORE);
        }
        sendbuf_starts.emplace_back(sendbuf.size());

        int n_sendbuf = sendbuf_procs.size();
        aligned_vector<MPI_Request> sendbuf_requests(n_sendbuf);
        for (int i = 0; i < n_sendbuf; i++)
        {
            int proc = sendbuf_procs[i];
            int start = sendbuf_starts[i];
            int end = sendbuf_starts[i+1];
            MPI_Isend(&(sendbuf[start]), end-start, MPI_INT, proc,
                    8642, MPI_COMM_WORLD, &(sendbuf_requests[i]));
        }
            
        for (int i = 0; i < num_sends; i++)
        {
            int proc = send_procs[i];
            if (proc == rank) 
                continue;
            int start = send_proc_starts[i];
            int end = send_proc_starts[i+1];
            MPI_Irecv(&(off_proc_col_to_proc[start]), end - start, MPI_INT, proc,
                    8642, MPI_COMM_WORLD, &(send_requests[i]));
        }

        MPI_Waitall(n_sendbuf, sendbuf_requests.data(), MPI_STATUSES_IGNORE);
        MPI_Waitall(num_sends, send_requests.data(), MPI_STATUSES_IGNORE);
    }

    index_t global_num_rows;
    index_t global_num_cols;
    int local_num_rows;
    int local_num_cols;
    index_t first_local_row;
    index_t first_local_col;
    index_t last_local_row;
    index_t last_local_col;

    index_t assumed_first_col;
    index_t assumed_last_col;
    int assumed_num_cols;
    aligned_vector<int> assumed_col_ptr;
    aligned_vector<int> assumed_col_procs;

    Topology* topology;

    int num_shared;  // Number of ParMatrix classes using partition

  };
}
#endif
