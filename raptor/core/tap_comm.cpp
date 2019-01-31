// Copyright (c) 2015-2017, RAPtor Developer Team
// License: Simplified BSD, http://opensource.org/licenses/BSD-2-Clause
#include "comm_pkg.hpp"

//#include <pmi.h>
//#include <rca_lib.h>

using namespace raptor;

/**************************************************************
*****   Split Off Proc Cols
**************************************************************
***** Splits off_proc_column_map into on_node_column_map and 
***** off_node_column map.  Also maps each of these columns to 
***** their corresponding process, and maps each local index
***** in on_node and off_node to off_proc
*****
***** Parameters
***** -------------
***** off_proc_column_map : aligned_vector<int>&
*****    Vector holding rank's off_proc_columns
***** off_proc_col_to_proc : aligned_vector<int>&
*****    Vector mapping rank's off_proc_columns to distant procs
***** on_node_column_map : aligned_vector<int>&
*****    Will be returned holding on_node columns
***** on_node_col_to_proc : aligned_vector<int>&
*****    Will be returned holding procs corresponding to on_node cols
***** on_node_to_off_proc : aligned_vector<int>&
*****    Will be returned holding map from on_node to off_proc
***** off_node_column_map : aligned_vector<int>&
*****    Will be returned holding off_node columns
***** off_node_col_to_node : aligned_vector<int>&
*****    Will be returned holding procs corresponding to off_node cols
***** off_node_to_off_proc : aligned_vector<int>&
*****    Will be returned holding map from off_node to off_proc
**************************************************************/
void TAPComm::split_off_proc_cols(const aligned_vector<int>& off_proc_column_map,
        const aligned_vector<int>& off_proc_col_to_proc,
        aligned_vector<int>& on_node_column_map,
        aligned_vector<int>& on_node_col_to_proc,
        aligned_vector<int>& on_node_to_off_proc,
        aligned_vector<int>& off_node_column_map,
        aligned_vector<int>& off_node_col_to_proc,
        aligned_vector<int>& off_node_to_off_proc)
{
    int rank, rank_node, num_procs;
    int proc;
    int node;
    int global_col;
    int off_proc_num_cols = off_proc_column_map.size();

    RAPtor_MPI_Comm_rank(RAPtor_MPI_COMM_WORLD, &rank);
    RAPtor_MPI_Comm_size(RAPtor_MPI_COMM_WORLD, &num_procs);
    rank_node = topology->get_node(rank);

    // Reserve size in vectors

    on_node_column_map.reserve(off_proc_num_cols);
    on_node_col_to_proc.reserve(off_proc_num_cols);
    off_node_column_map.reserve(off_proc_num_cols);
    off_node_col_to_proc.reserve(off_proc_num_cols);
    
    for (int i = 0; i < off_proc_num_cols; i++)
    {
        proc = off_proc_col_to_proc[i];
        node = topology->get_node(proc);
        global_col = off_proc_column_map[i];
        if (node == rank_node)
        {
            on_node_column_map.emplace_back(global_col);
            on_node_col_to_proc.emplace_back(topology->get_local_proc(proc));
            on_node_to_off_proc.emplace_back(i);
        }
        else
        {
            off_node_column_map.emplace_back(global_col);
            off_node_col_to_proc.emplace_back(proc);
            off_node_to_off_proc.emplace_back(i);
        }
    }
}


/**************************************************************
*****   Gather off node nodes
**************************************************************
***** Gathers nodes with which any local processes communicates
*****
***** Parameters
***** -------------
***** off_node_col_to_node : aligned_vector<int>&
*****    Vector holding rank's off_node_columns
***** recv_nodes : aligned_vector<int>&
*****    Returned holding all nodes with which any local
*****    process communicates (union of off_node_col_to_node)
**************************************************************/
void TAPComm::form_local_R_par_comm(const aligned_vector<int>& off_node_column_map,
        const aligned_vector<int>& off_node_col_to_proc,
        aligned_vector<int>& orig_procs, data_t* comm_t)
{
    int local_rank;
    RAPtor_MPI_Comm_rank(topology->local_comm, &local_rank);

    // Declare Variables
    int int_size = sizeof(int);
    
    int node;
    int num_recv_nodes;
    int local_proc;
    int idx, proc;
    int start_ctr, ctr;
    int local_num_sends;
    int recv_start, recv_end;
    int recv_proc, recv_size;
    int count;
    int off_node_num_cols = off_node_column_map.size();
    int N = topology->num_nodes / int_size;
    if (topology->num_nodes % int_size)
    {
        N++;
    }
    aligned_vector<int> tmp_recv_nodes(N, 0);
    aligned_vector<int> nodal_recv_nodes(N, 0);
    aligned_vector<int> node_sizes(topology->num_nodes, 0);
    aligned_vector<int> nodal_off_node_sizes;
    aligned_vector<int> node_to_local_proc;
    aligned_vector<int> local_recv_procs(topology->PPN, 0);
    aligned_vector<int> local_recv_sizes(topology->PPN, 0);
    aligned_vector<int> local_send_procs(topology->PPN);
    aligned_vector<int> proc_idx;
    aligned_vector<int> off_node_col_to_lcl_proc;
    aligned_vector<int> send_buffer;
    aligned_vector<int> recv_nodes;

    RAPtor_MPI_Status recv_status;

    NonContigData* local_R_recv = (NonContigData*) local_R_par_comm->recv_data;

    // Find nodes from which rank must recv, and the size of each recv
    for (aligned_vector<int>::const_iterator it = off_node_col_to_proc.begin();
            it != off_node_col_to_proc.end(); ++it)
    {
        node = topology->get_node(*it);
        int idx = node / int_size;
        int pos = node % int_size;
        tmp_recv_nodes[idx] |= 1 << pos;
        node_sizes[node]++;
    }

    // Allreduce among procs local to node to find nodes from which rank_node
    // recvs
    RAPtor_MPI_Allreduce(tmp_recv_nodes.data(), nodal_recv_nodes.data(), N, RAPtor_MPI_INT,
            RAPtor_MPI_BOR, topology->local_comm);

    // Add nodes from which rank_node must recv to recv_nodes
    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < int_size; j++)
        {
            if ((nodal_recv_nodes[i] >> j) & 1)
            {
                recv_nodes.emplace_back(i*int_size + j);
            }
        }
    }

    // Find the number of nodes from which rank node recvs
    num_recv_nodes = recv_nodes.size();

    // Find the size of each nodal recv
    if (num_recv_nodes)
    {
        // Collect the number of bytes sent to each node
        nodal_off_node_sizes.resize(num_recv_nodes);
        for (int i = 0; i < num_recv_nodes; i++)
        {
            node = recv_nodes[i];
            nodal_off_node_sizes[i] = node_sizes[node];
        }
        RAPtor_MPI_Allreduce(RAPtor_MPI_IN_PLACE, nodal_off_node_sizes.data(), num_recv_nodes, RAPtor_MPI_INT,
                RAPtor_MPI_SUM, topology->local_comm);

        // Sort nodes, descending by msg size (find permutation)
        aligned_vector<int> p(num_recv_nodes);
        std::iota(p.begin(), p.end(), 0);
        std::sort(p.begin(), p.end(), 
                [&](const int lhs, const int rhs)
                {
                    return nodal_off_node_sizes[lhs] > nodal_off_node_sizes[rhs];
                });

        // Sort recv nodes by total num bytes recvd from node
        aligned_vector<bool> done(num_recv_nodes);
        for (int i = 0; i < num_recv_nodes; i++)
        {
            if (done[i]) continue;

            done[i] = true;
            int prev_j = i;
            int j = p[i];
            while (i != j)
            {
                std::swap(recv_nodes[prev_j], recv_nodes[j]);
                std::swap(nodal_off_node_sizes[prev_j], nodal_off_node_sizes[j]);
                done[j] = true;
                prev_j = j;
                j = p[j];
            }
        }

        // Map recv nodes to local processes
        local_proc = 0;
        node_to_local_proc.resize(topology->num_nodes);
        for (aligned_vector<int>::iterator it = recv_nodes.begin();
                it != recv_nodes.end(); ++it)
        {
            node_to_local_proc[*it] = local_proc++ ;
            if (local_proc >= topology->PPN)
            {
                local_proc = 0;
            }
        }
    }

    if (num_recv_nodes)
    {
        proc_idx.resize(num_recv_nodes, 0);
    }
    if (off_node_num_cols)
    {
        off_node_col_to_lcl_proc.resize(off_node_num_cols);
    }

    // Find number of recvd indices per local proc
    for (int i = 0; i < off_node_num_cols; i++)
    {
        proc = off_node_col_to_proc[i];
        node = topology->get_node(proc);
        local_proc = node_to_local_proc[node];
        local_recv_sizes[local_proc]++;
        off_node_col_to_lcl_proc[i] = local_proc;
    }

    // Create displs based on local_recv_sizes
    recv_size = 0;
    aligned_vector<int> proc_to_idx(topology->PPN);
    for (int i = 0; i < topology->PPN; i++)
    {
        if (local_recv_sizes[i])
        {
            recv_size += local_recv_sizes[i];
            proc_to_idx[i] = local_R_recv->procs.size();
            local_R_recv->procs.emplace_back(i);
            local_R_recv->indptr.emplace_back(recv_size);
            local_recv_sizes[i] = 0;
            local_recv_procs[i] = 1;
        }
    }
    // Add columns to local_recv_indices in location according to
    local_R_recv->indices.resize(off_node_num_cols);
    for (int i = 0; i < off_node_num_cols; i++)
    {
        local_proc = off_node_col_to_lcl_proc[i];
        int proc_idx = proc_to_idx[local_proc];
        idx = local_R_recv->indptr[proc_idx] + local_recv_sizes[local_proc]++;
        local_R_recv->indices[idx] = i;
    }
    local_R_recv->num_msgs = local_R_recv->procs.size();
    local_R_recv->size_msgs = local_R_recv->indices.size();
    local_R_recv->finalize();

    // On node communication-- scalable to do all reduce to find number of
    // local processes to send to :)
    RAPtor_MPI_Allreduce(local_recv_procs.data(), local_send_procs.data(), topology->PPN, RAPtor_MPI_INT,
            RAPtor_MPI_SUM, topology->local_comm);
    local_num_sends = local_send_procs[local_rank];

    // Send recv_indices to each recv_proc along with their origin 
    // node
    if (local_R_recv->size_msgs)
    {
        send_buffer.resize(2*local_R_recv->size_msgs);
    }

    if (comm_t) *comm_t -= RAPtor_MPI_Wtime();

    ctr = 0;
    start_ctr = 0;
    for (int i = 0; i < local_R_recv->num_msgs; i++)
    {
        recv_proc = local_R_recv->procs[i];
        recv_start = local_R_recv->indptr[i];
        recv_end = local_R_recv->indptr[i+1];
        for (int j = recv_start; j < recv_end; j++)
        {
            idx = local_R_recv->indices[j];
            send_buffer[ctr++] = off_node_column_map[idx];
        }
        for (int j = recv_start; j < recv_end; j++)
        {
            idx = local_R_recv->indices[j];
            send_buffer[ctr++] = off_node_col_to_proc[idx];
        }
        RAPtor_MPI_Isend(&(send_buffer[start_ctr]), 2*(recv_end - recv_start),
                RAPtor_MPI_INT, recv_proc, 6543, topology->local_comm, 
                &(local_R_recv->requests[i]));
        start_ctr = ctr;
    }

    // Recv messages from local processes and add to send_data
    ctr = 0;
    for (int i = 0; i < local_num_sends; i++)
    {
        RAPtor_MPI_Probe(RAPtor_MPI_ANY_SOURCE, 6543, topology->local_comm, &recv_status);
        RAPtor_MPI_Get_count(&recv_status, RAPtor_MPI_INT, &count);
        proc = recv_status.RAPtor_MPI_SOURCE;
        int recvbuf[count];
        RAPtor_MPI_Recv(recvbuf, count, RAPtor_MPI_INT, proc, 6543, topology->local_comm,
                &recv_status);
        local_R_par_comm->send_data->add_msg(proc, count / 2, recvbuf);
        start_ctr = count / 2;
        // Add orig nodes for each recvd col (need to know this for
        // global communication setup)
        for (int j = start_ctr; j < count; j++)
        {
            orig_procs.emplace_back(recvbuf[j]);
        }
    }
    local_R_par_comm->send_data->finalize();

    // Wait for all sends to complete
    if (local_R_recv->num_msgs)
    {
        RAPtor_MPI_Waitall(local_R_recv->num_msgs,
                local_R_recv->requests.data(),
                RAPtor_MPI_STATUS_IGNORE);
    }

    if (comm_t) *comm_t += RAPtor_MPI_Wtime();

}   

/**************************************************************
*****   Find global comm procs
**************************************************************
***** Determine which processes with which rank will communicate
***** during inter-node communication
*****
***** Parameters
***** -------------
***** recv_nodes : aligned_vector<int>&
*****    All nodes with which any local process communicates 
***** send_procs : aligned_vector<int>&
*****    Returns with all off_node processes to which rank sends
***** recv_procs : aligned_vector<int>&
*****    Returns with all off_node process from which rank recvs
**************************************************************/
void TAPComm::form_global_par_comm(aligned_vector<int>& orig_procs,
        data_t* comm_t)
{
    int rank, num_procs;
    int local_rank;
    RAPtor_MPI_Comm_rank(RAPtor_MPI_COMM_WORLD, &rank);
    RAPtor_MPI_Comm_rank(topology->local_comm, &local_rank);
    RAPtor_MPI_Comm_size(RAPtor_MPI_COMM_WORLD, &num_procs);

    int n_sends;
    int proc, node;
    int finished, msg_avail;
    int recvbuf;
    int n_send_procs;
    int recv_size;
    int idx, node_idx;
    int ctr;
    int start, end, size;
    int count;
    RAPtor_MPI_Status recv_status;
    RAPtor_MPI_Request barrier_request;

    aligned_vector<int> node_list(topology->num_nodes, 0);
    aligned_vector<int> sendbuf;
    aligned_vector<int> sendbuf_sizes;
    aligned_vector<int> send_procs;
    aligned_vector<int> send_sizes(topology->PPN);
    aligned_vector<int> send_displs(topology->PPN+1);
    aligned_vector<int> node_sizes(topology->num_nodes, 0);
    aligned_vector<int> send_proc_sizes;
    aligned_vector<int> node_to_idx(topology->num_nodes, 0);
    aligned_vector<int> node_recv_idx_orig_procs;
    aligned_vector<int> send_buffer;

    NonContigData* global_recv = (NonContigData*) global_par_comm->recv_data;

    if (local_R_par_comm->send_data->size_msgs)
    {
        node_recv_idx_orig_procs.resize(local_R_par_comm->send_data->size_msgs);
    }

    // Find how many msgs must recv from each node
    for (int i = 0; i < local_R_par_comm->send_data->size_msgs; i++)
    {
        proc = orig_procs[i];
        node = topology->get_node(proc);
        node_sizes[node]++;
    }

    // Form recv procs and indptr, based on node_sizes
    recv_size = 0;
    for (int i = 0; i < topology->num_nodes; i++)
    {
        if (node_sizes[i])
        {
            recv_size += node_sizes[i];
            node_to_idx[i] = global_recv->procs.size();
            global_recv->indptr.emplace_back(recv_size);
            global_recv->procs.emplace_back(i);  // currently have node 
            node_sizes[i] = 0;
        }
    }
    global_recv->num_msgs = global_recv->procs.size();
    global_recv->size_msgs = recv_size;

    // Form recv indices, placing global column in correct position
    global_recv->indices.resize(recv_size);
    for (int i = 0; i < local_R_par_comm->send_data->size_msgs; i++)
    {
        proc = orig_procs[i];
        node = topology->get_node(proc);
        node_idx = node_to_idx[node];
        idx = global_recv->indptr[node_idx] + node_sizes[node]++;
        global_recv->indices[idx] = local_R_par_comm->send_data->indices[i];
        node_recv_idx_orig_procs[idx] = proc;
    }

    // Remove duplicates... Likely send same data to multiple local procs, but
    // only want to recv this data from a distant node once
    ctr = 0;
    start = global_recv->indptr[0];
    for (int i = 0; i < global_recv->num_msgs; i++)
    {
        proc = global_recv->procs[i];
        end = global_recv->indptr[i+1];
        size = end - start;
        if (size)
        {
            // Find permutation of node_recv_indices (between start and end)
            // in ascending order
            aligned_vector<int> p(size);
            std::iota(p.begin(), p.end(), 0);
            std::sort(p.begin(), p.end(),
                    [&] (int j, int k)
                    {
                        return global_recv->indices[j+start] 
                               < global_recv->indices[k+start];
                    });

            // Sort node_recv_indices and node_recv_idx_orig_procs together
            aligned_vector<bool> done(size);
            for (int j = 0; j < size; j++)
            {
                if (done[j]) continue;

                done[j] = true;
                int prev_k = j;
                int k = p[j];
                while (j != k)
                {
                    std::swap(global_recv->indices[prev_k+start],
                            global_recv->indices[k+start]);
                    std::swap(node_recv_idx_orig_procs[prev_k+start], 
                            node_recv_idx_orig_procs[k+start]);
                    done[k] = true;
                    prev_k = k;
                    k = p[k];
                }
            }
        }

        // Add msg to global_par_comm->recv_data
        node_recv_idx_orig_procs[ctr] = node_recv_idx_orig_procs[start];
        global_recv->indices[ctr++] 
                = global_recv->indices[start];
        for (int j = start+1; j < end; j++)
        {
            if (global_recv->indices[j] != global_recv->indices[j-1])
            {
                node_recv_idx_orig_procs[ctr] = node_recv_idx_orig_procs[j];
                global_recv->indices[ctr++] = global_recv->indices[j];
            }
        }
        global_recv->indptr[i + 1] = ctr;
        start = end;
    }
    global_recv->indices.resize(ctr);
    global_recv->size_msgs = ctr;
    global_recv->finalize();

    if (comm_t) *comm_t -= RAPtor_MPI_Wtime(); 
    aligned_vector<int> send_p(num_procs, 0);
    for (int i = 0; i < global_recv->num_msgs; i++)
    {
        node = global_recv->procs[i];
        proc = topology->get_global_proc(node, local_rank);
        send_p[proc] = 1;
    }
    RAPtor_MPI_Allreduce(RAPtor_MPI_IN_PLACE, send_p.data(), num_procs, RAPtor_MPI_INT,
            RAPtor_MPI_SUM, RAPtor_MPI_COMM_WORLD);
    int recv_n = send_p[rank];
    sendbuf.resize(recv_n);
    sendbuf_sizes.resize(recv_n);
    // Send recv sizes to corresponding local procs on appropriate nodes
    ctr = 0;
    for (int i = 0; i < global_recv->num_msgs; i++)
    {
        node = global_recv->procs[i];
        proc = topology->get_global_proc(node, local_rank);
        RAPtor_MPI_Isend(&(node_sizes[node]), 1, RAPtor_MPI_INT, proc, 9876, RAPtor_MPI_COMM_WORLD,
                &(global_recv->requests[i]));
    }
    for (int i = 0; i < recv_n; i++)
    {
        RAPtor_MPI_Probe(RAPtor_MPI_ANY_SOURCE, 9876, RAPtor_MPI_COMM_WORLD, &recv_status);
        proc = recv_status.RAPtor_MPI_SOURCE;
        RAPtor_MPI_Recv(&recvbuf, 1, RAPtor_MPI_INT, proc, 9876, RAPtor_MPI_COMM_WORLD, 
                &recv_status);
        sendbuf[i] = proc;
        sendbuf_sizes[i] = recvbuf;
    }
    RAPtor_MPI_Waitall(global_recv->num_msgs, global_recv->requests.data(),
            RAPtor_MPI_STATUSES_IGNORE);
    if (comm_t) *comm_t += RAPtor_MPI_Wtime();

    // Gather all procs to which node must send 
    n_sends = sendbuf.size();
    RAPtor_MPI_Allgather(&n_sends, 1, RAPtor_MPI_INT, send_sizes.data(), 1, RAPtor_MPI_INT, topology->local_comm);
    send_displs[0] = 0;
    for (int i = 0; i < topology->PPN; i++)
    {
        send_displs[i+1] = send_displs[i] + send_sizes[i];
    } 
    n_send_procs = send_displs[topology->PPN];
    send_procs.resize(n_send_procs);
    send_proc_sizes.resize(n_send_procs);
    RAPtor_MPI_Allgatherv(sendbuf.data(), n_sends, RAPtor_MPI_INT, send_procs.data(), 
            send_sizes.data(), send_displs.data(), RAPtor_MPI_INT, topology->local_comm);
    RAPtor_MPI_Allgatherv(sendbuf_sizes.data(), n_sends, RAPtor_MPI_INT, send_proc_sizes.data(), 
            send_sizes.data(), send_displs.data(), RAPtor_MPI_INT, topology->local_comm);

    // Permute send_procs based on send_proc_sizes
    aligned_vector<int> p(n_send_procs);
    std::iota(p.begin(), p.end(), 0);
    std::sort(p.begin(), p.end(), 
            [&](const int lhs, const int rhs)
            {
                return send_proc_sizes[lhs] > send_proc_sizes[rhs];
            });
    aligned_vector<bool> done(n_send_procs);
    for (int i = 0; i < n_send_procs; i++)
    {
        if (done[i]) continue;

        done[i] = true;
        int prev_j = i;
        int j = p[i];
        while (i != j)
        {
            std::swap(send_procs[prev_j], send_procs[j]);
            std::swap(send_proc_sizes[prev_j], send_proc_sizes[j]);
            done[j] = true;
            prev_j = j; 
            j = p[j];
        }
    }

    // Distribute send_procs across local procs
    n_sends = 0;
    for (size_t i = topology->PPN - local_rank - 1; i < send_procs.size(); i += topology->PPN)
    {
        global_par_comm->send_data->procs.emplace_back(send_procs[i]);
    }
    global_par_comm->send_data->num_msgs = global_par_comm->send_data->procs.size();
    global_par_comm->send_data->requests.resize(global_par_comm->send_data->num_msgs);


    if (comm_t) *comm_t -= RAPtor_MPI_Wtime();
    for (int i = 0; i < global_par_comm->send_data->num_msgs; i++)
    {
        proc = global_par_comm->send_data->procs[i];
        RAPtor_MPI_Isend(&(global_par_comm->send_data->procs[i]), 1, RAPtor_MPI_INT, proc, 6789, 
                RAPtor_MPI_COMM_WORLD, &(global_par_comm->send_data->requests[i]));
    }
    // Recv processes from which rank must recv
    for (int i = 0; i < global_recv->num_msgs; i++)
    {
        RAPtor_MPI_Probe(RAPtor_MPI_ANY_SOURCE, 6789, RAPtor_MPI_COMM_WORLD, &recv_status);
        proc = recv_status.RAPtor_MPI_SOURCE;
        node = topology->get_node(proc);
        RAPtor_MPI_Recv(&recvbuf, 1, RAPtor_MPI_INT, proc, 6789, RAPtor_MPI_COMM_WORLD, &recv_status);
        idx = node_to_idx[node];
        global_recv->procs[idx] = proc;
    }
    // Wait for sends to complete
    if (global_par_comm->send_data->num_msgs)
    {
        RAPtor_MPI_Waitall(global_par_comm->send_data->num_msgs, 
                global_par_comm->send_data->requests.data(), RAPtor_MPI_STATUSES_IGNORE);

    }
    if (comm_t) *comm_t += RAPtor_MPI_Wtime();


    for (int i = 0; i < global_recv->size_msgs; i++)
    {
        send_buffer.emplace_back(global_recv->indices[i]);
        send_buffer.emplace_back(node_recv_idx_orig_procs[i]);
    }


    // Send recv indices to each recv proc along with the process of
    // origin for each recv idx
    ctr = 0;
    
    if (comm_t) *comm_t -= RAPtor_MPI_Wtime();
    for (int i = 0; i < global_recv->num_msgs; i++)
    {
        proc = global_recv->procs[i];
        start = global_recv->indptr[i];
        end = global_recv->indptr[i+1];
        RAPtor_MPI_Isend(&(send_buffer[2*start]), 2*(end - start),
                RAPtor_MPI_INT, proc, 5432, RAPtor_MPI_COMM_WORLD,
                &(global_recv->requests[i]));
        
    }

    // Recv send data (which indices to send) to global processes
    orig_procs.clear();
    for (int i = 0; i < global_par_comm->send_data->num_msgs; i++)
    {
        proc = global_par_comm->send_data->procs[i];
        RAPtor_MPI_Probe(proc, 5432, RAPtor_MPI_COMM_WORLD, &recv_status);
        RAPtor_MPI_Get_count(&recv_status, RAPtor_MPI_INT, &count);
        int recvbuf[count];
        RAPtor_MPI_Recv(recvbuf, count, RAPtor_MPI_INT, proc, 5432, RAPtor_MPI_COMM_WORLD, &recv_status);
        for (int j = 0; j < count; j += 2)
        {
           global_par_comm->send_data->indices.emplace_back(recvbuf[j]);
           orig_procs.emplace_back(topology->get_local_proc(recvbuf[j+1]));
        }
        global_par_comm->send_data->indptr.emplace_back(
                global_par_comm->send_data->indices.size()); 
    }
    global_par_comm->send_data->num_msgs = global_par_comm->send_data->procs.size();
    global_par_comm->send_data->size_msgs = global_par_comm->send_data->indices.size();
    global_par_comm->send_data->finalize();

    if (global_recv->num_msgs)
    {
        RAPtor_MPI_Waitall(global_recv->num_msgs,
                global_recv->requests.data(),
                RAPtor_MPI_STATUS_IGNORE);
    }
    if (comm_t) *comm_t += RAPtor_MPI_Wtime();
}


/**************************************************************
*****   Form local_S_par_comm
**************************************************************
***** Find which local processes the values originating on rank
***** must be sent to, and which processes store values rank must
***** send as inter-node communication.
*****
***** Parameters
***** -------------
**************************************************************/
void TAPComm::form_local_S_par_comm(aligned_vector<int>& orig_procs,
        data_t* comm_t)
{
    int rank;
    int local_rank;
    RAPtor_MPI_Comm_rank(RAPtor_MPI_COMM_WORLD, &rank);
    RAPtor_MPI_Comm_rank(topology->local_comm, &local_rank);

    // Find local_col_starts for all procs local to node, and sort
    int start, end;
    int proc, proc_idx;
    
    int ctr, idx;
    int size;

    aligned_vector<int> local_procs(topology->PPN);
    aligned_vector<int> proc_sizes(topology->PPN, 0);
    aligned_vector<int> recv_procs(topology->PPN, 0);
    aligned_vector<int> proc_to_idx(topology->PPN);

    NonContigData* local_S_recv = (NonContigData*) local_S_par_comm->recv_data;

    if (global_par_comm->send_data->num_msgs)
    {
        local_S_recv->indices.resize(global_par_comm->send_data->size_msgs);
    }

    // Find all column indices originating on local procs
    for (int i = 0; i < global_par_comm->send_data->size_msgs; i++)
    {
        proc = orig_procs[i];
        proc_sizes[proc]++;
        recv_procs[proc] = 1;
    }

    // Reduce recv_procs to how many msgs rank will recv
    RAPtor_MPI_Allreduce(RAPtor_MPI_IN_PLACE, recv_procs.data(), topology->PPN, RAPtor_MPI_INT, RAPtor_MPI_SUM, topology->local_comm);
    int n_recvs = recv_procs[local_rank];

    // Form local_S_par_comm recv_data
    int recv_size = 0;
    for (int i = 0; i < topology->PPN; i++)
    {
        if (proc_sizes[i])
        {
            recv_size += proc_sizes[i];
            proc_to_idx[i] = local_S_recv->procs.size();
            local_S_recv->procs.emplace_back(i);
            local_S_recv->indptr.emplace_back(recv_size);
        }
        proc_sizes[i] = 0;
    }
    local_S_recv->num_msgs = local_S_recv->procs.size();
    for (int i = 0; i < global_par_comm->send_data->size_msgs; i++)
    {
        proc = orig_procs[i];
        proc_idx = proc_to_idx[proc];
        idx = local_S_recv->indptr[proc_idx] + proc_sizes[proc]++;
        local_S_recv->indices[idx] = global_par_comm->send_data->indices[i];
    }

    // Remove duplicate entries from local_S_par_comm recv_data (proc may have
    // to send the same data to multiple nodes, but should only recv values a
    // single time from each local proc)
    ctr = 0;
    start = local_S_recv->indptr[0];
    for (int i = 0; i < local_S_recv->num_msgs; i++)
    {
        end = local_S_recv->indptr[i+1];
        size = end - start;
        if (size)
        {
            std::sort(local_S_recv->indices.begin() + start, 
                    local_S_recv->indices.begin() + end);
            local_S_recv->indices[ctr++] = 
                local_S_recv->indices[start];
            for (int j = start+1; j < end; j++)
            {
                if (local_S_recv->indices[j] 
                        != local_S_recv->indices[j-1])
                {
                    local_S_recv->indices[ctr++] 
                        = local_S_recv->indices[j];
                }
            }
        }
        local_S_recv->indptr[i+1] = ctr;
        start = end;
    }
    local_S_recv->indices.resize(ctr);
    local_S_recv->size_msgs = ctr;
    local_S_recv->finalize();

    // Send messages to local procs, informing of what data to send
    if (comm_t) *comm_t -= RAPtor_MPI_Wtime();
    for (int i = 0; i < local_S_recv->num_msgs; i++)
    {
        proc = local_S_recv->procs[i];
        start = local_S_recv->indptr[i];
        end = local_S_recv->indptr[i+1];
        RAPtor_MPI_Isend(&(local_S_recv->indices[start]), 
                end - start, RAPtor_MPI_INT, proc, 4321, topology->local_comm,
                &(local_S_recv->requests[i]));
    }
    // Recv messages and form local_S_par_comm send_data
    int count;
    RAPtor_MPI_Status recv_status;
    for (int i = 0; i < n_recvs; i++)
    {
        RAPtor_MPI_Probe(RAPtor_MPI_ANY_SOURCE, 4321, topology->local_comm, &recv_status);
        RAPtor_MPI_Get_count(&recv_status, RAPtor_MPI_INT, &count);
        proc = recv_status.RAPtor_MPI_SOURCE;        
        int recvbuf[count];
        RAPtor_MPI_Recv(recvbuf, count, RAPtor_MPI_INT, proc, 4321, topology->local_comm, &recv_status);
        for (int j = 0; j < count; j++)
        {
            local_S_par_comm->send_data->indices.emplace_back(recvbuf[j]);
        }
        local_S_par_comm->send_data->indptr.emplace_back(
                local_S_par_comm->send_data->indices.size());
        local_S_par_comm->send_data->procs.emplace_back(proc);
    }
    local_S_par_comm->send_data->num_msgs = local_S_par_comm->send_data->procs.size();
    local_S_par_comm->send_data->size_msgs = local_S_par_comm->send_data->indices.size();
    local_S_par_comm->send_data->finalize();
    if (local_S_recv->num_msgs)
    {
        RAPtor_MPI_Waitall(local_S_recv->num_msgs,
                local_S_recv->requests.data(),
                RAPtor_MPI_STATUS_IGNORE);
    }
    if (comm_t) *comm_t += RAPtor_MPI_Wtime();
}


void TAPComm::adjust_send_indices(const int first_local_col)
{
    int idx, idx_pos, size;
    int local_S_idx, global_comm_idx;

    if (local_S_par_comm)
    {
        DuplicateData* local_S_recv = (DuplicateData*) local_S_par_comm->recv_data;
        // Update global row index with local row to send 
        for (int i = 0; i < local_S_par_comm->send_data->size_msgs; i++)
        {
            local_S_par_comm->send_data->indices[i] -= first_local_col;
        }

        // Update global_par_comm->send_data->indices (global rows) to 
        std::map<int, int> S_global_to_local;
        for (int i = 0; i < local_S_recv->size_msgs; i++)
        {
            S_global_to_local[local_S_recv->indices[i]] = i;
        }
        aligned_vector<int> local_S_num_pos;
        if (local_S_recv->size_msgs)
            local_S_num_pos.resize(local_S_recv->size_msgs, 0);
        for (int i = 0; i < global_par_comm->send_data->size_msgs; i++)
        {
            idx = global_par_comm->send_data->indices[i];
            local_S_idx = S_global_to_local[idx];
            global_par_comm->send_data->indices[i] = local_S_idx;
            local_S_num_pos[local_S_idx]++;
        }
        local_S_recv->indptr_T.resize(local_S_recv->size_msgs + 1);
        local_S_recv->indptr_T[0] = 0;
        size = 0;
        for (int i = 0; i < local_S_recv->size_msgs; i++)
        {
            size += local_S_num_pos[i];
            local_S_recv->indptr_T[i+1] = size;
            local_S_num_pos[i] = 0;
        }
        local_S_recv->indices.resize(size);
        for(int i = 0; i < global_par_comm->send_data->size_msgs; i++)
        {
            idx = global_par_comm->send_data->indices[i];
            idx_pos = local_S_recv->indptr_T[idx] + local_S_num_pos[idx]++;
            local_S_recv->indices[idx_pos] = i;
        }
    }
    else
    {
        for (int i = 0; i < global_par_comm->send_data->size_msgs; i++)
        {
            global_par_comm->send_data->indices[i] -= first_local_col;
        }
    }

    // Update local_R_par_comm->send_data->indices (global_rows)
    DuplicateData* global_recv = (DuplicateData*) global_par_comm->recv_data;
    std::map<int, int> global_to_local;
    for (int i = 0; i < global_recv->size_msgs; i++)
    {
        global_to_local[global_recv->indices[i]] = i;
    }
    aligned_vector<int> global_num_pos;
    if (global_recv->size_msgs)
        global_num_pos.resize(global_recv->size_msgs, 0);
    for (int i = 0; i < local_R_par_comm->send_data->size_msgs; i++)
    {
        idx = local_R_par_comm->send_data->indices[i];
        global_comm_idx = global_to_local[idx];
        local_R_par_comm->send_data->indices[i] = global_comm_idx;
        global_num_pos[global_comm_idx]++;
    }
    global_recv->indptr_T.resize(global_recv->size_msgs + 1);
    global_recv->indptr_T[0] = 0;
    size = 0;
    for (int i = 0; i < global_recv->size_msgs; i++)
    {
        size += global_num_pos[i];
        global_recv->indptr_T[i+1] = size;
        global_num_pos[i] = 0;
    }
    global_recv->indices.resize(size);
    for (int i = 0; i < local_R_par_comm->send_data->size_msgs; i++)
    {
        idx = local_R_par_comm->send_data->indices[i];
        idx_pos = global_recv->indptr_T[idx] + global_num_pos[idx]++;
        global_recv->indices[idx_pos] = i;
    }

}

/**************************************************************
*****  Form local_L_par_comm 
**************************************************************
***** Adjust send indices from global row index to index of 
***** global column in previous recv buffer.  
*****
***** Parameters
***** -------------
***** on_node_column_map : aligned_vector<int>&
*****    Columns corresponding to on_node processes
***** on_node_col_to_proc : aligned_vector<int>&
*****    On node process corresponding to each column
*****    in on_node_column_map
***** first_local_row : int
*****    First row local to rank 
**************************************************************/
void TAPComm::form_local_L_par_comm(const aligned_vector<int>& on_node_column_map,
        const aligned_vector<int>& on_node_col_to_proc, const int first_local_col,
        data_t* comm_t)
{
    int local_rank;
    RAPtor_MPI_Comm_rank(topology->local_comm, &local_rank);

    int on_node_num_cols = on_node_column_map.size();
    int prev_proc, prev_idx;
    int num_sends;
    int proc, start, end;
    int count;
    RAPtor_MPI_Status recv_status;
    aligned_vector<int> recv_procs(topology->PPN, 0);

    NonContigData* local_L_recv = (NonContigData*) local_L_par_comm->recv_data;

    if (on_node_num_cols)
    {
        prev_proc = on_node_col_to_proc[0];
        recv_procs[prev_proc] = 1;
        prev_idx = 0;
        for (int i = 1; i < on_node_num_cols; i++)
        {
            proc = on_node_col_to_proc[i];
            if (proc != prev_proc)
            {
                local_L_recv->add_msg(prev_proc, i - prev_idx);
                prev_proc = proc;
                prev_idx = i;
                recv_procs[proc] = 1;
            }
        }
        local_L_recv->add_msg(prev_proc, on_node_num_cols - prev_idx);
        local_L_recv->finalize();

        for (int i = 0; i < on_node_num_cols; i++)
        {
            local_L_recv->indices.emplace_back(i);
        }
    }

    RAPtor_MPI_Allreduce(RAPtor_MPI_IN_PLACE, recv_procs.data(), topology->PPN, RAPtor_MPI_INT, RAPtor_MPI_SUM, 
            topology->local_comm);
    num_sends = recv_procs[local_rank];

    if (comm_t) *comm_t -= RAPtor_MPI_Wtime();
    for (int i = 0; i < local_L_recv->num_msgs; i++)
    {
        proc = local_L_recv->procs[i];
        start = local_L_recv->indptr[i];
        end = local_L_recv->indptr[i+1];
        RAPtor_MPI_Isend(&(on_node_column_map[start]), end - start, RAPtor_MPI_INT, proc,
                7890, topology->local_comm, &(local_L_recv->requests[i]));
    }
    for (int i = 0; i < num_sends; i++)
    {
        RAPtor_MPI_Probe(RAPtor_MPI_ANY_SOURCE, 7890, topology->local_comm, &recv_status);
        RAPtor_MPI_Get_count(&recv_status, RAPtor_MPI_INT, &count);
        proc = recv_status.RAPtor_MPI_SOURCE;
        int recvbuf[count];
        RAPtor_MPI_Recv(recvbuf, count, RAPtor_MPI_INT, proc, 7890, topology->local_comm, &recv_status);
        for (int i = 0; i < count; i++)
        {
            recvbuf[i] -= first_local_col;
        }
        local_L_par_comm->send_data->add_msg(proc, count, recvbuf);
    }
    local_L_par_comm->send_data->finalize();
    
    if (local_L_recv->num_msgs)
    {
        RAPtor_MPI_Waitall(local_L_recv->num_msgs,
                local_L_recv->requests.data(), 
                RAPtor_MPI_STATUSES_IGNORE);
    }
    if (comm_t) *comm_t += RAPtor_MPI_Wtime();
}

void TAPComm::form_simple_R_par_comm(aligned_vector<int>& off_node_column_map,
        aligned_vector<int>& off_node_col_to_proc, data_t* comm_t)
{
    int rank, local_rank;
    RAPtor_MPI_Comm_rank(RAPtor_MPI_COMM_WORLD, &rank);
    RAPtor_MPI_Comm_rank(topology->local_comm, &local_rank);

    int proc, local_proc;
    int proc_idx, idx;
    int start, end;
    int count;
    RAPtor_MPI_Status recv_status;
    int off_node_num_cols = off_node_column_map.size();
    aligned_vector<int> local_proc_sizes(topology->PPN, 0);
    aligned_vector<int> proc_size_idx(topology->PPN);

    NonContigData* local_R_recv = (NonContigData*) local_R_par_comm->recv_data;

    // Form local_R_par_comm recv_data (currently with global recv indices)
    for (aligned_vector<int>::iterator it = off_node_col_to_proc.begin();
            it != off_node_col_to_proc.end(); ++it)
    {
        local_proc = topology->get_local_proc(*it);
        local_proc_sizes[local_proc]++;
    }

    local_R_recv->size_msgs = 0;
    local_R_recv->indptr[0] = local_R_recv->size_msgs;
    for (int i = 0; i < topology->PPN; i++)
    {
        if (local_proc_sizes[i])
        {
            local_R_recv->num_msgs++;
            local_R_recv->size_msgs += local_proc_sizes[i];
            local_proc_sizes[i] = 0;

            proc_size_idx[i] = local_R_recv->procs.size();
            local_R_recv->procs.emplace_back(i);
            local_R_recv->indptr.emplace_back(
                    local_R_recv->size_msgs);
        }
    }
    if (local_R_recv->size_msgs)
    {
        local_R_recv->indices.resize(local_R_recv->size_msgs);
    }

    for (int i = 0; i < off_node_num_cols; i++)
    {
        proc = off_node_col_to_proc[i];
        local_proc = topology->get_local_proc(proc);
        proc_idx = proc_size_idx[local_proc];
        idx = local_R_recv->indptr[proc_idx] + local_proc_sizes[local_proc]++;
        local_R_recv->indices[idx] = i;
    }
    local_R_recv->finalize();

    // Communicate local_R recv_data so send_data can be formed
    RAPtor_MPI_Allreduce(RAPtor_MPI_IN_PLACE, local_proc_sizes.data(), topology->PPN, RAPtor_MPI_INT,
            RAPtor_MPI_SUM, topology->local_comm);

    if (comm_t) *comm_t -= RAPtor_MPI_Wtime();
    local_R_par_comm->recv_data->send(off_node_column_map.data(), 6543, topology->local_comm);
    local_R_par_comm->send_data->probe(local_proc_sizes[local_rank], 6543, topology->local_comm);
    local_R_par_comm->recv_data->waitall();
    if (comm_t) *comm_t += RAPtor_MPI_Wtime();
}

void TAPComm::form_simple_global_comm(aligned_vector<int>& off_proc_col_to_proc,
        data_t* comm_t)
{
    int rank;
    int num_procs;
    int proc, start, end;
    int finished, msg_avail;
    int count;
    int idx, proc_idx;
    int global_idx;
    RAPtor_MPI_Status recv_status;
    RAPtor_MPI_Request barrier_request;

    RAPtor_MPI_Comm_rank(RAPtor_MPI_COMM_WORLD, &rank);
    RAPtor_MPI_Comm_size(RAPtor_MPI_COMM_WORLD, &num_procs);

    aligned_vector<int> proc_sizes(num_procs, 0);
    aligned_vector<int> proc_ctr;

    NonContigData* local_R_recv = (NonContigData*) local_R_par_comm->recv_data;
    NonContigData* global_recv = (NonContigData*) global_par_comm->recv_data;

    // Communicate processes on which each index originates
    if (comm_t) *comm_t -= RAPtor_MPI_Wtime();
    local_R_par_comm->communicate_T(off_proc_col_to_proc.data());
    aligned_vector<int>& int_buffer = local_R_par_comm->send_data->get_buffer<int>();
    if (comm_t) *comm_t += RAPtor_MPI_Wtime();

    for (int i = 0; i < local_R_par_comm->send_data->size_msgs; i++)
    {
        proc = int_buffer[i];
        if (proc_sizes[proc] == 0)
        {
            global_recv->procs.emplace_back(proc);
        }
        proc_sizes[proc]++;
    }

    global_recv->num_msgs = global_recv->procs.size();
    global_recv->indptr[0] = 0;
    global_recv->size_msgs = 0;
    for (int i = 0; i < global_recv->num_msgs; i++)
    {
        proc = global_recv->procs[i];
        global_recv->size_msgs += proc_sizes[proc];
        proc_sizes[proc] = i; // Will now use this for proc_idx
        global_recv->indptr.emplace_back(global_recv->size_msgs);
    }
    if (global_recv->size_msgs)
    {
        global_recv->indices.resize(global_recv->size_msgs);
        proc_ctr.resize(global_recv->num_msgs, 0);
    }

    for (int i = 0; i < local_R_par_comm->send_data->size_msgs; i++)
    {
        global_idx = local_R_par_comm->send_data->indices[i];
        proc = int_buffer[i];
        proc_idx = proc_sizes[proc];
        idx = global_recv->indptr[proc_idx] + proc_ctr[proc_idx]++;
        global_recv->indices[idx] = global_idx;
    }
    global_recv->finalize();

    // Communicate global recv_data so send_data can be formed (dynamic comm)
    aligned_vector<int> recv_sizes(num_procs, 0);
    for (int i = 0; i < global_recv->num_msgs; i++)
        recv_sizes[global_recv->procs[i]] = global_recv->indptr[i+1] - global_recv->indptr[i];
    RAPtor_MPI_Allreduce(RAPtor_MPI_IN_PLACE, recv_sizes.data(), num_procs, RAPtor_MPI_INT, RAPtor_MPI_SUM, RAPtor_MPI_COMM_WORLD);

    if (comm_t) *comm_t -= RAPtor_MPI_Wtime();
    for (int i = 0; i < global_recv->num_msgs; i++)
    {
        proc = global_recv->procs[i];
        start = global_recv->indptr[i];
        end = global_recv->indptr[i+1];
        RAPtor_MPI_Isend(&(global_recv->indices[start]), end - start, RAPtor_MPI_INT,
                proc, 6789, RAPtor_MPI_COMM_WORLD, &(global_recv->requests[i]));
    }
    global_par_comm->send_data->probe(recv_sizes[rank], 6789, RAPtor_MPI_COMM_WORLD);
    global_par_comm->recv_data->waitall();
    if (comm_t) *comm_t += RAPtor_MPI_Wtime();
}

void TAPComm::update_recv(const aligned_vector<int>& on_node_to_off_proc,
        const aligned_vector<int>& off_node_to_off_proc, bool update_L)
{
    int idx;

    // Determine size of final recvs (should be equal to 
    // number of off_proc cols)
    recv_size = local_R_par_comm->recv_data->size_msgs +
        local_L_par_comm->recv_data->size_msgs;
    NonContigData* local_R_recv = (NonContigData*) local_R_par_comm->recv_data;
    NonContigData* local_L_recv = (NonContigData*) local_L_par_comm->recv_data;
    if (recv_size)
    {
        // Want a single recv buffer local_R and local_L par_comms
        buffer.resize(recv_size);
        int_buffer.resize(recv_size);

        // Map local_R recvs to original off_proc_column_map
        if (local_R_recv->size_msgs)
        {
            for (int i = 0; i < local_R_recv->size_msgs; i++)
            {
                idx = local_R_recv->indices[i];
                local_R_recv->indices[i] = off_node_to_off_proc[idx];
            }
        }


        // Map local_L recvs to original off_proc_column_map
        if (update_L && local_L_recv->size_msgs)
        {
            for (int i = 0; i < local_L_recv->size_msgs; i++)
            {
                idx = local_L_recv->indices[i];
                local_L_recv->indices[i] = on_node_to_off_proc[idx];
            }
        }
    }
}




