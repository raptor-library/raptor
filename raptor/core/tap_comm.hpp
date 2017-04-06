// Copyright (c) 2015, Raptor Developer Team, University of Illinois at Urbana-Champaign
// License: Simplified BSD, http://opensource.org/licenses/BSD-2-Clause
#ifndef RAPTOR_CORE_TAPCOMM_HPP
#define RAPTOR_CORE_TAPCOMM_HPP

#include <mpi.h>
#include "par_comm.hpp"

// TODO - when using TAPSpMV, must either export 
// MPICH_RANK_REORDER_METHOD and PPN before running
// program, or set these defines to defaults
#define STANDARD_PPN 4
#define STANDARD_PROC_LAYOUT 1

namespace raptor
{
    class ParComm;

class TAPComm
{
public:
    /**************************************************************
    *****   TAPComm Class Constructor
    **************************************************************
    ***** Initializes an empty TAPComm, setting send and recv
    ***** sizes to 0
    ***** _key : int (optional)
    *****    Tag to be used in MPI Communication (default 0)
    **************************************************************/
    TAPComm(int _key = 0)
    {
        key = _key;

        local_S_par_comm = new ParComm();
        local_R_par_comm = new ParComm();
        local_L_par_comm = new ParComm();
        global_par_comm = new ParComm();
    }

    /**************************************************************
    *****   TAPComm Class Constructor
    **************************************************************
    ***** Initializes a TAPComm object based on the off_proc Matrix
    *****
    ***** Parameters
    ***** -------------
    ***** off_proc_column_map : std::vector<index_t>&
    *****    Maps local off_proc columns indices to global
    ***** first_local_row : index_t
    *****    Global row index of first row local to process
    ***** first_local_col : index_t
    *****    Global row index of first column to fall in local block
    ***** _key : int (optional)
    *****    Tag to be used in MPI Communication (default 9999)
    **************************************************************/
    TAPComm(std::vector<int>& off_proc_column_map,
            int first_local_row, 
            int first_local_col,
            int global_num_cols, 
            int local_num_cols,
            int _key = 9999,
            MPI_Comm comm = MPI_COMM_WORLD)
    {
        // Get MPI Information
        int rank, num_procs;
        MPI_Comm_rank(comm, &rank);
        MPI_Comm_size(comm, &num_procs);

        // Initialize class variables
        key = _key;
        local_S_par_comm = new ParComm(2345);
        local_R_par_comm = new ParComm(3456);
        local_L_par_comm = new ParComm(4567);
        global_par_comm = new ParComm(5678);

        // Map procs to nodes
        char* proc_layout_c = std::getenv("MPICH_RANK_REORDER_METHOD");
        char* PPN_c = std::getenv("PPN");
        int rank_node;
        if (PPN_c) 
        {
            PPN = atoi(PPN_c);
        }
        else
        {
            PPN = STANDARD_PPN;
        }

        if (proc_layout_c)
        {
            rank_ordering = atoi(proc_layout_c);
        }
        else
        {
            rank_ordering = STANDARD_PROC_LAYOUT;
        }

        num_nodes = num_procs / PPN;
        if (num_procs % PPN) num_nodes++;
        rank_node = get_node(rank);

        // Create intra-node communicator
        MPI_Comm_split(MPI_COMM_WORLD, rank_node, rank, &local_comm);
        int local_rank;
        MPI_Comm_rank(local_comm, &local_rank);

        int off_proc_num_cols = off_proc_column_map.size();
        int global_col;

        std::vector<int> off_proc_col_to_proc;
        std::vector<int> on_node_column_map;
        std::vector<int> on_node_col_to_proc;
        std::vector<int> off_node_column_map;
        std::vector<int> off_node_col_to_node;
        std::vector<int> on_node_to_off_proc;
        std::vector<int> off_node_to_off_proc;
        std::vector<int> recv_nodes;

        global_par_comm->form_col_to_proc(first_local_col, global_num_cols,
            local_num_cols, off_proc_column_map, off_proc_col_to_proc);

        // Partition off_proc cols into on_node and off_node
        split_off_proc_cols(off_proc_column_map, off_proc_col_to_proc,
               on_node_column_map, on_node_col_to_proc, on_node_to_off_proc,
               off_node_column_map, off_node_col_to_node, off_node_to_off_proc);

        int on_node_num_cols = on_node_column_map.size();
        int off_node_num_cols = off_node_column_map.size();

        gather_off_node_nodes(off_node_col_to_node, recv_nodes);

        std::vector<int> send_procs;
        std::vector<int> recv_procs;
        find_global_comm_procs(recv_nodes, send_procs, 
                recv_procs);

        std::vector<int> orig_nodes;
        form_local_R_par_comm(off_node_column_map, off_node_col_to_node,
                recv_nodes, orig_nodes);

        form_global_par_comm(send_procs, recv_procs, orig_nodes);
        form_local_S_par_comm(first_local_col);
        adjust_send_indices(first_local_row);
        form_local_L_par_comm(on_node_column_map, on_node_col_to_proc,
                first_local_row);

        // TODO there is a much cheaper way to do this
        // I know I recv from each local proc in order, and each recv is ordered
        // so I can figure out idx_R and idx_L for each off_proc_col
        int recv_size = local_R_par_comm->recv_data->size_msgs +
            local_L_par_comm->recv_data->size_msgs;
        if (recv_size)
        {
            recv_buffer.set_size(recv_size);
            R_to_orig.resize(local_R_par_comm->recv_data->size_msgs);
            for (int i = 0; i < local_R_par_comm->recv_data->size_msgs; i++)
            {
                int idx = local_R_par_comm->recv_data->indices[i];
                R_to_orig[i] = off_node_to_off_proc[idx];
            }
            L_to_orig.resize(local_L_par_comm->recv_data->size_msgs);
            for (int i = 0; i < local_L_par_comm->recv_data->size_msgs; i++)
            {
                int idx = local_L_par_comm->recv_data->indices[i];
                L_to_orig[i] = on_node_to_off_proc[idx];
            }
        }
    }

    TAPComm(TAPComm* tap_comm)
    {
        key = tap_comm->key;

        local_S_par_comm = new ParComm(tap_comm->local_S_par_comm);
        local_R_par_comm = new ParComm(tap_comm->local_R_par_comm);
        local_L_par_comm = new ParComm(tap_comm->local_L_par_comm);
        global_par_comm = new ParComm(tap_comm->global_par_comm);
    }

    /**************************************************************
    *****   ParComm Class Destructor
    **************************************************************
    ***** 
    **************************************************************/
    ~TAPComm()
    {
        delete global_par_comm;
        delete local_S_par_comm;
        delete local_R_par_comm;
        delete local_L_par_comm;
    };

    void split_off_proc_cols(std::vector<int>& off_proc_column_map,
            std::vector<int>& off_proc_col_to_proc,
            std::vector<int>& on_node_column_map,
            std::vector<int>& on_node_col_to_proc,
            std::vector<int>& on_node_to_off_proc,
            std::vector<int>& off_node_column_map,
            std::vector<int>& off_node_col_to_node,
            std::vector<int>& off_node_to_off_proc);

    void gather_off_node_nodes(std::vector<int>& off_node_col_to_node,
            std::vector<int>& recv_nodes);

    void find_global_comm_procs(std::vector<int>& recv_nodes,
            std::vector<int>& send_procs, std::vector<int>& recv_procs);

    void form_local_R_par_comm(std::vector<int>& off_node_column_map,
            std::vector<int>& off_node_col_to_node,
            std::vector<int>& recv_nodes,
            std::vector<int>& orig_nodes);

    void form_global_par_comm(std::vector<int>& send_procs,
            std::vector<int>& recv_procs, std::vector<int>& orig_nodes);

    void form_local_S_par_comm(int first_local_col);

    void adjust_send_indices(int first_local_row);

    void form_local_L_par_comm(std::vector<int>& on_node_column_map,
            std::vector<int>& on_node_col_to_proc, int first_local_row);
   
    int get_node(int proc)
    {
        int rank;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);

        if (rank_ordering == 0)
        {
            return proc % num_nodes;
        }
        else if (rank_ordering == 1)
        {
            return proc / PPN;
        }
        else if (rank_ordering == 2)
        {
            if ((proc / num_nodes) % 2 == 0)
            {
                return proc % num_nodes;
            }
            else
            {
                return num_nodes - (proc % num_nodes) - 1;
            }
        }
        else
        { 
            if (rank == 0)
            {
                printf("This MPI rank ordering is not supported!\n");
            }
            return -1;
        }
    }

    int get_local_proc(int proc)
    {
        int rank;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);

        if (rank_ordering == 0 || rank_ordering == 2)
        {
            return proc / num_nodes;
        }
        else if (rank_ordering == 1)
        {
            return proc % PPN;
        }
        else
        { 
            if (rank == 0)
            {
                printf("This MPI rank ordering is not supported!\n");
            }
            return -1;
        }
    }

    int get_node_master(int node)
    {
        int rank;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);

        if (rank_ordering == 0 || rank_ordering == 2)
        {
            return node;
        }
        else if (rank_ordering == 1)
        {
            return node * PPN;
        }
        else
        { 
            if (rank == 0)
            {
                printf("This MPI rank ordering is not supported!\n");
            }
            return -1;
        }
    }

    int get_global_proc(int node, int local_proc)
    {
        int rank;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);

        if (rank_ordering == 0)
        {
            return local_proc * num_nodes + node;
        }
        else if (rank_ordering == 1)
        {
            return local_proc + (node * PPN);
        }
        else if (rank_ordering == 2)
        {
            if ((rank / num_nodes) % 2 == 0)
            {
                return local_proc * num_nodes + node;
            }
            else
            {
                return local_proc * num_nodes + num_nodes - node - 1;                
            }
        }
        else
        { 
            if (rank == 0)
            {
                printf("This MPI rank ordering is not supported!\n");
            }
            return -1;
        }
    }

    int key;

    ParComm* local_S_par_comm;
    ParComm* local_R_par_comm;
    ParComm* local_L_par_comm;
    ParComm* global_par_comm;

    // Vector for combining L and R recv buffers
    Vector recv_buffer;

    // Map L_recv_buffer and R_recv_buffers to original
    // off_proc_column_map 
    std::vector<int> L_to_orig;
    std::vector<int> R_to_orig;

    MPI_Comm local_comm;

    int PPN; 
    int num_nodes;
    int rank_ordering;

};
}
#endif

