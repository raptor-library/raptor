// Copyright (c) 2015, Raptor Developer Team, University of Illinois at Urbana-Champaign
// License: Simplified BSD, http://opensource.org/licenses/BSD-2-Clause
#ifndef RAPTOR_CORE_TAPCOMM_HPP
#define RAPTOR_CORE_TAPCOMM_HPP

#include <mpi.h>
#include "par_comm.hpp"

// When using TAPSpMV, must either export 
// MPICH_RANK_REORDER_METHOD and PPN before running
// program, or set these defines to defaults
#define STANDARD_PPN 16
#define STANDARD_PROC_LAYOUT 1

/**************************************************************
 *****   TAPComm Class
 **************************************************************
 ***** This class constructs a topology-aware communication 
 ***** package, setting up parallel communication 
 *****
 ***** Attributes
 ***** -------------
 ***** local_S_par_comm : ParComm*
 *****    Communication package for initial distribution of values
 *****    among processes local to node
 ***** local_R_par_comm : ParComm*
 *****    Communication package for final redistribution of recvd
 ****     values among processes local to node
 ***** local_L_par_comm : ParComm*
 *****    Communication package for fully local communiation
 *****    (both origin and distination processes are on node)
 ***** global_par_comm : ParComm*
 *****    Communication package for inter-node communication
 ***** recv_buffer : Vector
 *****    Vector in which to return recvd values after communication
 ***** L_to_orig : std::vector<int> 
 *****    Maps on_node cols to off_proc
 ***** R_to_orig : std::vector<int>
 *****    Maps off_node cols to off_proc
 ***** local_comm : MPI_Comm
 *****    MPI Communicator for intra-node communication
 ***** PPN : int
 *****    Number of processes per node, set as an environment
 *****    variable or defaults to defined value
 ***** num_nodes : int
 *****    Number of nodes in partition.  Calculated with num_procs
 *****    and PPN.
 ***** rank_ordering : int
 *****    Method for mapping ranks to nodes.  Assumes SMP style
 *****    unless MPICH_RANK_REORDER_METHOD is set.
 ***** 
 ***** Methods
 ***** -------
 ***** split_off_proc_cols
 ***** gather_off_node_nodes
 ***** find_global_comm_procs
 ***** form_local_R_par_comm
 ***** form_global_par_comm
 ***** form_local_S_par_comm
 ***** adjust_send_indices
 ***** form_local_L_par_comm
 ***** get_node
 ***** get_local_proc
 ***** get_global_proc
 **************************************************************/
namespace raptor
{
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
    TAPComm()
    {
        local_S_par_comm = new ParComm();
        local_R_par_comm = new ParComm();
        local_L_par_comm = new ParComm();
        global_par_comm = new ParComm();
    }

    /**************************************************************
    *****   TAPComm Class Constructor
    **************************************************************
    ***** Initializes a TAPComm object based on the off_proc columns 
    *****
    ***** Parameters
    ***** -------------
    ***** off_proc_column_map : std::vector<int>&
    *****    Maps local off_proc columns indices to global
    ***** first_local_row : int
    *****    Global row index of first row local to process
    ***** first_local_col : int
    *****    Global row index of first column to fall in local block
    ***** global_num_cols : int
    *****    Number of global columns in matrix
    ***** local_num_cols : int
    *****    Number of columns local to rank
    **************************************************************/
    TAPComm(const std::vector<int>& off_proc_column_map,
            const int first_local_row, 
            const int first_local_col,
            const int global_num_cols, 
            const int local_num_cols,
            MPI_Comm comm = MPI_COMM_WORLD)
    {
        // Get MPI Information
        int rank, num_procs;
        MPI_Comm_rank(comm, &rank);
        MPI_Comm_size(comm, &num_procs);

        // Initialize class variables
        local_S_par_comm = new ParComm(2345);
        local_R_par_comm = new ParComm(3456);
        local_L_par_comm = new ParComm(4567);
        global_par_comm = new ParComm(5678);

        // Initialize Variables
        int global_col;
        int local_rank;
        int idx;
        int rank_node;
        int recv_size;
        std::vector<int> off_proc_col_to_proc;
        std::vector<int> on_node_column_map;
        std::vector<int> on_node_col_to_proc;
        std::vector<int> off_node_column_map;
        std::vector<int> off_node_col_to_node;
        std::vector<int> on_node_to_off_proc;
        std::vector<int> off_node_to_off_proc;
        std::vector<int> recv_nodes;
        std::vector<int> send_procs;
        std::vector<int> recv_procs;
        std::vector<int> orig_nodes;

        // Map procs to nodes -- Topology Aware Portion
        char* proc_layout_c = std::getenv("MPICH_RANK_REORDER_METHOD");
        char* PPN_c = std::getenv("PPN");
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
        MPI_Comm_rank(local_comm, &local_rank);

        // Find process on which vector value associated with each column is
        // stored
        global_par_comm->form_col_to_proc(first_local_col, global_num_cols,
            local_num_cols, off_proc_column_map, off_proc_col_to_proc);

        // Partition off_proc cols into on_node and off_node
        split_off_proc_cols(off_proc_column_map, off_proc_col_to_proc,
               on_node_column_map, on_node_col_to_proc, on_node_to_off_proc,
               off_node_column_map, off_node_col_to_node, off_node_to_off_proc);

        // Gather all nodes with which any local process must communication
        gather_off_node_nodes(off_node_col_to_node, recv_nodes);

        // Find global processes with which rank communications
        find_global_comm_procs(recv_nodes, send_procs, recv_procs);

        // Form local_R_par_comm: communication for redistribution of inter-node
        //  communication        
        form_local_R_par_comm(off_node_column_map, off_node_col_to_node,
                recv_nodes, orig_nodes);

        // Form inter-node communication
        form_global_par_comm(send_procs, recv_procs, orig_nodes);

        // Form local_S_par_comm: initial distribution of values among local
        // processes, before inter-node communication
        form_local_S_par_comm(first_local_col);

        // Adjust send indices (currently global vector indices) to be index 
        // of global vector value from previous recv
        adjust_send_indices(first_local_row);

        // Form local_L_par_comm: fully local communication (origin and
        // destination processes both local to node)
        form_local_L_par_comm(on_node_column_map, on_node_col_to_proc,
                first_local_row);

        // Determine size of final recvs (should be equal to 
        // number of off_proc cols)
        recv_size = local_R_par_comm->recv_data->size_msgs +
            local_L_par_comm->recv_data->size_msgs;
        if (recv_size)
        {
            // Want a single recv buffer local_R and local_L par_comms
            recv_buffer.set_size(recv_size);

            // Map local_R recvs to original off_proc_column_map
            R_to_orig.resize(local_R_par_comm->recv_data->size_msgs);
            for (int i = 0; i < local_R_par_comm->recv_data->size_msgs; i++)
            {
                idx = local_R_par_comm->recv_data->indices[i];
                R_to_orig[i] = off_node_to_off_proc[idx];
            }

            // Map local_L recvs to original off_proc_column_map
            L_to_orig.resize(local_L_par_comm->recv_data->size_msgs);
            for (int i = 0; i < local_L_par_comm->recv_data->size_msgs; i++)
            {
                idx = local_L_par_comm->recv_data->indices[i];
                L_to_orig[i] = on_node_to_off_proc[idx];
            }
        }
    }

    /**************************************************************
    *****   TAPComm Class Constructor
    **************************************************************
    ***** Create topology-aware communication class from 
    ***** original communication package (which processes rank
    ***** communication which, and what is sent to / recv from
    ***** each process.
    *****
    ***** Parameters
    ***** -------------
    ***** orig_comm : ParComm*
    *****    Existing standard communication package from which
    *****    to form topology-aware communicator
    **************************************************************/
    TAPComm(ParComm* orig_comm)
    {
        //TODO -- Write this constructor
    }

    
    /**************************************************************
    *****   TAPComm Class Constructor
    **************************************************************
    ***** Deep copy of an existing TAPComm object
    *****
    ***** Parameters
    ***** -------------
    ***** tap_comm : TAPComm*
    *****    Existing topology-aware communication package to copy
    **************************************************************/
    TAPComm(TAPComm* tap_comm)
    {
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
    ***** off_proc_column_map : std::vector<int>&
    *****    Vector holding rank's off_proc_columns
    ***** off_proc_col_to_proc : std::vector<int>&
    *****    Vector mapping rank's off_proc_columns to distant procs
    ***** on_node_column_map : std::vector<int>&
    *****    Will be returned holding on_node columns
    ***** on_node_col_to_proc : std::vector<int>&
    *****    Will be returned holding procs corresponding to on_node cols
    ***** on_node_to_off_proc : std::vector<int>&
    *****    Will be returned holding map from on_node to off_proc
    ***** off_node_column_map : std::vector<int>&
    *****    Will be returned holding off_node columns
    ***** off_node_col_to_node : std::vector<int>&
    *****    Will be returned holding procs corresponding to off_node cols
    ***** off_node_to_off_proc : std::vector<int>&
    *****    Will be returned holding map from off_node to off_proc
    **************************************************************/
    void split_off_proc_cols(const std::vector<int>& off_proc_column_map,
            const std::vector<int>& off_proc_col_to_proc,
            std::vector<int>& on_node_column_map,
            std::vector<int>& on_node_col_to_proc,
            std::vector<int>& on_node_to_off_proc,
            std::vector<int>& off_node_column_map,
            std::vector<int>& off_node_col_to_node,
            std::vector<int>& off_node_to_off_proc);

    /**************************************************************
    *****   Gather off node nodes
    **************************************************************
    ***** Gathers nodes with which any local processes communicates
    *****
    ***** Parameters
    ***** -------------
    ***** off_node_col_to_node : std::vector<int>&
    *****    Vector holding rank's off_node_columns
    ***** recv_nodes : std::vector<int>&
    *****    Returned holding all nodes with which any local
    *****    process communicates (union of off_node_col_to_node)
    **************************************************************/
    void gather_off_node_nodes(const std::vector<int>& off_node_col_to_node,
            std::vector<int>& recv_nodes);

    /**************************************************************
    *****   Find global comm procs
    **************************************************************
    ***** Determine which processes with which rank will communicate
    ***** during inter-node communication
    *****
    ***** Parameters
    ***** -------------
    ***** recv_nodes : std::vector<int>&
    *****    All nodes with which any local process communicates 
    ***** send_procs : std::vector<int>&
    *****    Returns with all off_node processes to which rank sends
    ***** recv_procs : std::vector<int>&
    *****    Returns wiht all off_node process from which rank recvs
    **************************************************************/
    void find_global_comm_procs(const std::vector<int>& recv_nodes,
            std::vector<int>& send_procs, std::vector<int>& recv_procs);

    /**************************************************************
    *****   Form local_R_par_comm
    **************************************************************
    ***** Find which local processes recv needed vector values
    ***** from inter-node communication
    *****
    ***** Parameters
    ***** -------------
    ***** off_node_column_map : std::vector<int>&
    *****    Columns that correspond to values stored off_node 
    ***** off_node_col_to_node : std::vector<int>&
    *****    Nodes corresponding to each value in off_node_column_map
    ***** recv_nodes : std::vector<int>&
    *****    All nodes with which any local process communicates 
    ***** orig_nodes : std::vector<int>&
    *****    Returns nodes on which local_R_par_comm->send_data->indices
    *****    originate (needed in forming global communication)
    **************************************************************/
    void form_local_R_par_comm(const std::vector<int>& off_node_column_map,
            const std::vector<int>& off_node_col_to_node,
            const std::vector<int>& recv_nodes,
            std::vector<int>& orig_nodes);

    /**************************************************************
    *****   Form global_par_comm
    **************************************************************
    ***** Form global communication package (for inter-node comm)
    *****
    ***** Parameters
    ***** -------------
    ***** send_procs : std::vector<int>&
    *****    Off_node processes to which rank sends 
    ***** recv_procs : std::vector<int>&
    *****    Off_node processes from which rank recvs
    ***** orig_nodes : std::vector<int>&
    *****    Nodes for each index in local_R_par_comm sends 
    **************************************************************/
    void form_global_par_comm(const std::vector<int>& send_procs,
            const std::vector<int>& recv_procs, const std::vector<int>& orig_nodes);

    /**************************************************************
    *****   Form local_S_par_comm
    **************************************************************
    ***** Find which local processes the values originating on rank
    ***** must be sent to, and which processes store values rank must
    ***** send as inter-node communication.
    *****
    ***** Parameters
    ***** -------------
    ***** first_local_col : int
    *****    First column local to rank 
    **************************************************************/
    void form_local_S_par_comm(const int first_local_col);

    /**************************************************************
    *****   Adjust Send Indices
    **************************************************************
    ***** Adjust send indices from global row index to index of 
    ***** global column in previous recv buffer.  
    *****
    ***** Parameters
    ***** -------------
    ***** first_local_row : int
    *****    First row local to rank 
    **************************************************************/
    void adjust_send_indices(const int first_local_row);

    /**************************************************************
    *****  Form local_L_par_comm 
    **************************************************************
    ***** Adjust send indices from global row index to index of 
    ***** global column in previous recv buffer.  
    *****
    ***** Parameters
    ***** -------------
    ***** on_node_column_map : std::vector<int>&
    *****    Columns corresponding to on_node processes
    ***** on_node_col_to_proc : std::vector<int>&
    *****    On node process corresponding to each column
    *****    in on_node_column_map
    ***** first_local_row : int
    *****    First row local to rank 
    **************************************************************/
    void form_local_L_par_comm(const std::vector<int>& on_node_column_map,
            const std::vector<int>& on_node_col_to_proc, const int first_local_row);
   
    /**************************************************************
    *****  Get Node 
    **************************************************************
    ***** Find node on which global rank lies
    *****
    ***** Returns
    ***** -------------
    ***** int : node on which proc lies
    *****
    ***** Parameters
    ***** -------------
    ***** proc : int
    *****    Global rank of process 
    **************************************************************/
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

    /**************************************************************
    *****  Get Local Proc 
    **************************************************************
    ***** Find rank local to node from global rank
    *****
    ***** Returns
    ***** -------------
    ***** int : rank local to processes on node
    *****
    ***** Parameters
    ***** -------------
    ***** proc : int
    *****    Global rank of process 
    **************************************************************/
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

    /**************************************************************
    *****  Get Global Proc 
    **************************************************************
    ***** Find global rank from node and local rank
    *****
    ***** Returns
    ***** -------------
    ***** int : Global rank of process
    *****
    ***** Parameters
    ***** -------------
    ***** node : int
    *****    Node on which process lies 
    ***** local_proc : int
    *****    Rank of process local to node
    **************************************************************/
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

    ParComm* local_S_par_comm;
    ParComm* local_R_par_comm;
    ParComm* local_L_par_comm;
    ParComm* global_par_comm;

    Vector recv_buffer;

    std::vector<int> L_to_orig;
    std::vector<int> R_to_orig;

    MPI_Comm local_comm;

    int PPN; 
    int num_nodes;
    int rank_ordering;

};
}
#endif

