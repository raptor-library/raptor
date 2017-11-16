// Copyright (c) 2015-2017, RAPtor Developer Team
// License: Simplified BSD, http://opensource.org/licenses/BSD-2-Clause
#ifndef TOPOLOGY_HPP 
#define TOPOLOGY_HPP

#include <mpi.h>
#include <math.h>
#include <set>

#include "types.hpp"

/**************************************************************
 *****   Topology Class
 **************************************************************
 ***** This class holds information about the topology of
 ***** the parallel computer on which Raptor is being run
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
  class Topology
  {
  public:
    Topology(int _PPN = 16, int _standard_rank_ordering = 1)
    {     
        int rank, num_procs;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

        int rank_node;

        char* proc_layout_c = getenv("MPICH_RANK_REORDER_METHOD");
        char* PPN_c = getenv("PPN");
        if (PPN_c) 
        {
            PPN = atoi(PPN_c);
        }
        else
        {
            PPN = _PPN;
        }

        if (proc_layout_c)
        {
            rank_ordering = atoi(proc_layout_c);
        }
        else
        {
            rank_ordering = _standard_rank_ordering;
        }

        num_nodes = num_procs / PPN;
        if (num_procs % PPN) num_nodes++;
        rank_node = get_node(rank);

        // Create intra-node communicator
        MPI_Comm_split(MPI_COMM_WORLD, rank_node, rank, &local_comm);
        num_shared = 0;
    }

    ~Topology()
    {
        MPI_Comm_free(&local_comm);
    }

    int get_node(int proc)
    {
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
            int rank;
            MPI_Comm_rank(MPI_COMM_WORLD, &rank);
            if (rank == 0)
            {
                printf("This MPI rank ordering is not supported!\n");
            }
            return -1;
        }
    }

    int get_local_proc(int proc)
    {
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
            int rank;
            MPI_Comm_rank(MPI_COMM_WORLD, &rank);
            if (rank == 0)
            {
                printf("This MPI rank ordering is not supported!\n");
            }
            return -1;
        }
    }

    int get_global_proc(int node, int local_proc)
    {
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
            if (local_proc % 2 == 0)
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
            int rank;
            MPI_Comm_rank(MPI_COMM_WORLD, &rank);
            if (rank == 0)
            {
                printf("This MPI rank ordering is not supported!\n");
            }
            return -1;
        }
    }

    int PPN;
    int rank_ordering;
    int num_shared;
    int num_nodes;

    MPI_Comm local_comm;
    MPI_Comm topology_comm;
  };
}

#endif
